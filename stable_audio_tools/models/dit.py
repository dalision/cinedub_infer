# Adapted from Stability AI's stable-audio-tools (MIT).
# See LICENSES/LICENSE_STABILITY.txt at the repository root.
import typing as tp
import math
import torch

from einops import rearrange
from torch import nn
from torch.nn import functional as F

from .blocks import FourierFeatures
from .transformer import ContinuousTransformer

from copy import deepcopy

class DiffusionTransformer(nn.Module):
    def __init__(self,
        io_channels=32,
        patch_size=1,
        embed_dim=768,
        cond_token_dim=0,
        project_cond_tokens=True,
        add_cond_token_dim=0,
        project_add_cond_tokens=True,
        global_cond_dim=0,
        additional_cond_dim=0,
        project_global_cond=True,
        project_additional_tokens=True,
        input_concat_dim=0,
        prepend_cond_dim=0,
        depth=12,
        num_heads=8,
        transformer_type: tp.Literal["x-transformers", "continuous_transformer"] = "x-transformers",
        global_cond_type: tp.Literal["prepend", "adaLN"] = "prepend",
        timestep_cond_type: tp.Literal["global", "input_concat"] = "global",
        timestep_embed_dim=None,
        diffusion_objective: tp.Literal["v", "rectified_flow"] = "v",
        **kwargs):

        super().__init__()
        
        self.cond_token_dim = cond_token_dim
        self.add_cond_token_dim = add_cond_token_dim

        # Timestep embeddings
        self.timestep_cond_type = timestep_cond_type

        timestep_features_dim = 256

        self.timestep_features = FourierFeatures(1, timestep_features_dim)

        if timestep_cond_type == "global":
            timestep_embed_dim = embed_dim
        elif timestep_cond_type == "input_concat":
            assert timestep_embed_dim is not None, "timestep_embed_dim must be specified if timestep_cond_type is input_concat"
            input_concat_dim += timestep_embed_dim

        self.to_timestep_embed = nn.Sequential(
            nn.Linear(timestep_features_dim, timestep_embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(timestep_embed_dim, timestep_embed_dim, bias=True),
        )
        
        if cond_token_dim > 0:
            # Conditioning tokens
            cond_embed_dim = cond_token_dim if not project_cond_tokens else embed_dim
            self.to_cond_embed = nn.Sequential(
                nn.Linear(cond_token_dim, cond_embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(cond_embed_dim, cond_embed_dim, bias=False)
            )
        else:
            cond_embed_dim = 0

        # Additional cross attention conditioning tokens (e.g., for trans_cap)
        if add_cond_token_dim > 0:
            add_cond_embed_dim = add_cond_token_dim if not project_add_cond_tokens else embed_dim
            self.to_add_cond_embed = nn.Sequential(
                nn.Linear(add_cond_token_dim, add_cond_embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(add_cond_embed_dim, add_cond_embed_dim, bias=False)
            )
        else:
            add_cond_embed_dim = 0

        if global_cond_dim > 0:
            # Global conditioning
            global_embed_dim = global_cond_dim if not project_global_cond else embed_dim
            self.to_global_embed = nn.Sequential(
                nn.Linear(global_cond_dim, global_embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(global_embed_dim, global_embed_dim, bias=False)
            )

        # for syncformer feature
        if additional_cond_dim > 0:
            additional_embed_dim = additional_cond_dim if not project_additional_tokens else embed_dim
            self.to_additional_embed = nn.Sequential(
                nn.Linear(additional_cond_dim, additional_embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(additional_embed_dim, additional_embed_dim, bias=False)
            )
            
        if prepend_cond_dim > 0:
            # Prepend conditioning
            self.to_prepend_embed = nn.Sequential(
                nn.Linear(prepend_cond_dim, embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim, bias=False)
            )

        self.input_concat_dim = input_concat_dim

        dim_in = io_channels + self.input_concat_dim

        self.patch_size = patch_size

        # Transformer

        self.transformer_type = transformer_type

        self.global_cond_type = global_cond_type

        if self.transformer_type == "x-transformers":
            
            from x_transformers import ContinuousTransformerWrapper, Encoder
            self.transformer = ContinuousTransformerWrapper(
                dim_in=dim_in * patch_size,
                dim_out=io_channels * patch_size,
                max_seq_len=0, #Not relevant without absolute positional embeds
                attn_layers = Encoder(
                    dim=embed_dim,
                    depth=depth,
                    heads=num_heads,
                    attn_flash = True,
                    cross_attend = cond_token_dim > 0,
                    dim_context=None if cond_embed_dim == 0 else cond_embed_dim,
                    zero_init_branch_output=True,
                    use_abs_pos_emb = False,
                    rotary_pos_emb=True,
                    ff_swish = True,
                    ff_glu = True,
                    **kwargs
                )
            )
        
        elif self.transformer_type == "continuous_transformer":
            global_dim = None

            if self.global_cond_type == "adaLN" or hasattr(self, 'to_additional_embed'):
                # The global conditioning is projected to the embed_dim already at this point
                # initial adaLN for additional condition or global condition
                global_dim = embed_dim

            self.transformer = ContinuousTransformer(
                dim=embed_dim,
                depth=depth,
                dim_heads=embed_dim // num_heads,
                dim_in=dim_in * patch_size,
                dim_out=io_channels * patch_size,
                cross_attend = cond_token_dim > 0,
                cond_token_dim = cond_embed_dim,
                add_cross_attend = add_cond_token_dim > 0,
                add_cond_token_dim = add_cond_embed_dim,
                global_cond_dim=global_dim,
                **kwargs
            )
        
        else:
            raise ValueError(f"Unknown transformer type: {self.transformer_type}")

        self.preprocess_conv = nn.Conv1d(dim_in, dim_in, 1, bias=False)
        nn.init.zeros_(self.preprocess_conv.weight)
        self.postprocess_conv = nn.Conv1d(io_channels, io_channels, 1, bias=False)
        nn.init.zeros_(self.postprocess_conv.weight)

        self.diffusion_objective = diffusion_objective

    def _forward(
        self,
        x,
        t,
        mask=None,
        cross_attn_cond=None,
        cross_attn_cond_mask=None,
        add_cross_attn_cond=None,
        add_cross_attn_cond_mask=None,
        input_concat_cond=None,
        global_embed=None,
        additional_embed=None,
        prepend_cond=None,
        prepend_cond_mask=None,
        return_info=False,
        **kwargs):

        if cross_attn_cond is not None:
            cross_attn_cond = self.to_cond_embed(cross_attn_cond)

        if add_cross_attn_cond is not None:
            add_cross_attn_cond = self.to_add_cond_embed(add_cross_attn_cond)

        if global_embed is not None:
            # Project the global conditioning to the embedding dimension
            global_embed = self.to_global_embed(global_embed)
        if additional_embed is not None:
            # Project the additional conditioning to the embedding dimension
            additional_embed = self.to_additional_embed(additional_embed)

        prepend_inputs = None 
        prepend_mask = None
        prepend_length = 0
        if prepend_cond is not None:
            # PretransformConditioner outputs [B, C, T], convert to [B, T, C] for Linear
            prepend_cond = prepend_cond.permute(0, 2, 1)
            prepend_cond = self.to_prepend_embed(prepend_cond)
            
            prepend_inputs = prepend_cond
            if prepend_cond_mask is not None:
                prepend_mask = prepend_cond_mask

        if input_concat_cond is not None:
            # Interpolate input_concat_cond to the same length as x
            if input_concat_cond.shape[2] != x.shape[2]: #B,D,T
                input_concat_cond = F.interpolate(input_concat_cond, (x.shape[2], ), mode='nearest')

            x = torch.cat([x, input_concat_cond], dim=1)

        # Get the batch of timestep embeddings
        timestep_embed = self.to_timestep_embed(self.timestep_features(t[:, None])) # (b, embed_dim)

        # Timestep embedding is considered a global embedding. Add to the global conditioning if it exists
        if self.timestep_cond_type == "global":
            if global_embed is not None:
                global_embed = global_embed + timestep_embed
            else:
                global_embed = timestep_embed
        elif self.timestep_cond_type == "input_concat":
            x = torch.cat([x, timestep_embed.unsqueeze(1).expand(-1, -1, x.shape[2])], dim=1)

        # Add the global_embed to the prepend inputs if there is no global conditioning support in the transformer
        if self.global_cond_type == "prepend" and global_embed is not None:
            if prepend_inputs is None:
                # Prepend inputs are just the global embed, and the mask is all ones
                prepend_inputs = global_embed.unsqueeze(1)
                prepend_mask = torch.ones((x.shape[0], 1), device=x.device, dtype=torch.bool)
            else:
                # Prepend inputs are the prepend conditioning + the global embed
                prepend_inputs = torch.cat([prepend_inputs, global_embed.unsqueeze(1)], dim=1)
                prepend_mask = torch.cat([prepend_mask, torch.ones((x.shape[0], 1), device=x.device, dtype=torch.bool)], dim=1)
            prepend_length = prepend_inputs.shape[1]
            extra_args = {"prepend_length": prepend_length}
        x = self.preprocess_conv(x) + x

        x = rearrange(x, "b c t -> b t c")

        extra_args = {}

        assert self.patch_size == 1, "Patch size must be 1 for VT2A DiffusionTransformer"
        
        if self.global_cond_type == "adaLN":
            assert global_embed.shape[1] == 1 or self.global_embed.shape[1] == x.shape[1], f"keep the global embed length {global_embed.shape} the same as the x {x.shape}"
            extra_args["global_cond"] = global_embed
        
        # for syncformer feature
        if additional_embed is not None:
            assert additional_embed.shape[1] == 1 or additional_embed.shape[1] == x.shape[1], f"keep the additional embed length {prepend_length.shape} the same as the x {x.shape}"
            extra_args["global_cond"] = extra_args["global_cond"] + additional_embed if "global_cond" in extra_args else additional_embed

        # 1. 通过在additional_embed前面padding 0的方式来处理序列长度不匹配的问题；这里我将sync feature通过additional_embed传入 global embed, 并且因为prepend_emb的问题，我把global embed前面填充prepend_length个0，来让他和x对齐；
        # 2. 对于concat_inputs的处理已经在上面做了，就是直接concat到x上面,他会在continueous transformer的输入端进行线性映射，到io_channels维度所以不会影响；
        # 3. 注意input_concat_cond是step-wise的，addition cat是block-wise
        # padding to x length
        if "global_cond" in extra_args:
            if extra_args["global_cond"].shape[1] != 1 and prepend_length != 0:
                extra_args["global_cond"] = F.pad(extra_args["global_cond"], (0, 0, prepend_length, 0), mode='constant', value=0)

        if self.patch_size > 1:
            x = rearrange(x, "b (t p) c -> b t (c p)", p=self.patch_size)

        if self.transformer_type == "x-transformers":
            output = self.transformer(x, prepend_embeds=prepend_inputs, context=cross_attn_cond, context_mask=cross_attn_cond_mask, mask=mask, prepend_mask=prepend_mask, **extra_args, **kwargs)
        elif self.transformer_type == "continuous_transformer":
            output = self.transformer(x, prepend_embeds=prepend_inputs, context=cross_attn_cond, context_mask=cross_attn_cond_mask, add_context=add_cross_attn_cond, add_context_mask=add_cross_attn_cond_mask, mask=mask, prepend_mask=prepend_mask, return_info=return_info, **extra_args, **kwargs)

            if return_info:
                output, info = output
        output = rearrange(output, "b t c -> b c t")[:,:,prepend_length:]

        if self.patch_size > 1:
            output = rearrange(output, "b (c p) t -> b c (t p)", p=self.patch_size)

        output = self.postprocess_conv(output) + output

        if return_info:
            return output, info

        return output

    def forward(
        self,
        x,
        t,
        cross_attn_cond=None,
        cross_attn_cond_mask=None,
        negative_cross_attn_cond=None,
        negative_cross_attn_mask=None,
        add_cross_attn_cond=None,
        add_cross_attn_cond_mask=None,
        negative_add_cross_attn_cond=None,
        negative_add_cross_attn_mask=None,
        input_concat_cond=None,
        global_embed=None,
        negative_global_embed=None,
        additional_embed=None,
        prepend_cond=None,
        prepend_cond_mask=None,
        cfg_scale=1.0,
        cfg_dropout_prob=0.0,
        cfg_interval = (0, 1),
        causal=False,
        scale_phi=0.0,
        mask=None,
        return_info=False,
        **kwargs):

        assert causal == False, "Causal mode is not supported for DiffusionTransformer"

        model_dtype = next(self.parameters()).dtype
        
        x = x.to(model_dtype)

        t = t.to(model_dtype)

        if cross_attn_cond is not None:
            cross_attn_cond = cross_attn_cond.to(model_dtype)

        if negative_cross_attn_cond is not None:
            negative_cross_attn_cond = negative_cross_attn_cond.to(model_dtype)

        if add_cross_attn_cond is not None:
            add_cross_attn_cond = add_cross_attn_cond.to(model_dtype)

        if negative_add_cross_attn_cond is not None:
            negative_add_cross_attn_cond = negative_add_cross_attn_cond.to(model_dtype)

        if input_concat_cond is not None:
            input_concat_cond = input_concat_cond.to(model_dtype)
        
        if global_embed is not None:
            global_embed = global_embed.to(model_dtype)

        if additional_embed is not None:
            additional_embed = additional_embed.to(model_dtype)

        if negative_global_embed is not None:
            negative_global_embed = negative_global_embed.to(model_dtype)

        if prepend_cond is not None:
            prepend_cond = prepend_cond.to(model_dtype)

        if cross_attn_cond_mask is not None:
            cross_attn_cond_mask = cross_attn_cond_mask.bool()
            cross_attn_cond_mask = None # Temporarily disabling conditioning masks due to kernel issue for flash attention

        if add_cross_attn_cond_mask is not None:
            add_cross_attn_cond_mask = add_cross_attn_cond_mask.bool()
            add_cross_attn_cond_mask = None # Temporarily disabling conditioning masks due to kernel issue for flash attention

        if prepend_cond_mask is not None:
            prepend_cond_mask = prepend_cond_mask.bool()
    
        # CFG dropout
        if cfg_dropout_prob > 0.0:
            dropout_mask = torch.bernoulli(
                torch.full((x.shape[0], 1, 1), cfg_dropout_prob, device=x.device)).to(torch.bool)

            if cross_attn_cond is not None:
                null_embed = torch.zeros_like(cross_attn_cond, device=cross_attn_cond.device)
                cross_attn_cond = torch.where(dropout_mask, null_embed, cross_attn_cond)

            if add_cross_attn_cond is not None:
                null_embed = torch.zeros_like(add_cross_attn_cond, device=add_cross_attn_cond.device)
                add_cross_attn_cond = torch.where(dropout_mask, null_embed, add_cross_attn_cond)

            if prepend_cond is not None:
                null_embed = torch.zeros_like(prepend_cond, device=prepend_cond.device)
                prepend_cond = torch.where(dropout_mask, null_embed, prepend_cond)

            if additional_embed is not None:
                null_embed = torch.zeros_like(additional_embed, device=additional_embed.device)
                additional_embed = torch.where(dropout_mask, null_embed, additional_embed)

        # Get the current t value from the first timestep
        step_t = t[0]

        if self.diffusion_objective == "v":
            sigma = torch.sin(step_t * math.pi / 2)
        elif self.diffusion_objective == "rectified_flow":
            sigma = step_t

        if cfg_scale != 1.0 and (cross_attn_cond is not None or prepend_cond is not None) and (cfg_interval[0] <= sigma <= cfg_interval[1]):
            # Classifier-free guidance
            # Concatenate conditioned and unconditioned inputs on the batch dimension            
            batch_inputs = torch.cat([x, x], dim=0)
            batch_timestep = torch.cat([t, t], dim=0)

            if global_embed is not None:
                batch_global_cond = torch.cat([global_embed, global_embed], dim=0)
            else:
                batch_global_cond = None

            if additional_embed is not None:
                null_embed = torch.zeros_like(additional_embed, device=additional_embed.device)
                batch_additional_cond = torch.cat([additional_embed, null_embed], dim=0)
            else:
                batch_additional_cond = None

            if input_concat_cond is not None:
                batch_input_concat_cond = torch.cat([input_concat_cond, input_concat_cond], dim=0)
            else:
                batch_input_concat_cond = None

            batch_cond = None
            batch_cond_masks = None
            
            # Handle CFG for cross-attention conditioning
            if cross_attn_cond is not None:

                null_embed = torch.zeros_like(cross_attn_cond, device=cross_attn_cond.device)

                # For negative cross-attention conditioning, replace the null embed with the negative cross-attention conditioning
                if negative_cross_attn_cond is not None:

                    # If there's a negative cross-attention mask, set the masked tokens to the null embed
                    if negative_cross_attn_mask is not None:
                        negative_cross_attn_mask = negative_cross_attn_mask.to(torch.bool).unsqueeze(2)

                        negative_cross_attn_cond = torch.where(negative_cross_attn_mask, negative_cross_attn_cond, null_embed)
                    
                    batch_cond = torch.cat([cross_attn_cond, negative_cross_attn_cond], dim=0)
                    # batch_cond=torch.cat([cross_attn_cond,cross_attn_cond],dim=0)

                else:
                    batch_cond = torch.cat([cross_attn_cond, null_embed], dim=0)
                    # batch_cond=torch.cat([cross_attn_cond,cross_attn_cond],dim=0)

                if cross_attn_cond_mask is not None:
                    batch_cond_masks = torch.cat([cross_attn_cond_mask, cross_attn_cond_mask], dim=0)
               
            batch_prepend_cond = None
            batch_prepend_cond_mask = None

            if prepend_cond is not None:

                null_embed = torch.zeros_like(prepend_cond, device=prepend_cond.device)

                batch_prepend_cond = torch.cat([prepend_cond, null_embed], dim=0)

                if prepend_cond_mask is not None:
                    batch_prepend_cond_mask = torch.cat([prepend_cond_mask, prepend_cond_mask], dim=0)

            # Handle CFG for additional cross-attention conditioning
            batch_add_cond = None
            batch_add_cond_masks = None

            if add_cross_attn_cond is not None:
                null_embed = torch.zeros_like(add_cross_attn_cond, device=add_cross_attn_cond.device)

                # For negative add cross-attention conditioning, replace the null embed with the negative conditioning
                if negative_add_cross_attn_cond is not None:
                    if negative_add_cross_attn_mask is not None:
                        negative_add_cross_attn_mask = negative_add_cross_attn_mask.to(torch.bool).unsqueeze(2)
                        negative_add_cross_attn_cond = torch.where(negative_add_cross_attn_mask, negative_add_cross_attn_cond, null_embed)

                    batch_add_cond = torch.cat([add_cross_attn_cond, negative_add_cross_attn_cond], dim=0)
                else:
                    batch_add_cond = torch.cat([add_cross_attn_cond, null_embed], dim=0)

                if add_cross_attn_cond_mask is not None:
                    batch_add_cond_masks = torch.cat([add_cross_attn_cond_mask, add_cross_attn_cond_mask], dim=0)

            if mask is not None:
                batch_masks = torch.cat([mask, mask], dim=0)
            else:
                batch_masks = None

            batch_output = self._forward(
                batch_inputs,
                batch_timestep,
                cross_attn_cond=batch_cond,
                cross_attn_cond_mask=batch_cond_masks,
                add_cross_attn_cond=batch_add_cond,
                add_cross_attn_cond_mask=batch_add_cond_masks,
                mask = batch_masks,
                input_concat_cond=batch_input_concat_cond,
                global_embed = batch_global_cond,
                additional_embed = batch_additional_cond,
                prepend_cond = batch_prepend_cond,
                prepend_cond_mask = batch_prepend_cond_mask,
                return_info = return_info,
                **kwargs)


            if return_info:
                batch_output, info = batch_output

            cond_output, uncond_output = torch.chunk(batch_output, 2, dim=0)

            cfg_output = uncond_output + (cond_output - uncond_output) * cfg_scale

            # CFG Rescale
            if scale_phi != 0.0:
                cond_out_std = cond_output.std(dim=1, keepdim=True)
                out_cfg_std = cfg_output.std(dim=1, keepdim=True)
                output = scale_phi * (cfg_output * (cond_out_std/out_cfg_std)) + (1-scale_phi) * cfg_output
            else:
                output = cfg_output
           
            if return_info:
                info["uncond_output"] = uncond_output
                return output, info
            return output
            
        else:
            output = self._forward(
                x,
                t,
                cross_attn_cond=cross_attn_cond,
                cross_attn_cond_mask=cross_attn_cond_mask,
                add_cross_attn_cond=add_cross_attn_cond,
                add_cross_attn_cond_mask=add_cross_attn_cond_mask,
                input_concat_cond=input_concat_cond,
                global_embed = global_embed,
                additional_embed = additional_embed,
                prepend_cond=prepend_cond,
                prepend_cond_mask=prepend_cond_mask,
                mask=mask,
                return_info=return_info,
                **kwargs
            )

            return output


class DiffusionTransformerCineDub(nn.Module):
    def __init__(self, 
        bad_model:DiffusionTransformer,
        io_channels=32, 
        patch_size=1,
        embed_dim=768,
        cond_token_dim=0,
        project_cond_tokens=True,
        global_cond_dim=0,
        project_global_cond=True,
        input_concat_dim=0,
        prepend_cond_dim=0,
        depth=12,
        num_heads=8,
        transformer_type: tp.Literal["x-transformers", "continuous_transformer"] = "x-transformers",
        global_cond_type: tp.Literal["prepend", "adaLN"] = "prepend",
        timestep_cond_type: tp.Literal["global", "input_concat"] = "global",
        timestep_embed_dim=None,
        diffusion_objective: tp.Literal["v", "rectified_flow"] = "v",
        **kwargs):

        super().__init__()
        
        pass

    def forward(
        self, 
        x, 
        t, 
        cross_attn_cond=None,
        cross_attn_cond_mask=None,
        negative_cross_attn_cond=None,
        negative_cross_attn_cond_mask=None,
        input_concat_cond=None,
        bad_cross_attn_cond=None,
        bad_cross_attn_cond_mask=None,
        bad_negative_cross_attn_cond=None,
        bad_negative_cross_attn_cond_mask=None,
        bad_input_concat_cond=None,
        bad_prepend_cond=None,
        bad_prepend_cond_mask=None,
        bad_global_embed=None,
        bad_negative_global_embed=None,
        global_embed=None,
        negative_global_embed=None,
        prepend_cond=None,
        prepend_cond_mask=None,
        cfg_scale=1.0,
        bad_cfg_scale=1.0,
        guidance_scale=1.0,
        cfg_dropout_prob=0.0,
        cfg_interval = (0, 1),
        causal=False,
        scale_phi=0.0,
        mask=None,
        return_info=False,
        **kwargs):

        pass




