from typing import Literal, Optional

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from open_clip import create_model_from_pretrained
from torchvision.transforms import Normalize
import os

# Optional MMAudio dependency: only needed if you want VAE / mel converter /
# Synchformer utilities from that repo. CineDub inference does not exercise
# them, so import failure is deferred until (and if) those code paths are
# actually invoked.
try:
    from mmaudio.ext.autoencoder import AutoEncoderModule
    from mmaudio.ext.mel_converter import get_mel_converter
    from mmaudio.ext.synchformer import Synchformer
    from mmaudio.model.utils.distributions import DiagonalGaussianDistribution
except ImportError:
    AutoEncoderModule = None
    get_mel_converter = None
    Synchformer = None
    DiagonalGaussianDistribution = None


# Local override for the DFN5B CLIP checkpoint: set CINEDUB_CLIP_LOCAL to a
# directory or the exact `open_clip_pytorch_model.bin` path if you have
# downloaded it locally (avoids re-hitting the HF hub on every launch).
_CINEDUB_CLIP_LOCAL = os.environ.get("CINEDUB_CLIP_LOCAL", None)


def patch_clip(clip_model):
    # a hack to make it output last hidden states
    # https://github.com/mlfoundations/open_clip/blob/fc5a37b72d705f760ebbc7915b84729816ed471f/src/open_clip/model.py#L269
    def new_encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        return F.normalize(x, dim=-1) if normalize else x

    clip_model.encode_text = new_encode_text.__get__(clip_model)
    return clip_model


class FeaturesUtils(nn.Module):

    def __init__(
        self,
        *,
        tod_vae_ckpt: Optional[str] = None,
        bigvgan_vocoder_ckpt: Optional[str] = None,
        synchformer_ckpt: Optional[str] = None,
        enable_conditions: bool = True,
        mode=Literal['16k', '44k'],
        need_vae_encoder: bool = True,
    ):
        super().__init__()

        if enable_conditions:
            # DFN5B CLIP: resolve via HF hub by default; users on air-gapped
            # machines can point CINEDUB_CLIP_LOCAL at a pre-downloaded copy of
            # open_clip_pytorch_model.bin (or the directory containing it).
            create_kwargs = {}
            if _CINEDUB_CLIP_LOCAL:
                if os.path.isdir(_CINEDUB_CLIP_LOCAL):
                    create_kwargs["cache_dir"] = _CINEDUB_CLIP_LOCAL
                else:
                    create_kwargs["pretrained"] = _CINEDUB_CLIP_LOCAL
            self.clip_model = create_model_from_pretrained(
                'hf-hub:apple/DFN5B-CLIP-ViT-H-14-384',
                **create_kwargs,
            )[0]

            self.clip_preprocess = Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                             std=[0.26862954, 0.26130258, 0.27577711])
            self.clip_model = patch_clip(self.clip_model)
            self.synchformer = None

            self.tokenizer = open_clip.get_tokenizer('ViT-H-14-378-quickgelu')  # same as 'ViT-H-14'
        else:
            self.clip_model = None
            self.synchformer = None
            self.tokenizer = None

    def compile(self):
        if self.clip_model is not None:
            self.clip_model.encode_image = torch.compile(self.clip_model.encode_image)
            self.clip_model.encode_text = torch.compile(self.clip_model.encode_text)
        if self.synchformer is not None:
            self.synchformer = torch.compile(self.synchformer)
        self.decode = torch.compile(self.decode)
        self.vocode = torch.compile(self.vocode)

    def train(self, mode: bool) -> None:
        return super().train(False)


    @torch.inference_mode()
    def encode_video_with_clip(self, x: torch.Tensor, frame_lens: torch.Tensor = None, batch_size: int = -1) -> torch.Tensor:
        assert self.clip_model is not None, 'CLIP is not loaded'
        # x: (B, T, C, H, W) H/W: 384
        b, t, c, h, w = x.shape
        assert c == 3 and h == 384 and w == 384
        x = self.clip_preprocess(x)
        if frame_lens is None:
            frame_lens = torch.full((b,), t, device=x.device)
        frames = []
        for i in range(b):
            frames.append(x[i, :frame_lens[i]]) #b * t, c, h, w
        x = torch.cat(frames, dim=0)
        assert x.shape[0] == sum(frame_lens)

        x = self.clip_model.encode_image(x, normalize=True)
        # Regroup features per sample
        encoded_features = []
        segment_start = 0

        for i, num_segs in enumerate(frame_lens):
            if num_segs == 0: #t, d
                encoded_features.append(torch.empty(0, x.size(-1), device=x.device))
            else:
                sample_features = x[segment_start:segment_start + num_segs]
                encoded_features.append(sample_features)
                segment_start += num_segs
        return encoded_features, frame_lens


    @torch.inference_mode()
    def encode_text(self, text: list[str]) -> torch.Tensor:
        assert self.clip_model is not None, 'CLIP is not loaded'
        assert self.tokenizer is not None, 'Tokenizer is not loaded'
        # x: (B, L)
        tokens = self.tokenizer(text).to(self.device)
        return self.clip_model.encode_text(tokens, normalize=True)

    @torch.inference_mode()
    def encode_audio(self, x):
        assert self.tod is not None, 'VAE is not loaded'
        # x: (B * L)
        mel = self.mel_converter(x)
        dist = self.tod.encode(mel)

        return dist

    @torch.inference_mode()
    def vocode(self, mel: torch.Tensor) -> torch.Tensor:
        assert self.tod is not None, 'VAE is not loaded'
        return self.tod.vocode(mel)

    @torch.inference_mode()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        assert self.tod is not None, 'VAE is not loaded'
        return self.tod.decode(z.transpose(1, 2))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype
