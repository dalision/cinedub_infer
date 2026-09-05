#Heavily influenced by https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/modules/conditioners.py

import torch
import logging, warnings
import string
import typing as tp
import gc
import math
import numpy as np

from .adp import NumberEmbedder
from ..inference.utils import set_audio_channels
from .factory import create_pretransform_from_config
from .pretransforms import Pretransform
from ..training.utils import copy_state_dict
from .utils import load_ckpt_state_dict
from .transformer import AbsolutePositionalEmbedding

from torch import nn, einsum, Tensor
from torch.nn import functional as F

from stable_audio_tools.models.naturalspeech2_pytorch.naturalspeech2_pytorch import PhonemeEncoder, DurationPitchPredictor, compute_pitch_pyworld, compute_pitch_pytorch, SpeechPromptEncoder
from stable_audio_tools.models.naturalspeech2_pytorch.naturalspeech2_pytorch import exists, AudioToMel, f0_to_coarse
from stable_audio_tools.models.naturalspeech2_pytorch.utils.utils import average_over_durations, create_mask
from stable_audio_tools.models.naturalspeech2_pytorch.utils.tokenizer import Tokenizer
from stable_audio_tools.models.naturalspeech2_pytorch.aligner import Aligner
import torch
from torch.nn.utils.rnn import pad_sequence

from einops import rearrange

from stable_audio_tools.models.transformer import ContinuousTransformer
from stable_audio_tools.models.adp import Conv1d

import time
import open_clip
from open_clip import create_model_from_pretrained
from torchvision.transforms import Normalize
from torchvision.transforms import v2
from moviepy.editor import VideoFileClip
import cv2
import sys
import tempfile
import subprocess
import os

# Synchformer visual features come from MMAudio (add to your PYTHONPATH, or
# install as an editable dep). If the import fails, the CineDub inference
# path still loads — Synchformer is only touched when the model config sets
# `online_extract_type: "mm_sync"` on the sync_feature branch.
try:
    from mmaudio.ext.synchformer import Synchformer
except ImportError:
    Synchformer = None


# Repo root, used to resolve weight paths shipped under `weights/`.
_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _resolve_weight_path(model_name: str) -> str:
    """Resolve `weights/<model_name>` under the repo root; fall back to
    the plain HF hub id (letting `AutoTokenizer/*.from_pretrained` handle
    the download) if the local dir is absent."""
    local = os.path.join(_PKG_ROOT, "weights", model_name)
    return local if os.path.isdir(local) else model_name


def _resolve_synchformer_ckpt(default_path: str) -> str:
    """Resolve the Synchformer checkpoint path.

    Priority:
      1. If ``CINEDUB_SYNCHFORMER_CKPT`` is set, use it — but require the
         file to actually exist (respecting the user's explicit intent).
      2. If the default local path exists, use it.
      3. Otherwise auto-download from ``hkchengrex/MMAudio`` via
         ``huggingface_hub`` (cached across runs).
    """
    env_path = os.environ.get("CINEDUB_SYNCHFORMER_CKPT")
    if env_path:
        if not os.path.exists(env_path):
            raise FileNotFoundError(
                f"CINEDUB_SYNCHFORMER_CKPT is set to '{env_path}' but that file does not exist. "
                f"Point it at the real synchformer_state_dict.pth or unset the variable to fall back to auto-download."
            )
        return env_path
    if os.path.exists(default_path):
        return default_path
    logging.info(
        f"[CineDub] Synchformer weight not found at {default_path}; "
        f"auto-downloading from hkchengrex/MMAudio (~40 MB)..."
    )
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(
            repo_id="hkchengrex/MMAudio",
            filename="ext_weights/synchformer_state_dict.pth",
        )
        logging.info(f"[CineDub] Synchformer weight cached at {p}")
        return p
    except Exception as e:
        raise FileNotFoundError(
            f"Synchformer weight missing at {default_path} and HF auto-download failed: {e}. "
            f"Manually download hkchengrex/MMAudio ext_weights/synchformer_state_dict.pth and place it at {default_path}, "
            f"or set CINEDUB_SYNCHFORMER_CKPT to a local path."
        ) from e

class Conditioner(nn.Module):
    def __init__(
            self,
            dim: int,
            output_dim: int, 
            project_out: bool = False,
            option: bool = False
            ):
        
        super().__init__()

        self.dim = dim
        self.output_dim = output_dim
        self.proj_out = nn.Linear(dim, output_dim) if (dim != output_dim or project_out) else nn.Identity()
        self.option = option

    def forward(self, x: tp.Any) -> tp.Any:
        raise NotImplementedError()

class IntConditioner(Conditioner):
    def __init__(self, 
                output_dim: int,
                min_val: int=0,
                max_val: int=512
                ):
        super().__init__(output_dim, output_dim)

        self.min_val = min_val
        self.max_val = max_val
        self.int_embedder = nn.Embedding(max_val - min_val + 1, output_dim).requires_grad_(True)

    def forward(self, ints: tp.List[int], device=None) -> tp.Any:
            
            #self.int_embedder.to(device)
    
            ints = torch.tensor(ints).to(device)
            ints = ints.clamp(self.min_val, self.max_val)
    
            int_embeds = self.int_embedder(ints).unsqueeze(1)
    
            return [int_embeds, torch.ones(int_embeds.shape[0], 1).to(device)]

class NumberConditioner(Conditioner):
    '''
        Conditioner that takes a list of floats, normalizes them for a given range, and returns a list of embeddings
    '''
    def __init__(self, 
                output_dim: int,
                min_val: float=0,
                max_val: float=1
                ):
        super().__init__(output_dim, output_dim)

        self.min_val = min_val
        self.max_val = max_val

        self.embedder = NumberEmbedder(features=output_dim)

    def forward(self, floats: tp.List[float], device=None) -> tp.Any:
    
            # Cast the inputs to floats
            floats = [float(x) for x in floats]

            floats = torch.tensor(floats).to(device)

            floats = floats.clamp(self.min_val, self.max_val)
    
            normalized_floats = (floats - self.min_val) / (self.max_val - self.min_val)

            # Cast floats to same type as embedder
            embedder_dtype = next(self.embedder.parameters()).dtype
            normalized_floats = normalized_floats.to(embedder_dtype)

            float_embeds = self.embedder(normalized_floats).unsqueeze(1)
    
            return [float_embeds, torch.ones(float_embeds.shape[0], 1).to(device)]

class ListConditioner(Conditioner):
    def __init__(self, 
                output_dim: int,
                options: tp.List[str]
                ):
        super().__init__(output_dim, output_dim)

        self.options = options
        self.embedder = nn.Embedding(len(options)+1, output_dim).requires_grad_(True)

    def forward(self, texts: tp.List[str], device=None) -> tp.Any:

        # Cast the inputs to floats, handling the case where the input is not in the options
        ints = [self.options.index(x) + 1 if x in self.options else 0 for x in texts]

        ints = torch.tensor(ints).to(device) # shape [batch_size]

        int_embeds = self.embedder(ints).unsqueeze(1) # shape [batch_size, 1, output_dim]

        return [int_embeds, torch.ones(int_embeds.shape[0], 1).to(device)]

class CLAPTextConditioner(Conditioner):
    def __init__(self, 
                 output_dim: int, 
                 clap_ckpt_path,
                 use_text_features = False,
                 feature_layer_ix: int = -1,
                 audio_model_type="HTSAT-base", 
                 enable_fusion=True,
                 project_out: bool = False,
                 finetune: bool = False,
                 option: bool = False):
        super().__init__(768 if use_text_features else 512, output_dim, project_out=project_out)

        self.use_text_features = use_text_features
        self.feature_layer_ix = feature_layer_ix
        self.finetune = finetune
        self.option = option

        # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                import laion_clap
                from laion_clap.clap_module.factory import load_state_dict as clap_load_state_dict
                
                model = laion_clap.CLAP_Module(enable_fusion=enable_fusion, amodel=audio_model_type, device='cpu')

                if self.finetune:
                    self.model = model
                else: 
                    self.__dict__["model"] = model

                state_dict = clap_load_state_dict(clap_ckpt_path)
                self.model.model.load_state_dict(state_dict, strict=False)

                if self.finetune:
                    self.model.model.text_branch.requires_grad_(True)
                    self.model.model.text_branch.train()
                else:
                    self.model.model.text_branch.requires_grad_(False)
                    self.model.model.text_branch.eval()

            finally:
                logging.disable(previous_level)

        # Log encoder loading info
        param_count = sum(p.numel() for p in self.model.model.text_branch.parameters())
        logging.info(f"Loaded CLAPTextConditioner: model={audio_model_type}, path={clap_ckpt_path}, "
                     f"params={param_count/1e6:.1f}M, trainable={finetune}")

        del self.model.model.audio_branch

        gc.collect()
        torch.cuda.empty_cache()

    def get_clap_features(self, prompts, layer_ix=-2, device: tp.Any = "cuda"):
        prompt_tokens = self.model.tokenizer(prompts)
        attention_mask = prompt_tokens["attention_mask"].to(device=device, non_blocking=True)
        prompt_features = self.model.model.text_branch(
            input_ids=prompt_tokens["input_ids"].to(device=device, non_blocking=True),
            attention_mask=attention_mask,
            output_hidden_states=True
        )["hidden_states"][layer_ix]

        return prompt_features, attention_mask

    def forward(self, texts: tp.List[str], device: tp.Any = "cuda") -> tp.Any:
        self.model.to(device)

        if self.use_text_features:
            if len(texts) == 1:
                text_features, text_attention_mask = self.get_clap_features([texts[0], ""], layer_ix=self.feature_layer_ix, device=device)
                text_features = text_features[:1, ...]
                text_attention_mask = text_attention_mask[:1, ...]
            else:
                text_features, text_attention_mask = self.get_clap_features(texts, layer_ix=self.feature_layer_ix, device=device)

            # Cast text feature to same type as proj_out, unless proj_out is Identity
            if not isinstance(self.proj_out, nn.Identity):
                proj_out_dtype = next(self.proj_out.parameters()).dtype
                text_features = text_features.to(proj_out_dtype)                        

            return [self.proj_out(text_features), text_attention_mask]

        # Fix for CLAP bug when only one text is passed
        if len(texts) == 1:
            text_embedding = self.model.get_text_embedding([texts[0], ""], use_tensor=True)[:1, ...]
        else:
            text_embedding = self.model.get_text_embedding(texts, use_tensor=True)

        text_embedding = text_embedding.unsqueeze(1).to(device)

        # Cast text embedding to same type as proj_out, unless proj_out is Identity
        if not isinstance(self.proj_out, nn.Identity):
            proj_out_dtype = next(self.proj_out.parameters()).dtype
            text_embedding = text_embedding.to(proj_out_dtype)

        return [self.proj_out(text_embedding), torch.ones(text_embedding.shape[0], 1).to(device)]

class CLAPAudioConditioner(Conditioner):
    def __init__(self, 
                 output_dim: int, 
                 clap_ckpt_path,
                 audio_model_type="HTSAT-base", 
                 enable_fusion=True,
                 project_out: bool = False):
        super().__init__(512, output_dim, project_out=project_out)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                import laion_clap
                from laion_clap.clap_module.factory import load_state_dict as clap_load_state_dict
                
                model = laion_clap.CLAP_Module(enable_fusion=enable_fusion, amodel=audio_model_type, device='cpu')

                if self.finetune:
                    self.model = model
                else: 
                    self.__dict__["model"] = model

                state_dict = clap_load_state_dict(clap_ckpt_path)
                self.model.model.load_state_dict(state_dict, strict=False)

                if self.finetune:
                    self.model.model.audio_branch.requires_grad_(True)
                    self.model.model.audio_branch.train()
                else:
                    self.model.model.audio_branch.requires_grad_(False)
                    self.model.model.audio_branch.eval()

            finally:
                logging.disable(previous_level)

        # Log encoder loading info
        param_count = sum(p.numel() for p in self.model.model.audio_branch.parameters())
        logging.info(f"Loaded CLAPAudioConditioner: model={audio_model_type}, path={clap_ckpt_path}, "
                     f"params={param_count/1e6:.1f}M")

        del self.model.model.text_branch

        gc.collect()
        torch.cuda.empty_cache()

    def forward(self, audios: tp.Union[torch.Tensor, tp.List[torch.Tensor], tp.Tuple[torch.Tensor]] , device: tp.Any = "cuda") -> tp.Any:

        self.model.to(device)

        if isinstance(audios, list) or isinstance(audios, tuple):
            audios = torch.cat(audios, dim=0)

        # Convert to mono
        mono_audios = audios.mean(dim=1)

        with torch.cuda.amp.autocast(enabled=False):
            audio_embedding = self.model.get_audio_embedding_from_data(mono_audios.float(), use_tensor=True)

        audio_embedding = audio_embedding.unsqueeze(1).to(device)

        # Cast audio embedding to same type as proj_out, unless proj_out is Identity

        if not isinstance(self.proj_out, nn.Identity):
            proj_out_dtype = next(self.proj_out.parameters()).dtype
            audio_embedding = audio_embedding.to(proj_out_dtype)

        return [self.proj_out(audio_embedding), torch.ones(audio_embedding.shape[0], 1).to(device)]


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class ConvNeXtV2Block(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        dilation: int = 1,
    ):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=7, padding=padding, groups=dim, dilation=dilation
        )  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)  # b n d -> b d n
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # b d n -> b n d
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x

def get_tokenizer(dataset_name, tokenizer: str = "pinyin"):
    """
    tokenizer   - "pinyin" do g2p for only chinese characters, need .txt vocab_file
                - "char" for char-wise tokenizer, need .txt vocab_file
                - "byte" for utf-8 tokenizer
                - "custom" if you're directly passing in a path to the vocab.txt you want to use
    vocab_size  - if use "pinyin", all available pinyin types, common alphabets (also those with accent) and symbols
                - if use "char", derived from unfiltered character & symbol counts of custom dataset
                - if use "byte", set to 256 (unicode byte range)
    """
    if tokenizer in ["pinyin", "char"]:
        # Ship your own vocab.txt (e.g. LibriTTS pinyin/char vocab) and point
        # CINEDUB_TTS_VOCAB at it. The CineDub inference path does not exercise
        # this branch — it is retained only for backwards compat with the
        # NaturalSpeech2 pytorch codebase we vendored.
        tokenizer_path = os.environ.get(
            "CINEDUB_TTS_VOCAB",
            os.path.join(_PKG_ROOT, "stable_audio_tools", "assets", "vocab.txt"),
        )
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)
        assert vocab_char_map[" "] == 0, "make sure space is of idx 0 in vocab.txt, cuz 0 is used for unknown char"

    elif tokenizer == "byte":
        vocab_char_map = None
        vocab_size = 256

    elif tokenizer == "custom":
        with open(dataset_name, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)

    return vocab_char_map, vocab_size

# char tokenizer, based on custom dataset's extracted .txt file
def list_str_to_idx(
    text: list[str] | list[list[str]],
    vocab_char_map: dict[str, int],  # {char: idx}
    padding_value=-1,
    seq_len=None,
):  # noqa: F722
    if seq_len is not None:
        # Handle empty strings by creating full-length -1 sequences
        list_idx_tensors = []
        for t in text:
            if t and t.strip():  # Non-empty string
                tensor = torch.tensor([vocab_char_map.get(c, 0) for c in t])
            else:  # Empty string - create seq_len -1s
                tensor = torch.full((seq_len,), -1, dtype=torch.long)
            list_idx_tensors.append(tensor)
    else:
        list_idx_tensors = [torch.tensor([vocab_char_map.get(c, 0) for c in t]) for t in text]  # pinyin or char style
    text = pad_sequence(list_idx_tensors, padding_value=padding_value, batch_first=True)
    return text
     
class TranscriptionConditioner(Conditioner):
    def __init__(self, 
                output_dim: int,
                mask_padding = True,
                conv_layers = 0,
                conv_mult = 2,
                seq_len = 250,
                option: bool = False,
                trans_dim: bool = True,
                ):
        super().__init__(output_dim, output_dim)

        self.vocab_char_map, self.vocab_size = get_tokenizer(None, "pinyin")
        self.trans_dim = trans_dim
        self.text_embed = nn.Embedding(self.vocab_size + 2, output_dim, padding_idx=0)  # use 0 as filler token
        self.option = option
        self.mask_padding = mask_padding  # mask filler and batch padding tokens or not
        
        self.seq_len = seq_len
        
        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = seq_len
            self.register_buffer("freqs_cis", self.precompute_freqs_cis(output_dim, self.precompute_max_pos), persistent=False)
            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(output_dim, output_dim * conv_mult) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False
        
    def precompute_freqs_cis(self, dim: int, end: int, theta: float = 10000.0, theta_rescale_factor=1.0):
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
        # https://github.com/lucidrains/rotary-embedding-torch/blob/main/rotary_embedding_torch/rotary_embedding_torch.py
        theta *= theta_rescale_factor ** (dim / (dim - 2))
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, device=freqs.device)  # type: ignore
        freqs = torch.outer(t, freqs).float()  # type: ignore
        freqs_cos = torch.cos(freqs)  # real part
        freqs_sin = torch.sin(freqs)  # imaginary part
        return torch.cat([freqs_cos, freqs_sin], dim=-1)
    
    def get_pos_embed_indices(self, start, length, max_pos, scale=1.0):
        # length = length if isinstance(length, int) else length.max()
        scale = scale * torch.ones_like(start, dtype=torch.float32)  # in case scale is a scalar
        pos = (
            start.unsqueeze(1)
            + (torch.arange(length, device=start.device, dtype=torch.float32).unsqueeze(0) * scale.unsqueeze(1)).long()
        )
        # avoid extra long error.
        pos = torch.where(pos < max_pos, pos, max_pos - 1)
        return pos
    
    def forward(self, text, device=None) -> tp.Any: #空字符全0，有字符按字转idx，silence取0，duration部分取1; start 默认取0
        assert "duration" in text[0] and "content" in text[0], "Each text item must be a dict with 'content' and 'duration' keys"
        contents = [t["content"] if t["content"] and t["content"].strip() else "" for t in text]
        durations = [int(round(d * 25)) if d is not None else 0 for d in (t["duration"] for t in text)]  #1s -> 25 frames
        # durations = [int(round(d * 25)) if d is not None else 0 for d in (0 for t in text)]
        
        # When duration=None, start should be automatically set to 0
        starts = []
        for i, t in enumerate(text):
            if t.get("duration") is None:
                starts.append(0)  # Auto set start to 0 when duration=None
            else:
                starts.append(int(round(t.get("start", 0) * 25)))
        
        # for duration must be zero gbl-vt2s
        # Check for duration=0 which should raise an error 
        # for i, t in enumerate(text):
        #     if t.get("duration") == 0:
        #         raise ValueError(f"Duration cannot be 0 for sample {i}. Use None for no duration limit or a positive value.")

        # dili
        # Track which samples have empty content for silence handling
        empty_content_mask = [not bool(content.strip()) for content in contents]
        # Track which samples have no duration limit (duration=None)
        no_duration_mask = [t.get("duration") is None for t in text]

        # Initialize final text tensor
        batch = len(contents)
        seq_len = self.seq_len
        text = torch.zeros((batch, seq_len), dtype=torch.long, device=device)

        dur_vals = torch.tensor(durations, device=device, dtype=torch.long).clamp(min=0, max=seq_len)
        start_vals = torch.tensor(starts, device=device, dtype=torch.long).clamp(min=0, max=seq_len)

        idx = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)

        # Create masks for start and duration ranges
        start_mask = idx >= start_vals.unsqueeze(1)

        # When duration is 0 (no duration specified), active range is from start to end of sequence
        # When duration > 0, active range is from start to start+duration
        has_duration_mask = dur_vals > 0
        end_mask = torch.where(
            has_duration_mask.unsqueeze(1),
            idx < (start_vals + dur_vals).unsqueeze(1),  # Has duration: use start+duration
            torch.ones_like(idx, dtype=torch.bool)       # No duration: goes to end
        )
        active_range_mask = start_mask & end_mask

        # Process each sample individually to handle start positioning
        for i, content in enumerate(contents):
            sample_active_mask = active_range_mask[i]
            active_positions = torch.where(sample_active_mask)[0]

            if no_duration_mask[i]:
                # When duration=None, first fill entire active range with 1s
                text[i, sample_active_mask] = 1
                # Then place content mapping if exists, overriding 1s
                if content and content.strip():
                    content_indices = [self.vocab_char_map.get(c, 0) for c in content]
                    content_tensor = torch.tensor(content_indices, device=device, dtype=torch.long) + 1
                    if len(active_positions) > 0 and len(content_tensor) > 0:
                        content_length = min(len(content_tensor), len(active_positions))
                        content_positions = active_positions[:content_length]
                        text[i, content_positions] = content_tensor[:content_length]
            elif content and content.strip():  # Non-empty content with specific duration
                # Convert content to indices
                content_indices = [self.vocab_char_map.get(c, 0) for c in content]
                content_tensor = torch.tensor(content_indices, device=device, dtype=torch.long) + 1

                if len(active_positions) > 0 and len(content_tensor) > 0:
                    # Place content starting from the first active position
                    content_length = min(len(content_tensor), len(active_positions))
                    content_positions = active_positions[:content_length]
                    text[i, content_positions] = content_tensor[:content_length]

                    # Fill remaining active positions with 1s
                    if len(active_positions) > content_length:
                        remaining_positions = active_positions[content_length:]
                        text[i, remaining_positions] = 1
                else:
                    # No active range, fill active range with 1s
                    text[i, sample_active_mask] = 1
            else:
                # Empty content with specific duration, fill entire active range with 1s
                text[i, sample_active_mask] = 1

        # Areas before start and after active range remain 0 (silence)

        # dili
        # For empty content, set entire sequence to 0 (silence) - do this AFTER all processing
        empty_content_tensor = torch.tensor(empty_content_mask, device=text.device).unsqueeze(1)
        text = text * (~empty_content_tensor).long()  # Zero out entire sequences for empty content
        
        if self.mask_padding:
            text_mask = text == 0

        # drop_text = self.drop_text
        # if drop_text:  # cfg for text
        #     text = torch.zeros_like(text)
        
        text = self.text_embed(text)  # b n -> b n d
        
        # possible extra modeling
        if self.extra_modeling:
            # sinus pos emb
            batch_start = torch.zeros((batch,), dtype=torch.long)
            pos_idx = self.get_pos_embed_indices(batch_start, self.seq_len, max_pos=self.precompute_max_pos)
            text_pos_embed = self.freqs_cis[pos_idx]
            text = text + text_pos_embed

            # convnextv2 blocks
            if self.mask_padding:
                text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)
                for block in self.text_blocks:
                    text = block(text)
                    text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)
            else:
                text = self.text_blocks(text)
        if self.trans_dim:
            text = text.permute(0, 2, 1)
        return [text, torch.ones(text.shape[0], 1).to(device)]


def _classify_special_texts(
    texts,
    clean_speech_embedding,
    non_speech_embedding,
):
    """Classify each sample in a batch as clean_speech / none_speech / encoder.

    Returns:
        (clean_indices, none_indices, encoder_indices)
    """
    clean_indices, none_indices, enc_indices = [], [], []
    for i, text in enumerate(texts):
        t = text.strip().lower()
        if t == "clean speech" and clean_speech_embedding is not None:
            clean_indices.append(i)
        elif t == "none_speech" and non_speech_embedding is not None:
            none_indices.append(i)
        else:
            enc_indices.append(i)
    return clean_indices, none_indices, enc_indices


def _make_learnable_emb(embedding, batch_size, max_length, device):
    """Create a batch of learnable embeddings (expand to all positions) with mask=[True, False, ...]."""
    emb = embedding.expand(batch_size, max_length, -1).to(device)
    mask = torch.zeros(batch_size, max_length, dtype=torch.bool, device=device)
    mask[:, 0] = True
    return emb, mask


def _fill_learnable_embedding(embedding, indices, out_embeddings, out_mask, max_length, device):
    """Fill output tensors with a learnable embedding at given batch indices."""
    if not indices or embedding is None:
        return
    for i in indices:
        out_embeddings[i] = embedding.squeeze(0).expand(max_length, -1).to(device)
        out_mask[i, 0] = True


def _route_and_encode(texts, device, per_sample_routing, clean_emb, none_emb, max_length, proj_out, output_dim, encode_fn):
    """Shared routing logic for T5Conditioner and GemmaT5Conditioner forward().

    Routes each sample to learnable embedding or text encoder based on text content.
    """
    batch_size = len(texts)

    if not per_sample_routing:
        # Legacy: only handle ALL-batch special texts
        if clean_emb is not None and all(t.strip().lower() == "clean speech" for t in texts):
            return _make_learnable_emb(clean_emb, batch_size, max_length, device)
        if none_emb is not None and all(t.strip().lower() == "none_speech" for t in texts):
            return _make_learnable_emb(none_emb, batch_size, max_length, device)
        return encode_fn(texts, device)

    # Per-sample routing
    clean_idx, none_idx, enc_idx = _classify_special_texts(texts, clean_emb, none_emb)

    # Fast paths: entire batch is one type
    if len(clean_idx) == batch_size:
        return _make_learnable_emb(clean_emb, batch_size, max_length, device)
    if len(none_idx) == batch_size:
        return _make_learnable_emb(none_emb, batch_size, max_length, device)
    if len(enc_idx) == batch_size:
        return encode_fn(texts, device)

    # Mixed batch
    out_dim = proj_out.out_features if hasattr(proj_out, 'out_features') else output_dim
    embeddings = torch.zeros(batch_size, max_length, out_dim, device=device)
    attn_mask = torch.zeros(batch_size, max_length, dtype=torch.bool, device=device)

    if enc_idx:
        enc_emb, enc_mask = encode_fn([texts[i] for i in enc_idx], device)
        for j, i in enumerate(enc_idx):
            embeddings[i] = enc_emb[j]
            attn_mask[i] = enc_mask[j]

    _fill_learnable_embedding(clean_emb, clean_idx, embeddings, attn_mask, max_length, device)
    _fill_learnable_embedding(none_emb, none_idx, embeddings, attn_mask, max_length, device)
    return embeddings, attn_mask


class T5Conditioner(Conditioner):

    T5_MODELS = ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b",
              "google/flan-t5-small", "google/flan-t5-base", "flan-t5-base", "flan-t5-large", "google/flan-t5-large",
              "google/flan-t5-xl", "google/flan-t5-xxl", "google/t5-v1_1-xl", "google/t5-v1_1-xxl"]

    T5_MODEL_DIMS = {
        "t5-small": 512,
        "t5-base": 768,
        "t5-large": 1024,
        "t5-3b": 1024,
        "t5-11b": 1024,
        "google/t5-v1_1-xl": 2048,
        "google/t5-v1_1-xxl": 4096,
        "google/flan-t5-small": 512,
        "google/flan-t5-base": 768,
        "flan-t5-base": 768,
        "flan-t5-large": 1024,
        "google/flan-t5-large": 1024,
        "google/flan-t5-3b": 1024,
        "google/flan-t5-11b": 1024,
        "google/flan-t5-xl": 2048,
        "google/flan-t5-xxl": 4096,
    }

    def __init__(
            self,
            output_dim: int,
            t5_model_name: str = "t5-base",
            max_length: str = 128,
            enable_grad: bool = False,
            project_out: bool = False,
            option: bool = False,
            clean_speech_embedding: bool = False,
            non_speech_embedding: bool = False,
            per_sample_routing: bool = False
    ):
        assert t5_model_name in self.T5_MODELS, f"Unknown T5 model name: {t5_model_name}"
        super().__init__(self.T5_MODEL_DIMS[t5_model_name], output_dim, project_out=project_out)

        from transformers import T5EncoderModel, AutoTokenizer
        self.option = option
        self.max_length = max_length
        self.enable_grad = enable_grad
        self.per_sample_routing = per_sample_routing

        # Learnable embedding for "clean speech" special token
        # Shape: (1, 1, output_dim)
        if clean_speech_embedding:
            self.clean_speech_embedding = nn.Parameter(
                torch.randn(1, 1, output_dim) * 0.02
            ) #for more stable training, we initialize it to small values
        else:
            self.clean_speech_embedding = None

        # Learnable embedding for "none_speech" special token
        # Shape: (1, 1, output_dim)
        if non_speech_embedding:
            self.non_speech_embedding = nn.Parameter(
                torch.randn(1, 1, output_dim) * 0.02
            )
        else:
            self.non_speech_embedding = None
        # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # self.tokenizer = T5Tokenizer.from_pretrained(t5_model_name, model_max_length = max_length)
                # model = T5EncoderModel.from_pretrained(t5_model_name, max_length=max_length).train(enable_grad).requires_grad_(enable_grad)

                local_path = _resolve_weight_path(t5_model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(local_path)
                if self.enable_grad:
                    dtype = torch.float32
                    model = T5EncoderModel.from_pretrained(local_path).train(enable_grad).requires_grad_(enable_grad).to(dtype)
                else:
                    dtype = torch.float16
                    model = T5EncoderModel.from_pretrained(local_path).train(enable_grad).requires_grad_(enable_grad).to(dtype)

            finally:
                logging.disable(previous_level)

        # Log encoder loading info
        param_count = sum(p.numel() for p in model.parameters())
        logging.info(f"Loaded T5Conditioner: model={t5_model_name}, path={local_path}, "
                     f"params={param_count/1e6:.1f}M, dtype={dtype}, trainable={enable_grad}")

        if self.enable_grad:
            self.model = model
        else:
            self.__dict__["model"] = model

    def _encode_texts(self, texts, device):
        """Encode texts using the T5 model."""
        self.model.to(device)
        self.proj_out.to(device)

        encoded = self.tokenizer(
            texts, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device).to(torch.bool)

        if not self.enable_grad:
            self.model.eval()

        with torch.cuda.amp.autocast(enabled=False), torch.set_grad_enabled(self.enable_grad):
            embeddings = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            )["last_hidden_state"]

        if not isinstance(self.proj_out, nn.Identity):
            embeddings = embeddings.to(next(self.proj_out.parameters()).dtype)
        embeddings = self.proj_out(embeddings)
        embeddings = embeddings * attention_mask.unsqueeze(-1).float()
        return embeddings, attention_mask

    def forward(self, texts, device):
        return _route_and_encode(
            texts, device, self.per_sample_routing,
            self.clean_speech_embedding, self.non_speech_embedding,
            self.max_length, self.proj_out, self.output_dim, self._encode_texts,
        )


def _detect_attn_implementation() -> str:
    """Auto-detect best attention implementation for T5Gemma.

    - CC >= 9.0 (Hopper/Blackwell) + flash_attn >= 2.6.0 -> flash_attention_2
    - Any GPU + flash_attn installed -> flash_attention_2
    - Fallback -> eager
    """
    if not torch.cuda.is_available():
        return "eager"
    try:
        import flash_attn
        from packaging.version import parse as V
        fa_version = getattr(flash_attn, "__version__", "0.0.0")
        major, _ = torch.cuda.get_device_capability()
        if major >= 9 and V(fa_version) >= V("2.6.0"):
            return "flash_attention_2"
        return "flash_attention_2"
    except ImportError:
        return "eager"


class GemmaT5Conditioner(Conditioner):
    """
    GemmaT5 text encoder conditioner based on google/t5gemma models.
    Uses T5GemmaModel encoder with bfloat16 precision (following ss-falcon pattern).
    """

    GEMMA_T5_MODELS = [
        "google/t5gemma-2b-2b-ul2",   # 2.6B encoder params, hidden_size=2304
        "google/t5gemma-ml-ml-ul2",   # medium-large, hidden_size=1152
        "google/t5gemma-l-l-ul2",     # large, hidden_size=1024
    ]

    GEMMA_T5_MODEL_DIMS = {
        "google/t5gemma-2b-2b-ul2": 2304,
        "google/t5gemma-ml-ml-ul2": 1152,
        "google/t5gemma-l-l-ul2": 1024,
    }

    def __init__(
            self,
            output_dim: int,
            model_name: str = "google/t5gemma-2b-2b-ul2",
            max_length: int = 128,
            enable_grad: bool = False,
            project_out: bool = False,
            option: bool = False,
            clean_speech_embedding: bool = False,
            non_speech_embedding: bool = False,
            per_sample_routing: bool = False
    ):
        assert model_name in self.GEMMA_T5_MODELS, f"Unknown GemmaT5 model name: {model_name}"
        super().__init__(self.GEMMA_T5_MODEL_DIMS[model_name], output_dim, project_out=project_out)

        from transformers import AutoTokenizer
        from transformers.models.t5gemma.modeling_t5gemma import T5GemmaModel

        self.option = option
        self.max_length = max_length
        self.enable_grad = enable_grad
        self.per_sample_routing = per_sample_routing

        # Learnable embedding for "clean speech" special token
        if clean_speech_embedding:
            self.clean_speech_embedding = nn.Parameter(
                torch.randn(1, 1, output_dim) * 0.02
            )
        else:
            self.clean_speech_embedding = None

        # Learnable embedding for "none_speech" special token
        if non_speech_embedding:
            self.non_speech_embedding = nn.Parameter(
                torch.randn(1, 1, output_dim) * 0.02
            )
        else:
            self.non_speech_embedding = None

        # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                local_path = _resolve_weight_path(model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(local_path)

                # Use bfloat16 for inference (ss-falcon pattern), float32 for training
                dtype = torch.float32 if self.enable_grad else torch.bfloat16
                attn_impl = _detect_attn_implementation()
                model = T5GemmaModel.from_pretrained(local_path, torch_dtype=dtype, attn_implementation=attn_impl)

                # Only keep encoder, delete decoder to save ~3B params
                del model.decoder
                gc.collect()

                model.train(enable_grad).requires_grad_(enable_grad)
            finally:
                logging.disable(previous_level)

        # Log encoder loading info
        param_count = sum(p.numel() for p in model.parameters())
        logging.info(f"Loaded T5GemmaConditioner (encoder-only): model={model_name}, path={local_path}, "
                     f"params={param_count/1e6:.1f}M, dtype={dtype}, trainable={enable_grad}, "
                     f"attn={attn_impl}")

        if self.enable_grad:
            self.model = model
        else:
            self.__dict__["model"] = model

    def _encode_texts(self, texts, device):
        """Encode texts using the GemmaT5 encoder."""
        self.model.to(device)
        self.proj_out.to(device)

        encoded = self.tokenizer(
            texts, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device).to(torch.bool)

        if not self.enable_grad:
            self.model.eval()

        with torch.cuda.amp.autocast(enabled=False), torch.set_grad_enabled(self.enable_grad):
            embeddings = self.model.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state

        if not isinstance(self.proj_out, nn.Identity):
            embeddings = embeddings.to(next(self.proj_out.parameters()).dtype)
        embeddings = self.proj_out(embeddings)
        embeddings = embeddings * attention_mask.unsqueeze(-1).float()
        return embeddings, attention_mask

    def forward(self, texts, device):
        return _route_and_encode(
            texts, device, self.per_sample_routing,
            self.clean_speech_embedding, self.non_speech_embedding,
            self.max_length, self.proj_out, self.output_dim, self._encode_texts,
        )


class PhonemeConditioner(Conditioner):
    """
    A conditioner that turns text into phonemes and embeds them using a lookup table
    Only works for English text

    Args:
        output_dim: the dimension of the output embeddings
        max_length: the maximum number of phonemes to embed
        project_out: whether to add another linear projection to the output embeddings
    """

    def __init__(
            self,
            output_dim: int,
            max_length: int = 1024,
            project_out: bool = False,
    ):
        super().__init__(output_dim, output_dim, project_out=project_out)
        
        import nltk
        # Ensure `punkt` is available; a fresh env may need this download once.
        try:
            nltk.download("punkt", quiet=True)
        except Exception:
            pass
        from g2p_en import G2p
        self.max_length = max_length
        self.g2p = G2p()

        # Reserving 0 for padding, 1 for ignored
        self.phoneme_embedder = nn.Embedding(len(self.g2p.phonemes) + 2, output_dim)


    def forward(self, texts: tp.List[str], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        
        self.phoneme_embedder.to(device)
        self.proj_out.to(device)

        batch_phonemes = [self.g2p(text) for text in texts] # shape [batch_size, length]
        
        phoneme_ignore = [" ", *string.punctuation]

        # Remove ignored phonemes and cut to max length
        batch_phonemes = [[p if p not in phoneme_ignore else "_" for p in phonemes] for phonemes in batch_phonemes]

        # Convert to ids
        phoneme_ids = [[self.g2p.p2idx[p] + 2 if p in self.g2p.p2idx else 1 for p in phonemes] for phonemes in batch_phonemes]

        #Pad to match longest and make a mask tensor for the padding
        longest = max([len(ids) for ids in phoneme_ids])
        phoneme_ids = [ids + [0] * (longest - len(ids)) for ids in phoneme_ids]
        
        phoneme_ids = torch.tensor(phoneme_ids).to(device)
        
        # Created By @JYuXuAN
        phoneme_ids = phoneme_ids.to(torch.long)
        
        # Convert to embeddings
        phoneme_embeds = self.phoneme_embedder(phoneme_ids)
        
        phoneme_embeds = self.proj_out(phoneme_embeds)

        return phoneme_embeds, torch.ones(phoneme_embeds.shape[0], phoneme_embeds.shape[1]).to(device)

class TokenizerLUTConditioner(Conditioner):
    """
    A conditioner that embeds text using a lookup table on a pretrained tokenizer's vocabulary

    Args:
        tokenizer_name: the name of the tokenizer from the Hugging Face transformers library
        output_dim: the dimension of the output embeddings
        max_length: the maximum length of the text to embed
        project_out: whether to add another linear projection to the output embeddings
    """

    def __init__(
            self,
            tokenizer_name: str, # Name of a tokenizer from the Hugging Face transformers library
            output_dim: int,
            max_length: int = 1024,
            use_abs_pos_emb = False,
            project_out: bool = False,
            special_tokens: tp.List[str] = []
    ):
        super().__init__(output_dim, output_dim, project_out=project_out)
        
        from transformers import AutoTokenizer

         # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            finally:
                logging.disable(previous_level)

        # Add special tokens
        if len(special_tokens) > 0:
            self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

        self.max_length = max_length

        self.token_embedder = nn.Embedding(len(self.tokenizer), output_dim)

        self.abs_pos_emb = None

        if use_abs_pos_emb:
            self.abs_pos_emb = AbsolutePositionalEmbedding(output_dim, max_length)

    def forward(self, texts: tp.List[str], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        self.proj_out.to(device)

        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device).to(torch.bool)
    
        embeddings = self.token_embedder(input_ids)
            
        embeddings = self.proj_out(embeddings)

        embeddings = embeddings * attention_mask.unsqueeze(-1).float()

        if self.abs_pos_emb is not None:
            embeddings = embeddings + self.abs_pos_emb(embeddings)

        return embeddings, attention_mask

class PretransformConditioner(Conditioner):
    """
    A conditioner that uses a pretransform's encoder for conditioning

    Args:
        pretransform: an instantiated pretransform to use for conditioning
        output_dim: the dimension of the output embeddings
    """
    def __init__(self, pretransform: Pretransform, output_dim: int, save_pretransform: bool = False, option: bool = False):
        super().__init__(pretransform.encoded_channels, output_dim, option=option)


        if not save_pretransform:
            self.__dict__["pretransform"] = pretransform
        else:
            self.pretransform = pretransform
        

    def forward(self, audio: tp.Union[torch.Tensor, tp.List[torch.Tensor], tp.Tuple[torch.Tensor]], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        
        self.pretransform.to(device)
        self.proj_out.to(device)

        if isinstance(audio, list) or isinstance(audio, tuple):
            audio = torch.stack(audio, dim=0)

        # Add batch dimension if needed
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)

        # Convert audio to pretransform input channels
        audio = set_audio_channels(audio, self.pretransform.io_channels)

        audio = audio.to(device)
        
        latents = self.pretransform.encode(audio)    # [B, C, T]
        latents = self.proj_out(latents)             # [B, C, T] (proj_out is Identity when dim==output_dim)

        return [latents, torch.ones(latents.shape[0], latents.shape[2]).to(latents.device)]

class SourceMixConditioner(Conditioner):
    """
    A conditioner that mixes projected audio embeddings from multiple sources

    Args:
        pretransform: an instantiated pretransform to use for conditioning
        output_dim: the dimension of the output embeddings
        source_keys: a list of keys for the potential sources in the metadata

    """
    def __init__(self, pretransform: Pretransform, output_dim: int, save_pretransform: bool = False, source_keys: tp.List[str] = [], pre_encoded: bool = False):
        super().__init__(pretransform.encoded_channels, output_dim)

        if not save_pretransform:
            self.__dict__["pretransform"] = pretransform
        else:
            self.pretransform = pretransform

        self.source_keys = source_keys

        self.source_heads = nn.ModuleList([nn.Conv1d(pretransform.encoded_channels, output_dim, kernel_size=1) for _ in source_keys])        

        self.pre_encoded = pre_encoded

    def forward(self, sources: tp.List[tp.Dict[str, torch.Tensor]], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:

        self.pretransform.to(device)
        self.proj_out.to(device)

        # Output has to be the batch of summed projections
        # Input is per-batch-item list of source audio

        mixes = []

        for source_dict in sources: # Iterate over batch items

            mix = None

            for key_ix, key in enumerate(self.source_keys): # Iterate over potential sources
                if key in source_dict:

                    source = source_dict[key]

                    if not self.pre_encoded:
                        audio = set_audio_channels(source, self.pretransform.io_channels)

                        audio = audio.to(device)

                        latents = self.pretransform.encode(audio)
                    else:
                        latents = source.to(device)           

                    if mix is None:
                        mix = self.source_heads[key_ix](latents)
                    else:
                        mix += self.source_heads[key_ix](latents)
            
            if mix is not None:
                mixes.append(mix)
            else:
                raise ValueError("No sources found for mix")

        mixes = torch.stack(mixes, dim=0)

        return [mixes, torch.ones(mixes.shape[0], mixes.shape[2]).to(mixes.device)]

class MultiConditioner(nn.Module):
    """
    A module that applies multiple conditioners to an input dictionary based on the keys

    Args:
        conditioners: a dictionary of conditioners with keys corresponding to the keys of the conditioning input dictionary (e.g. "prompt")
        default_keys: a dictionary of default keys to use if the key is not in the input dictionary (e.g. {"prompt_t5": "prompt"})
    """
    def __init__(self, conditioners: tp.Dict[str, Conditioner], default_keys: tp.Dict[str, str] = {}, pre_encoded_keys: tp.List[str] = []):
        super().__init__()

        self.conditioners = nn.ModuleDict(conditioners)
        self.default_keys = default_keys
        self.pre_encoded_keys = pre_encoded_keys

    def forward(self, batch_metadata: tp.List[tp.Dict[str, tp.Any]], device: tp.Union[torch.device, str]) -> tp.Dict[str, tp.Any]:
        output = {}

        for key, conditioner in self.conditioners.items():
            condition_key = key

            conditioner_inputs = []
            for x in batch_metadata: # Iterate each sample
                if condition_key not in x:
                    if condition_key in self.default_keys:
                        condition_key = self.default_keys[condition_key]
                    else:
                        if conditioner.option:
                            break
                        else:
                            raise ValueError(f"Conditioner key {condition_key} not found in batch metadata")

                #Unwrap the condition info if it's a single-element list or tuple, this is to support collation functions that wrap everything in a list
                if (isinstance(x[condition_key], list) or isinstance(x[condition_key], tuple)) and len(x[condition_key]) == 1:  # 修括号: 只 unwrap 单元素(多片段 list 如 [clip1,clip2] 不拆)
                    conditioner_input = x[condition_key][0]
                else:
                    conditioner_input = x[condition_key]

                conditioner_inputs.append(conditioner_input)

            # If option conditioner broke early (key not found in batch), skip it
            if conditioner.option and len(conditioner_inputs) < len(batch_metadata):
                continue

            if key in self.pre_encoded_keys:
                if isinstance(conditioner_inputs[0], list):
                    assert len(conditioner_inputs[0]) == 2, "Pre-encoded inputs must be a list of length 2"
                    feature = torch.stack([x[0] for x in conditioner_inputs], dim=0).to(device)
                    mask = torch.stack([x[1] for x in conditioner_inputs], dim=0).to(device)
                    output[key] = [feature, mask]
                else:
                    output[key] = [torch.stack(conditioner_inputs, dim=0).to(device), None]
            else:
                # VA task: skip text_prompt (audio description) but doesn't skip trans_cap (speech content)
                if (isinstance(conditioner, (T5Conditioner, GemmaT5Conditioner, CLAPTextConditioner))) and (x["task"] == "VA"):
                    # Only skip if this is text_prompt (clean_speech_embedding), not trans_cap (non_speech_embedding)
                    if isinstance(conditioner, (T5Conditioner, GemmaT5Conditioner)) and conditioner.non_speech_embedding is None:
                        continue
                # skip clip_feature for TTS task
                elif isinstance(conditioner,PostProcessConditioner) and x["task"] == "TTS" and key == "clip_feature":
                    continue

                # try:
                output[key] = conditioner(conditioner_inputs, device)
                # except Exception as e:
                #     logging.info(f"batch_metadata:{batch_metadata}")
                #     logging.error(f"Error in conditioner '{key}': {e}")

        #  Condition注入方式
        #  TA时 - T5和TTS的zero padding【option:a.TTS的zero padding】
        #  VTA时 - T5和clip和syncformer 【option:a.TTS的zero padding】
        #  VA时 - clip和syncformer 【option:a.TTS的zero padding】

        # TTS时 - TTS的channel concat 和 t5的空字符 padding 和 syncformer的zero padding
        # V2SA时 - TTS的channel concat 和 clip和syncformer
        # T2AS时 - T5 和 TTS的channel concat 和 syncformer的zero padding
        # VT2AS时 - clip和syncformer 和 T5 和 TTS的channel concat

        return output
    
class VideoFeatureConditioner(Conditioner):
    def __init__(
            self,
            output_dim: int,
            frame_per_second: int = 10,
            max_feature_length: int = 100,
            feature_dim: int = 768,
            enable_grad: bool = False,
            project_out: bool = False,
    ):  
        total_dim = max_feature_length * feature_dim
        super().__init__(total_dim, output_dim=output_dim, project_out=project_out)
        
        self.frame_per_second = frame_per_second
        self.max_feature_length = max_feature_length
        self.feature_dim = feature_dim
        self.enable_grad = enable_grad
        self.device = 'cpu'
        
    def set_device(self, device):
        self.to(device)
        self.device = device
    
    def forward(self, video_features: tp.List[tp.Dict[str, tp.Any]], device) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        # 这似乎是每个 batch 处理，有多个 vidfeats 同时传入，需要将他们合并为一个
        # TODO: 最好还是传入路径，在这里读取 video 文件！这样可以更方便地添加 demo！
        
        # 加入 padding
        self.set_device(device)
        padded_vidfeats = []
        attention_mask = torch.zeros((len(video_features), self.max_feature_length), dtype=torch.bool).to(self.device)
        
        for i, vi in enumerate(video_features):
            
            # == 如果已经读取了特征，则直接使用 ==
            if "loaded_feature" in vi:
                loaded_feature = vi["loaded_feature"]
                padded_vidfeats.append(loaded_feature)
                attention_mask_i = vi["attention_mask"]
                attention_mask[i] = attention_mask_i
            
            # == 如果没有读取特征，则读取特征 ==
            else:
                print('Reading video features in conditioners.py')
                # == 解析 ==
                # import pdb;pdb.set_trace()
                path = vi.get('path', None)
                seconds_start = vi.get('seconds_start', None)
                seconds_end = vi.get('seconds_end', None)
                assert path is not None and seconds_start is not None and seconds_end is not None
                
                # == 读取 ==
                try:
                    video_feature = torch.from_numpy(np.concatenate([np.load(p) for p in path], axis=0) if isinstance(path, list) else np.load(path)).to(self.device)  # list: 多片段特征沿时间维在线拼接
                except Exception as e:
                    # DEBUG
                    # TODO: 检查是否正确加载
                    raise(e)
                
                # == slice ==
                VID_FRAME_PER_SEC = self.frame_per_second
                MAX_VID_LEN = self.max_feature_length
                
                start_vid_frame = math.floor(seconds_start * VID_FRAME_PER_SEC)
                end_vid_frame = math.ceil(seconds_end * VID_FRAME_PER_SEC)
                
                m1 = start_vid_frame + MAX_VID_LEN
                m2 = video_feature.size(0)
                max_end_vid_frame = min(m1, m2)
                if end_vid_frame > max_end_vid_frame:
                    end_vid_frame = max_end_vid_frame
                    
                assert end_vid_frame <= start_vid_frame + MAX_VID_LEN and end_vid_frame <= video_feature.size(0), "INVALID END_VID_FRAME"
                
                # == slice and pad ==
                
                # 保留 [start_vid_frame, end_vid_frame) 之间的
                padded_vidfeat = torch.zeros((MAX_VID_LEN, self.feature_dim), 
                                            dtype=video_feature.dtype).to(self.device)
                                            
                length = end_vid_frame - start_vid_frame
                padded_vidfeat[:length, :] = video_feature[start_vid_frame:end_vid_frame, :]
                padded_vidfeats.append(padded_vidfeat)
                attention_mask[i, :length] = True
    
        padded_vidfeats = torch.stack(padded_vidfeats).to(self.device)
        
        return padded_vidfeats, attention_mask
    
class TransitionFeatureConditioner(Conditioner):
    def __init__(
            self,
            output_dim: int,
            frame_per_second: int = 10,
            max_feature_length: int = 98,
            feature_dim: int = 768,
            enable_grad: bool = False,
            project_out: bool = False,
    ):
        total_dim = max_feature_length * feature_dim
        super().__init__(total_dim, output_dim=output_dim, project_out=project_out)
        
        self.frame_per_second = frame_per_second
        self.max_feature_length = max_feature_length
        self.feature_dim = feature_dim
        self.enable_grad = enable_grad
        self.original_feature_length=self.max_feature_length+2 #这里+2是硬编码！！
        self.device = 'cpu'
        
    def set_device(self, device):
        self.to(device)
        self.device = device
    
    def get_transition_features(self, vid_feature, vid_mask):
        now_vid_feat = vid_feature[vid_mask.long()==1]
        transition_feats, transition_masks = [], []
        self.transition_steps=[2]
        self.transition_lens=[98]
        for transition_step, now_trans_len in zip(self.transition_steps, self.transition_lens):
            now_trans_feat = now_vid_feat[transition_step:, ...] - now_vid_feat[:-transition_step, ...]
            now_trans_mask = torch.ones((now_trans_feat.shape[0]))

            if now_trans_feat.shape[0] < now_trans_len:
                trans_pad_size = now_trans_len - now_trans_feat.shape[0]
                now_trans_feat = torch.nn.functional.pad(now_trans_feat, (0, 0, 0, trans_pad_size))
                now_trans_mask = torch.nn.functional.pad(now_trans_mask, (0, trans_pad_size))
            elif now_trans_feat.shape[0] > now_trans_len:
                now_trans_feat = now_trans_feat[:now_trans_len, ...]
                now_trans_mask = now_trans_mask[:now_trans_len, ...]

            # now_trans_feat*=5 # Needs ablation
            transition_feats.append(now_trans_feat)
            transition_masks.append(now_trans_mask)

        nested_transition_feats = torch.cat(transition_feats)
        nested_transition_masks = torch.cat(transition_masks)
        # import pdb;pdb.set_trace()
        # print(f"nested_transition_feats shape: {nested_transition_feats.shape}")
        # print(f"nested_transition_masks shape: {nested_transition_masks.shape}")
        return nested_transition_feats, nested_transition_masks
    
    def forward(self, video_features: tp.List[tp.Dict[str, tp.Any]], device) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        # 这似乎是每个 batch 处理，有多个 vidfeats 同时传入，需要将他们合并为一个
        # TODO: 最好还是传入路径，在这里读取 video 文件！这样可以更方便地添加 demo！
        
        # 加入 padding
        self.set_device(device)
        padded_vidfeats = []
        transition_features=[]
        attention_mask = torch.zeros((len(video_features), self.original_feature_length), dtype=torch.bool).to(self.device)
        transition_mask= torch.zeros((len(video_features), self.max_feature_length), dtype=torch.bool).to(self.device)
        for i, vi in enumerate(video_features):
            # import pdb;pdb.set_trace()
            # == 如果已经读取了特征，则直接使用 ==
            if "loaded_feature" in vi:
                loaded_feature = vi["loaded_feature"]
                transition_features.append(loaded_feature)
                attention_mask_i = vi["attention_mask"]
                transition_mask[i] = attention_mask_i
            
            # == 如果没有读取特征，则读取特征 ==
            else:
                print('Reading transition features in conditioners.py')
                # == 解析 ==
                # import pdb;pdb.set_trace()
                path = vi.get('path', None)
                seconds_start = vi.get('seconds_start', None)
                seconds_end = vi.get('seconds_end', None)
                assert path is not None and seconds_start is not None and seconds_end is not None
                
                # == 读取 ==
                try:
                    video_feature = torch.from_numpy(np.concatenate([np.load(p) for p in path], axis=0) if isinstance(path, list) else np.load(path)).to(self.device)  # list: 多片段特征沿时间维在线拼接
                except Exception as e:
                    # DEBUG
                    # TODO: 检查是否正确加载
                    raise(e)
                
                # == slice ==
                VID_FRAME_PER_SEC = self.frame_per_second
                MAX_VID_LEN = self.original_feature_length
                
                start_vid_frame = math.floor(seconds_start * VID_FRAME_PER_SEC)
                end_vid_frame = math.ceil(seconds_end * VID_FRAME_PER_SEC)
                
                m1 = start_vid_frame + MAX_VID_LEN
                m2 = video_feature.size(0)
                max_end_vid_frame = min(m1, m2)
                if end_vid_frame > max_end_vid_frame:
                    end_vid_frame = max_end_vid_frame
                    
                assert end_vid_frame <= start_vid_frame + MAX_VID_LEN and end_vid_frame <= video_feature.size(0), "INVALID END_VID_FRAME"
                
                # == slice and pad ==
                
                # 保留 [start_vid_frame, end_vid_frame) 之间的
                padded_vidfeat = torch.zeros((MAX_VID_LEN, self.feature_dim), 
                                            dtype=video_feature.dtype).to(self.device)
                                            
                length = end_vid_frame - start_vid_frame
                padded_vidfeat[:length, :] = video_feature[start_vid_frame:end_vid_frame, :]
                padded_vidfeats.append(padded_vidfeat)
                attention_mask[i, :length] = True

                # == 计算 transition features ==
                cnt_transition_feats, cnt_transition_masks = self.get_transition_features(padded_vidfeat, attention_mask[i])
                transition_features.append(cnt_transition_feats)
                transition_mask[i]=cnt_transition_masks
    
        transition_features = torch.stack(transition_features).to(self.device)
        
        # padded_vidfeats = torch.stack(padded_vidfeats).to(self.device) # No need
        # import pdb;pdb.set_trace()
        return transition_features, transition_mask

class PostProcessConditioner(Conditioner):

    def __init__(
            self,
            feature_dim: int,
            output_dim: int,
            project_out: bool = False,
            frame_per_second: int = None,
            target_per_second: int = None,
            use_abs_pos_emb: bool = False,
            max_duration: int = 10,
            option: bool = False,
            online_extract_type: str = None,
            synchformer_ckpt: str = os.path.join(_PKG_ROOT, "weights", "synchformer_state_dict.pth"),
            use_learnable_mask: bool = False,  # 是否使用可学习的 mask token 替代零
            ):
        super().__init__(feature_dim, output_dim=output_dim, project_out=project_out, option=option)
        self.feature_dim = feature_dim
        self.frame_per_second = frame_per_second
        self.target_per_second = target_per_second
        self.use_abs_pos_emb = use_abs_pos_emb
        self.max_duration = max_duration
        self.online_extract_type = online_extract_type
        self.synchformer_ckpt = synchformer_ckpt
        self.use_learnable_mask = use_learnable_mask

        # 可学习的 mask token，用于替换被 mask 掉的位置（全零帧）
        if use_learnable_mask:
            self.mask_token = nn.Parameter(torch.zeros(1, feature_dim))
            nn.init.normal_(self.mask_token, std=0.02)
            logging.info(f"PostProcessConditioner: 使用可学习 mask token, shape=[1, {feature_dim}]")

        if use_abs_pos_emb:
            max_len = max_duration * (target_per_second if target_per_second else frame_per_second)
            self.abs_pos_emb = AbsolutePositionalEmbedding(output_dim, max_len)

        # Initialize CLIP model for online extraction
        if self.online_extract_type == "mm_clip":
            self._init_clip_model()
        # Initialize Syncformer for online extraction
        elif self.online_extract_type == "mm_sync":
            self._init_syncformer()

    def _patch_clip(self, clip_model):
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

    def _init_clip_model(self):
        # DFN5B CLIP: resolve via HF hub by default. For offline / air-gapped
        # runs set CINEDUB_CLIP_LOCAL to either the directory containing
        # `open_clip_pytorch_model.bin` or the .bin file itself.
        clip_local = os.environ.get("CINEDUB_CLIP_LOCAL", None)
        create_kwargs = {}
        if clip_local:
            if os.path.isdir(clip_local):
                create_kwargs["cache_dir"] = clip_local
            else:
                create_kwargs["pretrained"] = clip_local
        self.clip_model = create_model_from_pretrained(
            'hf-hub:apple/DFN5B-CLIP-ViT-H-14-384',
            **create_kwargs,
        )[0]
        param_count = sum(p.numel() for p in self.clip_model.parameters())
        logging.info(f"Loaded CLIP: model=DFN5B-CLIP-ViT-H-14-384, params={param_count/1e6:.1f}M")


        # Initialize transforms matching vgg_sound_dys_clip.py
        _CLIP_SIZE = 384
        self.clip_transform = v2.Compose([
            v2.Resize((_CLIP_SIZE, _CLIP_SIZE), interpolation=v2.InterpolationMode.BICUBIC),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])

        self.clip_preprocess = Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                         std=[0.26862954, 0.26130258, 0.27577711])
        self.clip_model = self._patch_clip(self.clip_model)
        self.tokenizer = open_clip.get_tokenizer('ViT-H-14-378-quickgelu')  # same as 'ViT-H-14'

    def _init_syncformer(self):
        # Initialize Syncformer model similar to features_utils_dys.py
        assert self.synchformer_ckpt is not None, "synchformer_ckpt must be provided for mm_sync mode"

        # Resolve the checkpoint path: honor CINEDUB_SYNCHFORMER_CKPT, then the
        # local default, then auto-download from hkchengrex/MMAudio on HF.
        self.synchformer_ckpt = _resolve_synchformer_ckpt(self.synchformer_ckpt)

        self.synchformer = Synchformer()
        self.synchformer.load_state_dict(
            torch.load(self.synchformer_ckpt, weights_only=True, map_location='cpu'))
        param_count = sum(p.numel() for p in self.synchformer.parameters())
        logging.info(f"Loaded Synchformer: path={self.synchformer_ckpt}, params={param_count/1e6:.1f}M")

        # Initialize sync transform matching vgg_sound_dys.py
        _SYNC_SIZE = 224
        self.sync_transform = v2.Compose([
            v2.Resize(_SYNC_SIZE, interpolation=v2.InterpolationMode.BICUBIC),
            v2.CenterCrop(_SYNC_SIZE),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _read_video_with_torio(self, video_path: str, start_time=None, end_time=None, fps: float = 8.0):
        """Read video with moviepy (replacement for torio). Local files only —
        the internal BOS/S3 fetch branch was removed for the open-source release.
        """
        original_video_path = video_path  # kept for error messages

        try:
            with VideoFileClip(video_path) as clip:
                # Set time range
                if start_time is not None or end_time is not None:
                    start_t = start_time if start_time is not None else 0
                    end_t = end_time if end_time is not None else clip.duration
                    clip = clip.subclip(start_t, end_t)

                # Extract frames at specified fps
                frames = []
                duration = clip.duration
                frame_count = int(duration * fps)

                for i in range(frame_count):
                    t = i / fps
                    if t >= duration:
                        break
                    frame = clip.get_frame(t)  # Returns numpy array (H, W, C) in RGB format
                    frames.append(torch.from_numpy(frame))

                if frames:
                    result = torch.stack(frames)  # Shape: (T, H, W, C)
                    #(T, H, W, C)-> (T, C, H, W)
                    return result.permute(0, 3, 1, 2)
                else:
                    return None

        except Exception as e:
            print(f"Error reading video {original_video_path}: {e}")
            return None

    @torch.inference_mode()
    def _encode_video_with_clip(self, x: torch.Tensor, frame_lens: torch.Tensor = None):
        """Encode video with CLIP model, based on features_utils_clip.py"""
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
        # Reorganize features for each sample
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
    def _encode_video_with_sync(self, x: torch.Tensor, frame_lens: torch.Tensor = None):
        """Encode video with Synchformer model, based on features_utils_dys.py"""
        assert self.synchformer is not None, 'Synchformer is not loaded'

        b, t, c, h, w = x.shape
        assert c == 3 and h == 224 and w == 224

        segment_size = 16
        step_size = 8

        if frame_lens is None:
            frame_lens = torch.full((b,), t, device=x.device)

        all_segments = []
        segment_nums = []
        sample_indices = []  # 记录每个segment属于哪个sample

        # 为每个sample单独处理
        for i in range(b):
            original_len = min(frame_lens[i], t)

            if original_len < segment_size:
                segment_nums.append(0)
                continue

            # 计算该sample的segment数量
            num_segments = (original_len - segment_size) // step_size + 1
            max_usable_length = (num_segments - 1) * step_size + segment_size

            # 提取segments
            sample_x = x[i:i+1, :max_usable_length]  # (1, T', C, H, W)

            for j in range(num_segments):
                start_idx = j * step_size
                end_idx = start_idx + segment_size
                segment = sample_x[:, start_idx:end_idx]  # (1, 16, C, H, W)
                all_segments.append(segment)
                sample_indices.append(i)

            segment_nums.append(num_segments)

        if not all_segments:
            raise ValueError("没有有效的segments")

        # 批量编码所有segments
        all_segments = torch.cat(all_segments, dim=0)  # (total_segments, 16, C, H, W)
        all_segments = all_segments.unsqueeze(1)  # (total_segments, 1, 16, C, H, W)

        encoded_segments = self.synchformer(all_segments)  # (total_segments, 1, 16, D)
        encoded_segments = encoded_segments.squeeze(1)  # (total_segments, 16, D)

        # 重新组织为每个sample的特征
        encoded_features = []
        segment_start = 0

        for i, num_segs in enumerate(segment_nums):
            if num_segs == 0:
                encoded_features.append(torch.empty(0, encoded_segments.size(-1), device=x.device))
            else:
                sample_features = encoded_segments[segment_start:segment_start + num_segs]
                from einops import rearrange
                sample_features = rearrange(sample_features, 's t d -> (s t) d')  # (segment_time, 768)
                encoded_features.append(sample_features)
                segment_start += num_segs

        return encoded_features, segment_nums

    def _process_mp4_paths(self, mp4_paths, device):
        """Process mp4 paths to extract CLIP features"""
        batch_videos = []
        frame_lens = []
        clip_fps = 8.0  # CLIP frame rate

        for mp4_path in mp4_paths:
            # Read video
            video_frames = self._read_video_with_torio(mp4_path, fps=clip_fps)

            if video_frames is None:
                raise RuntimeError(f'Failed to read video {mp4_path}')

            # Apply clip_transform (matching vgg_sound_dys_clip.py)
            # video_frames is (T, H, W, C) uint8, clip_transform expects this format
            clip_chunk = self.clip_transform(video_frames)
            # After clip_transform: (T, C, H, W) float32 in [0,1], already resized to 384x384

            batch_videos.append(clip_chunk)
            frame_lens.append(clip_chunk.shape[0])

        # Pad videos to same length
        max_frames = max(frame_lens)
        padded_videos = []
        for video in batch_videos:
            if video.shape[0] < max_frames:
                padding = torch.zeros((max_frames - video.shape[0], video.shape[1], video.shape[2], video.shape[3]),
                                    dtype=video.dtype, device=video.device)
                video = torch.cat([video, padding], dim=0)
            padded_videos.append(video)

        # Stack to batch tensor: (B, T, C, H, W)
        batch_tensor = torch.stack(padded_videos, dim=0).to(device)
        frame_lens_tensor = torch.tensor(frame_lens, device=device)

        # Move CLIP model to device
        self.clip_model.to(device)

        # Extract CLIP features
        clip_features, _ = self._encode_video_with_clip(batch_tensor, frame_lens_tensor)

        # Convert clip features to the expected format
        # CLIP features are list of tensors with shape (T, clip_feature_dim)
        # We may need to project them to match self.feature_dim if different
        return clip_features

    def _process_mp4_paths_sync(self, mp4_paths, device):
        """Process mp4 paths to extract Sync features"""
        batch_videos = []
        frame_lens = []
        sync_fps = 25.0  # Sync frame rate

        for mp4_path in mp4_paths:
            # Read video for sync processing
            video_frames = self._read_video_with_torio(mp4_path, fps=sync_fps)

            if video_frames is None:
                raise RuntimeError(f'Failed to read video {mp4_path}')

            # Apply sync_transform (matching vgg_sound_dys.py)
            # video_frames is (T, H, W, C) uint8, sync_transform expects this format
            sync_chunk = self.sync_transform(video_frames)
            # After sync_transform: (T, C, H, W) float32 in [-1,1], already resized to 224x224

            batch_videos.append(sync_chunk)
            frame_lens.append(sync_chunk.shape[0])

        # Pad videos to same length
        max_frames = max(frame_lens)
        padded_videos = []
        for video in batch_videos:
            if video.shape[0] < max_frames:
                padding = torch.zeros((max_frames - video.shape[0], video.shape[1], video.shape[2], video.shape[3]),
                                    dtype=video.dtype, device=video.device)
                video = torch.cat([video, padding], dim=0)
            padded_videos.append(video)

        # Stack to batch tensor: (B, T, C, H, W)
        batch_tensor = torch.stack(padded_videos, dim=0).to(device)
        frame_lens_tensor = torch.tensor(frame_lens, device=device)

        # Move Synchformer to device
        self.synchformer.to(device)

        # Extract Sync features
        sync_features, _ = self._encode_video_with_sync(batch_tensor, frame_lens_tensor)

        return sync_features

    def forward(self, x: tp.Any, device: tp.Any = None) -> tp.Any:
        # x is a list of tensors, each with shape (T, C)

        # === STEP 0: Parse tuple inputs with mask_ranges ===
        # Items may be tuple(value, mask_ranges) from _parse_feature_with_mask
        # Extract mask_ranges and unwrap to plain values for downstream processing
        mask_ranges_list = [None] * len(x)
        parsed_x = []
        for i, item in enumerate(x):
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], list):
                parsed_x.append(item[0])       # path (str) or tensor
                mask_ranges_list[i] = item[1]   # [[start_sec, end_sec], ...]
            else:
                parsed_x.append(item)
        x = parsed_x

        # Check if we need to extract CLIP features from mp4 paths
        if (self.online_extract_type == "mm_clip" and
            len(x) > 0 and x[0] is not None and
            isinstance(x[0], str) and
            x[0].endswith('.mp4')):
            assert self.feature_dim == 1024, "For mm_clip extraction, feature_dim must be 1024"
            # Process mp4 paths to extract CLIP features
            x = self._process_mp4_paths(x, device)
        elif (self.online_extract_type == "mm_sync" and
            len(x) > 0 and x[0] is not None and
            isinstance(x[0], str) and
            x[0].endswith('.mp4')):
            assert self.feature_dim == 768, "For mm_sync extraction, feature_dim must be 768"

            # Process mp4 paths to extract Sync features
            # Returns list of tensors with shape (segment_time, self.feature_dim)
            x = self._process_mp4_paths_sync(x, device)

        # for demo generation with TA syncfeature define in training/utils.py
        if len(x) > 0 and x[0] is None:
            max_target_length = int(self.max_duration * self.target_per_second) if self.target_per_second else int(self.max_duration * self.frame_per_second)
            x = [torch.zeros((max_target_length, self.feature_dim), dtype=torch.float32).to(device) for _ in range(len(x))]
        # for demo generation with paths (non-mp4)
        elif len(x) > 0 and x[0] is not None and isinstance(x[0], (str, list)) and not (isinstance(x[0], str) and x[0].endswith('.mp4')):
            x = [torch.from_numpy(np.concatenate([np.load(p) for p in item], axis=0) if isinstance(item, list) else np.load(item)).to(device) for item in x]  # list: 多片段特征沿时间维在线拼接

        # === STEP 0.5: Apply mask_ranges — zero out time ranges before learnable mask ===
        # mask_ranges are pre-snapped to sync block boundaries (with overlap expansion)
        # Here we just convert seconds → frame indices using self.frame_per_second
        fps = self.frame_per_second
        for i, ranges in enumerate(mask_ranges_list):
            if ranges is not None and isinstance(x[i], torch.Tensor):
                T = x[i].shape[0]
                for start_sec, end_sec in ranges:
                    start_frame = max(0, int(math.floor(start_sec * fps)))
                    end_frame = min(int(math.ceil(end_sec * fps)), T)
                    if start_frame < end_frame:
                        x[i][start_frame:end_frame] = 0.0
                logging.info(f"PostProcessConditioner(fps={fps}): applied mask_ranges {ranges} "
                             f"to sample {i}, T={T}")

        # Get device from first tensor
        if device is None:
            device = x[0].device if len(x) > 0 and x[0] is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Get dimensions and original lengths
        batch_size = len(x)
        feature_dim = x[0].shape[1]
        original_lengths = torch.tensor([tensor.shape[0] for tensor in x], device=device)
        max_original_length = original_lengths.max().item()

        # === STEP 1: Batch pad to same length ===
        # Create batch tensor by padding shorter sequences
        batch_tensor = torch.zeros((batch_size, max_original_length, feature_dim), dtype=x[0].dtype, device=device)
        for i, tensor in enumerate(x):
            seq_len = tensor.shape[0]
            batch_tensor[i, :seq_len] = tensor.to(device)

        # === STEP 1.5: 用 learnable mask token 替换全零帧（在插值之前）===
        if self.use_learnable_mask:
            # 检测每个时间步是否全零 (被 mask 的位置)
            # batch_tensor: [B, T, C], is_zero_frame: [B, T]
            is_zero_frame = (batch_tensor == 0).all(dim=-1)

            # 只替换原始序列长度内的全零帧，padding 部分不替换
            for i in range(batch_size):
                seq_len = original_lengths[i].item()
                zero_mask = is_zero_frame[i, :seq_len]
                if zero_mask.any():
                    batch_tensor[i, :seq_len][zero_mask] = self.mask_token.to(batch_tensor.dtype)

        # === STEP 2: Batch interpolation (if needed) ===
        expected_lengths = original_lengths.clone()  # Will be updated if interpolation happens
        
        if self.target_per_second and self.frame_per_second:
            target_per_second, frame_per_second = self.target_per_second, self.frame_per_second
            if frame_per_second != target_per_second:
                # Calculate new lengths after interpolation
                ratio = target_per_second / frame_per_second
                expected_lengths = (original_lengths.float() * ratio).long()
                max_expected_length = expected_lengths.max().item()
                
                if max_expected_length > 0:
                    # Batch interpolation: (B, T, C) -> (B, C, T) -> interpolate -> (B, T, C)
                    batch_tensor = F.interpolate(
                        batch_tensor.transpose(1, 2),  # (B, T, C) -> (B, C, T)
                        size=max_expected_length,
                        mode='nearest-exact'
                    ).transpose(1, 2)  # (B, C, T) -> (B, T, C)
        
        # === STEP 3: Calculate final target length ===
        if self.target_per_second:
            max_target_length = int(self.max_duration * self.target_per_second)
        else:
            max_target_length = int(self.max_duration * self.frame_per_second)
        
        # === STEP 4: Batch padding/truncation to final length ===
        current_length = batch_tensor.shape[1]
        if current_length < max_target_length:
            # Batch padding
            pad_length = max_target_length - current_length
            batch_tensor = F.pad(batch_tensor, (0, 0, 0, pad_length), mode='constant', value=0)
        elif current_length > max_target_length:
            # Batch truncation
            batch_tensor = batch_tensor[:, :max_target_length, :]
            # Update expected lengths (clip to max_target_length)
            expected_lengths = torch.clamp(expected_lengths, max=max_target_length)
        
        # === STEP 5: Batch mask creation ===
        # Create mask efficiently using broadcasting
        sequence_indices = torch.arange(max_target_length, device=device).unsqueeze(0)  # (1, T)
        expected_lengths_expanded = expected_lengths.unsqueeze(1)  # (B, 1)
        mask = sequence_indices < expected_lengths_expanded  # (B, T) - broadcasting magic!
        
        # === STEP 6: Batch projection ===
        batch_tensor = self.proj_out(batch_tensor)
        
        # === STEP 7: Batch positional embedding ===
        if self.use_abs_pos_emb:
            batch_tensor = batch_tensor + self.abs_pos_emb(batch_tensor)
        
        # === STEP 8: Handle edge case ===
        if torch.all(batch_tensor == 0):
            mask = torch.zeros(batch_size, max_target_length, dtype=torch.bool, device=batch_tensor.device)

        return batch_tensor, mask

def create_multi_conditioner_from_conditioning_config(config: tp.Dict[str, tp.Any], pretransform=None) -> MultiConditioner:
    """
    Create a MultiConditioner from a conditioning config dictionary

    Args:
        config: the conditioning config dictionary
        device: the device to put the conditioners on
    """
    conditioners = {}
    cond_dim = config["cond_dim"]
    
    default_keys = config.get("default_keys", {})

    pre_encoded_keys = config.get("pre_encoded_keys", [])

    for conditioner_info in config["configs"]:
        id = conditioner_info["id"]

        conditioner_type = conditioner_info["type"]

        conditioner_config = conditioner_info["config"]
        
        if "output_dim" not in conditioner_config:
            conditioner_config.update({"output_dim": cond_dim})

        if conditioner_type == "t5":
            conditioners[id] = T5Conditioner(**conditioner_config)
        elif conditioner_type == "gemma_t5":
            conditioners[id] = GemmaT5Conditioner(**conditioner_config)
        elif conditioner_type == "transcription":
            conditioners[id] = TranscriptionConditioner(**conditioner_config)
        elif conditioner_type == "post_process":
            conditioners[id] = PostProcessConditioner(**conditioner_config)
        elif conditioner_type == "clap_text":
            conditioners[id] = CLAPTextConditioner(**conditioner_config)
        elif conditioner_type == "clap_audio":
            conditioners[id] = CLAPAudioConditioner(**conditioner_config)
        elif conditioner_type == "int":
            conditioners[id] = IntConditioner(**conditioner_config)
        elif conditioner_type == "number":
            conditioners[id] = NumberConditioner(**conditioner_config)
        elif conditioner_type == "list":
            conditioners[id] = ListConditioner(**conditioner_config)
        elif conditioner_type == "phoneme":
            conditioners[id] = PhonemeConditioner(**conditioner_config)
        elif conditioner_type == "lut":
            conditioners[id] = TokenizerLUTConditioner(**conditioner_config)
        elif conditioner_type == "vidfeat":
            conditioners[id] = VideoFeatureConditioner(**conditioner_config)
        elif conditioner_type == "transfeat":
            conditioners[id] = TransitionFeatureConditioner(**conditioner_config)
        elif conditioner_type == "pretransform":
            sample_rate = conditioner_config.pop("sample_rate", None)
            assert sample_rate is not None, "Sample rate must be specified for pretransform conditioners"

            use_model_pretransform = conditioner_config.pop("use_model_pretransform", False)

            if not use_model_pretransform:
                cond_pretransform = create_pretransform_from_config(conditioner_config.pop("pretransform_config"), sample_rate=sample_rate)
            else:
                assert pretransform is not None, "Model pretransform must be specified for pretransform conditioners"
                cond_pretransform = pretransform

            if conditioner_config.get("pretransform_ckpt_path", None) is not None:
                cond_pretransform.load_state_dict(load_ckpt_state_dict(conditioner_config.pop("pretransform_ckpt_path")))

            conditioners[id] = PretransformConditioner(cond_pretransform, **conditioner_config)
        elif conditioner_type == "source_mix":
            sample_rate = conditioner_config.pop("sample_rate", None)
            assert sample_rate is not None, "Sample rate must be specified for source_mix conditioners"

            use_model_pretransform = conditioner_config.pop("use_model_pretransform", False)

            if not use_model_pretransform:
                cond_pretransform = create_pretransform_from_config(conditioner_config.pop("pretransform_config"), sample_rate=sample_rate)
            else:
                assert pretransform is not None, "Model pretransform must be specified for source_mix conditioners if use_model_pretransform is True"
                cond_pretransform = pretransform

            if conditioner_config.get("pretransform_ckpt_path", None) is not None:
                cond_pretransform.load_state_dict(load_ckpt_state_dict(conditioner_config.pop("pretransform_ckpt_path")))

            conditioners[id] = SourceMixConditioner(cond_pretransform, **conditioner_config)
        else:
            raise ValueError(f"Unknown conditioner type: {conditioner_type}")

    return MultiConditioner(conditioners, default_keys=default_keys, pre_encoded_keys=pre_encoded_keys)