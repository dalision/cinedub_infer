import torch
from safetensors.torch import load_file

from torch.nn.utils import remove_weight_norm

def load_ckpt_state_dict(ckpt_path, use_ema=False):
    print(f"loading checkpoint... -> {ckpt_path}")
    if ckpt_path.endswith(".safetensors"):
        state_dict = load_file(ckpt_path)
    elif ckpt_path.endswith(".ckpt"):
        # PyTorch Lightning .ckpt files bundle callback state, EMA objects,
        # optimizer configs and DictConfig hparams alongside the actual
        # tensors. torch>=2.4 blocks these under weights_only=True with an
        # UnpicklingError before we can reach the ["state_dict"] key, so we
        # explicitly opt back into the full pickle loader for this branch.
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    else:
        # Bare .pth / .bin containing only tensors — safe to load in the
        # hardened weights_only mode.
        loaded = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state_dict = loaded["state_dict"] if isinstance(loaded, dict) and "state_dict" in loaded else loaded
    # Detect whether the checkpoint needs unwrapping — compatible with both
    # plain checkpoints and torch.compile-wrapped ones.
    #   plain:   "diffusion.model.model.xxx"
    #   compile: "diffusion.model._orig_mod.model.xxx"
    unwrap_keys = [key for key in state_dict.keys()
                   if "diffusion.model.model" in key or "diffusion.model._orig_mod.model" in key]
    if len(unwrap_keys) > 0:
        state_dict = unwrap_model(state_dict, use_ema=use_ema)
        print(f"Unwrap checkpoint... use_ema is {use_ema} -> {ckpt_path}")
    return state_dict

def unwrap_model(model_params, use_ema=True):
    unwrapped_params = {}

    for key, value in model_params.items():
        # Strip the torch.compile-added "_orig_mod." prefix if present, e.g.
        #   diffusion.model._orig_mod.model.xxx -> diffusion.model.model.xxx
        processed_key = key.replace("._orig_mod.", ".")

        new_key = processed_key

        if use_ema:
            # EMA-only export: drop the training tower, keep the EMA tower.
            if processed_key.startswith("diffusion.model."):
                continue
            elif processed_key.startswith("diffusion."):
                new_key = processed_key.replace("diffusion.", "", 1)
            elif processed_key.startswith("diffusion_ema.ema_model."):
                new_key = processed_key.replace("diffusion_ema.ema_model.", "model.", 1)
            else:
                continue
        else:
            # Non-EMA export: promote the training tower into model.*.
            if processed_key.startswith("diffusion.model."):
                new_key = processed_key.replace("diffusion.model.", "model.", 1)
            elif processed_key.startswith("diffusion."):
                new_key = processed_key.replace("diffusion.", "", 1)
            else:
                continue

        unwrapped_params[new_key] = value

    return unwrapped_params
    
def remove_weight_norm_from_model(model):
    for module in model.modules():
        if hasattr(module, "weight"):
            print(f"Removing weight norm from {module}")
            remove_weight_norm(module)

    return model

# Get torch.compile flag from environment variable ENABLE_TORCH_COMPILE

import os
enable_torch_compile = os.environ.get("ENABLE_TORCH_COMPILE", "0") == "1"

def compile(function, *args, **kwargs):
    
    if enable_torch_compile:
        # try:
        return torch.compile(function, *args, **kwargs)
        # except RuntimeError:
        #     return function

    return function

# Sampling functions copied from https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/utils/utils.py under MIT license
# License can be found in LICENSES/LICENSE_META.txt

def multinomial(input: torch.Tensor, num_samples: int, replacement=False, *, generator=None):
    """torch.multinomial with arbitrary number of dimensions, and number of candidates on the last dimension.

    Args:
        input (torch.Tensor): The input tensor containing probabilities.
        num_samples (int): Number of samples to draw.
        replacement (bool): Whether to draw with replacement or not.
    Keywords args:
        generator (torch.Generator): A pseudorandom number generator for sampling.
    Returns:
        torch.Tensor: Last dimension contains num_samples indices
            sampled from the multinomial probability distribution
            located in the last dimension of tensor input.
    """

    if num_samples == 1:
        q = torch.empty_like(input).exponential_(1, generator=generator)
        return torch.argmax(input / q, dim=-1, keepdim=True).to(torch.int64)

    input_ = input.reshape(-1, input.shape[-1])
    output_ = torch.multinomial(input_, num_samples=num_samples, replacement=replacement, generator=generator)
    output = output_.reshape(*list(input.shape[:-1]), -1)
    return output


def sample_top_k(probs: torch.Tensor, k: int) -> torch.Tensor:
    """Sample next token from top K values along the last dimension of the input probs tensor.

    Args:
        probs (torch.Tensor): Input probabilities with token candidates on the last dimension.
        k (int): The k in “top-k”.
    Returns:
        torch.Tensor: Sampled tokens.
    """
    top_k_value, _ = torch.topk(probs, k, dim=-1)
    min_value_top_k = top_k_value[..., [-1]]
    probs *= (probs >= min_value_top_k).float()
    probs.div_(probs.sum(dim=-1, keepdim=True))
    next_token = multinomial(probs, num_samples=1)
    return next_token


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Sample next token from top P probabilities along the last dimension of the input probs tensor.

    Args:
        probs (torch.Tensor): Input probabilities with token candidates on the last dimension.
        p (int): The p in “top-p”.
    Returns:
        torch.Tensor: Sampled tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort *= (~mask).float()
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def next_power_of_two(n):
    return 2 ** (n - 1).bit_length()

def next_multiple_of_64(n):
    return ((n + 63) // 64) * 64