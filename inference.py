import os

import math
import json

import torch
import torchaudio
# torchaudio 2.x prefers torchcodec, which needs a recent ffmpeg / glibc. On
# hosts with older toolchains torchcodec's .so may fail to load, so we fall
# back to soundfile (libsndfile) which is dependency-light and covers the
# formats used here (wav / flac / ogg). mp3 stays on the original backend.
try:
    import soundfile as _sf
    _orig_ta_load, _orig_ta_save = torchaudio.load, torchaudio.save
    def _ta_load(path, *a, **k):
        try:
            return _orig_ta_load(path, *a, **k)
        except Exception:
            d, sr = _sf.read(str(path), dtype="float32", always_2d=True)  # [T,C]
            return torch.from_numpy(d.T.copy()), sr                        # [C,T]
    def _ta_save(path, tensor, sample_rate, *a, **k):
        try:
            return _orig_ta_save(path, tensor, sample_rate, *a, **k)
        except Exception:
            arr = tensor.detach().cpu().numpy()
            _sf.write(str(path), arr.T if arr.ndim == 2 else arr, sample_rate)
    torchaudio.load, torchaudio.save = _ta_load, _ta_save
except ImportError:
    pass
from torch.nn import functional as F
import numpy as np
import typing as tp
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.utilities import rank_zero_only
import argparse
import sys
from stable_audio_tools.models.utils import unwrap_model
from accelerate import Accelerator
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.pretrained import get_pretrained_model
from stable_audio_tools.models.utils import load_ckpt_state_dict
from stable_audio_tools.inference.utils import prepare_audio
from stable_audio_tools.inference.sampling import sample, sample_k, sample_rf
from stable_audio_tools.training.utils import copy_state_dict, get_rank, get_world_size,replace_mp4_wav
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import warnings
from stable_audio_tools.data.utils import get_audio_duration_and_sr
from stable_audio_tools.data.jsonl_toolkit import JSONLToolkit, validate_rows
from stable_audio_tools.training.utils import generate_multimodal_tasks
import logging
import random
import sys
import tempfile
import uuid


# Echo the full invocation to make logs self-explanatory when replaying runs.
print(" ".join(sys.argv))

# ─── Audio Prompt 常量 ───────────────────────────────────────────────────────
_AP_SAMPLE_RATE = 16000
_AP_BLOCK_SIZE = 640  # VAE downsampling ratio
_AP_FIXED_SAMPLES = int(3.0 * _AP_SAMPLE_RATE / _AP_BLOCK_SIZE) * _AP_BLOCK_SIZE  # 48000，与训练保持一致

# seconds_total 上限 = sample_size(160000) / sample_rate(16000) = 10s。
# 训练样本全 ≤10s，推理传 >10 的 seconds_total 是分布外(OOD)值，会让模型(尤其多人 m2)整段生成崩。
# 必须钳到上限。换模型若 sample_size 变化需同步改。
_MAX_SECONDS_TOTAL = 10.0


def _load_audio_prompt_tensor(wav_path):
    """加载 wav，取前3秒并 pad 到 48000 samples（与训练时 _generate_audio_prompt 一致）。"""
    wav, sr = torchaudio.load(wav_path)
    if sr != _AP_SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, _AP_SAMPLE_RATE)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav / (wav.abs().max() + 1e-8)   # 峰值归一化，对齐训练 pad_crop_load(dataset.py:569)
    if wav.shape[1] >= _AP_FIXED_SAMPLES:
        wav = wav[:, :_AP_FIXED_SAMPLES]
    else:
        wav = F.pad(wav, (0, _AP_FIXED_SAMPLES - wav.shape[1]))
    return wav  # [1, 48000]


# ─── 多人 Audio Prompt（与训练 _generate_audio_prompt 多人路径对齐）────────────
_AP_PER_SPK_MAX = int(2.0 * _AP_SAMPLE_RATE / _AP_BLOCK_SIZE) * _AP_BLOCK_SIZE   # 32000 (2s) 每人上限
_AP_HALF = int(1.5 * _AP_SAMPLE_RATE / _AP_BLOCK_SIZE) * _AP_BLOCK_SIZE          # 24000 (1.5s) 默认平分


def _normalize_ref_entries(ref):
    """把 ref_wav 归一化成 List[entry]，entry 是 str 或 [path, start, end]。
    消歧：flat [path, start, end]（第一位 str、第二位 number）当作单个截段 entry。
    """
    if isinstance(ref, str):
        return [ref]
    if isinstance(ref, (list, tuple)):
        if len(ref) == 0:
            return []
        if isinstance(ref[0], str) and len(ref) >= 3 and isinstance(ref[1], (int, float)):
            return [list(ref)]   # flat [path, start, end] → 单个截段 entry
        return list(ref)
    raise ValueError(f"Unsupported ref_wav type: {type(ref)}")


def _load_one_ref_clip(entry):
    """加载单个 ref entry → [1, T] (16k mono)。entry: str 或 [path, start, end]（按时间戳截段）。"""
    if isinstance(entry, (list, tuple)):
        path = entry[0]
        start_sec, end_sec = float(entry[1]), float(entry[2])
    else:
        path, start_sec, end_sec = entry, None, None
    wav, sr = torchaudio.load(path)
    if sr != _AP_SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, _AP_SAMPLE_RATE)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav / (wav.abs().max() + 1e-8)   # 峰值归一化（先归一化整段再截，对齐训练 pad_crop_load）
    if start_sec is not None:
        s = max(0, int(start_sec * _AP_SAMPLE_RATE))
        e = int(end_sec * _AP_SAMPLE_RATE)
        wav = wav[:, s:e]
    return wav


def _allocate_multi_budget(lens):
    """确定性分配每人取样点数（无 random）。lens: 各 clip 实际样本数。
    - 1 人：取 ≤3s
    - 2 人：都 ≥1.5s → 平分 1.5+1.5；有不足 → 不足的取满 cap(≤2s)，富余的补到总 3s(≤2s)，仍不足则 pad
    """
    n = len(lens)
    if n == 1:
        return [min(lens[0], _AP_FIXED_SAMPLES)]
    caps = [min(l, _AP_PER_SPK_MAX) for l in lens]
    if caps[0] >= _AP_HALF and caps[1] >= _AP_HALF:
        return [_AP_HALF, _AP_HALF]
    if caps[0] + caps[1] <= _AP_FIXED_SAMPLES:
        return caps  # 都吃满，剩下 pad
    small_idx = 0 if caps[0] <= caps[1] else 1
    big_idx = 1 - small_idx
    res = [0, 0]
    res[small_idx] = caps[small_idx]
    res[big_idx] = min(caps[big_idx], _AP_FIXED_SAMPLES - caps[small_idx])
    return res


def _load_multi_audio_prompt_tensor(ref):
    """多人 audio prompt：ref 是 str 或 list（最多取前 2 人）。返回 [1, 48000]。
    每人从头取分配长度（block 对齐），按 list 顺序 concat，pad/截到 48000。
    结构与训练多人 prompt（[spk1][spk2] 拼接）一致。
    """
    entries = _normalize_ref_entries(ref)
    if not entries:
        raise ValueError("empty ref_wav for multi audio prompt")
    entries = entries[:2]                       # 最多 2 人
    clips = [_load_one_ref_clip(e) for e in entries]
    targets = _allocate_multi_budget([c.shape[1] for c in clips])
    pieces = []
    for clip, t in zip(clips, targets):
        t = (t // _AP_BLOCK_SIZE) * _AP_BLOCK_SIZE   # block 对齐
        t = min(t, clip.shape[1])
        if t > 0:
            pieces.append(clip[:, :t])               # 从头取
    if not pieces:
        return torch.zeros(1, _AP_FIXED_SAMPLES)
    concat = torch.cat(pieces, dim=1)
    if concat.shape[1] < _AP_FIXED_SAMPLES:
        concat = F.pad(concat, (0, _AP_FIXED_SAMPLES - concat.shape[1]))
    else:
        concat = concat[:, :_AP_FIXED_SAMPLES]
    return concat


def inject_audio_prompt(task2dict):
    """
    向 task2dict 中所有 conditioning dict 注入 _audio_prompt_path（路径或 list）。
    实际 tensor 加载延迟到 process_batch 中按 batch 进行，避免全量加载导致 OOM。
    优先级：item 自带的 _ref_wav_path（JSONL 里的 ref_wav 字段） > _wav_path（自身 wav）
    多人模式下 _ref_wav_path 可以是 list（[路径...] 或含 [path,start,end] 截段），原样透传。
    """
    for task_id, conds in task2dict.items():
        for cond in conds:
            wav_path = cond.get("_ref_wav_path") or cond.get("_wav_path", "")
            if wav_path:
                cond["_audio_prompt_path"] = wav_path
            else:
                logging.warning(f"No ref_wav or wav for {cond.get('id','?')}, skipping audio_prompt")
    return task2dict


def check_single_item(item):
    """检查单个item的特征对齐情况"""
    try:
        clip_feature = np.load(item["clip_feature"])
        sync_feature = np.load(item["sync_feature"])
        audio_duration, sample_rate = get_audio_duration_and_sr(item["path"])
        # SYNC_FPS default 24 matches the offline sync-feature extractor used
        # during training. Override with env CINEDUB_SYNC_FPS if a different
        # extractor pipeline was used.
        _SYNC_FPS = float(os.environ.get("CINEDUB_SYNC_FPS", "24"))
        clip_duration = clip_feature.shape[0] / 8
        sync_duration = sync_feature.shape[0] / _SYNC_FPS
        duration_diff = abs(clip_duration - sync_duration)
        audio_diff_1 = abs(audio_duration - sync_duration)
        audio_diff_2 = abs(audio_duration - clip_duration)

        # if duration_diff > 0.5 or audio_diff_1 > 0.5 or audio_diff_2 > 0.5:
        if duration_diff > 0.5 :
            # 返回包含详细信息的字典
            return {
                **item,
                "clip_duration": clip_duration,
                "sync_duration": sync_duration,
                "duration_diff": duration_diff,
                "audio_duration": audio_duration,
                "audio_sync_diff": audio_diff_1,
                "audio_clip_diff": audio_diff_2
            }
        return None
    except Exception as e:
        logging.info(f"Error processing {item.get('path', 'unknown')}: {e}")
        return {**item, "error": str(e)}

def check_vfeature_alignment(items, max_workers=128):
    """多线程检查视频特征对齐情况"""
    if not items:
        return []
    
    bad_items = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(check_single_item, items))
        bad_items = [item for item in results if item is not None]
    
    logging.info(f"Processed {len(items)} items, found {len(bad_items)} bad items")
    
    # 打印坏的items的详细信息
    if bad_items:
        logging.info("\nBad items details:")
        logging.info("-" * 80)
        for i, item in enumerate(bad_items):
            logging.info(f"Bad item {i+1}:")
            logging.info(f"  Path: {item.get('path', 'unknown')}")
            logging.info(f"  Clip feature: {item.get('clip_feature', 'unknown')}")
            logging.info(f"  Sync feature: {item.get('sync_feature', 'unknown')}")
            
            if 'error' in item:
                logging.info(f"  Error: {item['error']}")
            else:
                logging.info(f"  Audio duration: {item.get('audio_duration', 'N/A'):.3f}s")
                logging.info(f"  Clip duration: {item.get('clip_duration', 'N/A'):.3f}s")
                logging.info(f"  Sync duration: {item.get('sync_duration', 'N/A'):.3f}s")
                logging.info(f"  Duration diff: {item.get('duration_diff', 'N/A'):.3f}s")
                logging.info(f"  Audio-Clip diff: {item.get('audio_clip_diff', 'N/A'):.3f}s")
                logging.info(f"  Audio-Sync diff: {item.get('audio_sync_diff', 'N/A'):.3f}s")
    
    return bad_items

def load_model(model_config=None, model_ckpt_path=None, device="cuda", model_half=False,video_online_extract=False, use_ema=False):

    global model, sample_rate, sample_size, model_type

    logging.info(f"Creating model from config")
    # model_config["bad_model"]=None
    model = create_model_from_config(model_config)

    logging.info(f"Loading model checkpoint from {model_ckpt_path}")

    logging.info(f"using ema: {use_ema}")
    # Load checkpoint
    model_params = load_ckpt_state_dict(model_ckpt_path, use_ema=use_ema)
    
    model.load_state_dict(model_params, strict=True) if not video_online_extract else model.load_state_dict(model_params, strict=False)

    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    model_type = model_config["model_type"]


    model.to(device).eval().requires_grad_(False)

    if model_half:
        model.to(torch.float16)
        
    logging.info(f"Done loading model")

    return model

def generate_diffusion_cond(
        model,
        steps: int = 250,
        cfg_scale=6,
        conditioning: dict = None,
        conditioning_tensors: tp.Optional[dict] = None,
        negative_conditioning: dict = None,
        negative_conditioning_tensors: tp.Optional[dict] = None,
        batch_size: int = 1,
        sample_size: int = 2097152,
        sample_rate: int = 48000,
        seed: int = -1,
        device: str = "cuda",
        init_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        init_noise_level: float = 1.0,
        return_latents = False,
        **sampler_kwargs
        ) -> torch.Tensor: 
    """
    Generate audio from a prompt using a diffusion model.
    
    Args:
        model: The diffusion model to use for generation.
        steps: The number of diffusion steps to use.
        cfg_scale: Classifier-free guidance scale 
        conditioning: A dictionary of conditioning parameters to use for generation.
        conditioning_tensors: A dictionary of precomputed conditioning tensors to use for generation.
        batch_size: The batch size to use for generation.
        sample_size: The length of the audio to generate, in samples.
        sample_rate: The sample rate of the audio to generate (Deprecated, now pulled from the model directly)
        seed: The random seed to use for generation, or -1 to use a random seed.
        device: The device to use for generation.
        init_audio: A tuple of (sample_rate, audio) to use as the initial audio for generation.
        init_noise_level: The noise level to use when generating from an initial audio sample.
        return_latents: Whether to return the latents used for generation instead of the decoded audio.
        **sampler_kwargs: Additional keyword arguments to pass to the sampler.    
    """

    # The length of the output in audio samples 
    audio_sample_size = sample_size

    # If this is latent diffusion, change sample_size instead to the downsampled latent size
    if model.pretransform is not None:
        sample_size = sample_size // model.pretransform.downsampling_ratio
        
    # Seed
    # The user can explicitly set the seed to deterministically generate the same output. Otherwise, use a random seed.
    seed = seed if seed != -1 else np.random.randint(0, 2**32 - 1)
    # logging.info(seed)
    torch.manual_seed(seed)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False

    # Conditioning
    assert conditioning is not None or conditioning_tensors is not None, "Must provide either conditioning or conditioning_tensors"
    if conditioning_tensors is None:
        conditioning_tensors = model.conditioner(conditioning, device)
    conditioning_inputs = model.get_conditioning_inputs(conditioning_tensors)

    if negative_conditioning is not None or negative_conditioning_tensors is not None:
        
        if negative_conditioning_tensors is None:
            negative_conditioning_tensors = model.conditioner(negative_conditioning, device)
            
        negative_conditioning_tensors = model.get_conditioning_inputs(negative_conditioning_tensors, negative=True)
    else:
        negative_conditioning_tensors = {}
    # import pdb;pdb.set_trace()
    if init_audio is not None:
        # The user supplied some initial audio (for inpainting or variation). Let us prepare the input audio.
        in_sr, init_audio = init_audio

        io_channels = model.io_channels

        # For latent models, set the io_channels to the autoencoder's io_channels
        if model.pretransform is not None:
            io_channels = model.pretransform.io_channels

        # Prepare the initial audio for use by the model
        init_audio = prepare_audio(init_audio, in_sr=in_sr, target_sr=model.sample_rate, target_length=audio_sample_size, target_channels=io_channels, device=device)

        # For latent models, encode the initial audio into latents
        if model.pretransform is not None:
            init_audio = model.pretransform.encode(init_audio)

        init_audio = init_audio.repeat(batch_size, 1, 1)

        sampler_kwargs["sigma_max"] = init_noise_level        

    model_dtype = next(model.model.parameters()).dtype
    
    # num_sample (batch size) is determined by condition tensor size
    num_sample = list(conditioning_tensors.values())[0][0].shape[0]
    
    # Define the initial noise immediately after setting the seed
    noise = torch.randn([num_sample, model.io_channels, sample_size], device=device)
    noise = noise.type(model_dtype)
    # import pdb;pdb.set_trace()
    conditioning_inputs = {k: v.type(model_dtype) if v is not None else v for k, v in conditioning_inputs.items()}
    # Now the generative AI part:
    # k-diffusion denoising process go!

    diff_objective = model.diffusion_objective

    if diff_objective == "v":    
        # k-diffusion denoising process go!
        sampled = sample_k(model.model, noise, init_audio, steps, **sampler_kwargs, **conditioning_inputs, **negative_conditioning_tensors, cfg_scale=cfg_scale, batch_cfg=True, rescale_cfg=True, device=device)
        # sampled = sample(model.model, noise, steps, 0, **conditioning_inputs, cfg_scale=cfg_scale)
    elif diff_objective == "rectified_flow":

        if "sigma_min" in sampler_kwargs:
            del sampler_kwargs["sigma_min"]

        if "rho" in sampler_kwargs:
            del sampler_kwargs["rho"]

        sampled = sample_rf(model.model, noise, init_data=init_audio, steps=steps, **sampler_kwargs, **conditioning_inputs, **negative_conditioning_tensors, dist_shift=model.dist_shift, cfg_scale=cfg_scale, batch_cfg=True, rescale_cfg=True, device=device)

    # v-diffusion: 
    #sampled = sample(model.model, noise, steps, 0, **conditioning_tensors, embedding_scale=cfg_scale)
    del noise
    del conditioning_tensors
    del conditioning_inputs
    torch.cuda.empty_cache()
    # Denoising process done. 
    # If this is latent diffusion, decode latents back into audio
    if model.pretransform is not None and not return_latents:
        #cast sampled latents to pretransform dtype
        sampled = sampled.to(next(model.pretransform.parameters()).dtype)
        sampled = model.pretransform.decode(sampled)

    # Return audio
    return sampled

def snap_mask_ranges_to_sync_blocks(mask_ranges, sync_fps=24, sync_block_size=8, overlap_expand=1):
    """
    Snap time ranges to sync block boundaries and expand for Syncformer overlap.

    Because Syncformer uses overlapping sliding windows (segment=16, step=8),
    adjacent blocks also contain information from the masked region.
    So we expand by overlap_expand blocks on each side.

    Args:
        mask_ranges: list of [start_sec, end_sec]
        sync_fps: sync feature fps (default 24)
        sync_block_size: frames per block (default 8, matching training sync_block_size)
        overlap_expand: blocks to expand on each side (default 1)

    Returns:
        list of [snapped_start_sec, snapped_end_sec] — aligned to sync block boundaries
    """
    snapped = []
    block_duration = sync_block_size / sync_fps  # 8/24 ≈ 0.333s per block
    for start_sec, end_sec in mask_ranges:
        start_block = int(math.floor(start_sec / block_duration))
        end_block = int(math.ceil(end_sec / block_duration))
        # Expand for overlap
        start_block = max(0, start_block - overlap_expand)
        end_block = end_block + overlap_expand
        snapped_start = start_block * block_duration
        snapped_end = end_block * block_duration
        snapped.append([snapped_start, snapped_end])
        logging.info(f"  Mask range [{start_sec:.2f}s, {end_sec:.2f}s] → "
                     f"snapped to blocks [{start_block}, {end_block}) = "
                     f"[{snapped_start:.3f}s, {snapped_end:.3f}s]")
    return snapped


def _parse_feature_with_mask(feature_val):
    """
    Parse feature value that may contain mask ranges.
    If mask_ranges present, return tuple (value, snapped_ranges) to survive
    MultiConditioner's list unwrapping (only unwraps list, not tuple of len>1).

    Supports:
        - str: plain path → str (unchanged)
        - list: [path_or_mp4, [[start, end], ...]] → tuple(path, snapped_ranges)
    """
    if isinstance(feature_val, list) and len(feature_val) == 2 and isinstance(feature_val[1], list):
        path = feature_val[0]
        mask_ranges = feature_val[1]
        snapped = snap_mask_ranges_to_sync_blocks(mask_ranges)
        return (path, snapped)  # tuple won't be unwrapped by MultiConditioner
    return feature_val


def load_dataset(jsonl_path,tasks,re_duration=False,demo_ids=None,clamp_seconds_total=True):
    # print("&&&&&&&&&&&& tasks")
    # print(tasks)
    if re_duration:
        logging.info("caculate duration from jsonl")
        items = JSONLToolkit.add_duration(jsonl_path,jsonl_path,use_threading=True,num_threads=128)
    else:
        items = JSONLToolkit.load(jsonl_path)
    converted_data = []

    for item in items:
        # dur 若是 list (如 [d1,d2] 多片段拼接各自时长) → 求和成总时长, 供 seconds_total 用
        if isinstance(item.get("dur"), list):
            item["dur"] = sum(item["dur"])
        # Check if this is demo_cond format (has wav_path instead of wav)
        is_demo_cond_format = "wav_path" in item

        if is_demo_cond_format:
            # Handle demo_cond format
            # Extract id from wav_path if not present
            item_id = item.get("id")
            if not item_id:
                wav_filename = os.path.basename(item["wav_path"])
                item_id = os.path.splitext(wav_filename)[0]

            # Convert demo_cond format to internal format
            # _parse_feature_with_mask: if [path, ranges] → tuple(path, snapped_ranges)
            converted_item = {
                "id": item_id,
                "path": item["video_path"],
                "clip_feature": _parse_feature_with_mask(item.get("clip_feature")) if item.get("clip_feature") else None,
                "sync_feature": _parse_feature_with_mask(item.get("sync_feature")) if item.get("sync_feature") else None,
                "seconds_start": 0.0,
                "seconds_total": (min(item.get("seconds_total", 10.0), _MAX_SECONDS_TOTAL) if clamp_seconds_total else item.get("seconds_total", 10.0)),  # --clamp_seconds_total 开启才钳到上限
                "text_prompt": item.get("text_prompt", "clean speech"),
                "trans_cap": item["trans_cap"],
                "_wav_path": item.get("wav", ""),            # 自身wav，供audio_prompt fallback
                "_ref_wav_path": item.get("ref_wav", ""),   # 显式参考音频，优先于_wav_path
            }

        else:
            # Auto-derive on-the-fly feature paths / id / wav when the JSONL row omits them.
            # PostProcessConditioner extracts CLIP + Synchformer features from mp4 at inference time.
            if "feature" not in item:
                item["feature"] = {}
            _vp = item.get("video_path", "")
            _vp0 = _vp[0] if isinstance(_vp, list) and _vp else (_vp if isinstance(_vp, str) else "")
            item["feature"].setdefault("clip", _vp0)
            item["feature"].setdefault("sync", _vp0)
            item.setdefault("id", os.path.splitext(os.path.basename(_vp0))[0] if _vp0 else "")
            item.setdefault("wav", "")
            # _parse_feature_with_mask: if [path, ranges] → tuple(path, snapped_ranges)
            clip_val = _parse_feature_with_mask(item["feature"]["clip"])
            sync_val = _parse_feature_with_mask(item["feature"]["sync"])

            # audio_duration, sample_rate =  get_audio_duration_and_sr(item["wav"])
            if tasks == ["VTS"]:
                converted_item = {
                    "id": item["id"],
                    "path": (item["video_path"][0] if isinstance(item["video_path"], list) else item["video_path"]),  # 多片段 video list → 取首个作命名/去重key(评测all_demo=0不贴回视频); 贴回多视频需另处理
                    "_video_paths": (item["video_path"] if isinstance(item["video_path"], list) else [item["video_path"]]),  # 完整多视频列表(供贴回concat)
                    "clip_feature": clip_val,
                    "sync_feature": sync_val,
                    "seconds_start": 0.0,  # 默认从0开始
                    "seconds_total": (min(item.get("dur",10), _MAX_SECONDS_TOTAL) if clamp_seconds_total else item.get("dur",10)),  # --clamp_seconds_total 开启才钳到上限
                    "text_prompt": item["cap"][0][0] if "cap" in item else item.get("text_prompt", "clean speech"),
                    "trans_cap": item["trans_cap"],
                    "_wav_path": item.get("wav", ""),          # 自身wav，供audio_prompt fallback
                    "_ref_wav_path": item.get("ref_wav", ""),  # 显式参考音频，优先于_wav_path
                }
            else:
                converted_item = {
                    "id": item["id"],
                    "path": item["wav"],
                    "clip_feature": clip_val,
                    "sync_feature": sync_val,
                    "seconds_start": 0.0,  # 默认从0开始
                    "seconds_total": (min(item.get("dur",10), _MAX_SECONDS_TOTAL) if clamp_seconds_total else item.get("dur",10)),  # --clamp_seconds_total 开启才钳到上限
                    "text_prompt": (item["cap"][0][0] if "cap" in item and item["cap"] else item.get("text_prompt", "")),  # first caption from item["cap"], else fall back to item["text_prompt"] (VTA JSONLs)
                    "trans_cap": item.get("trans_cap", "none_speech")  # honour JSONL trans_cap; default to none_speech when absent
                }
        converted_data.append(converted_item)

    if demo_ids and len(demo_ids) != 0 and len([item for item in converted_data if item["id"] in demo_ids]) != 0:
        #demo_ids 根据demo_ids 把converted_data分成两个部分
        demo_data = [item for item in converted_data if item["id"] in demo_ids]
        non_demo_data = [item for item in converted_data if item["id"] not in demo_ids]
        return generate_multimodal_tasks(demo_data,tasks, add_id=True), generate_multimodal_tasks(non_demo_data,tasks, add_id=True)
    else:
        return generate_multimodal_tasks(converted_data,tasks, add_id=True)

    # if "V" in "".join(tasks):
    #     check_vfeature_alignment(task2dict["VTA"] if "VTA" in task2dict else task2dict["VA"])
    
    

def get_args():
    """CineDub inference CLI.

    Two entry points:

    1. Single file (quick demo):
         python inference.py --input clip.mp4 --task v2a --text_prompt "Rain on tin roof."
         python inference.py --input clip.mp4 --task v2s --ref_wav ref.wav --trans_cap "<S>hi<E>"
         python inference.py --input clip.mp4 --task v2sa \
             --ref_wav spkA.wav,spkB.wav --trans_cap "..." --text_prompt "..."

    2. Batch (JSONL):
         python inference.py --input examples/v2a.jsonl --task v2a

    The old ``--dataset_config`` / ``--model_ckpt_path`` / ``--save_dir`` /
    ``--cfg_scale`` / ``--tasks`` names are still accepted so existing
    scripts keep working, but ``--input / --task / --output / --ckpt / --cfg``
    are the preferred, documented names.
    """
    p = argparse.ArgumentParser(
        description="CineDub inference (V2A / V2S / V2SA)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ─── User-facing (preferred) ─────────────────────────────────────────
    p.add_argument('--input', type=str, default=None,
                   help="Path to a .mp4 (single-file mode) or .jsonl (batch mode). Auto-detected by suffix.")
    p.add_argument('--task', type=str, default=None, choices=['v2a', 'v2s', 'v2sa', 'v2as'],
                   help="High-level mode. v2a=SFX only, v2s=dubbed speech, v2sa=speech+SFX. "
                        "('v2as' is accepted as a deprecated alias for 'v2sa'.) "
                        "Sets --tasks and --use_audio_prompt / --use_multi_audio_prompt for you. "
                        "Mutually exclusive with the legacy --tasks flag.")
    p.add_argument('--output', type=str, default=None,
                   help="Output directory. Overrides --save_dir when both are given.")
    p.add_argument('--ckpt', type=str, default=None,
                   help="Model checkpoint (.ckpt). Overrides --model_ckpt_path when both are given. "
                        "Falls back to $CINEDUB_CKPT (or $CINEDUB_CKPT_MULTISPEAKER for v2sa).")
    p.add_argument('--cfg', type=float, default=None,
                   help="Classifier-free guidance scale. Default depends on --task: v2a=2.5, v2s=7.0, v2sa=7.0.")
    p.add_argument('--seed', type=int, default=-1,
                   help="Random seed. -1 means non-deterministic (default). Any int >=0 is honoured.")

    # Single-file shortcuts (only used when --input is a .mp4)
    p.add_argument('--ref_wav', type=str, default=None,
                   help="For v2s: single wav path. For v2sa: two wav paths separated by ',' "
                        "(e.g. spkA.wav,spkB.wav). Only used with --input <mp4>.")
    p.add_argument('--text_prompt', type=str, default=None,
                   help="SFX / ambient caption. Only used with --input <mp4>.")
    p.add_argument('--trans_cap', type=str, default=None,
                   help="Speech transcript wrapped as <S>utterance<E>. Only used with --input <mp4>.")

    # ─── Legacy / advanced (still supported) ─────────────────────────────
    p.add_argument('--model_config', type=str, default=None)
    p.add_argument('--model_ckpt_path', type=str, default=None,
                   help="Legacy alias for --ckpt.")
    p.add_argument('--dataset_config', type=str, default=None,
                   help="Legacy alias for --input (JSONL only).")
    p.add_argument('--save_dir', type=str, default=None,
                   help="Legacy alias for --output.")
    p.add_argument('--cfg_scale', type=float, default=None,
                   help="Legacy alias for --cfg (same auto-default rule).")
    p.add_argument('--batch_size', type=int, default=4, help="Batch size of sampling.")
    p.add_argument('--tasks', type=str, default=None,
                   help="Legacy explicit task string (VTA / VTS / VTAS / VTA+VA / ...). "
                        "Prefer --task. If neither is given, defaults to 'VTA VA TA' as before.")
    p.add_argument('--re_duration', action='store_true', help="Whether to re-calculate duration.")
    p.add_argument('--demoid_path', type=str, default=None, help="Path to the demo IDs file.")
    p.add_argument('--steps', type=int, default=200, help="Diffusion sampling steps.")
    p.add_argument('--all_demo', action='store_true', help="Run all data as demo mode.")
    p.add_argument('--video_online_extract', action='store_true',
                   help="Force online CLIP+Sync extraction. Auto-enabled if any JSONL row lacks .npy features.")
    p.add_argument('--copy_video_low_quality', action='store_true', help="Whether to copy low quality video when making mp4.")
    p.add_argument('--cache_dir', type=str, default=None, help="Path to the cache directory.")
    p.add_argument('--exp_prefix', type=str, default="",
                   help="Optional path prefix stripped from save_path before applying --cache_dir. "
                        "Leave empty (default) to skip any prefix substitution.")
    p.add_argument('--use_ema', action='store_true', help="Whether to use EMA model weights.")
    p.add_argument('--add_id', action='store_true', default=True, help="Use item id as save filename.")
    p.add_argument('--resume', action='store_true', help="Skip already generated files.")
    p.add_argument('--no_add_id', action='store_false', dest='add_id', help="Use original filename instead of item id.")
    p.add_argument('--use_audio_prompt', action='store_true',
                   help="Enable single-speaker audio prompt. Auto-set by --task v2s/v2sa; usually leave unset.")
    p.add_argument('--use_multi_audio_prompt', action='store_true',
                   help="Enable multi-speaker audio prompt. Auto-set by --task v2sa or when any row's ref_wav is a list.")
    p.add_argument('--no_clamp_seconds_total', action='store_false', dest='clamp_seconds_total',
                   help="Disable clamping seconds_total to the model's 10s cap. Not recommended; long clips (>10s) risk OOD collapse.")
    args = p.parse_args()
    # v2as -> v2sa deprecation shim (paper uses V2SA; keep old spelling working).
    if getattr(args, 'task', None) == 'v2as':
        import sys as _sys
        print('[inference] WARNING: --task v2as is a deprecated alias for --task v2sa (paper terminology); continuing as v2sa.', file=_sys.stderr)
        args.task = 'v2sa'
    return args


# ─── User-facing preflight helpers ──────────────────────────────────────────

def _derive_id_from_video_path(video_path: str) -> str:
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return stem or "clip"


def _parse_ref_wav_cli(ref_wav_str: str):
    """Parse --ref_wav CLI value: single path or comma-separated list of paths."""
    if not ref_wav_str:
        return None
    parts = [p.strip() for p in ref_wav_str.split(",") if p.strip()]
    if len(parts) == 1:
        return parts[0]
    return parts


def _synthesize_single_row_jsonl(args) -> str:
    """--input <mp4> shortcut: build a 1-row JSONL to a tmp path and return the path."""
    video_path = args.input
    task = args.task
    row = {"video_path": video_path}

    if task == "v2a":
        if not args.text_prompt:
            raise ValueError(
                "task=v2a with --input <mp4> requires --text_prompt (SFX caption). "
                "Hint: --text_prompt 'A fast, melodic run on an electric keyboard.'"
            )
        row["trans_cap"] = "none_speech"
        row["text_prompt"] = args.text_prompt
    elif task == "v2s":
        if not args.trans_cap:
            raise ValueError(
                "task=v2s with --input <mp4> requires --trans_cap 'transcript with <S>...<E>'. "
                "Hint: use examples/prompts/singlespeaker_caption.txt to build one via Gemini."
            )
        if not args.ref_wav:
            raise ValueError(
                "task=v2s with --input <mp4> requires --ref_wav <path.wav> for voice cloning."
            )
        row["trans_cap"] = args.trans_cap
        row["text_prompt"] = "clean speech"
        row["ref_wav"] = _parse_ref_wav_cli(args.ref_wav)
    elif task == "v2sa":
        if not args.trans_cap or not args.text_prompt or not args.ref_wav:
            raise ValueError(
                "task=v2sa with --input <mp4> requires --trans_cap, --text_prompt, and --ref_wav. "
                "Hint: --ref_wav spkA.wav,spkB.wav (comma separates speakers)."
            )
        row["trans_cap"] = args.trans_cap
        row["text_prompt"] = args.text_prompt
        row["ref_wav"] = _parse_ref_wav_cli(args.ref_wav)
    else:
        raise ValueError(f"--task must be one of v2a/v2s/v2sa (got {task!r})")

    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, f"cinedub_infer_{uuid.uuid4().hex[:8]}.jsonl")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    logging.info(f"[inference] synthesized single-file JSONL at {tmp_path}: {row}")
    return tmp_path


def _preflight_and_rewrite_jsonl(jsonl_path: str, task: str) -> tuple:
    """Load rows, validate, auto-derive id / dur / wav_path / feature, write to a
    tmp JSONL, and return (tmp_path, any_row_needs_online_extract, any_row_ref_wav_is_list).

    The rewrite lets the existing load_dataset() see the demo_cond schema
    (wav_path key) so no changes to that function are needed.
    """
    from stable_audio_tools.data.jsonl_toolkit import _ffprobe_duration  # local import

    rows = JSONLToolkit.load(jsonl_path)

    # Validate first (fail fast before checkpoint load).
    validate_rows(rows, task)

    any_online = False
    any_multi_ref = False
    fixed_rows = []
    for i, r in enumerate(rows):
        r = dict(r)  # copy
        vp = r["video_path"]

        # 1) id (derive from video_path stem if absent)
        if not r.get("id"):
            r["id"] = _derive_id_from_video_path(vp)

        # 2) dur -> seconds_total (ffprobe if missing)
        dur = r.get("dur") or r.get("seconds_total")
        if dur in (None, "") or (isinstance(dur, float) and dur <= 0):
            dur = _ffprobe_duration(vp)
            if dur is None:
                logging.warning(
                    f"[inference] {r['id']}: ffprobe failed for {vp}; using default seconds_total=10.0"
                )
                dur = 10.0
        r["seconds_total"] = float(dur)

        # 3) wav_path — trigger the demo_cond format branch in load_dataset.
        #    Point wav_path at the video (audio track lives inside the mp4);
        #    the model does not read wav_path for VTA / VTS pipelines.
        if not r.get("wav_path"):
            r["wav_path"] = vp

        # 4) feature — if absent or explicitly points at mp4, use online extraction.
        feat = r.get("feature")
        clip_f = r.get("clip_feature")
        sync_f = r.get("sync_feature")
        if not clip_f and not sync_f:
            if isinstance(feat, dict) and feat.get("clip") and feat.get("sync"):
                clip_f = feat["clip"]
                sync_f = feat["sync"]
            else:
                clip_f = vp
                sync_f = vp
        r["clip_feature"] = clip_f
        r["sync_feature"] = sync_f
        if str(clip_f).endswith(".mp4") or str(sync_f).endswith(".mp4"):
            any_online = True

        # 5) auto-detect multi-speaker ref_wav shape
        ref = r.get("ref_wav")
        if isinstance(ref, list) and len(ref) > 1:
            # A [path,start,end] triple counts as SINGLE speaker (one clip cut).
            first = ref[0]
            if isinstance(first, str) or (isinstance(first, (list, tuple))
                                          and len(first) >= 3
                                          and isinstance(first[1], (int, float))):
                # If first entry is a triple AND the whole outer list contains
                # multiple triples/strs → multi-speaker.
                if any(isinstance(x, str) for x in ref) or \
                        len([x for x in ref if isinstance(x, (list, tuple))]) > 1:
                    any_multi_ref = True

        fixed_rows.append(r)

    # Write to a tmp file next to the input.
    tmp_path = os.path.join(
        tempfile.gettempdir(), f"cinedub_infer_prepared_{uuid.uuid4().hex[:8]}.jsonl"
    )
    JSONLToolkit.save(fixed_rows, tmp_path)
    logging.info(f"[inference] validated {len(fixed_rows)} rows for task={task}; "
                 f"prepared jsonl at {tmp_path} (online_extract={any_online}, multi_ref={any_multi_ref})")
    return tmp_path, any_online, any_multi_ref


def _cfg_default_for_task(task):
    """Return the default classifier-free guidance scale for a given --task.

    Rule (author):
        * v2a (SFX only)                      → 2.5
        * v2s / v2sa / v2as (speech outputs)  → 7.0
    Unknown task falls back to 2.5 with a warning.
    """
    if task in ("v2s", "v2sa", "v2as"):
        return 7.0
    if task == "v2a":
        return 2.5
    print(f"[CineDub] warn: unknown task {task!r} for cfg default; using 2.5", flush=True)
    return 2.5


def _resolve_task_dispatch(args):
    """Set args.tasks + args.use_audio_prompt / args.use_multi_audio_prompt from --task."""
    if args.task is None:
        # Legacy path — trust --tasks or the default.
        if args.tasks is None:
            args.tasks = "VTA VA TA"
        return

    if args.tasks:
        raise ValueError("--task and --tasks are mutually exclusive; pick one.")

    if args.task == "v2a":
        args.tasks = "VTA"
        # neither audio prompt flag
    elif args.task == "v2s":
        args.tasks = "VTS"
        args.use_audio_prompt = True
    elif args.task == "v2sa":
        args.tasks = "VTS"
        # will be flipped to multi in the preflight if any row's ref_wav is a list.
        args.use_multi_audio_prompt = True
    else:
        raise ValueError(f"unknown task {args.task!r}")


def _resolve_ckpt(args):
    """Resolve --ckpt / $CINEDUB_CKPT / $CINEDUB_CKPT_MULTISPEAKER into model_ckpt_path."""
    if args.ckpt:
        args.model_ckpt_path = args.ckpt
        return
    if args.model_ckpt_path:
        return
    env_key = "CINEDUB_CKPT_MULTISPEAKER" if args.task == "v2sa" else "CINEDUB_CKPT"
    env_val = os.environ.get(env_key)
    if env_val:
        args.model_ckpt_path = env_val


    
rank = getattr(rank_zero_only, "rank", None)
world_size = torch.cuda.device_count()
_rank = f":{rank}/{world_size}"

@rank_zero_only
def rank_zero_logconfig(rank):
    logging.basicConfig(
    level="INFO", 
    force=True,
    format=f"[{os.uname()[1].split('.')[0]}:{rank}]"
        f" %(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",)


def process_batch(conds_i, generate_args, save_path, mp4_dir=None, task_id=None,copy_video_low_quality=False, add_id=True, use_multi_audio_prompt=False):
    """Process a single batch of conditions"""
    # For MP4 processing, always use path
    path_i = [item["path"].split("/")[-1].replace(".mp4", ".wav") for item in conds_i]
    # Lazy-load audio_prompt tensors for this batch only
    for item in conds_i:
        ap_path = item.pop("_audio_prompt_path", None)
        if ap_path:
            try:
                if use_multi_audio_prompt:
                    item["audio_prompt"] = _load_multi_audio_prompt_tensor(ap_path)
                else:
                    item["audio_prompt"] = _load_audio_prompt_tensor(ap_path)
            except Exception as e:
                logging.warning(f"Failed to load audio_prompt for {item.get('id','?')}: {e}")
    generate_args["conditioning"] = conds_i
    audio = generate_diffusion_cond(**generate_args)
    # 修复torchaudio 2.9.1的int16保存bug：直接保存float32（范围[-1,1]）
    audio = (audio.to(torch.float32)
           .div(torch.max(torch.abs(audio)))
           .clamp(-1, 1)
           .cpu())

    # Save audio files - use id if available, otherwise use filename
    for b_i, filename in enumerate(path_i):
        # Use id if add_id is True and id exists, otherwise use filename
        item_id = conds_i[b_i].get("id") if add_id else None
        save_filename = (item_id + ".wav") if item_id else filename
        audio_path = os.path.join(save_path, save_filename)
        torchaudio.save(audio_path, audio[b_i], sample_rate)
        # Create MP4 if demo mode
        if mp4_dir and task_id:
            mp4_path = os.path.join(mp4_dir, save_filename.replace(".wav", f"_{task_id}.mp4"))
            # import pdb;pdb.set_trace()
            replace_mp4_wav(conds_i[b_i]["path"], audio_path, mp4_path,copy_video_low_quality=copy_video_low_quality)
            mp4_gt = mp4_path.replace(f"_{task_id}.mp4", "_GT.mp4")
            # if not os.path.exists(mp4_gt) and os.path.exists(conds_i[b_i]['path']):
            replace_mp4_wav(mp4_path=conds_i[b_i]["path"], output_path=mp4_gt,copy_video_low_quality=copy_video_low_quality)

def generate_samples(task2dict, generate_args, save_dir, cfg_scale, rank, world_size, batch_size, is_demo=False, steps=200, copy_video_low_quality=False, cache_dir=None, add_id=True, resume=False, use_multi_audio_prompt=False, exp_prefix=""):
    """Generate samples for tasks"""
    batch_size = 6 if is_demo else batch_size
    generate_args["batch_size"] = batch_size
    infer_steps = generate_args["steps"]
    for task_id, conditions in task2dict.items():
        logging.info(f"{'Demo' if is_demo else 'Processing'} {task_id} with {len(conditions)} samples with batch size: {batch_size} infer steps:{infer_steps}")
        save_path = os.path.join(save_dir, f"{task_id}_cfg{round(cfg_scale, 2)}")
        mp4_dir = os.path.join(save_dir, f"cfg{round(cfg_scale, 2)}_mp4") if is_demo else None

        # Optional relocation of save_path: strip --exp_prefix (default empty) and re-root under cache_dir.
        # When both are empty (the common inference case) save_path is used as-is.
        if cache_dir and exp_prefix and save_path.startswith(exp_prefix):
            save_path = save_path.replace(exp_prefix, cache_dir, 1)

        # if infer_steps != 200:
        save_path = save_path + f"_{infer_steps}"
        mp4_dir = mp4_dir + f"_{infer_steps}" if mp4_dir else None
        os.makedirs(save_path, exist_ok=True)
        if mp4_dir:
            os.makedirs(mp4_dir, exist_ok=True)

        conds_rank = conditions[rank::world_size]
        if resume:
            conds_rank = [c for c in conds_rank if not os.path.exists(
                os.path.join(save_path, (c.get("id") + ".wav") if add_id and c.get("id") else c["path"].split("/")[-1].replace(".mp4", ".wav"))
            )]
        n_iter = math.ceil(len(conds_rank) / batch_size) if conds_rank else 0
        logging.info(f"Generate {len(conds_rank)} samples on rank {rank}, world size {world_size}")
        for idx in range(n_iter):
            start_idx = idx * batch_size
            conds_i = conds_rank[start_idx:start_idx + batch_size]
            process_batch(conds_i, generate_args, save_path, mp4_dir, task_id if is_demo else None,copy_video_low_quality=copy_video_low_quality, add_id=add_id, use_multi_audio_prompt=use_multi_audio_prompt)
            
            if not is_demo and (idx + 1) % 10 == 0:
                total = (idx + 1) * batch_size * world_size
                logging.info(f"Processed {idx + 1}/{n_iter} batches, {total} samples generated")

def main():

    rank = getattr(rank_zero_only, "rank", None)
    world_size = torch.cuda.device_count()
    _rank = f":{rank}/{world_size}"
    
    # Setup logging
    if rank != None and rank != 0:
        logging.basicConfig(
        level="ERROR",
        force=True,
        format=f"[{os.uname()[1].split('.')[0]}:{rank}/{world_size}]"
        f" %(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",)
    else:
        rank_zero_logconfig(_rank)
        
    logging.info(f"Rank {rank}, world size {world_size}")

    torch.set_float32_matmul_precision('high')
    warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
    warnings.filterwarnings("ignore", category=UserWarning, module="torchsde")
    warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
    
    args = get_args()

    # ─── User-facing preflight ───────────────────────────────────────────
    # Resolve --task → --tasks + audio-prompt flags.
    _resolve_task_dispatch(args)
    # Resolve --cfg / --output / --ckpt aliases (new names take precedence).
    # Precedence: --cfg > --cfg_scale > per-task auto-default.
    if args.cfg is not None:
        args.cfg_scale = args.cfg
    elif args.cfg_scale is not None:
        args.cfg = args.cfg_scale
    else:
        if args.task:
            _auto = _cfg_default_for_task(args.task)
            print(f"[CineDub] auto-cfg: task={args.task} \u2192 cfg={_auto}", flush=True)
        else:
            _auto = 2.5
            print("[CineDub] warn: no --task specified, using cfg=2.5 default; pass --cfg to override", flush=True)
        args.cfg = _auto
        args.cfg_scale = _auto
    if args.output is not None:
        args.save_dir = args.output
    _resolve_ckpt(args)

    # Resolve --input into a JSONL dataset_config.
    input_path = args.input or args.dataset_config
    if not input_path:
        raise SystemExit("--input <mp4|jsonl> (or legacy --dataset_config) is required")

    if input_path.endswith(".mp4"):
        if not args.task:
            raise SystemExit("--input <mp4> requires --task {v2a,v2s,v2sa}")
        input_path = _synthesize_single_row_jsonl(args)
        args.all_demo = True  # single file → always demo mode (save mp4)

    # If we have a --task, run the validator + auto-derive pipeline.
    if args.task:
        input_path, any_online, any_multi_ref = _preflight_and_rewrite_jsonl(input_path, args.task)
        if any_online and not args.video_online_extract:
            logging.info("[inference] auto-enabling --video_online_extract (rows have mp4 features).")
            args.video_online_extract = True
        # v2sa: if any row's ref_wav is a list of paths, multi-speaker mode.
        # v2s: single-speaker is already set by _resolve_task_dispatch.
        if args.task == "v2sa" and not args.use_multi_audio_prompt:
            args.use_multi_audio_prompt = True
        if any_multi_ref:
            if args.task == "v2sa":
                args.use_multi_audio_prompt = True
            elif not args.use_multi_audio_prompt:
                logging.warning(
                    "[CineDub] row(s) have list-form ref_wav but --task is not v2sa; "
                    "treating first entry as single-speaker. "
                    "Pass --use_multi_audio_prompt or --task v2sa to enable dual-speaker cloning."
                )
        args.dataset_config = input_path
    else:
        args.dataset_config = input_path

    # Default save_dir if user didn't set one.
    if not args.save_dir:
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = args.task or "run"
        args.save_dir = os.path.join("output", f"{tag}_{ts}")
        logging.info(f"[inference] --output not set; defaulting to {args.save_dir}")

    # Seed reproducibility: honour args.seed globally for numpy/random too.
    if args.seed is not None and args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        try:
            torch.cuda.manual_seed_all(args.seed)
        except Exception:
            pass
        logging.info(f"[inference] seed pinned to {args.seed}")

    cfg_scale = args.cfg_scale
    batch_size = args.batch_size
    all_demo = args.all_demo
    video_online_extract = args.video_online_extract
    # load_config
    model_ckpt_path = args.model_ckpt_path
    save_dir = args.save_dir
    tasks = args.tasks.replace("\"","").replace("\'","").split("+")
    if args.model_config is not None:
        model_config = args.model_config
        logging.info(f"->-> Loading model config from {model_config}...")
        with open(model_config) as f:
            model_config = json.load(f)
    else:
        exp_dir = ("/").join(model_ckpt_path.split("/")[:-2])
        model_config = os.path.join(exp_dir, "model_config.json")
        logging.info(f"->-> Loading dataset from {model_config}...")
        with open(model_config, 'r') as f:
            model_config = json.load(f)

    if args.video_online_extract:
        # Find clip_feature and sync_feature configs dynamically
        configs = model_config["model"]["conditioning"]["configs"]
        clip_config = None
        sync_config = None

        for config in configs:
            if config["id"] == "clip_feature":
                clip_config = config
            elif config["id"] == "sync_feature":
                sync_config = config

        assert clip_config is not None, "clip_feature config not found"
        assert clip_config["type"] == "post_process", "clip_feature must be post_process type"
        clip_config["config"]["online_extract_type"] = "mm_clip"

        assert sync_config is not None, "sync_feature config not found"
        assert sync_config["type"] == "post_process", "sync_feature must be post_process type"
        sync_config["config"]["online_extract_type"] = "mm_sync"
    
    logging.info(f"->-> Loading model from {model_ckpt_path}...")
    accel = Accelerator()
    device = accel.device
    model = load_model(model_config, model_ckpt_path, model_half=False, device=device,video_online_extract=args.video_online_extract, use_ema=args.use_ema)

    logging.info(f"->-> Loading dataset from {args.dataset_config}...")    
    
    #conditions and check
    demo_ids = None
    if args.demoid_path:
        with open(args.demoid_path, 'r') as f:
            demo_ids = [line.strip() for line in f]
    
    if all_demo:
        task2dict = load_dataset(args.dataset_config,tasks,args.re_duration,clamp_seconds_total=args.clamp_seconds_total)
    else:
        if demo_ids:
            demo_task2dict, task2dict = load_dataset(args.dataset_config,tasks,args.re_duration,demo_ids,clamp_seconds_total=args.clamp_seconds_total)
        else:
            task2dict = load_dataset(args.dataset_config,tasks,args.re_duration,clamp_seconds_total=args.clamp_seconds_total)

    if args.use_audio_prompt or args.use_multi_audio_prompt:
        mode = "multi-speaker" if args.use_multi_audio_prompt else "single"
        logging.info(f"->-> Injecting audio_prompt ({mode}) into conditions...")
        if all_demo:
            task2dict = inject_audio_prompt(task2dict)
        else:
            if demo_ids:
                demo_task2dict = inject_audio_prompt(demo_task2dict)
            task2dict = inject_audio_prompt(task2dict)

    generate_args = {
        "model": model, "conditioning": None, "negative_conditioning": None,
        "steps": args.steps, "cfg_scale": args.cfg_scale, "cfg_interval": (0.0, 1.0),
        "batch_size": args.batch_size, "sample_size": sample_size, "seed": args.seed,
        "device": device, "sampler_type": "dpmpp-2m-sde",
        "sigma_min": 0.03, "sigma_max": 1000, "init_audio": None,
        "init_noise_level": 1.0, "callback": None, "scale_phi": 0.0,
    }
    
    if all_demo:
        generate_samples(task2dict, generate_args, args.save_dir, args.cfg_scale,
                        rank, world_size, args.batch_size, is_demo=True, steps=args.steps,copy_video_low_quality=args.copy_video_low_quality,cache_dir=args.cache_dir, add_id=args.add_id, resume=args.resume, use_multi_audio_prompt=args.use_multi_audio_prompt, exp_prefix=args.exp_prefix)
    else:
        if demo_ids:
            generate_samples(demo_task2dict, generate_args, args.save_dir, args.cfg_scale,
                            rank, world_size, args.batch_size, is_demo=True,copy_video_low_quality=args.copy_video_low_quality,cache_dir=args.cache_dir, add_id=args.add_id, resume=args.resume, use_multi_audio_prompt=args.use_multi_audio_prompt, exp_prefix=args.exp_prefix)

        generate_samples(task2dict, generate_args, args.save_dir, args.cfg_scale,
                        rank, world_size, args.batch_size, is_demo=False,copy_video_low_quality=args.copy_video_low_quality,cache_dir=args.cache_dir, add_id=args.add_id, resume=args.resume, use_multi_audio_prompt=args.use_multi_audio_prompt, exp_prefix=args.exp_prefix)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

if __name__ == '__main__':
    main()
