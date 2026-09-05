# NOTE: this file survives from the training tree solely because a handful of
# inference-side helpers (copy_state_dict, get_rank, get_world_size,
# replace_mp4_wav, generate_multimodal_tasks) live here and are imported by
# pyscripts/generate_v2a_cond.py. The rest of the training subtree (factory,
# diffusion, autoencoders, lm, losses) was removed for the inference release.
from pytorch_lightning.loggers import WandbLogger, CometLogger,TensorBoardLogger
from ..interface.aeiou import pca_point_cloud
import logging
import wandb
import torch
import os
import numpy as np
from PIL import Image
import warnings
import torchaudio
import subprocess
import json
import tempfile
import shutil
import random
def get_rank():
    """Get rank of current process."""
    # logging.info(os.environ.keys())
    if "SLURM_PROCID" in os.environ:
        return int(os.environ["SLURM_PROCID"])

    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 0

    return torch.distributed.get_rank()


# Defined by @JYuXuAN
def get_world_size():
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 1
    else:
        return torch.distributed.get_world_size()

class MultiStageInverseLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Implements a multi-stage inverse decay learning rate schedule.
    
    Stage 1: Linear warmup + Inverse decay (optional warmup)
    Stage 2+: Direct start from specified lr + Inverse decay  
    
    Supports 2 or more stages dynamically.
    """
    
    def __init__(self, optimizer, stage_configs, last_epoch=-1, verbose=False):
        # Validate stage configs
        if len(stage_configs) < 2:
            raise ValueError("Must provide at least 2 stage configurations")
        
        # Validate first stage can have linear_warmup_steps (optional)
        # Other stages should not have warmup
        
        # Validate required parameters for each stage
        required_params = ['steps', 'lr', 'inv_gamma', 'power', 'final_lr']
        for i, config in enumerate(stage_configs):
            for param in required_params:
                if param not in config:
                    raise ValueError(f"Stage {i+1} missing required parameter: {param}")
            
            # Only first stage can have warmup
            if i > 0 and 'linear_warmup_steps' in config:
                raise ValueError(f"Stage {i+1} cannot have 'linear_warmup_steps'. Only first stage supports warmup.")
        
        self.stage_configs = stage_configs
        self.num_stages = len(stage_configs)
        
        # Calculate cumulative steps for stage transitions
        self.stage_boundaries = []
        cumulative_steps = 0
        for config in stage_configs:
            cumulative_steps += config['steps']
            self.stage_boundaries.append(cumulative_steps)
        
        self.total_steps = cumulative_steps
        try:
            super().__init__(optimizer, last_epoch, verbose)
        except:
            super().__init__(optimizer, last_epoch)

    def get_current_stage_and_local_step(self):
        """Get current stage (0, 1, 2, ...) and step within that stage"""
        current_step = self.last_epoch + 1
        
        # Find which stage we're in
        for stage_idx in range(self.num_stages):
            if current_step <= self.stage_boundaries[stage_idx]:
                if stage_idx == 0:
                    return stage_idx, current_step - 1  # First stage, 0-indexed local step
                else:
                    return stage_idx, current_step - self.stage_boundaries[stage_idx - 1] - 1
        
        # If we're beyond all stages, return the last stage
        return self.num_stages - 1, current_step - self.stage_boundaries[-2] - 1
    
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                         "please use `get_last_lr()`.")
        return self._get_closed_form_lr()
    
    def _get_closed_form_lr(self):
        """Calculate learning rate based on current stage and step"""
        stage, local_step = self.get_current_stage_and_local_step()
        config = self.stage_configs[stage]
        
        if stage == 0 and 'linear_warmup_steps' in config:
            # First stage with warmup: Linear warmup + Inverse decay
            return self._stage1_lr(local_step, config)
        else:
            # All other stages: Direct start + Inverse decay
            return self._stage_inverse_lr(local_step, config)
    
    def _stage1_lr(self, local_step, config):
        """First stage: Optional linear warmup followed by inverse decay"""
        warmup_steps = config.get('linear_warmup_steps', 0)  # Default to 0 if not specified
        target_lr = config['lr']
        inv_gamma = config['inv_gamma']
        power = config['power']
        final_lr = config['final_lr']
        
        if warmup_steps > 0 and local_step < warmup_steps:
            # Linear warmup phase
            warmup_factor = (local_step + 1) / warmup_steps
            current_lr = target_lr * warmup_factor
        else:
            # Inverse decay phase (starts from warmup end or immediately)
            decay_step = local_step - warmup_steps if warmup_steps > 0 else local_step
            lr_mult = (1 + decay_step / inv_gamma) ** -power
            current_lr = max(final_lr, target_lr * lr_mult)
        
        return [current_lr * (base_lr / self.base_lrs[0]) if self.base_lrs[0] > 0 else current_lr
                for base_lr in self.base_lrs]

    def _stage_inverse_lr(self, local_step, config):
        """All non-first stages: Direct inverse decay from specified lr"""
        target_lr = config['lr']
        inv_gamma = config['inv_gamma']
        power = config['power']
        final_lr = config['final_lr']

        # Inverse decay starts immediately
        lr_mult = (1 + local_step / inv_gamma) ** -power
        current_lr = max(final_lr, target_lr * lr_mult)

        return [current_lr * (base_lr / self.base_lrs[0]) if self.base_lrs[0] > 0 else current_lr
                for base_lr in self.base_lrs]

class InverseLR(torch.optim.lr_scheduler._LRScheduler):
    """Implements an inverse decay learning rate schedule with an optional exponential
    warmup. When last_epoch=-1, sets initial lr as lr.
    inv_gamma is the number of steps/epochs required for the learning rate to decay to
    (1 / 2)**power of its original value.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        inv_gamma (float): Inverse multiplicative factor of learning rate decay. Default: 1.
        power (float): Exponential factor of learning rate decay. Default: 1.
        warmup (float): Exponential warmup factor (0 <= warmup < 1, 0 to disable)
            Default: 0.
        final_lr (float): The final learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, logging.infos a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(self, optimizer, inv_gamma=1., power=1., warmup=0., final_lr=0.,
                 last_epoch=-1, verbose=False):
        self.inv_gamma = inv_gamma
        self.power = power
        if not 0. <= warmup < 1:
            raise ValueError('Invalid value for warmup')
        self.warmup = warmup
        self.final_lr = final_lr
        try:
            super().__init__(optimizer, last_epoch, verbose)
        except:
            super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            import warnings
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        warmup = 1 - self.warmup ** (self.last_epoch + 1)
        lr_mult = (1 + self.last_epoch / self.inv_gamma) ** -self.power
        return [warmup * max(self.final_lr, base_lr * lr_mult)
                for base_lr in self.base_lrs]

def copy_state_dict(model, state_dict):
    """Load state_dict to model, but only for keys that match exactly.

    Args:
        model (nn.Module): model to load state_dict.
        state_dict (OrderedDict): state_dict to load.
    """
    model_state_dict = model.state_dict()
    key_not_in_model_state_dict = []
    size_mismatch_keys = []
    for key in state_dict:
        if key in model_state_dict and state_dict[key].shape == model_state_dict[key].shape:
            if isinstance(state_dict[key], torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                state_dict[key] = state_dict[key].data
            model_state_dict[key] = state_dict[key]
        else:
            if key not in model_state_dict:
                key_not_in_model_state_dict.append(key)
            else:
                size_mismatch_keys.append(key)

    # 检查模型中有但checkpoint中没有的keys
    key_not_in_checkpoint = [key for key in model_state_dict.keys() if key not in state_dict]

    model.load_state_dict(model_state_dict, strict=False)
    logging.info(f" Warning: The following key in the checkpoint is not presented in the model: {key_not_in_model_state_dict}")
    logging.info(f" Warning: These keys have different size between checkpoint and current model: {size_mismatch_keys}")
    logging.info(f" Warning: The following key in the model is not presented in the checkpoint: {key_not_in_checkpoint}")

def create_optimizer_from_config(optimizer_config, parameters):
    """Create optimizer from config.

    Args:
        parameters (iterable): parameters to optimize.
        optimizer_config (dict): optimizer config.

    Returns:
        torch.optim.Optimizer: optimizer.
    """

    optimizer_type = optimizer_config["type"]

    if optimizer_type == "FusedAdam":
        from deepspeed.ops.adam import FusedAdam
        optimizer = FusedAdam(parameters, **optimizer_config["config"])
    else:
        optimizer_fn = getattr(torch.optim, optimizer_type)
        optimizer = optimizer_fn(parameters, **optimizer_config["config"])
    return optimizer

def create_scheduler_from_config(scheduler_config, optimizer):
    """Create scheduler from config.

    Args:
        scheduler_config (dict): scheduler config.
        optimizer (torch.optim.Optimizer): optimizer.

    Returns:
        torch.optim.lr_scheduler._LRScheduler: scheduler.
    """
    if scheduler_config["type"] == "InverseLR":
        scheduler_fn = InverseLR
    elif scheduler_config["type"] == "MultiStageInverseLR":
        scheduler_fn = MultiStageInverseLR
    else:
        scheduler_fn = getattr(torch.optim.lr_scheduler, scheduler_config["type"])
    scheduler = scheduler_fn(optimizer, **scheduler_config["config"] if isinstance(scheduler_config, dict) else scheduler_config)
    return scheduler

def logger_project_name(logger) -> str:
    if isinstance(logger, WandbLogger):
        return logger.experiment.project
    elif isinstance(logger, CometLogger):
        return logger.name

def log_metric(logger, key, value, step=None):
    if isinstance(logger, WandbLogger):
        logger.experiment.log({key: value})
    elif isinstance(logger, CometLogger):
        logger.experiment.log_metrics({key: value}, step=step)
    elif isinstance(logger, TensorBoardLogger):
        logger.experiment.add_scalar(key, value, global_step=step)

def log_audio(logger, key, audio_path, sample_rate, caption=None):
    if isinstance(logger, WandbLogger):
        logger.experiment.log({key: wandb.Audio(audio_path, sample_rate=sample_rate, caption=caption)})
    elif isinstance(logger, CometLogger):
        logger.experiment.log_audio(audio_path, file_name=key, sample_rate=sample_rate)
    elif isinstance(logger, TensorBoardLogger):
        # 读取音频文件
        if isinstance(audio_path, str):
            # 如果是文件路径，需要读取音频数据
            import torchaudio
            audio_data, sr = torchaudio.load(audio_path)
            if sr != sample_rate:
                # 重采样到目标采样率
                resampler = torchaudio.transforms.Resample(sr, sample_rate)
                audio_data = resampler(audio_data)
        else:
            # 假设audio_path实际上是音频数据
            audio_data = audio_path
        
        # 确保音频数据是正确的格式 (1, num_samples) 或 (num_samples,)
        if audio_data.dim() == 1:
            audio_data = audio_data.unsqueeze(0)
        elif audio_data.dim() == 2 and audio_data.shape[0] > 1:
            # 如果是多声道，取第一个声道
            audio_data = audio_data[0:1]
            
        logger.experiment.add_audio(key, audio_data, sample_rate=sample_rate)

def log_image(logger, key, img_data):
    if isinstance(logger, WandbLogger):
        logger.experiment.log({key: wandb.Image(img_data)})
    elif isinstance(logger, CometLogger):
        logger.experiment.log_image(img_data, name=key)
    elif isinstance(logger, TensorBoardLogger):
        # 处理不同类型的图像数据
        if isinstance(img_data, torch.Tensor):
            # 如果是PyTorch tensor
            if img_data.dim() == 3:
                # (C, H, W) format
                logger.experiment.add_image(key, img_data)
            elif img_data.dim() == 2:
                # (H, W) format, 添加channel维度
                logger.experiment.add_image(key, img_data.unsqueeze(0))
            else:
                raise ValueError(f"Unsupported tensor dimension: {img_data.dim()}")
        elif isinstance(img_data, np.ndarray):
            # 如果是numpy数组
            if img_data.ndim == 3:
                # (H, W, C) -> (C, H, W)
                img_data = np.transpose(img_data, (2, 0, 1))
            elif img_data.ndim == 2:
                # (H, W) -> (1, H, W)
                img_data = np.expand_dims(img_data, axis=0)
            
            # 转换为tensor
            img_tensor = torch.from_numpy(img_data).float()
            logger.experiment.add_image(key, img_tensor)
        elif isinstance(img_data, Image.Image):
            # 如果是PIL Image
            img_array = np.array(img_data)
            if img_array.ndim == 3:
                img_array = np.transpose(img_array, (2, 0, 1))
            elif img_array.ndim == 2:
                img_array = np.expand_dims(img_array, axis=0)
            
            img_tensor = torch.from_numpy(img_array).float() / 255.0
            logger.experiment.add_image(key, img_tensor)
        else:
            raise ValueError(f"Unsupported image data type: {type(img_data)}")

def log_point_cloud(logger, key, tokens, caption=None):
    if isinstance(logger, WandbLogger):
        point_cloud = pca_point_cloud(tokens)
        logger.experiment.log({key: point_cloud})
    elif isinstance(logger, CometLogger):
        point_cloud = pca_point_cloud(tokens, rgb_float=True, output_type="points")
        #logger.experiment.log_points_3d(scene_name=key, points=point_cloud)

def generate_multimodal_tasks(original_data, tasks=None, add_id=False):
    """
    根据原始数据生成 VTS, TTS, VTA, VA 四种任务的数据格式
    逐条检查每个条目，根据其属性判断应该生成哪些任务，并自动去重

    Args:
        original_data: 包含完整信息的原始数据列表，每个条目应包含：
            - path/wav_path/video_path: 音频/视频路径（至少一个）
            - text_prompt: 音频描述文本（可选）
            - trans_cap: 语音转录标注（可选）
            - clip_feature: CLIP 视频特征路径（可选，VTA/VA/VTS需要）
            - sync_feature: 同步特征路径（可选）
            - seconds_start: 起始时间（可选，默认0.0）
            - seconds_total: 总时长（可选，默认10.0）
            - id: 条目ID（可选）
        tasks: 要生成的任务类型列表，默认为 ["VTS", "TTS", "VTA", "VA"]
        add_id: 是否在生成的任务中添加id字段，默认为False

    Returns:
        dict: 包含指定任务数据的字典，键为任务类型，值为数据列表

    任务规则（逐条判断）:
        - VTS (Video+Text+Speech): 有 trans_cap 内容（非 "none_speech"）且有视频特征，text_prompt 可以是任意值
        - TTS (Text+Speech): text_prompt="clean speech" 且有 trans_cap 内容（非 "none_speech"），纯语音无背景音
        - VTA (Video+Text+Audio): trans_cap="none_speech" 且有 text_prompt 内容（非 "clean speech"）且有视频特征
        - VA (Video+Audio): trans_cap="none_speech" 且有视频特征
    """
    if tasks is None:
        tasks = ["VTS", "TTS", "VTA", "VA"]

    # 验证输入的任务类型
    valid_tasks = {"VTA", "VA", "TTS", "VTS"}
    invalid_tasks = set(tasks) - valid_tasks
    if invalid_tasks:
        raise ValueError(f"Invalid task types: {invalid_tasks}. Valid options are: {valid_tasks}. Your tasks: {tasks}.")

    if not original_data:
        return {}

    # 初始化结果字典和去重集合
    result = {task: [] for task in tasks}
    seen_keys = {task: set() for task in tasks}

    # 逐条检查每个条目
    for item in original_data:
        # 提取路径（用于去重）
        path = item.get("path") or item.get("video_path") or item.get("wav_path")
        if not path:
            continue

        # 提取条目的属性
        text_prompt = item.get("text_prompt", "")
        trans_cap = item.get("trans_cap", "")
        has_clip_feature = "clip_feature" in item
        has_sync_feature = "sync_feature" in item

        # VTS任务: Video+Text+Speech
        # 条件: 有 trans_cap 内容（非 "none_speech"）且有视频特征
        # text_prompt 可以是 "clean speech" 或背景音效描述
        if "VTS" in tasks:
            if (trans_cap and trans_cap != "none_speech" and
                has_clip_feature and has_sync_feature):

                # 去重检查：基于 (path, trans_cap, text_prompt) 组合
                vts_key = (path, trans_cap, text_prompt)
                if vts_key not in seen_keys["VTS"]:
                    vts_item = {
                        "task": "VTS",
                        "path": path,
                        "clip_feature": item["clip_feature"],
                        "sync_feature": item["sync_feature"],
                        "text_prompt": text_prompt,  # 保留原始 text_prompt
                        "trans_cap": trans_cap,
                        "seconds_start": item.get("seconds_start", 0.0),
                        "seconds_total": item.get("seconds_total", 10.0)
                    }
                    # 如果 add_id 为 True 且 item 有 id，则添加 id 字段
                    if add_id and "id" in item:
                        vts_item["id"] = item["id"]
                    # 透传所有内部辅助字段（以 _ 开头，如 _wav_path, _ref_wav_path）
                    vts_item.update({k: v for k, v in item.items() if k.startswith("_")})
                    result["VTS"].append(vts_item)
                    seen_keys["VTS"].add(vts_key)

        # TTS任务: Text+Speech (无视频特征，纯语音)
        # 条件: text_prompt="clean speech" 且有 trans_cap 内容（非 "none_speech"）
        if "TTS" in tasks:
            if text_prompt == "clean speech" and trans_cap and trans_cap != "none_speech":

                # 去重检查：基于 (path, trans_cap, text_prompt) 组合
                tts_key = (path, trans_cap, text_prompt)
                if tts_key not in seen_keys["TTS"]:
                    tts_item = {
                        "task": "TTS",
                        "path": path,
                        "sync_feature": None,  # TTS 可能用到 sync_feature
                        "text_prompt": "clean speech",
                        "trans_cap": trans_cap,
                        "seconds_start": item.get("seconds_start", 0.0),
                        "seconds_total": item.get("seconds_total", 10.0)
                    }
                    # 如果 add_id 为 True 且 item 有 id，则添加 id 字段
                    if add_id and "id" in item:
                        tts_item["id"] = item["id"]
                    # 透传所有内部辅助字段（以 _ 开头，如 _wav_path, _ref_wav_path）
                    tts_item.update({k: v for k, v in item.items() if k.startswith("_")})
                    result["TTS"].append(tts_item)
                    seen_keys["TTS"].add(tts_key)

        # VTA任务: Video+Text+Audio
        # 条件: trans_cap="none_speech" 且有 text_prompt 内容（非 "clean speech"）且有 clip_feature
        if "VTA" in tasks:
            if (trans_cap == "none_speech" and
                text_prompt and text_prompt != "clean speech" and
                has_clip_feature and has_sync_feature):

                # 去重检查：基于 (path, text_prompt) 组合（trans_cap 都是 "none_speech"）
                vta_key = (path, text_prompt)
                if vta_key not in seen_keys["VTA"]:
                    vta_item = {
                        "task": "VTA",
                        "path": path,
                        "clip_feature": item["clip_feature"],
                        "sync_feature": item["sync_feature"],
                        "text_prompt": text_prompt,
                        "trans_cap": "none_speech",
                        "seconds_start": item.get("seconds_start", 0.0),
                        "seconds_total": item.get("seconds_total", 10.0)
                    }
                    # 如果 add_id 为 True 且 item 有 id，则添加 id 字段
                    if add_id and "id" in item:
                        vta_item["id"] = item["id"]
                    result["VTA"].append(vta_item)
                    seen_keys["VTA"].add(vta_key)

        # VA任务: Video+Audio (无文本提示)
        # 条件: trans_cap="none_speech" 且有 clip_feature
        if "VA" in tasks:
            if trans_cap == "none_speech" and has_clip_feature and has_sync_feature:

                # 去重检查：只基于 path（VA 没有 text_prompt，trans_cap 都是 "none_speech"）
                va_key = path
                if va_key not in seen_keys["VA"]:
                    va_item = {
                        "task": "VA",
                        "path": path,
                        "clip_feature": item["clip_feature"],
                        "sync_feature": item["sync_feature"],
                        "trans_cap": "none_speech",
                        "seconds_start": item.get("seconds_start", 0.0),
                        "seconds_total": item.get("seconds_total", 10.0)
                        # 注意: VA任务不包含 text_prompt
                    }
                    # 如果 add_id 为 True 且 item 有 id，则添加 id 字段
                    if add_id and "id" in item:
                        va_item["id"] = item["id"]
                    result["VA"].append(va_item)
                    seen_keys["VA"].add(va_key)

    # 移除空任务
    result = {k: v for k, v in result.items() if v}

    # 强制TTS任务只取前4个样本
    if "TTS" in result:
        result["TTS"] = result["TTS"][:8]

    # # 如果最后只有VA和VTA任务，则去除其中的trans_cap字段
    # if set(result.keys()).issubset({"VA", "VTA"}):
    #     for task_type in result:
    #         for item in result[task_type]:
    #             if "trans_cap" in item:
    #                 del item["trans_cap"]

    return result

def generate_multimodal_tasks_fromVTA(vta_data, task):
    """
    从VTA数据中提取指定任务类型的数据
    
    Args:
        vta_data: VTA格式的完整数据列表
        task: 要提取的任务类型 ("VTA", "TA", "VA")
    
    Returns:
        list: 指定任务格式的数据列表
    """
    if task == "VTA":
        return vta_data
    elif task == "TA":
        return [{
            "task": "TA",
            "path": item["path"],
            "sync_feature": None,
            "seconds_start": item["seconds_start"],
            "seconds_total": item["seconds_total"],
            "text_prompt": item["text_prompt"]
        } for item in vta_data]
    elif task == "VA":
        return [{
            "task": "VA",
            "path": item["path"],
            "clip_feature": item["clip_feature"],
            "sync_feature": item["sync_feature"],
            "seconds_start": item["seconds_start"],
            "seconds_total": item["seconds_total"]
        } for item in vta_data]
    else:
        raise ValueError(f"Invalid task type: {task}. Valid options are: VTA, TA, VA")

def get_video_info(video_path: str):
    """
    使用ffprobe获取视频信息
    
    Args:
        video_path: 视频文件路径
    
    Returns:
        dict: 视频信息字典
    """
    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        
        # 提取视频流信息
        video_stream = None
        audio_stream = None
        
        for stream in info['streams']:
            if stream['codec_type'] == 'video' and video_stream is None:
                video_stream = stream
            elif stream['codec_type'] == 'audio' and audio_stream is None:
                audio_stream = stream
        
        duration = float(info['format']['duration'])
        
        return {
            'success': True,
            'duration': duration,
            'video_stream': video_stream,
            'audio_stream': audio_stream,
            'format': info['format']
        }
        
    except subprocess.CalledProcessError as e:
        return {
            'success': False,
            'error': f"ffprobe执行失败: {e.stderr}",
            'duration': None
        }
    except json.JSONDecodeError:
        return {
            'success': False,
            'error': "无法解析ffprobe输出",
            'duration': None
        }
    except Exception as e:
        return {
            'success': False,
            'error': f"获取视频信息失败: {str(e)}",
            'duration': None
        }


def process_audio_for_video(wav_path: str, target_duration: float, 
                          temp_audio_path: str):
    """
    使用torchaudio处理音频以匹配视频时长
    
    Args:
        wav_path: 输入音频文件路径
        target_duration: 目标时长（秒）
        temp_audio_path: 临时音频输出路径
    
    Returns:
        dict: 处理结果
    """
    try:
        # 使用torchaudio加载音频
        audio_tensor, sample_rate = torchaudio.load(wav_path)
        audio_duration = audio_tensor.shape[1] / sample_rate
        
        # print(f"原始音频时长: {audio_duration:.2f}秒，目标时长: {target_duration:.2f}秒")
        
        # 处理音频长度
        processing_type = 'unchanged'
        
        if abs(audio_duration - target_duration) < 0.01:  # 差异小于10ms
            processed_audio = audio_tensor
            # print("音频和视频时长相同，无需调整")
            
        elif audio_duration > target_duration:
            # 音频较长，截断
            target_samples = int(target_duration * sample_rate)
            processed_audio = audio_tensor[:, :target_samples]
            processing_type = 'truncated'
            # print(f"音频较长，截断到 {target_duration:.2f}秒")
            
        else:
            # 音频较短，填充静音
            silence_duration = target_duration - audio_duration
            silence_samples = int(silence_duration * sample_rate)
            n_channels = audio_tensor.shape[0]
            
            # 创建静音张量
            silence_tensor = torch.zeros(n_channels, silence_samples, dtype=audio_tensor.dtype)
            
            # 拼接原音频和静音
            processed_audio = torch.cat([audio_tensor, silence_tensor], dim=1)
            processing_type = 'padded'
            # print(f"音频较短，用静音填充 {silence_duration:.2f}秒")
        
        # 使用torchaudio保存处理后的音频
        torchaudio.save(temp_audio_path, processed_audio, sample_rate)
        
        final_duration = processed_audio.shape[1] / sample_rate
        
        return {
            'success': True,
            'original_duration': audio_duration,
            'final_duration': final_duration,
            'processing_type': processing_type,
            'sample_rate': sample_rate
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f"音频处理失败: {str(e)}"
        }


def replace_audio_with_ffmpeg(video_path: str, audio_path: str, output_path: str,
                             video_codec: str = 'copy', audio_codec: str = 'libmp3lame',
                             audio_bitrate: str = '192k', audio_sample_rate = None, copy_video_low_quality: bool = False):
    """
    使用ffmpeg替换视频中的音频轨道

    Args:
        video_path: 输入视频路径
        audio_path: 输入音频路径
        output_path: 输出视频路径
        video_codec: 视频编码器 ('copy' 表示不重新编码)
        audio_codec: 音频编码器
        audio_bitrate: 音频比特率
        audio_sample_rate: 音频采样率 (Hz)，None表示自动检测
        copy_video_low_quality: 是否启用低质量模式，降低FPS到24和长边到320

    Returns:
        dict: 处理结果
    """
    try:
        cmd = [
            'ffmpeg',
            '-i', video_path,      # 输入视频
            '-i', audio_path,      # 输入音频
        ]

        # 处理低质量模式
        if copy_video_low_quality:
            # 强制使用libx264编码器以支持过滤器
            video_codec = 'libx264'
            # 添加视频过滤器：降低FPS到24，缩放长边到320，保持宽高比
            cmd.extend([
                '-vf', 'fps=24,scale=\'if(gt(iw,ih),320,-2):if(gt(iw,ih),-2,320)\'',
                '-r', '24'  # 确保输出帧率为24fps
            ])

        cmd.extend([
            '-c:v', video_codec,   # 视频编码器
            '-c:a', audio_codec,   # 音频编码器
            '-b:a', audio_bitrate, # 音频比特率
        ])

        # 如果指定了采样率，添加到命令中
        if audio_sample_rate is not None:
            cmd.extend(['-ar', str(audio_sample_rate)])

        cmd.extend([
            '-map', '0:v:0',       # 使用第一个输入的视频流
            '-map', '1:a:0',       # 使用第二个输入的音频流
            '-shortest',           # 以最短的流为准
            '-y',                  # 覆盖输出文件
            output_path
        ])
        
        # print(f"执行ffmpeg命令: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        return {
            'success': True,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
    except subprocess.CalledProcessError as e:
        return {
            'success': False,
            'error': f"ffmpeg执行失败: {e.stderr}",
            'returncode': e.returncode
        }
    except FileNotFoundError:
        return {
            'success': False,
            'error': "未找到ffmpeg命令，请确保已安装ffmpeg并添加到PATH"
        }
    except Exception as e:
        return {
            'success': False,
            'error': f"ffmpeg处理失败: {str(e)}"
        }


def reencode_mp4_audio(mp4_path: str, output_path: str,
                      video_codec: str = 'copy', audio_codec: str = 'libmp3lame',
                      audio_bitrate: str = '16k', copy_video_low_quality: bool = False):
    """
    重新编码MP4文件的音频轨道，保持视频轨道不变

    Args:
        mp4_path: 原始MP4文件路径
        output_path: 输出MP4文件路径
        video_codec: 视频编码器
        audio_codec: 音频编码器
        audio_bitrate: 音频比特率，默认16k
        copy_video_low_quality: 是否启用低质量模式，降低FPS到24和长边到320

    Returns:
        dict: 包含处理结果和信息的字典
    """
    try:
        # 获取视频信息
        video_info = get_video_info(mp4_path)
        if not video_info['success']:
            return {
                'success': False,
                'error': video_info['error'],
                'video_duration': None,
                'audio_duration': None
            }

        # 检查MP4是否包含音频轨道
        if video_info.get('audio_stream') is None:
            logging.info(f"MP4文件不存在音频轨道: {mp4_path}")
            return {
                'success': False,
                'error': f"MP4文件不存在音频轨道: {mp4_path}",
                'video_duration': video_info['duration'],
                'audio_duration': None
            }

        # 如果输出路径和输入路径相同，先备份到临时文件
        temp_input = None
        if output_path == mp4_path:
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                temp_input = temp_file.name
            shutil.copy2(mp4_path, temp_input)
            input_file = temp_input
        else:
            input_file = mp4_path

        # 构建FFmpeg命令
        cmd = [
            'ffmpeg', '-y',
            '-i', input_file,
        ]

        # 处理低质量模式
        if copy_video_low_quality:
            # 强制使用libx264编码器以支持过滤器
            video_codec = 'libx264'
            # 添加视频过滤器：降低FPS到24，缩放长边到320，保持宽高比
            cmd.extend([
                '-vf', 'fps=24,scale=\'if(gt(iw,ih),320,-2):if(gt(iw,ih),-2,320)\'',
                '-r', '24'  # 确保输出帧率为24fps
            ])

        cmd.extend([
            '-c:v', video_codec,
            '-c:a', audio_codec,
            '-b:a', audio_bitrate,
            output_path
        ])
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            error_msg = f"FFmpeg重编码失败: {result.stderr}"
            return {
                'success': False,
                'error': error_msg,
                'video_duration': video_info['duration'],
                'audio_duration': None
            }

        return {
            'success': True,
            'output_path': output_path,
            'video_duration': video_info['duration'],
            'audio_duration': video_info['duration'],
            'final_duration': video_info['duration'],
            'processing': 'audio_reencoded',
            'video_codec': video_codec,
            'audio_codec': audio_codec
        }

    except Exception as e:
        return {
            'success': False,
            'error': f"重编码过程中出现错误: {str(e)}",
            'video_duration': None,
            'audio_duration': None
        }
    finally:
        # 清理临时文件
        if temp_input and os.path.exists(temp_input):
            try:
                os.unlink(temp_input)
            except:
                pass


def replace_mp4_wav(mp4_path: str, wav_path: str = None, output_path: str = None,
                   video_codec: str = 'copy', audio_codec: str = 'libmp3lame',
                   audio_bitrate: str = '16k', keep_temp: bool = False, copy_video_low_quality: bool = False):
    """
    使用TorchAudio + FFmpeg替换MP4文件中的音频轨道
    处理策略：音频长则截断，短则用静音填充
    如果wav_path为None，则只重新编码原MP4中的音频

    Local-file only: the object-storage fetch branch that lived in the
    training tree was removed for the open-source inference release.

    Args:
        mp4_path: 原始MP4文件路径（本地路径）
        wav_path: 新的WAV音频文件路径，如果为None则使用原MP4的音频
        output_path: 输出MP4文件路径，如果为None则覆盖原文件
        video_codec: 视频编码器，'copy'表示不重新编码（推荐）
        audio_codec: 音频编码器
        audio_bitrate: 音频比特率，默认16k
        keep_temp: 是否保留临时文件（调试用）
        copy_video_low_quality: 是否启用低质量模式，降低FPS到24和长边到320

    Returns:
        dict: 包含处理结果和信息的字典
    """
    temp_audio_path = None
    original_mp4_path = mp4_path  # 保存原始路径用于日志

    try:
        # 检查输入文件是否存在
        if not os.path.exists(mp4_path):
            return {
                'success': False,
                'error': f"MP4文件不存在: {mp4_path}",
                'video_duration': None,
                'audio_duration': None
            }


        # 如果没有指定wav_path，则使用原MP4的音频
        if wav_path is None:
            # 直接重新编码原MP4的音频
            ffmpeg_result = reencode_mp4_audio(
                mp4_path, output_path, video_codec, audio_codec, audio_bitrate, copy_video_low_quality=copy_video_low_quality
            )
            return ffmpeg_result

        if not os.path.exists(wav_path):
            return {
                'success': False,
                'error': f"WAV文件不存在: {wav_path}",
                'video_duration': None,
                'audio_duration': None
            }
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # print(f"正在处理: {os.path.basename(mp4_path)} + {os.path.basename(wav_path)}")
        
        # 1. 使用ffprobe获取视频信息
        video_info = get_video_info(mp4_path)
        if not video_info['success']:
            return {
                'success': False,
                'error': video_info['error'],
                'video_duration': None,
                'audio_duration': None
            }
        
        video_duration = video_info['duration']
        
        # 2. 创建临时音频文件
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_audio_path = temp_file.name
        
        # 3. 使用torchaudio处理音频
        audio_result = process_audio_for_video(wav_path, video_duration, temp_audio_path)
        if not audio_result['success']:
            return {
                'success': False,
                'error': audio_result['error'],
                'video_duration': video_duration,
                'audio_duration': None
            }
        
        # 4. 使用ffmpeg合成最终视频
        ffmpeg_result = replace_audio_with_ffmpeg(
            mp4_path, temp_audio_path, output_path,
            video_codec, audio_codec, audio_bitrate,
            audio_sample_rate=audio_result['sample_rate'],
            copy_video_low_quality=copy_video_low_quality
        )
        
        if not ffmpeg_result['success']:
            return {
                'success': False,
                'error': ffmpeg_result['error'],
                'video_duration': video_duration,
                'audio_duration': audio_result['original_duration']
            }
        
        # print(f"✓ 成功生成: {output_path}")
        
        return {
            'success': True,
            'output_path': output_path,
            'video_duration': video_duration,
            'audio_duration': audio_result['original_duration'],
            'final_duration': audio_result['final_duration'],
            'processing': audio_result['processing_type'],
            'video_codec': video_codec,
            'audio_codec': audio_codec,
            'temp_audio_path': temp_audio_path if keep_temp else None
        }
        
    except Exception as e:
        error_msg = f"处理过程中出现错误: {str(e)}"
        print(error_msg)
        return {
            'success': False,
            'error': error_msg,
            'video_duration': None,
            'audio_duration': None
        }
    
    finally:
        # 清理临时文件
        if temp_audio_path and not keep_temp and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except Exception as e:
                print(f"清理临时音频文件失败: {e}")