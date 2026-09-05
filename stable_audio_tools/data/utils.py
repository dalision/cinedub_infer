import logging
import math
import random
import torch

from torch import nn
from typing import Tuple
import random
import re
from torchaudio import transforms as T
import sys
import subprocess
import json

# The upstream training tree pulled `ex_8_2_filter_music` from an internal sibling
# repo (Synchformer/local/vat_caption/03_VA_recaption) to filter music-heavy clips
# during data curation. That module is not shipped with the open-source release and
# is only referenced by training-time meta filtering, so we stub it out with no-ops
# to keep any legacy imports harmless at inference time.
def check_music_pattern(*args, **kwargs):
    return False


def advanced_music_pattern_check(*args, **kwargs):
    return False


def create_music_keywords(*args, **kwargs):
    return []


music_keywords = []

def get_audio_duration_and_sr(mp4_path: str, replace_list = None) -> Tuple[float, int]:
    """
    获取MP4文件的音频时长和采样率
    
    Args:
        mp4_path: MP4文件路径
    
    Returns:
        Tuple[float, int]: (音频时长(秒), 采样率(Hz))
    """
    if replace_list:  # optional per-call [(regex, org_dir, target_dir), ...]
        for match_rule, org_dir, target_dir in replace_list:
            if re.match(match_rule, mp4_path):
                mp4_path = mp4_path.replace(org_dir, target_dir)
                break
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_streams',
        '-show_format',
        '-select_streams', 'a:0',  # 只选择第一个音频流
        mp4_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    
    audio_duration = 0.0
    sample_rate = 0
    
    # 从音频流中获取时长和采样率
    for stream in data.get('streams', []):
        if stream.get('codec_type') == 'audio':
            # 获取时长
            duration = stream.get('duration')
            if duration is not None:
                audio_duration = float(duration)
            
            # 获取采样率
            sr = stream.get('sample_rate')
            if sr is not None:
                sample_rate = int(sr)
            
            break
    
    # 如果音频流没有duration信息，从format中获取总时长
    if audio_duration == 0.0 and 'format' in data and 'duration' in data['format']:
        audio_duration = float(data['format']['duration'])
    
    return audio_duration, sample_rate

def get_video_duration_precise(mp4_path: str, replace_list = None) -> float:
    """
    精确获取视频/音频文件时长（带备选方案）

    Args:
        mp4_path: 视频或音频文件路径（支持 .mp4, .wav 等格式）

    Returns:
        float: 文件时长，单位：秒
    """
    if replace_list:  # optional per-call [(regex, org_dir, target_dir), ...]
        for match_rule, org_dir, target_dir in replace_list:
            if re.match(match_rule, mp4_path):
                mp4_path = mp4_path.replace(org_dir, target_dir)
                break

    # 判断是音频文件还是视频文件
    is_audio_file = mp4_path.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'))

    if is_audio_file:
        # 对于音频文件，选择音频流，同时获取 format 信息作为 fallback
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-select_streams', 'a:0',        # 只选择第一个音频流
            '-show_streams',
            '-show_format',                  # 同时获取 format 信息
            mp4_path
        ]
        target_codec_type = 'audio'
    else:
        # 对于视频文件，选择视频流
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-select_streams', 'v:0',        # 只选择第一个视频流
            '-show_streams',
            mp4_path
        ]
        target_codec_type = 'video'

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)

    # 从目标流获取时长
    duration = 0.0
    for stream in data.get('streams', []):
        if stream.get('codec_type') == target_codec_type:
            stream_duration = stream.get('duration')
            if stream_duration is not None:
                duration = float(stream_duration)
                break

    # 对于音频文件，如果 stream 中没有 duration，从 format 中获取
    if duration == 0.0 and is_audio_file:
        if 'format' in data and 'duration' in data['format']:
            duration = float(data['format']['duration'])

    return duration

def get_mp4_durations_with_wav_sr(mp4_path: str, replace_list = None) -> Tuple[float, float, int]:
    """
    获取MP4文件的视频时长、音频时长和音频采样率
    
    Args:
        mp4_path: MP4文件路径
    
    Returns:
        Tuple[float, float, int]: (视频时长, 音频时长, 音频采样率) 单位：秒, 秒, Hz
    """

    if replace_list:  # optional per-call [(regex, org_dir, target_dir), ...]
        for match_rule, org_dir, target_dir in replace_list:
            if re.match(match_rule, mp4_path):
                mp4_path = mp4_path.replace(org_dir, target_dir)
                break
            
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_streams',
        '-show_format',
        mp4_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    
    video_duration = 0.0
    audio_duration = 0.0
    wav_sr = 0  # 音频采样率
    
    # 从format信息中获取总时长
    if 'format' in data and 'duration' in data['format']:
        total_duration = float(data['format']['duration'])
    else:
        total_duration = 0.0
    
    # 分别获取视频和音频流的时长和采样率
    for stream in data.get('streams', []):
        codec_type = stream.get('codec_type')
        duration = stream.get('duration')
        
        if duration is not None:
            duration = float(duration)
            
            if codec_type == 'video' and video_duration == 0.0:
                video_duration = duration
            elif codec_type == 'audio' and audio_duration == 0.0:
                audio_duration = duration
                
                # 获取音频采样率
                sample_rate = stream.get('sample_rate')
                if sample_rate is not None:
                    wav_sr = int(sample_rate)
    
    # 如果某个流没有duration信息，使用总时长
    if video_duration == 0.0:
        video_duration = total_duration
    if audio_duration == 0.0:
        audio_duration = total_duration
    
    return video_duration, audio_duration, wav_sr

def judge_music(caption):
    is_basic_match, basic_details = check_music_pattern(caption, music_keywords, 1, False)
    is_advanced_match, advanced_info = advanced_music_pattern_check(caption)
    return  is_basic_match or is_advanced_match


class PadCrop(nn.Module):
    def __init__(self, n_samples, randomize=True):
        super().__init__()
        self.n_samples = n_samples
        self.randomize = randomize

    def __call__(self, signal):
        n, s = signal.shape
        start = 0 if (not self.randomize) else torch.randint(0, max(0, s - self.n_samples) + 1, []).item()
        end = start + self.n_samples
        output = signal.new_zeros([n, self.n_samples])
        output[:, :min(s, self.n_samples)] = signal[:, start:end]
        return output

class PadCrop_Normalized_T(nn.Module):
    
    def __init__(self, n_samples: int, sample_rate: int, randomize: bool = True):
        
        super().__init__()
        
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.randomize = randomize

    def __call__(self, source: torch.Tensor) -> Tuple[torch.Tensor, float, float, int, int]:
        
        n_channels, n_samples = source.shape
        
        # If the audio is shorter than the desired length, pad it
        upper_bound = max(0, n_samples - self.n_samples)
        
        # If randomize is False, always start at the beginning of the audio
        offset = 0
        if(self.randomize and n_samples > self.n_samples):
            offset = random.randint(0, upper_bound)

        # Calculate the start and end times of the chunk
        t_start = offset / (upper_bound + self.n_samples)
        t_end = (offset + self.n_samples) / (upper_bound + self.n_samples)

        # Create the chunk
        chunk = source.new_zeros([n_channels, self.n_samples])

        # Copy the audio into the chunk
        chunk[:, :min(n_samples, self.n_samples)] = source[:, offset:offset + self.n_samples]
        
        # Calculate the start and end times of the chunk in seconds
        seconds_start = math.floor(offset / self.sample_rate)
        seconds_total = math.ceil(n_samples / self.sample_rate)

        # Create a mask the same length as the chunk with 1s where the audio is and 0s where it isn't
        padding_mask = torch.zeros([self.n_samples])
        padding_mask[:min(n_samples, self.n_samples)] = 1
        
        
        return (
            chunk,
            t_start,
            t_end,
            seconds_start,
            seconds_total,
            padding_mask
        )

class PhaseFlipper(nn.Module):
    "Randomly invert the phase of a signal"
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    def __call__(self, signal):
        return -signal if (random.random() < self.p) else signal
        
class Mono(nn.Module):
  def __call__(self, signal):
    return torch.mean(signal, dim=0, keepdims=True) if len(signal.shape) > 1 else signal

class Stereo(nn.Module):
  def __call__(self, signal):
    signal_shape = signal.shape
    # Check if it's mono
    if len(signal_shape) == 1: # s -> 2, s
        signal = signal.unsqueeze(0).repeat(2, 1)
    elif len(signal_shape) == 2:
        if signal_shape[0] == 1: #1, s -> 2, s
            signal = signal.repeat(2, 1)
        elif signal_shape[0] > 2: #?, s -> 2,s
            signal = signal[:2, :]    

    return signal

class VolumeNorm(nn.Module):
    "Volume normalization and augmentation of a signal [LUFS standard]"
    def __init__(self, params=[-16, 2], sample_rate=16000, energy_threshold=1e-6):
        super().__init__()
        self.loudness = T.Loudness(sample_rate)
        self.value = params[0]
        self.gain_range = [-params[1], params[1]]
        self.energy_threshold = energy_threshold

    def __call__(self, signal):
        """
        signal: torch.Tensor [channels, time]
        """
        # avoid do normalisation for silence
        energy = torch.mean(signal**2)
        if energy < self.energy_threshold:
            return signal
        
        input_loudness = self.loudness(signal)
        # Generate a random target loudness within the specified range
        target_loudness = self.value + (torch.rand(1).item() * (self.gain_range[1] - self.gain_range[0]) + self.gain_range[0])
        delta_loudness = target_loudness - input_loudness
        gain = torch.pow(10.0, delta_loudness / 20.0)
        output = gain * signal

        # Check for potentially clipped samples
        if torch.max(torch.abs(output)) >= 1.0:
            output = self.declip(output)

        return output

    def declip(self, signal):
        """
        Declip the signal by scaling down if any samples are clipped
        """
        max_val = torch.max(torch.abs(signal))
        if max_val > 1.0:
            signal = signal / max_val
            signal *= 0.95
        return signal

def extract_clap_score(score):
    """
    从score中提取CLAP分数
    """
    if isinstance(score, (int, float)):
        return float(score)
    
    if isinstance(score, str):
        # 提取数字部分，例如 "flash:0.33" -> 0.33
        match = re.search(r'[\d.]+', score)
        if match:
            return float(match.group())
    
    return 0.0  # 默认返回0.0

def get_caption_priority(caption):
    """
    获取caption的优先级
    返回值越小优先级越高
    """
    if not isinstance(caption, str):
        return 2  # 非字符串类型优先级最低
   
    caption_lower = caption.lower().strip()
    if caption_lower.startswith('pro'):
        return 1  # pro前缀优先级最高
    elif caption_lower.startswith('flash'):
        return 2  # flash前缀次之
    else:
        return 2  # 无前缀优先级次之

def select_caption_with_probability(data, top_n_rate=0.7, m_clap_thr=None, a_clap_thr=None):
    """
    根据概率和阈值选择caption
    
    Args:
        data: (captions, scores) 元组
        top_n_rate: 选择优先级最高caption的概率（0-1），剩余概率随机选择
        m_clap_thr: 音乐caption的CLAP阈值，None表示不过滤
        a_clap_thr: 音频caption的CLAP阈值，None表示不过滤
    
    Returns:
        选中的caption字符串，如果没有有效caption则返回None
    """
    captions, scores = data
    
    if not captions or not scores:
        return None
       
    if len(captions) != len(scores):
        raise ValueError("captions和scores长度不匹配")
    
    # 步骤1：如果设置了阈值，先进行过滤
    filtered_captions = []
    filtered_scores = []

    if m_clap_thr is not None or a_clap_thr is not None:        
        for caption, score in zip(captions, scores):
            try:
                # 提取CLAP分数
                clap_score = extract_clap_score(score)
                
                # 判断是音乐还是音频（假设judge_music函数已定义）
                is_music = judge_music(caption)
                
                # 应用对应的阈值过滤
                should_keep = True
                
                if is_music and m_clap_thr is not None:
                    # 音乐类型且设置了音乐阈值
                    should_keep = float(clap_score) >= m_clap_thr
                elif not is_music and a_clap_thr is not None:
                    # 音频类型且设置了音频阈值
                    should_keep = float(clap_score) >= a_clap_thr
                # 如果没有对应的阈值设置，保留该caption
                
                if should_keep:
                    filtered_captions.append(caption)
                    filtered_scores.append(score)
                    
            except (ValueError, TypeError) as e:
                # 如果处理出错，跳过这个caption-score对
                logging.info(f"Error processing caption '{caption}': {e}")
                continue
        
        # 使用过滤后的数据
        captions, scores = filtered_captions, filtered_scores
        # logging.info(f"After filtering: {len(captions)} captions remain")
    
    # 检查过滤后是否还有有效数据
    if not captions or not scores:
        # logging.info("No captions remain after filtering")
        return None

    # 步骤2：根据top_n_rate决定选择策略
    if random.random() < top_n_rate:
        # 随机选择
        selected = random.choice(captions)
        return selected
    else:
        # 选择优先级最高的
        if len(captions) == 1:
            return captions[0]
        
        caption_data = []
        for i, (caption, score) in enumerate(zip(captions, scores)):
            try:
                # 提取CLAP分数
                clap_score = extract_clap_score(score)
                priority = get_caption_priority(caption)
                caption_data.append({
                    'index': i,
                    'caption': caption,
                    'score': score,
                    'clap_score': round(clap_score, 1),
                    'priority': priority,
                })
            except (ValueError, TypeError):
                # 如果无法提取分数，给一个很低的分数
                raise ValueError(f"Invalid score format in item {i}: {score}")
        
        # 排序规则：
        # 1. 优先级（数值越小越优先）
        # 2. CLAP分数（越高越好）
        caption_data.sort(key=lambda x: (
            x['priority'],       # 优先级，数值小的优先
            -x['clap_score'],    # CLAP分数，负号表示降序
            not isinstance(x['caption'], str),  # 非字符串类型排在最后
        ))

        selected = caption_data[0]

        return selected['caption']
        



# =============================================================================
# Path rewriting: passthrough (open-source release).
#
# The internal training tree carried a large table of legacy path-rewrite rules
# that translated dataset paths between two cluster layouts. Those rules were
# specific to the internal environment and are irrelevant for the public
# release, where JSONL entries are expected to point directly at the user's
# local files (see examples/*.jsonl).
#
# We keep the `apply_replace_list` symbol as a passthrough because the training
# codebase called it in a few loader helpers; at inference time it just returns
# the path unchanged.
# =============================================================================

REPLACE_LIST = []


def apply_replace_list(path, replace_list=None, keep_ext=False):
    """Passthrough — returns `path` unchanged.

    Kept as a stub so any historical caller (feature/wav loader helpers) still
    imports cleanly. All internal path-rewrite rules were stripped for the
    open-source inference release; users pass fully-qualified local paths in
    their JSONL rows.
    """
    return path
