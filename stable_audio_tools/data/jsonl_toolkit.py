"""Minimal JSONL utilities used by the CineDub inference entry point.

The training-side repo has a much richer JSONL toolkit; this file exposes
only the two static methods the inference script actually calls, so the
release stays self-contained.

Also exports ``validate_row`` — a per-task schema check that fails FAST
(before the ~3 GB checkpoint load) with an actionable error message.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional


# ---------------------------------------------------------------------------
# ffprobe helper
# ---------------------------------------------------------------------------

def _ffprobe_duration(media_path: str) -> Optional[float]:
    """Return duration in seconds via ffprobe, or None if it fails."""
    if not media_path or not os.path.isfile(media_path):
        return None
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                media_path,
            ],
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
        return float(out.decode().strip())
    except Exception:
        return None


class JSONLToolkit:
    """Static helpers for loading and lightly transforming JSONL rows."""

    @staticmethod
    def load(jsonl_path: str, validate: Optional[Callable] = None) -> List[dict]:
        items: List[dict] = []
        with open(jsonl_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        if validate is not None:
            validate(items)
        return items

    @staticmethod
    def save(items: List[dict], jsonl_path: str) -> None:
        parent = os.path.dirname(os.path.abspath(jsonl_path)) or "."
        os.makedirs(parent, exist_ok=True)
        with open(jsonl_path, "w", encoding="utf-8") as fh:
            for it in items:
                fh.write(json.dumps(it, ensure_ascii=False) + "\n")

    @staticmethod
    def add_duration(
        src: str,
        dst: str,
        use_threading: bool = True,
        num_threads: int = 4,
    ) -> List[dict]:
        """Populate a per-row 'dur' field via ffprobe on 'video_path' (or 'wav').

        - Rows that already carry a 'dur' field are left untouched.
        - If ffprobe fails and no fallback is present, the row is kept as-is.
        - Writes the (possibly enriched) list back to 'dst' and returns it.
        """
        items = JSONLToolkit.load(src)

        def _needs(item: dict) -> bool:
            return "dur" not in item or item.get("dur") in (None, "")

        def _fill(item: dict) -> dict:
            if not _needs(item):
                return item
            probe_path = (
                item.get("video_path")
                or item.get("wav")
                or item.get("wav_path")
            )
            dur = _ffprobe_duration(probe_path) if probe_path else None
            if dur is not None:
                item["dur"] = dur
            return item

        targets = [i for i, it in enumerate(items) if _needs(it)]
        if targets:
            if use_threading and num_threads > 1:
                with ThreadPoolExecutor(max_workers=num_threads) as pool:
                    filled = list(pool.map(_fill, (items[i] for i in targets)))
                for i, item in zip(targets, filled):
                    items[i] = item
            else:
                for i in targets:
                    items[i] = _fill(items[i])

        if dst != src or targets:
            if dst != src and not targets:
                shutil.copyfile(src, dst)
            else:
                JSONLToolkit.save(items, dst)
        return items


# ---------------------------------------------------------------------------
# Per-task schema validator (fail fast, before model load)
# ---------------------------------------------------------------------------

# Accept both spellings the CLI uses.
_TASK_ALIASES = {
    "v2a": "v2a", "VTA": "v2a", "VA": "v2a",
    "v2s": "v2s", "VTS": "v2s",
    # Paper terminology is V2SA; "v2as" / "VTAS" are kept as deprecated aliases.
    "v2sa": "v2sa", "VTSA": "v2sa", "V2SA": "v2sa",
    "v2as": "v2sa", "VTAS": "v2sa",
}


def _row_label(row: dict, row_idx: int) -> str:
    ident = row.get("id") or row.get("video_path") or f"row_{row_idx}"
    return f"row {row_idx} ({ident})"


def _check_ref_wav_shape(ref, label: str) -> None:
    """Reject nonsense ref_wav shapes with a clear message."""
    if ref is None:
        return  # absence is caller's concern
    if isinstance(ref, str):
        return
    if isinstance(ref, list):
        if len(ref) == 0:
            raise ValueError(
                f"{label}: ref_wav is an empty list. "
                "Hint: pass a wav path (single-speaker) or a list of wav paths (multi-speaker)."
            )
        for i, entry in enumerate(ref):
            if isinstance(entry, str):
                continue
            if isinstance(entry, (list, tuple)) and len(entry) == 3 \
                    and isinstance(entry[0], str) \
                    and isinstance(entry[1], (int, float)) \
                    and isinstance(entry[2], (int, float)):
                continue
            raise ValueError(
                f"{label}: ref_wav[{i}] has bad shape ({entry!r}). "
                "Hint: each entry must be either a str path or a "
                "[path, start_sec, end_sec] triple."
            )
        return
    raise ValueError(
        f"{label}: ref_wav has unsupported type {type(ref).__name__}. "
        "Hint: use a str path (single-speaker), a list of str paths (multi-speaker), "
        "or a list of [path, start, end] triples."
    )


def validate_row(row: dict, task: str, row_idx: int = 0) -> None:
    """Validate a single JSONL row for the given task. Raises ValueError.

    task ∈ {v2a, v2s, v2sa} (accepts VTA/VTS/VTSA aliases too; "v2as" is a deprecated alias for "v2sa").
    """
    t = _TASK_ALIASES.get(task, task)
    if t not in ("v2a", "v2s", "v2sa"):
        raise ValueError(f"validate_row: unknown task {task!r}. Use one of v2a/v2s/v2sa.")
    label = f"{t.upper()} {_row_label(row, row_idx)}"

    # video_path always required
    vp = row.get("video_path")
    if not vp or not isinstance(vp, str):
        raise ValueError(
            f"{label}: missing required field 'video_path' (str). "
            "Hint: point video_path at your .mp4 file (relative to repo root or absolute)."
        )

    trans_cap = row.get("trans_cap")
    text_prompt = row.get("text_prompt")

    if t == "v2a":
        # V2A: SFX-only. trans_cap must be sentinel; text_prompt must be a real caption.
        if trans_cap is not None and trans_cap != "none_speech":
            raise ValueError(
                f"{label}: task=v2a requires trans_cap=='none_speech' (got {trans_cap!r}). "
                "Hint: use --task v2s or --task v2sa for speech-conditioned modes."
            )
        if not text_prompt or not isinstance(text_prompt, str):
            raise ValueError(
                f"{label}: task=v2a is missing required field 'text_prompt' (non-empty str). "
                "Hint: set text_prompt to a SFX caption (e.g. 'Rain on a tin roof')."
            )
        if text_prompt == "clean speech":
            raise ValueError(
                f"{label}: task=v2a with text_prompt=='clean speech' would disable both streams. "
                "Hint: use a real SFX caption, or pick --task v2s if you want speech only."
            )
    elif t == "v2s":
        # V2S: speech only. trans_cap must be a real transcript; text_prompt must be sentinel.
        if not trans_cap or not isinstance(trans_cap, str) or trans_cap == "none_speech":
            raise ValueError(
                f"{label}: task=v2s requires trans_cap to be a non-empty transcript "
                "(wrap each utterance in <S>...<E>). "
                "Hint: run examples/prompts/singlespeaker_caption.txt through Gemini 2.5 Pro to build one."
            )
        if "<S>" not in trans_cap or "<E>" not in trans_cap:
            raise ValueError(
                f"{label}: task=v2s trans_cap must contain at least one <S>...<E>-wrapped utterance. "
                "Hint: e.g. 'A young man says: <S>hello world<E>'."
            )
        if text_prompt not in (None, "clean speech"):
            raise ValueError(
                f"{label}: task=v2s requires text_prompt=='clean speech' (got {text_prompt!r}). "
                "Hint: use --task v2sa if you want joint speech + ambient SFX."
            )
        _check_ref_wav_shape(row.get("ref_wav"), label)
    elif t == "v2sa":
        # V2SA: both streams live.
        if not trans_cap or not isinstance(trans_cap, str) or trans_cap == "none_speech":
            raise ValueError(
                f"{label}: task=v2sa requires trans_cap to be a non-empty transcript "
                "(one <S>...<E> per speaker turn). "
                "Hint: run examples/prompts/multispeaker_caption.txt through Gemini 2.5 Pro."
            )
        if "<S>" not in trans_cap or "<E>" not in trans_cap:
            raise ValueError(
                f"{label}: task=v2sa trans_cap must contain <S>...<E>-tagged utterances."
            )
        if not text_prompt or not isinstance(text_prompt, str) or text_prompt == "clean speech":
            raise ValueError(
                f"{label}: task=v2sa requires a real text_prompt (SFX caption); "
                f"got {text_prompt!r}. Hint: use --task v2s if you want speech only."
            )
        _check_ref_wav_shape(row.get("ref_wav"), label)


def validate_rows(rows: List[dict], task: str) -> None:
    """Validate every row; raise on the first failure."""
    for i, row in enumerate(rows):
        validate_row(row, task, row_idx=i)
