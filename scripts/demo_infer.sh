#!/usr/bin/env bash
# ============================================================
# CineDub — one-liner demo wrapper.
#
# Usage:  bash scripts/demo_infer.sh <mode>
#   mode ∈ { v2a | v2s | v2sa }
#
# This is a thin wrapper around the real Python entrypoint:
#     python inference.py --task <mode> --input examples/<mode>.jsonl ...
#
# All knobs can be overridden by env:
#   CKPT              path to the model .ckpt   (default: weights/cinedub/checkpoint/step=300000.ckpt)
#   CKPT_MULTISPEAKER path to the v2sa .ckpt    (default: same as CKPT — one unified checkpoint)
#   MODEL_CONFIG      path to model_config.json (default: shipped config)
#   SAVE_DIR          output dir                (default: exp/demo)
#   CFG               cfg-scale                 (default: auto — 2.5 for v2a, 7 for v2s/v2sa)
#   STEPS             diffusion steps           (default: 100)
#   SEED              random seed (int)         (default: -1 = random)
# ============================================================

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    sed -n '2,22p' "$0" | sed -e 's|^# \{0,1\}||'
    exit 0
fi

MODE="${1:-}"
if [[ "${MODE}" == "v2as" ]]; then
    echo "[CineDub] warning: mode 'v2as' is a deprecated alias for 'v2sa' (paper terminology); continuing as v2sa." >&2
    MODE="v2sa"
fi
if [[ -z "${MODE}" || ( "${MODE}" != "v2a" && "${MODE}" != "v2s" && "${MODE}" != "v2sa" ) ]]; then
    echo "usage: bash scripts/demo_infer.sh {v2a|v2s|v2sa}" >&2
    exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_CONFIG="${MODEL_CONFIG:-${ROOT_DIR}/stable_audio_tools/configs/model_configs/txt2audio/stable_audio_ours_vt2as_translarge_cleanemb_ap.json}"
CKPT="${CKPT:-${ROOT_DIR}/weights/cinedub/checkpoint/step=300000.ckpt}"
CKPT_MULTISPEAKER="${CKPT_MULTISPEAKER:-${CKPT}}"
SAVE_DIR="${SAVE_DIR:-${ROOT_DIR}/exp/demo}"
# CFG env var overrides the per-task auto-default (2.5 for v2a, 7 for v2s/v2sa)
CFG_FLAG=()
[[ -n "${CFG:-}" ]] && CFG_FLAG=(--cfg "${CFG}")
STEPS="${STEPS:-100}"

case "${MODE}" in
    v2a)   INPUT="${INPUT:-${ROOT_DIR}/examples/v2a.jsonl}";  RUN_CKPT="${CKPT}" ;;
    v2s)   INPUT="${INPUT:-${ROOT_DIR}/examples/v2s.jsonl}";  RUN_CKPT="${CKPT}" ;;
    v2sa)  INPUT="${INPUT:-${ROOT_DIR}/examples/v2sa.jsonl}"; RUN_CKPT="${CKPT_MULTISPEAKER}" ;;
esac

# Fail fast with a friendly message rather than a torch stack.
if [[ ! -f "${MODEL_CONFIG}" ]]; then
    echo "[CineDub] MODEL_CONFIG not found: ${MODEL_CONFIG}" >&2
    exit 1
fi
if [[ ! -f "${RUN_CKPT}" ]]; then
    echo "[CineDub] Checkpoint not found: ${RUN_CKPT}" >&2
    echo "[CineDub] Download with:" >&2
    echo "[CineDub]   huggingface-cli download Dalision/cinedub --local-dir weights/cinedub" >&2
    exit 1
fi
if [[ ! -f "${INPUT}" ]]; then
    echo "[CineDub] input JSONL not found: ${INPUT}" >&2
    exit 1
fi

mkdir -p "${SAVE_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export ENABLE_TORCH_COMPILE=0
export ENABLE_GRADIENT_CHECKPOINT=0
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline

SEED_ARG=""
if [[ -n "${SEED:-}" ]]; then
    SEED_ARG="--seed ${SEED}"
fi

echo "[CineDub] mode=${MODE}  input=${INPUT}  ckpt=${RUN_CKPT}"
echo "[CineDub] cfg=${CFG:-auto}  steps=${STEPS}  save_dir=${SAVE_DIR}"

python "${ROOT_DIR}/inference.py" \
    --task "${MODE}" \
    --input "${INPUT}" \
    --output "${SAVE_DIR}" \
    --model_config "${MODEL_CONFIG}" \
    --ckpt "${RUN_CKPT}" \
    "${CFG_FLAG[@]}" \
    --steps "${STEPS}" \
    --all_demo \
    --copy_video_low_quality \
    ${SEED_ARG}
