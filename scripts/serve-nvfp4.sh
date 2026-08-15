#!/usr/bin/env bash
# Serve Unsloth Qwen3.8-27B-NVFP4 on 2x RTX 3080 20GB.
# First-pass: text-only. Default gpu-mem 0.93.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  # shellcheck disable=SC1091
  set -a && source .env && set +a
fi

IMAGE="${IMAGE:-vllm-qwen38-27b:ampere}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEM="${GPU_MEM:-0.93}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
HOST_MODELS="${HOST_MODELS:-${ROOT}/models}"
CONTAINER_MODEL="${CONTAINER_MODEL:-/models/Qwen3.8-27B-NVFP4}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-qwen38-nvfp4}"

if [[ ! -f "${HOST_MODELS}/Qwen3.8-27B-NVFP4/config.json" ]]; then
  echo "target checkpoint missing under ${HOST_MODELS}/Qwen3.8-27B-NVFP4" >&2
  exit 1
fi

exec docker run --rm --name "${CONTAINER_NAME}" \
  --gpus all \
  --ipc=host \
  --shm-size=16g \
  -p "${PORT}:8000" \
  -v "${HOST_MODELS}:/models:ro" \
  -e NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-0,1}" \
  "$IMAGE" \
  "${CONTAINER_MODEL}" \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size "${TP_SIZE}" \
  --dtype auto \
  --kv-cache-dtype auto \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEM}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --language-model-only \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --trust-remote-code \
  --linear-backend marlin \
  "$@"
