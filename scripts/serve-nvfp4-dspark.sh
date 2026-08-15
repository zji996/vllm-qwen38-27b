#!/usr/bin/env bash
# Serve NVFP4 target + RadixArk DSpark draft. Experimental.
# RadixArk ships architectures=["DSparkDraftModel"], which vLLM maps to
# DeepSeek-V4. prepare-dspark-for-vllm.py rewrites that to Qwen3DSparkModel.
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
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
# 0.95 + max-num-seqs=10 OOMs during CUDA-graph warmup on 2×3080.
# seqs=2 could hold 0.95; seqs=10 needs graph headroom → 0.93.
GPU_MEM="${GPU_MEM:-0.93}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-10}"
HOST_MODELS="${HOST_MODELS:-${ROOT}/models}"
CONTAINER_MODEL="${CONTAINER_MODEL:-/models/Qwen3.8-27B-NVFP4}"
DRAFT_NAME="${DRAFT_NAME:-Qwen3.8-27B-DSpark}"
DRAFT_HOST="${HOST_MODELS}/${DRAFT_NAME}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-qwen38-nvfp4-dspark}"
export NUM_SPEC="${NUM_SPEC:-7}"
export DRAFT_CONTAINER="${DRAFT_CONTAINER:-/models/${DRAFT_NAME}}"

python3 "${ROOT}/scripts/prepare-dspark-for-vllm.py" "${DRAFT_HOST}"

SPEC_CONFIG=$(python3 - <<'PY'
import json, os
print(json.dumps({
    "method": "dspark",
    "model": os.environ["DRAFT_CONTAINER"],
    "num_speculative_tokens": int(os.environ.get("NUM_SPEC", "7")),
    "draft_sample_method": "probabilistic",
}))
PY
)

exec docker run --rm --name "${CONTAINER_NAME}" \
  --gpus all \
  --ipc=host \
  --shm-size=16g \
  -p "${PORT}:8000" \
  -v "${HOST_MODELS}:/models:ro" \
  -e NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-0,1}" \
  -e NUM_SPEC="${NUM_SPEC}" \
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
  --speculative-config "${SPEC_CONFIG}" \
  "$@"
