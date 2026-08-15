#!/usr/bin/env bash
# CPU RTN of DSpark -> NVFP4A16. Uses the vLLM image (host has no torch).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  # shellcheck disable=SC1091
  set -a && source .env && set +a
fi

IMAGE="${IMAGE:-vllm-qwen38-27b:ampere}"

exec docker run --rm --name dspark-nvfp4a16-rtn \
  --user "$(id -u):$(id -g)" \
  --entrypoint python3 \
  -v "${ROOT}:/work" \
  -w /work \
  "$IMAGE" \
  /work/scripts/quantize_dspark_nvfp4a16.py \
  "$@"
