#!/usr/bin/env bash
# NVFP4 target + native Qwen3.8 MTP (1 layer in this checkpoint).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export NUM_SPEC="${NUM_SPEC:-3}"
export CONTAINER_NAME="${CONTAINER_NAME:-vllm-qwen38-nvfp4-mtp}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

SPEC_CONFIG=$(python3 - <<'PY'
import json, os
print(json.dumps({
    "method": "mtp",
    "num_speculative_tokens": int(os.environ.get("NUM_SPEC", "3")),
}))
PY
)

exec "${ROOT}/scripts/serve-nvfp4.sh" --speculative-config "${SPEC_CONFIG}"
