#!/usr/bin/env bash
# Start DSpark, record KV capacity, stop. No generation bench.
# Override draft with DRAFT_NAME=Qwen3.8-27B-DSpark (default) or ...-NVFP4A16.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8000}"
DRAFT_NAME="${DRAFT_NAME:-Qwen3.8-27B-DSpark}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-qwen38-nvfp4-dspark}"
LOG="${1:?log path}"
shift

stop_all() {
  docker stop vllm-qwen38-nvfp4 vllm-qwen38-nvfp4-mtp vllm-qwen38-nvfp4-dspark \
    vllm-qwen38-nvfp4-dspark-nvfp4 2>/dev/null || true
  local i
  for i in $(seq 1 25); do
    if ! ss -ltn | grep -q ":${PORT} "; then
      return 0
    fi
    sleep 1
  done
}

stop_all
mkdir -p "$(dirname "$LOG")"
: >"$LOG"
DRAFT_NAME="${DRAFT_NAME}" \
  CONTAINER_NAME="${CONTAINER_NAME}" \
  "${ROOT}/scripts/serve-nvfp4-dspark.sh" "$@" >>"$LOG" 2>&1 &
pid=$!

ok=0
for i in $(seq 1 180); do
  if grep -q "GPU KV cache size:" "$LOG" 2>/dev/null; then
    ok=1
    break
  fi
  if grep -qiE "CUDA out of memory|Engine core initialization failed|ValueError: .*(KV|memory)" "$LOG" 2>/dev/null; then
    break
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    break
  fi
  sleep 2
done

python3 - "$LOG" <<'PY'
import re, sys
text = open(sys.argv[1], errors="replace").read()
kv = re.search(r"GPU KV cache size:\s*([0-9,]+)\s*tokens, Maximum concurrency for ([0-9,]+) tokens per request: ([0-9.]+)x", text)
avail = re.search(r"Available KV cache memory:\s*([0-9.]+)\s*GiB", text)
weight = re.search(r"Model loading took\s+([0-9.]+)\s*GiB", text)
pad = re.findall(r"Add (\d+) padding layers, may waste at most ([0-9.]+)%", text)
print("weight_gib", weight.group(1) if weight else None)
print("kv_gib", avail.group(1) if avail else None)
if kv:
    print("kv_tokens", kv.group(1).replace(",", ""))
    print("max_model_len", kv.group(2).replace(",", ""))
    print("concurrency", kv.group(3))
else:
    print("kv_tokens", None)
    err = None
    for pat in ("CUDA out of memory", "Engine core initialization failed", "ValueError"):
        if pat.lower() in text.lower():
            err = pat
            break
    print("error", err or "timeout/no-kv-line")
print("padding", pad)
PY

stop_all
wait "$pid" 2>/dev/null || true
