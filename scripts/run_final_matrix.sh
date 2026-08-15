#!/usr/bin/env bash
# Final DSpark serving matrix: production defaults, one serve, then bench.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PORT="${PORT:-8000}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-qwen38-nvfp4-dspark}"
STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${ROOT}/logs/final-matrix-${STAMP}"
mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/serve.log"
JSON="${OUT_DIR}/results.json"

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
: >"$LOG"
# Production defaults from serve-nvfp4-dspark.sh (0.93 / 131k / seqs=10).
# 0.95 + seqs=10 OOMed during warmup; see logs/final-matrix-20260815-114537.
"${ROOT}/scripts/serve-nvfp4-dspark.sh" >>"$LOG" 2>&1 &
pid=$!

ready=0
for i in $(seq 1 240); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if grep -qiE "CUDA out of memory|Engine core initialization failed" "$LOG" 2>/dev/null; then
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
kv = re.search(
    r"GPU KV cache size:\s*([0-9,]+)\s*tokens, Maximum concurrency for ([0-9,]+) tokens per request: ([0-9.]+)x",
    text,
)
avail = re.search(r"Available KV cache memory:\s*([0-9.]+)\s*GiB", text)
weight = re.search(r"Model loading took\s+([0-9.]+)\s*GiB", text)
print("weight_gib", weight.group(1) if weight else None)
print("kv_gib", avail.group(1) if avail else None)
if kv:
    print("kv_tokens", kv.group(1).replace(",", ""))
    print("max_model_len", kv.group(2).replace(",", ""))
    print("concurrency", kv.group(3))
else:
    print("kv_tokens", None)
PY

if [[ "$ready" != "1" ]]; then
  echo "server failed to start" >&2
  tail -n 80 "$LOG" >&2
  stop_all
  wait "$pid" 2>/dev/null || true
  exit 1
fi

set +e
python3 "${ROOT}/scripts/bench_final_matrix.py" \
  --config-name dspark-bf16-final \
  --out "$JSON" \
  --wait 60
bench_status=$?
set -e

if [[ -f "$JSON" ]]; then
  python3 - "$JSON" "$LOG" <<'PY'
import json, re, sys
path, log = sys.argv[1], sys.argv[2]
data = json.loads(open(path).read())
text = open(log, errors="replace").read()
m = re.search(r"GPU KV cache size:\s*([0-9,]+)\s*tokens, Maximum concurrency for ([0-9,]+) tokens per request: ([0-9.]+)x", text)
avail = re.search(r"Available KV cache memory:\s*([0-9.]+)\s*GiB", text)
weight = re.search(r"Model loading took\s+([0-9.]+)\s*GiB", text)
if m:
    data["kv_tokens"] = int(m.group(1).replace(",", ""))
    data["max_model_len"] = int(m.group(2).replace(",", ""))
    data["kv_concurrency"] = float(m.group(3))
if avail:
    data["kv_gib"] = float(avail.group(1))
if weight:
    data["weight_gib"] = float(weight.group(1))
open(path, "w").write(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
print(f"wrote {path} runs={len(data.get('runs', []))}")
PY
fi

stop_all
wait "$pid" 2>/dev/null || true
exit "$bench_status"
