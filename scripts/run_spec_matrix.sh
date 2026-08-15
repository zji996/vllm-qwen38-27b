#!/usr/bin/env bash
# Short speculative-decoding matrix. Drops cold start (Triton JIT / first request).
# Fair speed: same max-model-len / gpu-mem / max-num-seqs. KV capacity still
# comes from the leftover-memory log line, independent of the 32k cap.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export GPU_MEM="${GPU_MEM:-0.93}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
PORT="${PORT:-8000}"
STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${ROOT}/logs/spec-matrix-${STAMP}"
mkdir -p "$OUT_DIR"

stop_all() {
  docker stop vllm-qwen38-nvfp4 vllm-qwen38-nvfp4-mtp vllm-qwen38-nvfp4-dspark \
    vllm-qwen38-nvfp4-dspark-nvfp4 2>/dev/null || true
  local i
  for i in $(seq 1 20); do
    if ! ss -ltn | grep -q ":${PORT} "; then
      return 0
    fi
    sleep 1
  done
  echo "port ${PORT} still busy after docker stop" >&2
}

parse_log() {
  local log="$1"
  python3 - "$log" <<'PY'
import re, sys, json
text = open(sys.argv[1], errors="replace").read()
kv = None
weight = None
m = re.search(r"GPU KV cache size:\s*([0-9,]+)\s*tokens", text)
if m:
    kv = int(m.group(1).replace(",", ""))
m = re.search(r"Model loading took\s+([0-9.]+)\s*GiB", text)
if m:
    weight = float(m.group(1))
avail = None
m = re.search(r"Available KV cache memory:\s*([0-9.]+)\s*GiB", text)
if m:
    avail = float(m.group(1))
print(json.dumps({"kv_tokens": kv, "weight_gib": weight, "kv_gib": avail}))
PY
}

run_one() {
  local name="$1"
  local log="${OUT_DIR}/${name}.log"
  local json="${OUT_DIR}/${name}.json"
  shift
  echo "=== ${name} ==="
  stop_all
  "$@" >"$log" 2>&1 &
  local pid=$!
  if ! python3 "${ROOT}/scripts/bench_spec_matrix.py" \
      --config-name "$name" \
      --out "$json" \
      --wait 720 \
      --warmup 3; then
    echo "bench failed for ${name}; last log:" >&2
    tail -n 80 "$log" >&2
    kill "$pid" 2>/dev/null || true
    stop_all
    return 1
  fi
  local meta
  meta="$(parse_log "$log")"
  python3 - "$json" "$meta" <<'PY'
import json, sys
path, meta = sys.argv[1], json.loads(sys.argv[2])
data = json.loads(open(path).read())
data.update(meta)
open(path, "w").write(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({"config": data["config"], **meta, "runs": [
    {"name": r["name"], "tok_s": r["tok_s"], "accept": r["spec"]["mean_accept_length"]}
    for r in data["runs"]
]}, ensure_ascii=False))
PY
  stop_all
  wait "$pid" 2>/dev/null || true
}

chmod +x "${ROOT}/scripts/serve-nvfp4.sh" \
  "${ROOT}/scripts/serve-nvfp4-mtp.sh" \
  "${ROOT}/scripts/serve-nvfp4-dspark.sh" \
  "${ROOT}/scripts/quantize-dspark-nvfp4a16.sh" 2>/dev/null || true

CONFIGS="${MATRIX_CONFIGS:-none,mtp,dspark-bf16,dspark-nvfp4a16}"
IFS=',' read -r -a CONFIG_ARR <<< "$CONFIGS"

for name in "${CONFIG_ARR[@]}"; do
  case "$name" in
    none)
      run_one none "${ROOT}/scripts/serve-nvfp4.sh" || echo "FAILED none"
      ;;
    mtp)
      run_one mtp "${ROOT}/scripts/serve-nvfp4-mtp.sh" || echo "FAILED mtp"
      ;;
    dspark-bf16)
      run_one dspark-bf16 "${ROOT}/scripts/serve-nvfp4-dspark.sh" || echo "FAILED dspark-bf16"
      ;;
    dspark-nvfp4a16)
      if [[ -f "${ROOT}/models/Qwen3.8-27B-DSpark-NVFP4A16/model.safetensors" ]]; then
        run_one dspark-nvfp4a16 \
          env DRAFT_NAME=Qwen3.8-27B-DSpark-NVFP4A16 \
              CONTAINER_NAME=vllm-qwen38-nvfp4-dspark-nvfp4 \
              "${ROOT}/scripts/serve-nvfp4-dspark.sh" \
          || echo "FAILED dspark-nvfp4a16"
      else
        echo "skip dspark-nvfp4a16 (checkpoint missing)"
      fi
      ;;
    *)
      echo "unknown config: ${name}" >&2
      ;;
  esac
done

python3 - "$OUT_DIR" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*.json")):
    if path.name == "summary.json":
        continue
    rows.append(json.loads(path.read_text()))
summary = {"dir": str(root), "configs": rows}
(root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
print(f"wrote {root / 'summary.json'} ({len(rows)} configs)")
PY
