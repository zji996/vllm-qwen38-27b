#!/usr/bin/env bash
# Start DSpark BF16, check KV, run pong + medium prefill + decode. Stop.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8000}"
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
DRAFT_NAME="${DRAFT_NAME:-Qwen3.8-27B-DSpark}" \
  CONTAINER_NAME="${CONTAINER_NAME}" \
  "${ROOT}/scripts/serve-nvfp4-dspark.sh" "$@" >>"$LOG" 2>&1 &
pid=$!

ready=0
for i in $(seq 1 180); do
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

set +e
python3 - "$LOG" "$ready" "$PORT" <<'PY'
import json, re, sys, urllib.request

log, ready, port = sys.argv[1], sys.argv[2] == "1", sys.argv[3]
text = open(log, errors="replace").read()
kv = re.search(
    r"GPU KV cache size:\s*([0-9,]+)\s*tokens, Maximum concurrency for ([0-9,]+) tokens per request: ([0-9.]+)x",
    text,
)
avail = re.search(r"Available KV cache memory:\s*([0-9.]+)\s*GiB", text)
full = re.search(r"or `--kv-cache-memory=[0-9]+` \(([0-9.]+) GiB\) to fully utilize", text)
oom = bool(re.search(r"CUDA out of memory|Engine core initialization failed", text, re.I))
print("ready", ready)
print("oom_init", oom)
print("kv_gib", avail.group(1) if avail else None)
print("fully_utilize_gib", full.group(1) if full else None)
if kv:
    print("kv_tokens", kv.group(1).replace(",", ""))
    print("concurrency", kv.group(3))
else:
    print("kv_tokens", None)
    sys.exit(0 if not ready else 1)

if not ready:
    sys.exit(0)

base = f"http://127.0.0.1:{port}"
model = "/models/Qwen3.8-27B-NVFP4"

def chat(name, content, max_tokens):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = json.load(resp)
        usage = body.get("usage") or {}
        text_out = ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        print(
            f"gen_{name}",
            "ok",
            "prompt", usage.get("prompt_tokens"),
            "completion", usage.get("completion_tokens"),
            "head", text_out[:60].replace("\n", " "),
        )
        return True
    except Exception as exc:
        print(f"gen_{name}", "FAIL", type(exc).__name__, str(exc)[:200])
        return False

ok = True
ok &= chat("pong", "Reply with the single word: pong", 16)
ok &= chat(
    "count",
    "Count from 1 to 30, integers only, space-separated, no other text.",
    128,
)
# ~2k-token prefill (chunked at 2048) + 64 decode
pad = ("The quick brown fox jumps over the lazy dog. ") * 400
ok &= chat("prefill2k", pad + "\nReply with the single word: pong", 16)
ok &= chat(
    "decode256",
    "Write 20 short numbered sentences about tea. No extra commentary.",
    256,
)
sys.exit(0 if ok else 2)
PY
status=$?
set -e

nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader || true
if grep -qiE "CUDA out of memory" "$LOG"; then
  echo "oom_in_log yes"
else
  echo "oom_in_log no"
fi
echo "exit $status"
stop_all
wait "$pid" 2>/dev/null || true
exit "$status"
