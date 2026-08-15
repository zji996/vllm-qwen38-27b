#!/usr/bin/env bash
# Hit the OpenAI-compatible server after it is up.
set -euo pipefail

PORT="${PORT:-8000}"
MODEL="${MODEL:-/models/Qwen3.8-27B-NVFP4}"
BASE="http://127.0.0.1:${PORT}"

curl -fsS "${BASE}/v1/models" | python3 -m json.tool
python3 - "$BASE" "$MODEL" <<'PY'
import json, sys, urllib.request

base, model = sys.argv[1], sys.argv[2]
payload = {
    "model": model,
    "messages": [{"role": "user", "content": "Reply with the single word: pong"}],
    "max_tokens": 32,
    "temperature": 0.0,
    "chat_template_kwargs": {"enable_thinking": False},
}
req = urllib.request.Request(
    f"{base}/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=120) as resp:
    print(json.dumps(json.load(resp), indent=2, ensure_ascii=False))
PY
