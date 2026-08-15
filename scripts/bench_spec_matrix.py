#!/usr/bin/env python3
"""Warm up a live vLLM server, then time a short prompt set.

Cold-start / Triton JIT is discarded. Spec metrics are Prometheus counter
deltas around each measured request.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

METRIC_KEYS = (
    "vllm:spec_decode_num_drafts",
    "vllm:spec_decode_num_draft_tokens",
    "vllm:spec_decode_num_accepted_tokens",
)

PROMPTS = [
    {
        "name": "count-1-30",
        "messages": [
            {
                "role": "user",
                "content": "Count from 1 to 30, integers only, space-separated, no other text.",
            }
        ],
        "temperature": 0.0,
        "max_tokens": 128,
        "enable_thinking": False,
    },
    {
        "name": "python-generator",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Write a Python generator function `countdown(n)` that yields "
                    "n, n-1, ..., 1. Only the function, no explanation."
                ),
            }
        ],
        "temperature": 0.7,
        "max_tokens": 128,
        "enable_thinking": False,
    },
]


def http_json(url: str, payload: dict[str, Any] | None = None, timeout: int = 180) -> Any:
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method="GET" if payload is None else "POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def http_text(url: str, timeout: int = 30) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode()


def parse_metrics(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in METRIC_KEYS:
        total = 0.0
        pattern = re.compile(
            rf"^{re.escape(key)}(?:_total)?(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)\s*$"
        )
        for line in text.splitlines():
            match = pattern.match(line)
            if match:
                total += float(match.group(1))
        out[key] = total
    return out


def spec_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    drafts = after["vllm:spec_decode_num_drafts"] - before["vllm:spec_decode_num_drafts"]
    draft_tokens = (
        after["vllm:spec_decode_num_draft_tokens"]
        - before["vllm:spec_decode_num_draft_tokens"]
    )
    accepted = (
        after["vllm:spec_decode_num_accepted_tokens"]
        - before["vllm:spec_decode_num_accepted_tokens"]
    )
    mean_accept = (accepted / drafts) if drafts > 0 else 0.0
    accept_rate = (accepted / draft_tokens) if draft_tokens > 0 else 0.0
    return {
        "drafts": drafts,
        "draft_tokens": draft_tokens,
        "accepted_tokens": accepted,
        "mean_accept_length": round(mean_accept, 4),
        "draft_accept_rate": round(accept_rate, 4),
    }


def wait_ready(base: str, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        try:
            http_json(f"{base}/v1/models", timeout=5)
            return
        except Exception as exc:  # noqa: BLE001
            last = str(exc)
            time.sleep(2)
    raise SystemExit(f"server not ready after {timeout_s}s: {last}")


def one_chat(base: str, model: str, prompt: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": prompt["messages"],
        "max_tokens": prompt["max_tokens"],
        "temperature": prompt["temperature"],
        "chat_template_kwargs": {"enable_thinking": prompt["enable_thinking"]},
    }
    metrics_before = parse_metrics(http_text(f"{base}/metrics"))
    t0 = time.perf_counter()
    body = http_json(f"{base}/v1/chat/completions", payload, timeout=180)
    elapsed = time.perf_counter() - t0
    metrics_after = parse_metrics(http_text(f"{base}/metrics"))
    usage = body.get("usage") or {}
    completion = int(usage.get("completion_tokens") or 0)
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    text = ""
    choices = body.get("choices") or []
    if choices:
        text = ((choices[0].get("message") or {}).get("content") or "")[:240]
    tok_s = (completion / elapsed) if elapsed > 0 and completion > 0 else 0.0
    spec = spec_delta(metrics_before, metrics_after)
    if spec["drafts"] == 0 and spec["draft_tokens"] == 0:
        sample = [
            line
            for line in http_text(f"{base}/metrics").splitlines()
            if "spec_decode" in line or "speculative" in line
        ][:12]
        if sample:
            print("spec metric sample:", sample, file=sys.stderr)
    return {
        "name": prompt["name"],
        "elapsed_s": round(elapsed, 4),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion,
        "tok_s": round(tok_s, 2),
        "text_head": text.replace("\n", "\\n"),
        "spec": spec,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="/models/Qwen3.8-27B-NVFP4")
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--wait", type=int, default=600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    wait_ready(args.base, args.wait)
    warmup_spec = {
        "name": "warmup-pong",
        "messages": [{"role": "user", "content": "Reply with the single word: pong"}],
        "temperature": 0.0,
        "max_tokens": 16,
        "enable_thinking": False,
    }
    warmup_runs = []
    for i in range(args.warmup):
        spec = dict(warmup_spec)
        spec["name"] = f"warmup-{i}"
        warmup_runs.append(one_chat(args.base, args.model, spec))
        print(f"warmup {i}: {warmup_runs[-1]['tok_s']} tok/s", file=sys.stderr)

    measured = [one_chat(args.base, args.model, spec) for spec in PROMPTS]
    for row in measured:
        print(
            f"{row['name']}: {row['completion_tokens']} tok / {row['elapsed_s']} s "
            f"= {row['tok_s']} tok/s  accept={row['spec']['mean_accept_length']}",
            file=sys.stderr,
        )

    result = {
        "config": args.config_name,
        "model": args.model,
        "warmup": warmup_runs,
        "runs": measured,
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
