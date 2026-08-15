#!/usr/bin/env python3
"""Final serving matrix: NVFP4 + BF16 DSpark, streaming TTFT / prefill / decode.

Length axis: conc=1 at ~1k/8k/32k/96k. Concurrency axis: 2/4/8/10 at ~1k.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

METRIC_KEYS = (
    "vllm:spec_decode_num_drafts",
    "vllm:spec_decode_num_draft_tokens",
    "vllm:spec_decode_num_accepted_tokens",
)

FILLER_EN = (
    "The agency published a quarterly briefing on procurement delays, "
    "staffing gaps, and revised delivery dates for regional infrastructure "
    "projects. "
)
FILLER_CODE = (
    "# padding: interval bookkeeping notes and fixture comments for the "
    "module under test.\n"
)

TASKS = {
    "translate": {
        "filler": FILLER_EN,
        "instruction": (
            "Translate the following English paragraph into Chinese. "
            "Output only the translation, no notes.\n\n"
            "The committee postponed the vote until Friday, citing incomplete "
            "budget figures and unanswered questions from the audit office."
        ),
    },
    "code": {
        "filler": FILLER_CODE,
        "instruction": (
            "Write a Python function `merge_intervals(intervals)` that merges "
            "overlapping inclusive [start, end] ranges. Return a new sorted "
            "list of lists. Only the function, no explanation."
        ),
    },
}

LENGTHS = (1024, 8192, 32768, 98304)
CONC_LEVELS = (2, 4, 8, 10)
CONC_PREFILL = 1024


def http_json(
    url: str, payload: dict[str, Any] | None = None, timeout: int = 180
) -> Any:
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    req = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method="GET" if payload is None else "POST",
    )
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


def count_tokens(base: str, model: str, content: str) -> int:
    body = http_json(
        f"{base}/tokenize",
        {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=120,
    )
    return int(body["count"])


def pad_content(
    base: str, model: str, instruction: str, filler: str, target: int
) -> tuple[str, int]:
    inst_n = count_tokens(base, model, instruction)
    if inst_n >= target:
        return instruction, inst_n
    mixed = filler + instruction
    mixed_n = count_tokens(base, model, mixed)
    per = max(1, mixed_n - inst_n)
    n = max(1, (target - inst_n) // per)
    content = filler * n + "\n\n" + instruction
    for _ in range(4):
        actual = count_tokens(base, model, content)
        if abs(actual - target) <= max(8, target // 50):
            return content, actual
        if actual < target:
            n += max(1, (target - actual) // per)
        else:
            n = max(0, n - max(1, (actual - target) // per))
        content = filler * n + "\n\n" + instruction
    return content, count_tokens(base, model, content)


def timeout_s(prefill: int, conc: int) -> int:
    if prefill >= 90000:
        return 600
    if prefill >= 30000:
        return 300
    if conc >= 8:
        return 300
    return 180


def stream_chat(
    base: str,
    model: str,
    content: str,
    max_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    t_first: float | None = None
    t_last = t0
    text_parts: list[str] = []
    usage: dict[str, Any] = {}
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = (choices[0].get("delta") or {}).get("content") or ""
            if delta:
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                t_last = now
                text_parts.append(delta)
    t_end = time.perf_counter()
    if t_first is None:
        t_first = t_end
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    ttft_s = t_first - t0
    e2e_s = t_end - t0
    decode_s = t_last - t_first
    prefill_tps = (prompt_tokens / ttft_s) if ttft_s > 0 else 0.0
    decode_tps = (
        ((completion - 1) / decode_s) if decode_s > 0 and completion > 1 else 0.0
    )
    e2e_tps = (completion / e2e_s) if e2e_s > 0 and completion > 0 else 0.0
    text = "".join(text_parts)
    return {
        "elapsed_s": round(e2e_s, 4),
        "ttft_s": round(ttft_s, 4),
        "ttft_ms": round(ttft_s * 1000, 1),
        "prefill_tps": round(prefill_tps, 2),
        "decode_tps": round(decode_tps, 2),
        "e2e_tps": round(e2e_tps, 2),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion,
        "text_head": text[:180].replace("\n", "\\n"),
    }


def one_cell_single(
    base: str, model: str, name: str, content: str, prefill: int, max_tokens: int
) -> dict[str, Any]:
    before = parse_metrics(http_text(f"{base}/metrics"))
    row = stream_chat(base, model, content, max_tokens, timeout_s(prefill, 1))
    after = parse_metrics(http_text(f"{base}/metrics"))
    row["name"] = name
    row["task"] = name.split("-")[0]
    row["prefill_target"] = prefill
    row["concurrency"] = 1
    row["spec"] = spec_delta(before, after)
    return row


def one_cell_conc(
    base: str,
    model: str,
    name: str,
    content: str,
    prefill: int,
    conc: int,
    max_tokens: int,
) -> dict[str, Any]:
    before = parse_metrics(http_text(f"{base}/metrics"))
    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=conc) as pool:
        futs = [
            pool.submit(
                stream_chat,
                base,
                model,
                content,
                max_tokens,
                timeout_s(prefill, conc),
            )
            for _ in range(conc)
        ]
        for fut in as_completed(futs):
            try:
                rows.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{type(exc).__name__}: {exc}"[:240])
    batch_wall = time.perf_counter() - t0
    after = parse_metrics(http_text(f"{base}/metrics"))
    ok = [r for r in rows]
    completion_sum = sum(r["completion_tokens"] for r in ok)
    return {
        "name": name,
        "task": name.split("-")[0],
        "prefill_target": prefill,
        "concurrency": conc,
        "batch_wall_s": round(batch_wall, 4),
        "batch_output_tps": round(
            (completion_sum / batch_wall) if batch_wall > 0 else 0.0, 2
        ),
        "mean_ttft_ms": round(
            sum(r["ttft_ms"] for r in ok) / len(ok), 1
        )
        if ok
        else None,
        "mean_decode_tps": round(
            sum(r["decode_tps"] for r in ok) / len(ok), 2
        )
        if ok
        else None,
        "mean_prefill_tps": round(
            sum(r["prefill_tps"] for r in ok) / len(ok), 2
        )
        if ok
        else None,
        "spec": spec_delta(before, after),
        "errors": errors,
        "replicas": ok,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="/models/Qwen3.8-27B-NVFP4")
    parser.add_argument("--config-name", default="dspark-bf16")
    parser.add_argument("--out", default="")
    parser.add_argument("--wait", type=int, default=720)
    parser.add_argument("--max-tokens", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    wait_ready(args.base, args.wait)
    print("ready", file=sys.stderr)

    cache: dict[tuple[str, int], tuple[str, int]] = {}
    for task_name, spec in TASKS.items():
        for target in sorted(set(LENGTHS) | {CONC_PREFILL}):
            print(f"pad {task_name} {target}", file=sys.stderr)
            cache[(task_name, target)] = pad_content(
                args.base, args.model, spec["instruction"], spec["filler"], target
            )
            print(
                f"  actual {cache[(task_name, target)][1]} tokens",
                file=sys.stderr,
            )

    warmup_pong = TASKS["translate"]["instruction"]
    for i in range(3):
        row = stream_chat(args.base, args.model, "Reply with the single word: pong", 16, 120)
        print(f"warmup pong {i}: ttft={row['ttft_ms']} ms", file=sys.stderr)
    for task_name in TASKS:
        content, _ = cache[(task_name, 1024)]
        row = stream_chat(args.base, args.model, content, 32, 180)
        print(f"warmup {task_name}-1k: ttft={row['ttft_ms']} ms", file=sys.stderr)
    pong = "Reply with the single word: pong"
    with ThreadPoolExecutor(max_workers=10) as pool:
        list(pool.map(lambda _: stream_chat(args.base, args.model, pong, 8, 180), range(10)))
    print("warmup conc=10 done", file=sys.stderr)

    runs: list[dict[str, Any]] = []
    for task_name in TASKS:
        for target in LENGTHS:
            name = f"{task_name}-{target}-c1"
            content, actual = cache[(task_name, target)]
            print(f"=== {name} (actual {actual}) ===", file=sys.stderr)
            try:
                row = one_cell_single(
                    args.base, args.model, name, content, target, args.max_tokens
                )
            except Exception as exc:  # noqa: BLE001
                row = {
                    "name": name,
                    "task": task_name,
                    "prefill_target": target,
                    "concurrency": 1,
                    "error": f"{type(exc).__name__}: {exc}"[:400],
                }
            runs.append(row)
            print(json.dumps({k: row.get(k) for k in (
                "name", "ttft_ms", "prefill_tps", "decode_tps", "error"
            ) if k in row or row.get("error")}, ensure_ascii=False), file=sys.stderr)

    for task_name in TASKS:
        content, actual = cache[(task_name, CONC_PREFILL)]
        for conc in CONC_LEVELS:
            name = f"{task_name}-{CONC_PREFILL}-c{conc}"
            print(f"=== {name} (actual {actual}) ===", file=sys.stderr)
            try:
                row = one_cell_conc(
                    args.base,
                    args.model,
                    name,
                    content,
                    CONC_PREFILL,
                    conc,
                    args.max_tokens,
                )
            except Exception as exc:  # noqa: BLE001
                row = {
                    "name": name,
                    "task": task_name,
                    "prefill_target": CONC_PREFILL,
                    "concurrency": conc,
                    "error": f"{type(exc).__name__}: {exc}"[:400],
                }
            runs.append(row)
            print(json.dumps({k: row.get(k) for k in (
                "name", "batch_output_tps", "mean_ttft_ms", "mean_decode_tps", "error"
            ) if k in row or row.get("error")}, ensure_ascii=False), file=sys.stderr)

    result = {
        "config": args.config_name,
        "model": args.model,
        "pad_actual": {f"{k[0]}-{k[1]}": v[1] for k, v in cache.items()},
        "runs": runs,
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    failed = [r for r in runs if r.get("error") or r.get("errors")]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
