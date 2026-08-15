# Serving

实测日期 2026-08-15。详细数字见 `docs/experiments/`。DSpark 默认 `--gpu-memory-utilization 0.93`（与 target-only / MTP 相同）。

## NVFP4 + DSpark（速度优先）

```bash
./scripts/serve-nvfp4-dspark.sh
```

脚本会：改 draft `architectures` 为 `Qwen3DSparkModel`、钉 `partial_rotary_factor=1.0`，然后 TP=2 + Marlin 启动。

固定参数：

- `--tensor-parallel-size 2`
- `--linear-backend marlin`
- `--max-model-len 131072`
- `--gpu-memory-utilization 0.93`
- `--max-num-seqs 10`（翻译短请求并发；不是 10×131k 预留）
- `--kv-cache-dtype auto`（checkpoint 的 FP8 KV scheme 仍会启用）
- `--language-model-only`
- speculative：`method=dspark`，`num_speculative_tokens=7`，`draft_sample_method=probabilistic`

停：`docker stop vllm-qwen38-nvfp4-dspark`

默认草稿是 BF16。gpu-mem **0.93** + `NUM_SPEC=7` + `max-num-seqs=10`。KV 池 **152,739** tokens，131k 并发 **1.17×**。`0.95` + seqs=10 会在 CUDA graph warmup 时 OOM（seqs=2 时 0.95 仍可用，池 166,916）。0.97 会在生成时 OOM。不要把 DSpark 开到 262k。

NVFP4A16 草稿可选（每卡少 0.72 GiB，代码 accept 略差），见 ADR 0002。保持 `NUM_SPEC=7`。

## MTP（更长上下文，仍比 none 快）

```bash
./scripts/serve-nvfp4-mtp.sh
```

用 checkpoint 自带的 1 层 MTP，不需要独立草稿。32k 公平对比 KV 约 200k，计数约 2.2×。

停：`docker stop vllm-qwen38-nvfp4-mtp`

## 只要更长上下文：关掉投机

```bash
MAX_MODEL_LEN=262144 ./scripts/serve-nvfp4.sh
```

gpu-mem 0.93 时 none KV **297,339 tokens**，原生 262k 单请求够用。

## 上下文怎么选

| 目标 | 命令 |
| --- | --- |
| 加速（推荐，BF16 131k） | `./scripts/serve-nvfp4-dspark.sh` |
| 较长上下文 + 加速 | `./scripts/serve-nvfp4-mtp.sh` |
| 尽量长上下文 | `MAX_MODEL_LEN=262144 ./scripts/serve-nvfp4.sh` |

DSpark 不要开 262k。BF16 草稿在 0.93 + seqs=10 下 131k 并发 1.17×（池 153k）。MTP 大约到 200k。

去冷启动对比：`docs/experiments/20260815-spec-matrix-gpu093.md`。
终态 matrix（长度 + 1k 并发）：`docs/experiments/20260815-final-dspark-matrix.md`。

## 客户端

thinking 默认 `temperature=1.0, top_p=0.95, top_k=20`。关思考：

```json
"chat_template_kwargs": {"enable_thinking": false}
```

冒烟：`./scripts/smoke-chat.sh`
