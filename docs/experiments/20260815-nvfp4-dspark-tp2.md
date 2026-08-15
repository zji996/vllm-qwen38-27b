# 2026-08-15 NVFP4 + DSpark TP2 Marlin

## 设置
- 镜像：`vllm-qwen38-27b:ampere` / vLLM `0.27.2rc1.dev77+gac7509e2b`
- target：`models/Qwen3.8-27B-NVFP4`
- draft：`models/Qwen3.8-27B-DSpark`（`prepare-dspark-for-vllm.py` 把 `DSparkDraftModel` 改成 `Qwen3DSparkModel`，并钉 `partial_rotary_factor=1.0`）
- flags：TP=2，`--linear-backend marlin`，`--max-model-len 131072`，`--gpu-memory-utilization 0.90`，`--language-model-only`，`num_speculative_tokens=7`，`draft_sample_method=probabilistic`
- 日志：`logs/serve-nvfp4-dspark.log`

## 命令
```bash
MAX_MODEL_LEN=131072 GPU_MEM=0.90 ./scripts/serve-nvfp4-dspark.sh
```

## 结果
- 加载成功。每卡 12.23 GiB（target-only 是 10.49 GiB）。架构解析为 `Qwen3DSparkModel`。aux layers `(5, 17, 29, 41, 53)`。
- KV **3.93 GiB → 133,746 tokens**；131,072 context 并发 1.02x。
- nvidia-smi 约 18321 / 18297 MiB。
- 冒烟：`pong`。
- 计数 1–30（greedy）：81 completion / 0.657 s ≈ **123 tok/s**（target-only 同任务 49.9 tok/s，约 **2.5×**）。
  - Spec：mean accept length **7.08**，draft accept **86.9%**（任务极好猜，偏乐观）。
- Python 生成器（t=0.7）：46 completion / 0.482 s ≈ **95 tok/s**。
  - Spec：mean accept length **5.22**，draft accept **60.3%**，per-pos 0.778/0.778/0.778/0.556/0.556/0.444/0.333。

## 结论
RadixArk DSpark 在 vLLM 里能跑，且比 Marlin NVFP4 基线明显更快。默认 **BF16 草稿 + 131072**。gpu-mem 0.93 重测：KV 池 **153,808**，并发 **1.17×**（日志 `logs/dspark-ctx-spec-20260815/kv-bf16-131k-s7.log`）。不要开 262k。
