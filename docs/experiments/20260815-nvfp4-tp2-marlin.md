# 2026-08-15 NVFP4 TP2 Marlin 加载与冒烟

## 设置
- 镜像：`vllm-qwen38-27b:ampere` / vLLM `0.27.2rc1.dev77+gac7509e2b`
- 模型：`models/Qwen3.8-27B-NVFP4`（22G + 811M MTP），无 DSpark
- flags：`--tensor-parallel-size 2 --linear-backend marlin --max-model-len 32768 --gpu-memory-utilization 0.90 --language-model-only --kv-cache-dtype auto`
- 硬件：2× RTX 3080 20GB SM 8.6，驱动 580.105.08
- 日志：`logs/serve-nvfp4-tp2-marlin.log`

## 命令
```bash
MAX_MODEL_LEN=32768 GPU_MEM=0.90 ./scripts/serve-nvfp4.sh --linear-backend marlin
./scripts/smoke-chat.sh
```

## 结果
- 加载成功。每卡权重 10.49 GiB。`MarlinNvFp4LinearKernel` + FP8 注意力 Marlin W8A16 回退（日志有预期 warning）。
- 无 GPU P2P，custom allreduce 关闭，走 PYNCCL。
- 可用 KV **5.23 GiB → 264,571 tokens**；32,768 context 时并发 8.07x。
- FlashInfer 把 KV 解析成 `float8_e4m3fn`（来自 checkpoint 的 `kv_cache_scheme`，不是 CLI `--kv-cache-dtype fp8`）。
- 冒烟：`pong`，prompt 19 / completion 2。
- 计数 1–30：completion 81 tokens，端到端 1.623 s ≈ **49.9 tok/s**（含 prefill）。
- nvidia-smi 服务空闲时约 17267 / 17239 MiB。

## 结论
Ampere + Marlin 可以跑这份 NVFP4。混合注意力让 KV 很便宜，原生 262k 在 target-only 下看起来装得下（264k tokens KV）。下一步加 DSpark，再量一次 KV。
