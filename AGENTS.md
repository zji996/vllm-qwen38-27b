# Agent 入口

本仓目标：在 2× RTX 3080 20GB 上用 vLLM 跑 Qwen3.8-27B NVFP4，并尝试 DSpark。

## 必读

- `docs/current.md` — 当前焦点与阻塞
- `docs/reference/hardware.md` — Ampere / NVFP4 / 显存预算
- `docs/reference/docker.md` — 镜像分层，不要轻易全量编译
- `docs/decision/0001-ampere-overlay-image.md` — 为什么 overlay 而不是从源码编 CUDA

## 命令

```bash
./scripts/build-image.sh              # overlay，几分钟
./scripts/serve-nvfp4.sh              # 第一优先：target-only
./scripts/serve-nvfp4-dspark.sh       # 第二优先：DSpark
./scripts/smoke-chat.sh
./scripts/build-from-source.sh        # 仅当 overlay 不够；数小时
```

## 约束

- 权重在 `models/`，永不提交。
- 不要默认 `--kv-cache-dtype fp8`：3080 没有原生 FP8。
- 不要默认 `--linear-backend flashinfer_cutedsl`：那是 Blackwell NVFP4 路径。
- 官方 recipe 的 NVFP4 + 262k 在 **target-only** 下 KV 测到 297k（gpu-mem 0.93），可以试；**DSpark 默认 BF16 + 131072 + gpu-mem 0.93 + max-num-seqs 10**。不要 0.95+seqs=10（warmup OOM）。不要 0.97。不要靠减小 `NUM_SPEC` 提速。
- `DSparkDraftModel` 在上游 registry 指向 DeepSeek-V4；启动前必须跑 `scripts/prepare-dspark-for-vllm.py`。
- 未经用户明确要求不要 force push、不要改 git config、不要提交 secrets。
- 用户未要求时不要创建 GitHub Actions。

## 验证

构建 overlay 后至少确认：`docker image inspect vllm-qwen38-27b:ampere`。权重下完前不要把 serve 失败写成「镜像不可用」。
