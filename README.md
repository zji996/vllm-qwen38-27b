# vllm-qwen38-27b

在 **2× RTX 3080 20GB（Ampere SM 8.6）** 上用 vLLM 跑 Qwen3.8-27B NVFP4，并实验 DSpark 投机解码。

## 快速路径

```bash
# 1. 官方 nightly 已由本仓脚本拉取；打 Ampere overlay
./scripts/build-image.sh

# 2. 权重下完后启动（text-only，无投机解码）
./scripts/serve-nvfp4.sh

# 3. 另开终端
./scripts/smoke-chat.sh
```

DSpark（实验，checkpoint 格式可能对不上）：

```bash
./scripts/serve-nvfp4-dspark.sh
```

## 仓库布局

| 路径 | 内容 |
| --- | --- |
| `docs/` | 实验记录、当前状态、硬件/镜像/模型事实 |
| `docker/Dockerfile` | 基于 `vllm/vllm-openai:nightly` 的 overlay |
| `third_party/vllm` | vLLM 源码 submodule，对照官方 Dockerfile |
| `models/` | 本地权重（gitignore） |
| `scripts/` | 构建、启动、冒烟 |

先读 [`docs/current.md`](docs/current.md)。硬件与量化约束见 [`docs/reference/hardware.md`](docs/reference/hardware.md)。
