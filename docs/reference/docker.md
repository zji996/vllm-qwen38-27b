# Docker

## 基座

本机无预置镜像。选用：

| Tag | 用途 |
| --- | --- |
| `vllm/vllm-openai:nightly` | 默认基座。已拉取 digest `sha256:c96082d33456ceeae7ec0d4faf2b5e47fb806a103decf94f9fbc9b35fd7d6b25`（2026-08-14，commit `ac7509e2`，CUDA 13） |
| `vllm/vllm-openai:cu129-nightly` | 若 CUDA 13 运行时出问题再退 |
| `vllm/vllm-openai:qwen38` | 2026-08-12 day-0，比 nightly 旧，不默认 |

驱动 580.105.08 报 CUDA 13.0，与 nightly 匹配。官方 Dockerfile 默认 `CUDA_VERSION=13.0.3`，`TORCH_CUDA_ARCH_LIST` 含 **8.6**。

对照源码：`third_party/vllm/docker/Dockerfile`。OpenAI 目标的 ENTRYPOINT 是 `["vllm", "serve"]`。

## 本仓镜像

`docker/Dockerfile`

| target | 做什么 |
| --- | --- |
| `runtime`（默认） | `FROM nightly` + `transformers>=5.8` + Ampere 环境变量 |
| `overlay` | 再 `uv pip install --no-deps` `third_party/vllm` 的 Python 树 |

```bash
./scripts/build-image.sh
TARGET=overlay ./scripts/build-image.sh
./scripts/build-from-source.sh   # 数小时，TORCH_CUDA_ARCH_LIST=8.6
```

产出 tag：`vllm-qwen38-27b:ampere`。

## 运行约束

- `--gpus all --ipc=host --shm-size=16g`
- 权重 bind-mount `./models:/models:ro`
- 不要把 20GB+ 权重 COPY 进镜像
