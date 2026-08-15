# ADR 0001 — 用官方 nightly overlay，而不是默认从源码编 CUDA

- 状态：accepted
- 日期：2026-08-15
- 范围：本仓 Docker 交付方式

## 背景

Qwen3.8 是 2026-08 的新架构（`Qwen3_5ForConditionalGeneration` + hybrid Gated DeltaNet）。需要新于 0.17 的 vLLM。本机是 2× RTX 3080 20GB，SM 8.6，没有 Blackwell 原生 FP4。

用户要求：拉最新 vLLM 镜像，把源码放进 `third_party/`，对照官方 Dockerfile 打本仓镜像。

## 决策

默认镜像是 **官方 `vllm/vllm-openai:nightly` 的薄 overlay**：

- 基座已包含 sm_86 kernel（`third_party/vllm/docker/Dockerfile` 的 `TORCH_CUDA_ARCH_LIST` 含 8.6）
- 只加 Qwen3.8 需要的 Python 依赖（`transformers>=5.8.0`）
- 可选 target `overlay`：把 `third_party/vllm` 的 Python 包叠上去，不重编 CUDA
- 全量 CUDA 重建留作 `scripts/build-from-source.sh`，`TORCH_CUDA_ARCH_LIST=8.6`

## 为什么

- 官方 OpenAI 镜像 ENTRYPOINT 已是 `vllm serve`；从源码编 CUDA 在 12 核主机上要数小时，且容易把 PyTorch/NCCL 编偏
- Ampere 上的 NVFP4 路径是 Marlin 软件回退，不依赖 Blackwell CUTLASS/CuTeDSL。缺的是「够新的 Python 模型代码」，不是「为本卡重编 kernel」
- submodule 让对照与打补丁有源码，但不把 117MB+ 的测试/文档打进默认 runtime 层

## 后果

- 镜像 tag 跟 nightly 漂移；构建时应记录 `docker inspect` 的 `VLLM_BUILD_COMMIT` / digest
- Python overlay 与基座 CUDA 扩展 ABI 不一致时会运行失败，那时才走 from-source
- 不解决 DSpark checkpoint 格式问题；那是模型配置/权重映射，不是镜像编译问题

## 替代方案（未采用）

| 方案 | 未采用原因 |
| --- | --- |
| 始终 `docker build -f third_party/vllm/docker/Dockerfile` | 成本高，当前没有必须重编的 kernel |
| 直接 `docker run vllm/vllm-openai:nightly` 不打 overlay | 无法钉 transformers、无法固化本机 Ampere 环境变量与脚本 |
| 专用 `vllm/vllm-openai:qwen38` tag | 2026-08-12 的 day-0 镜像，比 08-14 nightly 旧 |
