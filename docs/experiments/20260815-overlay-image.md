# 2026-08-15 overlay 镜像构建

## 设置
- 主机：Ubuntu 24.04，驱动 580.105.08，2× RTX 3080 20GB SM 8.6
- 基座：`vllm/vllm-openai:nightly`
  - digest `sha256:c96082d33456ceeae7ec0d4faf2b5e47fb806a103decf94f9fbc9b35fd7d6b25`
  - 对应 commit `ac7509e2`
- 本仓镜像：`vllm-qwen38-27b:ampere` target `runtime`
- submodule：`third_party/vllm` @ `925ea7e`（比 nightly 新的 main HEAD；runtime 层未 overlay 源码）

## 命令
```bash
docker pull vllm/vllm-openai:nightly
./scripts/build-image.sh
docker run --rm --entrypoint python3 vllm-qwen38-27b:ampere -c \
  'import vllm, transformers, torch; print(vllm.__version__, transformers.__version__, torch.__version__, torch.version.cuda)'
```

## 结果
- pull 成功，约 11.5 min
- overlay 构建成功
- 镜像内：vLLM `0.27.2rc1.dev77+gac7509e2b`，transformers `5.15.0`，torch `2.13.0+cu130`
- 未跑 serve：NVFP4 / DSpark 权重当时仍在下载

## 结论
阶段 1 的「镜像可构建」已过。下一步等权重齐套后跑 `./scripts/serve-nvfp4.sh`。
