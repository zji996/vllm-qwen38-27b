# Ampere overlay for Qwen3.8-27B NVFP4 (+ DSpark) on 2x RTX 3080 20GB.
#
# Default target `runtime` starts from the official OpenAI image. CUDA kernels
# in that image already include sm_86 (see third_party/vllm/docker/Dockerfile
# TORCH_CUDA_ARCH_LIST default: 7.5 8.0 8.6 ...).
#
# Target `overlay` reinstalls the Python package from third_party/vllm without
# rebuilding CUDA extensions. Use it when local Python patches must ship.
#
# Full CUDA rebuild (hours): scripts/build-from-source.sh

ARG VLLM_BASE_IMAGE=vllm/vllm-openai:nightly

FROM ${VLLM_BASE_IMAGE} AS runtime

USER root

ENV CUDA_DEVICE_ORDER=PCI_BUS_ID \
    NCCL_IB_DISABLE=1 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    UV_HTTP_TIMEOUT=500 \
    VLLM_USAGE_SOURCE=qwen38-3080-overlay

COPY docker/requirements-extra.txt /tmp/requirements-extra.txt
# Do not pass -U: nightly already meets the floor; -U was bumping numpy as a side effect.
RUN uv pip install --system -r /tmp/requirements-extra.txt \
    && python3 -c "import vllm, transformers; print('vllm', getattr(vllm, '__version__', '?')); print('transformers', transformers.__version__)"

WORKDIR /workspace
# Base image already sets ENTRYPOINT ["vllm", "serve"]

FROM runtime AS overlay

COPY third_party/vllm /opt/vllm-src
WORKDIR /opt/vllm-src
# Python-only overlay: compiled CUDA extensions stay from the base image.
# --no-deps avoids pulling a second PyTorch/CUDA stack.
RUN uv pip install --system --no-build-isolation --no-deps .
WORKDIR /workspace
