#!/usr/bin/env bash
# Full CUDA rebuild of vLLM from third_party/vllm. Hours on this host.
# Prefer scripts/build-image.sh unless we need Ampere-only kernels or a
# source patch that cannot overlay Python-only.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT}/third_party/vllm"
IMAGE="${IMAGE:-vllm-qwen38-27b:ampere-from-source}"

if [[ ! -f "${SRC}/docker/Dockerfile" ]]; then
  echo "missing ${SRC}/docker/Dockerfile; run: git submodule update --init --depth 1" >&2
  exit 1
fi

# 2x RTX 3080 is SM 8.6. Building only this arch shrinks compile time vs the
# official TORCH_CUDA_ARCH_LIST (7.5 8.0 8.6 8.9 9.0 10.0 11.0 12.0).
export DOCKER_BUILDKIT=1
docker build \
  -f "${SRC}/docker/Dockerfile" \
  --target vllm-openai \
  --build-arg torch_cuda_arch_list="8.6" \
  --build-arg max_jobs="${MAX_JOBS:-4}" \
  --build-arg nvcc_threads="${NVCC_THREADS:-2}" \
  -t "$IMAGE" \
  "$SRC"
