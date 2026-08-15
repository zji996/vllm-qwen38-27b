#!/usr/bin/env bash
# Build the Ampere overlay image from official vLLM nightly.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

IMAGE="${IMAGE:-vllm-qwen38-27b:ampere}"
TARGET="${TARGET:-runtime}"
VLLM_BASE_IMAGE="${VLLM_BASE_IMAGE:-vllm/vllm-openai:nightly}"

echo "Building ${IMAGE} (target=${TARGET}, base=${VLLM_BASE_IMAGE})"
docker build \
  -f docker/Dockerfile \
  --target "$TARGET" \
  --build-arg "VLLM_BASE_IMAGE=${VLLM_BASE_IMAGE}" \
  -t "$IMAGE" \
  .
