# third_party

本仓不自行维护 vLLM 内核，只把上游源码作为 **git submodule** 钉在这里，用来：

1. 对照官方 `docker/Dockerfile` 写本仓 overlay 镜像
2. 查 Qwen3.8 / NVFP4 / DSpark 的实际实现
3. 需要时用 Python overlay 或全量 CUDA 重建打补丁

| 路径 | 上游 | 用途 |
| --- | --- | --- |
| `vllm/` | [vllm-project/vllm](https://github.com/vllm-project/vllm) | 推理引擎源码（shallow clone） |

更新：

```bash
git submodule update --init --depth 1
git -C third_party/vllm fetch --depth 1 origin main
git -C third_party/vllm checkout origin/main
```
