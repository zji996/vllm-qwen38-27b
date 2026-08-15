# 当前状态

更新日期：2026-08-15

## 焦点

2× RTX 3080 20GB 上，vLLM nightly overlay 跑 Qwen3.8-27B NVFP4。默认 **DSpark BF16**、**gpu-mem 0.93**、`--max-model-len 131072`、**max-num-seqs 10**（翻译短请求并发）。

## 已落地（implementation）

- overlay 镜像 `vllm-qwen38-27b:ampere`
- `scripts/serve-nvfp4.sh` / `serve-nvfp4-mtp.sh` / `serve-nvfp4-dspark.sh`
- `scripts/quantize_dspark_nvfp4a16.py`：MLP+`fc` NVFP4A16，存大 `weight_global_scale`，gate/up 共享
- `scripts/run_final_matrix.sh` / `bench_final_matrix.py`（生产默认 matrix）
- `scripts/run_spec_matrix.sh` / `bench_spec_matrix.py`（旧公平对比，不再当默认入口）

## 已验证（validation）

终态 matrix：`docs/experiments/20260815-final-dspark-matrix.md`。

| 项 | 值 |
| --- | --- |
| KV 池 | 152,739 tokens（4.49 GiB，131k 并发 1.17×） |
| 翻译 1k TTFT / decode | 719 ms / 44 tok/s |
| 代码 1k TTFT / decode | 748 ms / 154 tok/s |
| 翻译 1k conc=10 batch | 24 tok/s |
| 代码 1k conc=10 batch | 119 tok/s |
| 96k TTFT | 翻译 71 s / 代码 73 s |

`gpu-mem 0.95` + `seqs=10` 在 CUDA graph warmup 时 OOM。seqs=2 时 0.95 仍可用（池 166,916），见 `docs/experiments/20260815-dspark-gpumem.md`。不要 0.97。

投机公平对比（gpu-mem 0.93、32k、seqs=2）：`docs/experiments/20260815-spec-matrix-gpu093.md`。

| 配置 | 每卡权重 | KV tokens（32k） | 计数 tok/s | 相对 none |
| --- | ---: | ---: | ---: | ---: |
| none | 10.49 GiB | 297,339 | 51 | 1.0× |
| MTP | 10.91 GiB | 200,248 | 111 | 2.2× |
| DSpark BF16 | 12.23 GiB | 99,166 | 140 | 2.7× |
| DSpark NVFP4A16 RTN | 11.51 GiB | 114,860 | 140 | 2.7× |

默认草稿用 **BF16**：代码更快。NVFP4A16 只省 0.72 GiB，不作为默认。

## 上下文结论

| 目标 | 命令 |
| --- | --- |
| 加速（默认，BF16 草稿 131k） | `./scripts/serve-nvfp4-dspark.sh` |
| 较长上下文 + 加速 | `./scripts/serve-nvfp4-mtp.sh` |
| 尽量长上下文 | `MAX_MODEL_LEN=262144 ./scripts/serve-nvfp4.sh` |

## 验证入口

```bash
./scripts/serve-nvfp4-dspark.sh
./scripts/smoke-chat.sh
```

停：`docker stop vllm-qwen38-nvfp4-dspark`

终态 matrix：`./scripts/run_final_matrix.sh`

## 下一步

1. 短翻译 DSpark mean accept ~0.77，是否值得为翻译路径关投机或换任务分布
2. 可选：`enable_adaptive_verification=true`（N=7 为上限，按 confidence 少校验）
3. DFlash fused KV 若支持 packed 权重，再考虑压 attn

## 非目标（本轮）

- 视觉输入
- 从源码编译 CUDA
- DSpark + 262k 同开
