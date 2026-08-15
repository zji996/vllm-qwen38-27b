# 2026-08-15 gpu-mem 0.93 + MTP / DSpark 短 matrix（去冷启动）

## 设置
- 镜像：`vllm-qwen38-27b:ampere` / vLLM `0.27.2rc1.dev77+gac7509e2b`
- target：`models/Qwen3.8-27B-NVFP4`
- 草稿：`models/Qwen3.8-27B-DSpark`（BF16）与 `models/Qwen3.8-27B-DSpark-NVFP4A16`（本仓 RTN，MLP+`fc`）
- 公平速度：`--max-model-len 32768`，`--gpu-memory-utilization 0.93`，`--max-num-seqs 2`，TP=2，Marlin
- 客户端：`scripts/bench_spec_matrix.py`，先 2 次 warmup 丢掉 Triton JIT / 首包
- 任务：计数 1–30（greedy，关 thinking）；Python generator（t=0.7）
- 日志：`logs/spec-matrix-20260815-060344/`（第一轮 tok/s）、`logs/spec-matrix-20260815-rerun/`（错误 scale 的 NVFP4A16）、`logs/spec-matrix-20260815-nvfp4a16-fix/`（修正 scale 后）

## 命令
```bash
GPU_MEM=0.93 MAX_MODEL_LEN=32768 MAX_NUM_SEQS=2 ./scripts/run_spec_matrix.sh
MATRIX_CONFIGS=mtp,dspark-bf16,dspark-nvfp4a16 ./scripts/run_spec_matrix.sh
./scripts/quantize-dspark-nvfp4a16.sh
```

## 结果

第一轮 warmup 更热，tok/s 用来做速度对比。第二轮修了 Prometheus `_total` 后缀，accept 数字以第二轮为准。NVFP4A16 只跑了第二轮。

| 配置 | 每卡权重 GiB | KV tokens | 计数 tok/s | 代码 tok/s | 计数 mean accept | 代码 mean accept |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 10.49 | 297,339 | **51.05** | **46.83** | — | — |
| mtp (`num_speculative_tokens=3`) | 10.91 | 200,248 | **110.91** | **76.86** | 3.00（100%） | 3.00（100%） |
| dspark-bf16 | 12.23 | 99,166 | **139.8** | **92.93** | 6.74（96%） | 3.92（56%） |
| dspark-nvfp4a16（错误 scale） | 11.51 | 114,860 | 23.48 | 25.09 | 0 | 0 |
| dspark-nvfp4a16（修正 scale） | 11.51 | 114,860 | **140.48** | **81.06** | 7.00（100%） | 3.50（50%） |

相对 none（计数）：MTP **2.17×**，DSpark BF16 **2.74×**，修正后的 NVFP4A16 **2.75×**。错误 scale 那版是 0.46×，已作废。

gpu-mem 0.93 相对此前 0.90 的 none KV：264,571 → **297,339**（+12%），262k 更有余量。

MTP 第二轮更冷（计数只有 65 tok/s），accept 仍是 3/3。DSpark 第二轮 98 tok/s，accept 与此前 131k 冒烟一致（计数很好猜，代码 ~56%）。

### NVFP4A16 RTN（错误 scale，已作废）

- 脚本：`scripts/quantize_dspark_nvfp4a16.py`。盘上 2.72 GB → **只压 MLP+`fc`** 1.40 GB。attn 必须 BF16：DFlash 要切稠密 `qkv_proj.weight`。
- 第一版按 vLLM 注释把 `weight_global_scale` 存成 divisor（`amax/(6*448)` ≈ 4.7e-5）。Unsloth 存的是大 scale（layer0 gate/up **6400**）。vLLM 加载一律 `1.0/scale`，divisor 再取倒数 → 权重大约 10⁸。
- 结果：accept **0%**，计数 23 tok/s。草稿 GEMM 仍在跑（`Drafted throughput: 111 tok/s`），慢是投机税。E2M1 打包本身没问题（回环 relMAE ≈ 0.09）。

### 修正后重跑（20260815-nvfp4a16-fix）

量化器改为存大 scale `(6*448)/amax`，同一层 gate/up 共用 max-amax。3 次 warmup。Marlin W4A16 极限约 3.8×。

- 计数 **140.48 tok/s**，accept **7.00 / 100%**（与 BF16 持平）
- 代码 **81.06 tok/s**，accept **3.50 / 50%**（BF16 第一轮 92.93 / 3.92，略降）
- 每卡 **11.51 GiB**（BF16 草稿 12.23，省 0.72），KV **114,860**（BF16 99,166）
- 日志不再出现 fused global-scale mismatch warning

## 结论（取舍）

1. **默认要速度：DSpark BF16 或修正后的 NVFP4A16。** 计数都约 2.7×。NVFP4A16 每卡省 0.72 GiB、KV 99k→115k；代码 accept 略低。
2. **要更长上下文、又不想挂独立草稿：MTP。** 多 0.42 GiB/卡，KV 约 200k，短任务 2.2×。
3. **要原生 262k：关掉投机**，gpu-mem 0.93 的 none KV 297k 够用。
4. attn 仍受 DFlash fused KV 限制，只能压 MLP+`fc`。错误的 divisor `weight_global_scale` 不要再用。
