# 硬件

探测日期：2026-08-15。命令：`nvidia-smi --query-gpu=index,name,memory.total,compute_cap,driver_version --format=csv`

| GPU | 型号 | 显存 | Compute | 驱动 | 主机 CUDA |
| --- | --- | ---: | --- | --- | --- |
| 0 | GeForce RTX 3080 | 20480 MiB | 8.6 | 580.105.08 | 13.0 |
| 1 | GeForce RTX 3080 | 20480 MiB | 8.6 | 580.105.08 | 13.0 |

主机：Ubuntu 24.04，12 CPU，62.7 GiB RAM，Docker 28.5.1，runtime `nvidia`。

## 架构含义

RTX 3080 是 Ampere，不是 Ada（8.9）或 Blackwell（10.0）。

| 能力 | 本机 | 官方 Qwen3.8 NVFP4 recipe 假设 |
| --- | --- | --- |
| 原生 NVFP4 GEMM | 无 | Blackwell，`--linear-backend flashinfer_cutedsl` |
| 原生 FP8 tensor core | 无（Ada 起） | `--kv-cache-dtype fp8` |
| NVFP4 权重加载 | 有，Marlin W4A16 回退，min SM 7.5 | W4A4 |
| FP8 权重加载 | 有，W8A16 回退（scheme min 8.9 失败后） | W8A8 |
| FA3 / CuTeDSL | 不作为默认 | Hopper/Blackwell |

vLLM 会打类似日志：GPU 无原生 FP4，改用 Marlin weight-only。这是预期，不是故障。

## 显存预算（2×20GB = 40GB）

`unsloth/Qwen3.8-27B-NVFP4` 的 `model.safetensors.index.json`：`total_size` ≈ **23.42 GB**（混合 NVFP4 MLP + FP8 注意力；视觉块未量化）。

粗算 TP=2：

| 项 | 量级 | 备注 |
| --- | --- | --- |
| Target 权重 | ~11.7 GB / GPU | 均匀切 |
| 视觉塔 | ~1 GB 合计 | `--language-model-only` 可去掉 |
| DSpark BF16 草稿 | ~4.8 GB | 常见是每卡复制，不是 TP |
| CUDA / 框架 | ~1–2 GB / GPU | |
| KV + 线性注意力状态 | 剩余 | 先 4k–8k，不要 262k |

加上 DSpark 后每卡大约只剩 ~4–5 GB 给 KV。实测（2026-08-15，gpu-mem **0.93**，32k 公平对比）：

- NVFP4 only：5.88 GiB KV → **297,339 tokens**
- NVFP4 + MTP：5.72 GiB KV → **200,248 tokens**
- NVFP4 + DSpark BF16：gpu-mem **0.93**、seqs=10、131k 时 KV **4.49 GiB → 152,739** tokens，并发 **1.17×**（seqs=2 @0.95 池 166,916，但 0.95+seqs=10 warmup OOM；0.97 生成 OOM）
- NVFP4 + DSpark NVFP4A16：5.23 GiB KV → 32k cap **114,860**（3.51×）；单请求 0.93 可到 **188k**，0.95 可到 **197k**

Marlin W4A16 有 kernel 加速（极限约 3.8×），不是只省权重。

混合注意力只有 16 层 full attention，再加 checkpoint 自带的 FP8 KV，上下文比「23GB 权重塞 40GB」直觉能开的长得多。
