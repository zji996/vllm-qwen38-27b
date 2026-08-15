# ADR 0002 — DSpark 草稿可选用 NVFP4A16（MLP+fc RTN）

- 状态：accepted（scale 约定修正后重测）
- 日期：2026-08-15
- 范围：RadixArk Qwen3.8-27B-DSpark 在 2×3080 上的显存与速度
- 证据：`docs/experiments/20260815-spec-matrix-gpu093.md`

## 背景

target 已经是 Unsloth NVFP4。DSpark 草稿是 BF16。问题：能不能把草稿压成 NVFP4 把空间要回来，并且用上 Marlin W4A16 加速。

## 事实

### Marlin W4A16 有加速

Ampere 上 NVFP4 走 Marlin **W4A16 weight-only**。kernel 有加速，极限大约 **3.8×**。前提是 checkpoint 的 `weight_global_scale` 跟 Unsloth 一样存**大 scale** `(6*448)/amax`：vLLM 加载时会再做 `1.0/scale`。第一版 RTN 存了 divisor，权重炸了 10⁸，accept 0%。

### 草稿权重

attn 必须保持 BF16：DFlash fused KV 要切稠密 `qkv_proj.weight`。只压 MLP + `fc`。同一层 gate/up 必须共享 global scale。

### 修正后实测（gpu-mem 0.93，32k）

| 草稿 | 每卡权重 | KV | 计数 tok/s | 代码 tok/s | 代码 accept |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 12.23 GiB | 99,166 | 139.8 | 92.93 | 3.92 |
| NVFP4A16 RTN | 11.51 GiB | 114,860 | 140.48 | 81.06 | 3.50 |

计数与 BF16 持平（2.75× vs none）。每卡省 0.72 GiB，KV +16k。代码略慢、accept 略低。

## 决策

1. **默认用 BF16 草稿**，`--max-model-len 131072`，gpu-mem **0.93**，**max-num-seqs 10**。KV 池 152,739，并发 1.17×。`0.95` + seqs=10 warmup OOM；seqs=2 时 0.95 仍可用（池 166,916）。0.97 生成会 OOM。
2. NVFP4A16 只作为可选（每卡省 0.72 GiB，代码略慢）。不改默认。
3. 262k 仍然要关掉 DSpark。MTP KV 约 200k。不要靠减小 `NUM_SPEC` 换速度。
4. 量化器必须存大 scale，fused gate/up 共用；不要按 vLLM 注释去存 divisor。

## 后果

- `models/Qwen3.8-27B-DSpark-NVFP4A16` 可作为可选草稿
- attn 量化仍等 DFlash fused KV 支持 packed 权重
