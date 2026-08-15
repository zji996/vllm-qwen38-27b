# 2026-08-15 DSpark NVFP4A16 上下文上限与 NUM_SPEC

## 设置
- target：`Qwen3.8-27B-NVFP4`；草稿：`Qwen3.8-27B-DSpark-NVFP4A16`
- TP=2 Marlin，`--max-num-seqs 2`，默认 gpu-mem **0.93**
- 日志：`logs/dspark-ctx-spec-20260815/`（KV probe）、`logs/spec-matrix-20260815-spec{3,5}/`（速度）
- 对照：`logs/spec-matrix-20260815-nvfp4a16-fix/`（NUM_SPEC=7 @ 32k）

## 32k 的 114k 不是上下文上限

公平速度 matrix 把 `--max-model-len` 钉在 32768。日志里的 `GPU KV cache size: 114,860 tokens` 是 **3.51 路 × 32k** 的池子，单请求仍被 cap 在 32k。

同一份 5.23 GiB KV，把 max-model-len 拉高，池子会按页重排，单请求上限跟着涨。

## KV probe（NUM_SPEC=7，gpu-mem 0.93，除非注明）

| max-model-len | gpu-mem | NUM_SPEC | KV GiB | 池子 tokens | 并发 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32,768 | 0.93 | 7 | 5.23 | 114,860 | 3.51× |
| 131,072 | 0.93 | 7 | 5.23 | 178,150 | 1.36× |
| 163,840 | 0.93 | 7 | 5.23 | 184,944 | 1.13× |
| 176,128 | 0.93 | 7 | 5.23 | 187,681 | 1.07× |
| 188,416 | 0.93 | 7 | 5.23 | 188,699 | **1.00×** |
| 176,128 | 0.93 | 3 | 5.24 | 199,708 | 1.13× |
| 196,608 | 0.95 | 7 | 5.63 | 203,731 | 1.04× |

峰值激活始终 1.33 GiB。草稿 5 层 full attn 让 hybrid 分组 `group_size=5`，target 16 层 full 要垫 4 层（浪费 25%）、48 层 GDN 垫 2 层（4.17%）。这是结构税，跟 NUM_SPEC 无关。

GDN 走 `mamba_cache_mode=align`，每组常驻 `2 + NUM_SPEC` 页。N=7→3 只多约 **6%** KV。

262k 仍然装不下（池子顶到 ~189k @0.93 / ~204k @0.95）。

## 速度 vs NUM_SPEC（32k cap，3 次 warmup）

DSpark 一次并行 draft N 个 token（checkpoint `block_size=7`），再让 target 一次校验。

| NUM_SPEC | 计数 tok/s | 代码 tok/s | 计数 accept | 代码 accept | 32k 池子 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 78.92 | 63.19 | 3.00（100%） | 2.63（88%） | 145,024 |
| 5 | 109.22 | 78.05 | 4.86（97%） | 3.50（70%） | 126,765 |
| 7 | **140.48** | **81.06** | 7.00（100%） | 3.50（50%） | 114,860 |

计数任务几乎 100% 命中，N 越大每步收下的 token 越多，越快。减 N **不能提速**。代码任务 N=5 与 N=7 的 mean accept 都是 3.5，速度几乎一样（78 vs 81）。

## 结论

1. **上下文能提。** 默认 `MAX_MODEL_LEN=131072` 在 NVFP4A16 + 0.93 下是 1.36×，还能往上。舒适：`176128`（1.07×）；顶格：`188416`（1.00×，不要给第二路请求留余量）。`GPU_MEM=0.95 MAX_MODEL_LEN=196608` 能到 197k。
2. **不要靠减预测条数提速。** 保持 `NUM_SPEC=7`。减到 3 只换约 6% KV，计数从 140 掉到 79 tok/s。
3. 还要更长：关掉 DSpark 用 MTP（KV ~200k）或 none（297k / 262k）。
4. 未测：`enable_adaptive_verification=true`（checkpoint 有 confidence head）。那是「N=7 为上限、低置信度少校验」，不是把 N 钉死变小。

## 命令

```bash
DRAFT_NAME=Qwen3.8-27B-DSpark-NVFP4A16 MAX_MODEL_LEN=176128 \
  ./scripts/serve-nvfp4-dspark.sh

DRAFT_NAME=Qwen3.8-27B-DSpark-NVFP4A16 GPU_MEM=0.95 MAX_MODEL_LEN=196608 \
  ./scripts/serve-nvfp4-dspark.sh
```
