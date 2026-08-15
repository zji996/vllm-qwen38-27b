# 2026-08-15 DSpark BF16 gpu-mem 上限（要稳定不炸）

## 设置
- `./scripts/serve-nvfp4-dspark.sh`，BF16 草稿，`--max-model-len 131072`，`NUM_SPEC=7`，`max-num-seqs=2`
- 每档：pong、计数 1–30、约 4k prefill、256 token decode
- 日志：`logs/dspark-gpumem-20260815/`

## 结果

| gpu-mem | KV GiB | 池子 tokens | 131k 并发 | nvidia-smi | 生成 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0.93 | 4.52 | 153,808 | 1.17× | — | 稳定（此前测过） |
| **0.95** | 4.91 | 166,916 | 1.27× | 19695 / 20480 | 四项都过，无 OOM |
| 0.96 | 5.11 | 173,603 | 1.32× | 19895 / 20480 | 短测过了，但 KV 已超过 vLLM「fully utilize」4.95 GiB |
| 0.97 | 5.30 | 180,290 | 1.38× | 19931 / 20480 | 请求仍 200，分配器 OOM（GPU0 只剩 6 MiB 时要 82 MiB） |

vLLM 在 0.93 就提示 fully utilize KV = **4.95 GiB**。0.95 的 4.91 贴着这条线；0.96 起超标。

## 结论

DSpark BF16 默认提到 **0.95**。不要 0.97。0.96 不作为默认。

后续：生产默认改成 **max-num-seqs=10** 后，0.95 在 CUDA graph warmup 时 OOM，默认改回 **0.93**。见 `docs/experiments/20260815-final-dspark-matrix.md`。
