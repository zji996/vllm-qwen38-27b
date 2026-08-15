# 2026-08-15 终态 DSpark serving matrix

## 设置

- 镜像 `vllm-qwen38-27b:ampere`，vLLM `0.27.2rc1.dev77+gac7509e2b`
- Target：`models/Qwen3.8-27B-NVFP4`；草稿：BF16 `Qwen3.8-27B-DSpark`
- TP=2，Marlin，`--max-model-len 131072`，`NUM_SPEC=7`，`--max-num-seqs 10`
- 任务：英→中翻译、写 `merge_intervals`。`temperature=0`，关 thinking，`max_tokens=256`
- 流式：`stream=true`，`stream_options.include_usage=true`
  - TTFT = 发请求 → 第一个非空 `delta.content`
  - prefill tok/s = `prompt_tokens / TTFT`
  - decode tok/s = `(completion_tokens - 1) / (t_last - t_first)`
  - 并发 `batch_output_tok_s` = 各路 `completion_tokens` 之和 / batch 墙钟
- 硬件：2× RTX 3080 20GB

## 命令

```bash
./scripts/run_final_matrix.sh
```

成功跑：`logs/final-matrix-20260815-114959/`（gpu-mem **0.93**）。
0.95 启动失败：`logs/final-matrix-20260815-114537/serve.log`。

## 启动：0.95 炸，0.93 稳

| gpu-mem | seqs | KV GiB | 池子 tokens | 131k 并发 | 结果 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0.95 | 10 | 4.88 | 166,113 | 1.27× | warmup `compile_or_warm_up_model` CUDA OOM（GPU0 剩 10 MiB，要 30 MiB） |
| **0.93** | **10** | **4.49** | **152,739** | **1.17×** | 16 格全部完成 |

seqs=2 时 0.95 能起（池 166,916）。seqs=10 多占 CUDA graph，KV 仍按 0.95 预留，warmup 时没有余量。生产默认改为 **0.93 + seqs=10**。KV 相对 seqs=2 @0.93（153,808）几乎没掉。

## 长度轴（conc=1）

| 任务 | prefill | TTFT | prefill tok/s | decode tok/s | 完成 tokens | mean accept |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 翻译 | 1,018 | 719 ms | 1,417 | 44.0 | 22 | 0.77 |
| 翻译 | 8,190 | 5.59 s | 1,465 | 41.6 | 22 | 0.77 |
| 翻译 | 32,764 | 21.8 s | 1,503 | 39.9 | 22 | 0.77 |
| 翻译 | 98,302 | 71.2 s | 1,380 | 37.7 | 21 | 0.69 |
| 代码 | 1,017 | 748 ms | 1,360 | 153.8 | 113 | 5.22 |
| 代码 | 8,191 | 5.91 s | 1,386 | 146.1 | 113 | 4.89 |
| 代码 | 32,756 | 22.4 s | 1,465 | 147.2 | 133 | 4.58 |
| 代码 | 98,291 | 72.7 s | 1,351 | 159.7 | 113 | 5.59 |

prefill 吞吐约 **1.35–1.50k tok/s**，随长度几乎平坦（投机把 `max_num_scheduled_tokens` 钉在 2048，长上下文分块）。TTFT 近似线性：1k ≈ 0.72 s，96k ≈ 71 s。

翻译 decode 慢是因为草稿几乎不中（mean accept 0.7、draft accept ≈11%），完成只有约 22 token。代码草稿中（mean accept ≈5.2），decode **147–160 tok/s**，随 prefill 不掉。

## 并发轴（全部 ~1k prefill）

conc=1 用单请求 e2e tok/s（`completion / 墙钟`）当基线。

| 任务 | conc | batch tok/s | 墙钟 s | 均 TTFT | 均 decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 翻译 | 1 | 18.4 | 1.20 | 719 ms | 44.0 |
| 翻译 | 2 | 21.3 | 2.02 | 1.47 s | 39.0 |
| 翻译 | 4 | 23.5 | 3.65 | 2.04 s | 16.5 |
| 翻译 | 8 | 24.0 | 7.24 | 3.63 s | 11.7 |
| 翻译 | 10 | 24.3 | 8.98 | 4.50 s | 9.8 |
| 代码 | 1 | 76.6 | 1.48 | 748 ms | 153.8 |
| 代码 | 2 | 100.6 | 2.25 | 1.47 s | 144.4 |
| 代码 | 4 | 114.5 | 3.95 | 2.05 s | 70.0 |
| 代码 | 8 | 115.3 | 7.84 | 3.73 s | 53.3 |
| 代码 | 10 | 118.7 | 9.52 | 4.65 s | 47.7 |

短翻译 batch 吞吐几乎不随并发涨（每路只有 ~22 token，草稿拒稿）。代码 4 路已接近打满（~115 tok/s），10 路 119 tok/s，单路 decode 被摊薄。10×(1k+256) 远小于 153k 池，无 OOM。

## 结论

1. 生产默认：**NVFP4 + BF16 DSpark，gpu-mem 0.93，131k，seqs=10，N=7**。不要 0.95+seqs=10。
2. 翻译服务主路径（大量 1k 短请求）：TTFT ~0.72 s；开到 10 路 batch 仍只有 ~24 tok/s，瓶颈是短输出 + DSpark 拒稿，不是 KV。
3. 代码类中等输出：单路 decode ~154 tok/s；4–10 路共享约 115–119 tok/s。
4. 96k 单请求能出字，TTFT ~72 s。不要用 DSpark 打满 131k 并发长请求。
