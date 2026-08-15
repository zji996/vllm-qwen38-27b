# Models

Local Hugging Face snapshots used for Qwen3.8-27B vLLM adaptation.

| Directory | Source | Notes |
| --- | --- | --- |
| `Qwen3.8-27B-NVFP4` | [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) | NVFP4 quantized 27B (~23 GB) |
| `Qwen3.8-27B-DSpark` | [RadixArk/Qwen3.8-27B-DSpark](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark) | DSpark speculative draft head (~4.8 GB) |

Weights are gitignored. Re-download with:

```bash
huggingface-cli download unsloth/Qwen3.8-27B-NVFP4 --local-dir models/Qwen3.8-27B-NVFP4
huggingface-cli download RadixArk/Qwen3.8-27B-DSpark --local-dir models/Qwen3.8-27B-DSpark
```
