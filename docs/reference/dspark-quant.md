# DSpark 草稿量化

默认服务用 BF16 草稿，不要走这条。NVFP4A16 只作为可选。

```bash
./scripts/quantize-dspark-nvfp4a16.sh
DRAFT_NAME=Qwen3.8-27B-DSpark-NVFP4A16 ./scripts/serve-nvfp4-dspark.sh
```

只压 MLP 和 `fc`。attn 保持 BF16。`weight_global_scale` 必须存大 scale `(6*448)/amax`（跟 Unsloth 一样）；vLLM 加载时会再 `1/scale`。同一层 gate/up 共用一个 scale。

修正后（gpu-mem 0.93）：每卡 11.51 GiB，32k 池子 115k，计数与 BF16 打平、代码更慢。见 ADR 0002。
