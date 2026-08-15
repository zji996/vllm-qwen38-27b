# 模型

本地目录，权重 gitignore。来源见 `models/README.md`。

## Qwen3.8-27B-NVFP4

- HF：`unsloth/Qwen3.8-27B-NVFP4`
- 架构：`Qwen3_5ForConditionalGeneration` / `model_type=qwen3_5`
- 27B dense，64 层，每 4 层一次 full attention（16 层 GQA + 48 层 Gated DeltaNet）
- 原生多模态；本仓第一阶段 `--language-model-only`
- 量化：`compressed-tensors` mixed-precision
  - MLP：`nvfp4-pack-quantized`，group 16
  - 注意力投影与末几层 MLP：FP8 `float-quantized`
  - 视觉块在 `ignore` 里，保持 BF16
- 权重约 23.4 GB；官方 recipe 里另一份 `Inferact/Qwen3.8-27B-NVFP4` 标 32 GB W4A4
- vLLM 注册名：`Qwen3_5ForConditionalGeneration` → `qwen3_5`
- 需要 vLLM ≥ 0.17 与 transformers ≥ 5.8（本仓 nightly + overlay）

采样（model card）：thinking `t=1.0, top_p=0.95, top_k=20`；instruct `t=0.7, top_p=0.8, presence_penalty=1.5`。

## Qwen3.8-27B-DSpark

- HF：`RadixArk/Qwen3.8-27B-DSpark`
- 训练目标是 **`Qwen/Qwen3.8-27B-FP8`**，不是 NVFP4。同一 hidden size 时仍可对 NVFP4 target 做无损投机，但接受率可能变。
- 发布栈：SpecForge 训练 + **SGLang** 服务
- `architectures`: `["DSparkDraftModel"]`
- `auto_map.AutoModel`: `dspark.DSparkDraftModel`（依赖 `specforge`）
- 草稿约 1.36B BF16，5 层 GQA，block size 7
- aux layers：4, 16, 28, 40, 52

### 与 vLLM 的缝

上游 `third_party/vllm/vllm/model_executor/models/registry.py`：

```
"DSparkDraftModel" -> DSparkDeepseekV4ForCausalLM
"Qwen3DSparkModel" -> Qwen3DSparkForCausalLM
```

直接 `--speculative-config method=dspark` 会走错 loader。`scripts/prepare-dspark-for-vllm.py` 把本地 `config.json` 改成 `Qwen3DSparkModel`。BF16 权重已验证可加载并加速。

## Qwen3.8-27B-DSpark-NVFP4A16

本仓 `scripts/quantize_dspark_nvfp4a16.py` 对 MLP+`fc` 做 weight-only RTN（大 `weight_global_scale`，gate/up 共享）。attn / Markov / confidence 保持 BF16。计数速度与 BF16 草稿持平，KV 多约 16k。见 ADR 0002。
