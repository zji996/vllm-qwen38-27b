# Roadmap

## 阶段 1 — 能加载（进行中）

- [x] 文档目录与实验入口
- [x] vLLM submodule
- [x] overlay 镜像构建成功（`vllm-qwen38-27b:ampere`，vLLM 0.27.2rc1.dev77+gac7509e2b）
- [x] NVFP4 target 在 2×3080 上完成加载（Marlin，KV 264k tokens）
- [x] `smoke-chat.sh` 返回非空文本
- [x] DSpark 加载并加速（accept length 5.2–7.1，131k context）

## 阶段 2 — 投机解码

- [x] 确认 RadixArk DSpark 权重能被 `Qwen3DSparkForCausalLM` 吃进去
- [x] 测 acceptance length（计数 6.7 / 代码 3.9；对照 model card ~3.4）
- [x] 短 matrix：none / MTP / DSpark BF16 / DSpark NVFP4A16 RTN（修正 scale 后计数 2.7×）

## 阶段 3 — 显存与吞吐

- [x] 默认 gpu-mem 0.93；none 297k KV，DSpark 配 131k，MTP 约 200k
- [x] 对比：无投机 / MTP / DSpark（去冷启动）
- [ ] 需要时再考虑 Ampere-only 源码重建
- [ ] 长文本 / thinking 的 accept length

## 明确不做（除非另开目标）

- DSpark 与 262k 同开（KV 不够）
- 把本机当成 Blackwell NVFP4 配方机（`flashinfer_cutedsl`、CLI `--kv-cache-dtype fp8`）
- 把 attn 打成 packed NVFP4（DFlash fused KV 还要稠密 `qkv_proj.weight`）
