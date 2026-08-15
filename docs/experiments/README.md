# 实验记录

每个实验一个 markdown，文件名 `YYYYMMDD-short-slug.md`。不要把数字只写在聊天里。

模板：

```markdown
# YYYY-MM-DD 标题

## 设置
- 镜像 digest / vLLM commit
- 模型路径与是否 DSpark
- 关键 flags（tp, max-model-len, kv dtype, language-model-only）
- 驱动 / nvidia-smi

## 命令
```bash
# 实际执行的命令
```

## 结果
- 是否加载成功
- nvidia-smi 峰值
- 延迟 / tok/s（若有）
- 日志摘要（OOM、Marlin warning、architecture mismatch）

## 结论
- 下一步改哪一个旋钮
```

数字必须带来源：日志路径、日期、硬件。定性结论可以链到 `docs/current.md`。
