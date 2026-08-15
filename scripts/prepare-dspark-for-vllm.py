#!/usr/bin/env python3
"""Rewrite RadixArk DSpark config so vLLM loads Qwen3DSparkForCausalLM.

RadixArk/Qwen3.8-27B-DSpark ships architectures=["DSparkDraftModel"].
Upstream vLLM maps that name to DeepSeek-V4, not the Qwen3 DSpark loader.
This only patches the local gitignored snapshot; it does not change weights.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <dspark-dir>", file=sys.stderr)
        return 2

    root = Path(sys.argv[1])
    config_path = root / "config.json"
    if not config_path.is_file():
        print(f"missing {config_path}", file=sys.stderr)
        return 1

    config = json.loads(config_path.read_text())
    original = list(config.get("architectures") or [])
    if original == ["Qwen3DSparkModel"]:
        print(f"{config_path}: already Qwen3DSparkModel")
        return 0

    backup = root / "config.radixark.json"
    if not backup.exists():
        backup.write_text(config_path.read_text())
        print(f"backed up original config to {backup}")

    config["architectures"] = ["Qwen3DSparkModel"]
    # vLLM Qwen3 DSpark reads these at top level; RadixArk nests some under
    # dflash_config. Keep both so either loader path can see them.
    dflash = dict(config.get("dflash_config") or {})
    if "target_layer_ids" in dflash and "target_layer_ids" not in config:
        config["target_layer_ids"] = dflash["target_layer_ids"]
    if "mask_token_id" in dflash and "mask_token_id" not in config:
        config["mask_token_id"] = dflash["mask_token_id"]
    # Qwen3.8 target uses partial RoPE 0.25; this draft is a Qwen3 GQA head
    # trained with full-dim RoPE. Leaving the family default (or inheriting
    # 0.25) collapses acceptance. Pin 1.0 like other public Qwen DSpark heads.
    config["partial_rotary_factor"] = 1.0

    config_path.write_text(json.dumps(config, indent=2) + "\n")
    print(f"{config_path}: {original} -> {config['architectures']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
