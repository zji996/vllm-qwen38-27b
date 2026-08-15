#!/usr/bin/env python3
"""NVFP4A16 weight-only RTN for RadixArk Qwen3.8-27B-DSpark.

Packs MLP and fc the way Unsloth NVFP4 + vLLM Marlin actually load:

  weight_packed        uint8 [out, in//2]     two E2M1 nibbles / byte, low first
  weight_scale         float8_e4m3fn [out, in//16]   group-16
  weight_global_scale  float32 [1]            LARGE scale (6*448)/amax

vLLM CompressedTensorsW4A4Fp4 always does `1.0 / weight_global_scale` at load.
Unsloth checkpoints store the large scale (e.g. 6400), not the divisor. Storing
1/scale here made Marlin inflate weights by ~1e8 and dropped DSpark accept to 0.

Attention stays BF16: DFlash fused KV slices dense qkv_proj.weight.
gate_proj / up_proj of the same layer share one global scale so fused
gate_up_proj does not hit the mismatched-scale warning.

Run inside the vLLM image (host has no torch)::

    ./scripts/quantize-dspark-nvfp4a16.sh
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)  # 448
GROUP = 16
E2M1_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

QUANT_WEIGHT_RE = re.compile(
    r"(?:^|\.)(?:"
    r"layers\.\d+\.mlp\.(?:gate|up|down)_proj"
    r"|fc"
    r")\.weight$"
)
GATE_UP_RE = re.compile(r"^(?P<prefix>.*layers\.\d+\.mlp\.)(?P<which>gate|up)_proj\.weight$")

COPY_FILES = ("dspark.py", "dflash.py", "README.md", ".gitattributes")

QUANTIZATION_CONFIG = {
    "config_groups": {
        "group_0": {
            "format": "nvfp4-pack-quantized",
            "input_activations": None,
            "output_activations": None,
            "targets": [
                r"re:.*mlp\.(gate_up_proj|gate_proj|up_proj|down_proj)$",
                r"re:.*\.fc$",
            ],
            "weights": {
                "actorder": None,
                "block_structure": None,
                "dynamic": False,
                "group_size": GROUP,
                "num_bits": 4,
                "observer": "minmax",
                "observer_kwargs": {},
                "scale_dtype": "torch.float8_e4m3fn",
                "strategy": "tensor_group",
                "symmetric": True,
                "type": "float",
                "zp_dtype": None,
            },
        }
    },
    "format": "nvfp4-pack-quantized",
    "ignore": [
        "lm_head",
        "re:.*embed_tokens.*",
        "re:.*norm.*",
        "re:.*markov.*",
        "re:.*confidence.*",
        "re:.*self_attn.*",
    ],
    "quant_method": "compressed-tensors",
    "quantization_status": "compressed",
}


def cast_to_fp4(x: torch.Tensor) -> torch.Tensor:
    sign = torch.sign(x)
    x = torch.abs(x)
    out = torch.zeros_like(x)
    out[(x > 0.25) & (x < 0.75)] = 0.5
    out[(x >= 0.75) & (x <= 1.25)] = 1.0
    out[(x > 1.25) & (x < 1.75)] = 1.5
    out[(x >= 1.75) & (x <= 2.5)] = 2.0
    out[(x > 2.5) & (x < 3.5)] = 3.0
    out[(x >= 3.5) & (x <= 5.0)] = 4.0
    out[x > 5.0] = 6.0
    return out * sign


def encode_e2m1(values: torch.Tensor) -> torch.Tensor:
    levels = E2M1_LEVELS.to(device=values.device, dtype=values.dtype)
    mag = values.abs()
    idx = (mag.unsqueeze(-1) - levels).abs().argmin(dim=-1).to(torch.uint8)
    sign = (values < 0).to(torch.uint8) << 3
    return sign | idx


def pack_e2m1(values: torch.Tensor) -> torch.Tensor:
    codes = encode_e2m1(values)
    low = codes[:, 0::2]
    high = codes[:, 1::2]
    return low | (high << 4)


def global_scale_from_amax(amax: torch.Tensor) -> torch.Tensor:
    amax = amax.to(torch.float32).clamp_min(1e-8)
    return (FLOAT4_E2M1_MAX * FLOAT8_E4M3_MAX) / amax


def quantize_nvfp4a16(
    weight: torch.Tensor, global_scale: torch.Tensor | None = None
) -> dict[str, torch.Tensor]:
    if weight.ndim != 2:
        raise ValueError(f"expected 2D weight, got {tuple(weight.shape)}")
    out_f, in_f = weight.shape
    if in_f % GROUP != 0:
        raise ValueError(f"in_features {in_f} not divisible by {GROUP}")

    w = weight.detach().to(torch.float32)
    if global_scale is None:
        global_scale = global_scale_from_amax(w.abs().amax())
    else:
        global_scale = global_scale.to(torch.float32).reshape(())

    # Store the LARGE scale. vLLM inverts it at load to match Unsloth NVFP4.
    stored_global = global_scale.detach().to(torch.float32).reshape(1).clone()

    grouped = w.view(out_f, in_f // GROUP, GROUP)
    vec_max = grouped.abs().amax(dim=-1, keepdim=True)
    scale = (global_scale * (vec_max * (1.0 / FLOAT4_E2M1_MAX))).clamp(
        min=-FLOAT8_E4M3_MAX, max=FLOAT8_E4M3_MAX
    )
    scale_fp8 = scale.to(torch.float8_e4m3fn)
    scale_f32 = scale_fp8.to(torch.float32)
    output_scale = torch.where(
        scale_f32 == 0,
        torch.zeros_like(scale_f32),
        global_scale / scale_f32,
    )
    scaled = (grouped * output_scale).clamp(-FLOAT4_E2M1_MAX, FLOAT4_E2M1_MAX)
    packed = pack_e2m1(cast_to_fp4(scaled.reshape(out_f, in_f)))
    return {
        "weight_packed": packed.contiguous(),
        "weight_scale": scale_fp8.squeeze(-1).contiguous(),
        "weight_global_scale": stored_global.contiguous(),
    }


def dequant_vllm_load(
    packed: torch.Tensor, scale_fp8: torch.Tensor, stored_global: torch.Tensor
) -> torch.Tensor:
    """CPU stand-in for vLLM invert-then-multiply: fp4 * sf / stored_large_scale."""
    inverted = 1.0 / stored_global.float().reshape(())
    flat = packed.flatten()
    codes = torch.stack((flat & 0x0F, (flat & 0xF0) >> 4), dim=1).flatten()
    signs = (codes & 0x08).bool()
    mag = (codes & 0x07).long()
    levels = E2M1_LEVELS.to(dtype=torch.float32)
    vals = levels[mag] * torch.where(signs, -1.0, 1.0)
    out_f, packed_k = packed.shape
    inn = packed_k * 2
    fp4 = vals.reshape(out_f, inn)
    sf = scale_fp8.to(torch.float32)
    grouped = fp4.reshape(out_f, inn // GROUP, GROUP)
    return (grouped * (sf * inverted).unsqueeze(-1)).reshape(out_f, inn)


def should_quantize(name: str) -> bool:
    return QUANT_WEIGHT_RE.search(name) is not None


def fused_global_scales(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """One global scale per gate+up pair, from max amax, matching fused gate_up_proj."""
    pairs: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for name, tensor in state.items():
        match = GATE_UP_RE.match(name)
        if match:
            pairs[match.group("prefix")][match.group("which")] = tensor
    shared: dict[str, torch.Tensor] = {}
    for prefix, parts in pairs.items():
        if "gate" not in parts or "up" not in parts:
            continue
        amax = torch.maximum(parts["gate"].abs().amax(), parts["up"].abs().amax())
        gs = global_scale_from_amax(amax)
        shared[f"{prefix}gate_proj.weight"] = gs
        shared[f"{prefix}up_proj.weight"] = gs
    return shared


def load_all(src: Path) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(src), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensors[key] = handle.get_tensor(key)
    return tensors


def quantize_state(
    state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[str]]:
    out: dict[str, torch.Tensor] = {}
    quantized: list[str] = []
    shared_gs = fused_global_scales(state)
    for name, tensor in state.items():
        if not should_quantize(name):
            out[name] = tensor
            continue
        packed = quantize_nvfp4a16(tensor, global_scale=shared_gs.get(name))
        prefix = name[: -len("weight")]
        for suffix, value in packed.items():
            out[prefix + suffix] = value
        dq = dequant_vllm_load(
            packed["weight_packed"],
            packed["weight_scale"],
            packed["weight_global_scale"],
        )
        w = tensor.detach().float()
        rel = (dq - w).abs().mean() / w.abs().mean().clamp_min(1e-8)
        gs = packed["weight_global_scale"].item()
        tag = "shared-gs" if name in shared_gs else "per-tensor"
        quantized.append(
            f"{name} {tuple(tensor.shape)} gs={gs:.1f} ({tag}) "
            f"vllm-load relMAE={rel:.4f} -> packed {tuple(packed['weight_packed'].shape)}"
        )
    return out, quantized


def write_config(src_config: Path, dst: Path) -> None:
    config = json.loads(src_config.read_text())
    config["architectures"] = ["Qwen3DSparkModel"]
    config["quantization_config"] = QUANTIZATION_CONFIG
    config["partial_rotary_factor"] = float(config.get("partial_rotary_factor") or 1.0)
    dst.write_text(json.dumps(config, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=Path,
        default=root / "models" / "Qwen3.8-27B-DSpark",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=root / "models" / "Qwen3.8-27B-DSpark-NVFP4A16",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    src = args.src.resolve()
    dst = args.dst.resolve()
    weight_path = src / "model.safetensors"
    if not weight_path.is_file():
        print(f"missing {weight_path}", file=sys.stderr)
        return 1
    if not (src / "config.json").is_file():
        print(f"missing {src / 'config.json'}", file=sys.stderr)
        return 1

    print(f"load {weight_path}")
    state = load_all(weight_path)
    print(f"tensors {len(state)}")
    out, quantized = quantize_state(state)
    if not quantized:
        print("no matching Linear weights", file=sys.stderr)
        return 1
    for line in quantized:
        print(f"  {line}")

    dst.mkdir(parents=True, exist_ok=True)
    out_path = dst / "model.safetensors"
    print(f"write {out_path} ({len(out)} tensors)")
    save_file(out, str(out_path))
    write_config(src / "config.json", dst / "config.json")
    for name in COPY_FILES:
        src_file = src / name
        if src_file.is_file():
            shutil.copy2(src_file, dst / name)

    src_bytes = weight_path.stat().st_size
    dst_bytes = out_path.stat().st_size
    print(
        f"done: {len(quantized)} linears, "
        f"{src_bytes / 1e9:.2f} GB BF16 -> {dst_bytes / 1e9:.2f} GB NVFP4A16"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
