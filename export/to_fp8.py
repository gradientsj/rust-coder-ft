#!/usr/bin/env python3
"""FP8 (W8A8) export for vLLM serving, via llm-compressor compressed-tensors.

Run with the QUANT venv (llm-compressor pins its own torch/transformers,
incompatible with the training venv — keep them separate):
  .venv-quant/bin/python export/to_fp8.py export/qwen3-8b-ft --out export/qwen3-8b-ft-fp8

Scheme: FP8_DYNAMIC — static per-channel RTN weight quant + dynamic per-token
activation quant. DATA-FREE: no calibration set, no forward passes; an 8B
fits a single H100 with room to spare. `lm_head` stays in bf16 (standard for
dense Qwen3; MoE variants would need a larger ignore list).

vLLM auto-detects the result via config.json quant_method=compressed-tensors;
no --quantization flag needed at serve time.

Note: serving FP8 needs compute capability >= 8.9 (Ada/Hopper). H100 = 9.0.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", help="HF id or local path (bf16 fine-tune)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ignore", nargs="*", default=["lm_head"],
                    help="modules to keep unquantized")
    args = ap.parse_args()

    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} exists and is not empty")

    import torch  # noqa: F401  (fail fast if venv is wrong)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    print(f"loading {args.model} (dtype=auto keeps bf16 fine-tunes bf16)...")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    recipe = QuantizationModifier(
        targets="Linear", scheme="FP8_DYNAMIC", ignore=args.ignore)
    oneshot(model=model, recipe=recipe)  # data-free: no dataset args

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out), save_compressed=True)
    tokenizer.save_pretrained(str(args.out))

    # verify the result is what vLLM will auto-detect
    cfg = json.loads((args.out / "config.json").read_text())
    qc = cfg.get("quantization_config", {})
    assert qc.get("quant_method") == "compressed-tensors", \
        f"unexpected quantization_config: {qc}"
    size_gb = sum(f.stat().st_size for f in args.out.glob("*.safetensors")) / 1e9
    print(json.dumps({
        "out": str(args.out),
        "quant_method": qc.get("quant_method"),
        "format": qc.get("format"),
        "weights_gb": round(size_gb, 2),
        "ignored": args.ignore,
    }, indent=2))


if __name__ == "__main__":
    main()
