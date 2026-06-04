#!/usr/bin/env python3
"""Consolidate a training run / checkpoint into a clean, servable HF model dir.

Run with the TRAINING venv (.venv — needs axolotl for the DCP merge path):
  .venv/bin/python export/merge_and_save.py outputs/qwen3-8b-fft --out export/qwen3-8b-ft
  .venv/bin/python export/merge_and_save.py outputs/qwen3-8b-fft/checkpoint-150 \
      --out export/qwen3-8b-ft-s150 --tokenizer-from outputs/qwen3-8b-fft

Two input cases, detected automatically (verified against the installed
transformers 5.5.0 / accelerate 1.13.0 / axolotl 0.16.1 source):

  A. Already-consolidated HF dir (our configs use state_dict_type
     FULL_STATE_DICT, so BOTH the final output_dir AND intermediate
     checkpoint-N dirs contain full model*.safetensors + config + tokenizer)
     -> copy model files, strip training debris (optimizer.bin,
        pytorch_model_fsdp_0.bin, rng/scheduler state), fix the
        FSDP-prefixed `architectures` field axolotl only fixes on final saves.

  B. DCP-sharded checkpoint (state_dict_type SHARDED_STATE_DICT:
     a pytorch_model_fsdp_0/ dir of __*_*.distcp shards, no safetensors)
     -> consolidate via axolotl's merge_fsdp_weights (bf16 cast + 5GB
        safetensors shards + index), then pull config/tokenizer from
        --tokenizer-from (DCP checkpoints don't carry them).

Always ends with a structural validation (config + tokenizer + weight index
load, parameter count); --smoke-generate additionally loads to GPU 0 and
generates a few tokens.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

TRAINING_DEBRIS = (
    "optimizer.bin", "scheduler.pt", "trainer_state.json", "training_args.bin",
    "pytorch_model_fsdp_0.bin", "scaler.pt",
)
TOKENIZER_FILES = (
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "vocab.json", "merges.txt", "added_tokens.json", "chat_template.jinja",
)


def is_consolidated(d: Path) -> bool:
    return (d / "config.json").exists() and (
        any(d.glob("model*.safetensors")) or (d / "pytorch_model.bin").exists())


def is_dcp_sharded(d: Path) -> bool:
    sub = d / "pytorch_model_fsdp_0"
    return sub.is_dir() and any(sub.glob("*.distcp"))


def fix_architectures(config_path: Path) -> None:
    """Intermediate FULL_STATE_DICT saves can carry FSDP-prefixed arch names;
    axolotl only strips them on the final output_dir (train.py:305-318)."""
    cfg = json.loads(config_path.read_text())
    archs = cfg.get("architectures") or []
    fixed = [a[4:] if a.startswith("FSDP") else a for a in archs]
    if fixed != archs:
        cfg["architectures"] = fixed
        config_path.write_text(json.dumps(cfg, indent=2))
        print(f"  fixed architectures: {archs} -> {fixed}")


def copy_consolidated(src: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.name in TRAINING_DEBRIS or f.name.startswith("rng_state"):
            continue
        if f.is_dir():
            continue  # checkpoint subdirs (optimizer_0/ etc.) are never model files
        shutil.copy2(f, out / f.name)
    fix_architectures(out / "config.json")


def merge_dcp(src: Path, out: Path, tokenizer_from: Path | None) -> None:
    from axolotl.cli.merge_sharded_fsdp_weights import merge_fsdp_weights
    merge_fsdp_weights(str(src / "pytorch_model_fsdp_0"), str(out),
                       remove_checkpoint_dir=False)
    if tokenizer_from is None:
        sys.exit("--tokenizer-from is required for DCP checkpoints "
                 "(they carry no config/tokenizer)")
    for name in ("config.json", "generation_config.json", *TOKENIZER_FILES):
        f = tokenizer_from / name
        if f.exists():
            shutil.copy2(f, out / name)
    fix_architectures(out / "config.json")


def validate(out: Path, smoke: bool) -> dict:
    from transformers import AutoConfig, AutoTokenizer
    cfg = AutoConfig.from_pretrained(out)
    tok = AutoTokenizer.from_pretrained(out)
    idx = out / "model.safetensors.index.json"
    n_params = None
    if idx.exists():
        meta = json.loads(idx.read_text())
        n_files = len(set(meta["weight_map"].values()))
    else:
        n_files = len(list(out.glob("model*.safetensors")))
    from safetensors import safe_open
    n_params = 0
    for f in sorted(out.glob("model*.safetensors")):
        with safe_open(f, framework="pt") as sf:
            for k in sf.keys():
                shape = sf.get_slice(k).get_shape()
                n = 1
                for s in shape:
                    n *= s
                n_params += n
    report = {"model_type": cfg.model_type, "weight_files": n_files,
              "n_params": n_params, "vocab_size": len(tok)}
    if smoke:
        import torch
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            out, dtype=torch.bfloat16, device_map="cuda:0")
        msgs = [{"role": "user", "content": "Write a Rust function that adds two i32s."}]
        try:   # match training/serving format: non-thinking
            text = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                           enable_thinking=False, tokenize=False)
        except TypeError:
            text = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                           tokenize=False)
        enc = tok(text, return_tensors="pt").to("cuda:0")
        gen = model.generate(**enc, max_new_tokens=48, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        report["smoke_generation"] = tok.decode(
            gen[0, enc.input_ids.shape[1]:], skip_special_tokens=True)[:200]
        del model
        torch.cuda.empty_cache()
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", type=Path, help="output_dir or checkpoint-N dir")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tokenizer-from", type=Path, default=None,
                    help="dir with config/tokenizer (needed for DCP checkpoints)")
    ap.add_argument("--smoke-generate", action="store_true")
    args = ap.parse_args()

    if not args.src.exists():
        sys.exit(f"{args.src} does not exist")
    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"{args.out} exists and is not empty — refusing to overwrite")

    if is_consolidated(args.src):
        print(f"[case A] consolidated HF dir: {args.src}")
        copy_consolidated(args.src, args.out)
    elif is_dcp_sharded(args.src):
        print(f"[case B] DCP-sharded checkpoint: {args.src}")
        merge_dcp(args.src, args.out, args.tokenizer_from)
    else:
        sys.exit(f"{args.src} is neither a consolidated HF dir nor a DCP "
                 "checkpoint (looked for model*.safetensors / "
                 "pytorch_model_fsdp_0/*.distcp)")

    report = validate(args.out, args.smoke_generate)
    print(json.dumps(report, indent=2))
    print(f"OK: {args.out}")


if __name__ == "__main__":
    main()
