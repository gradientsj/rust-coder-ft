#!/usr/bin/env python3
"""Held-out Rust eval: generate completions, verdict them with cargo.

Honest-metric design: extracted crate functions often reference private,
crate-internal items, so "does the generation compile standalone" is only a
fair metric when the REFERENCE solution itself compiles standalone. Pipeline:

  1. reference-compile filter: cargo-check every reference; pairs whose
     reference compiles form the `compile_subset`
  2. generate a completion for every pair (greedy, non-thinking chat template)
  3. metrics:
       compile_subset -> compile_pass_rate, clippy_clean_rate
       all pairs      -> edit_similarity (difflib, code-normalized), exact_match

Usage:
  python eval/domain_eval.py --model Qwen/Qwen3-8B --n 20
  python eval/domain_eval.py --model outputs/qwen3-8b-fft [--pairs ...]
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from compile_check.runner import check_many  # noqa: E402

RUST_BLOCK = re.compile(r"```rust\n(.*?)```", re.S)


def extract_code(text: str) -> str:
    m = RUST_BLOCK.search(text)
    return (m.group(1) if m else text).strip()


def norm(code: str) -> str:
    return " ".join(code.split())


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def generate(model_id: str, prompts: list[str], max_new_tokens: int) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="flash_attention_2")
    model.eval()

    outs = []
    for i, p in enumerate(prompts):
        msgs = [{"role": "user", "content": p}]
        try:  # Qwen3: force non-thinking so the SFT format matches
            text = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                           enable_thinking=False, tokenize=False)
        except TypeError:  # template without the kwarg
            text = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                           tokenize=False)
        enc = tok(text, return_tensors="pt").to("cuda:0")
        ids = enc.input_ids
        with torch.inference_mode():
            out = model.generate(ids, attention_mask=enc.attention_mask,
                                 max_new_tokens=max_new_tokens,
                                 do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        outs.append(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
        if (i + 1) % 10 == 0:
            print(f"  generated {i + 1}/{len(prompts)}", file=sys.stderr)
    del model
    torch.cuda.empty_cache()
    return outs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pairs", type=Path,
                    default=Path("data/splits/heldout_domain.jsonl"))
    ap.add_argument("--n", type=int, default=None, help="limit #pairs")
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    pairs = [json.loads(l) for l in args.pairs.open()][: args.n]
    if not pairs:
        sys.exit(f"no pairs in {args.pairs} — at tiny scale heldout_domain "
                 "can be empty; pass --pairs explicitly")
    prompts = [p["messages"][0]["content"] for p in pairs]
    refs = [extract_code(p["messages"][1]["content"]) for p in pairs]

    print(f"[1/3] reference-compile filter over {len(pairs)} pairs...")
    ref_results = check_many([{"code": r} for r in refs], clippy=False)
    compile_subset = [i for i, r in enumerate(ref_results) if r.compiles]
    print(f"      {len(compile_subset)}/{len(pairs)} references compile "
          f"standalone -> compile_subset")

    print(f"[2/3] generating with {args.model}...")
    gens = generate(args.model, prompts, args.max_new_tokens)
    codes = [extract_code(g) for g in gens]

    print("[3/3] cargo verdicts on generations (compile_subset only)...")
    gen_results = {i: r for i, r in zip(
        compile_subset,
        check_many([{"code": codes[i]} for i in compile_subset]))}

    items = []
    for i, p in enumerate(pairs):
        r = gen_results.get(i)
        items.append({
            "id": p.get("id"), "crate": p.get("crate"),
            "pair_type": p.get("pair_type"),
            "in_compile_subset": i in gen_results,
            "compiles": r.compiles if r else None,
            "clippy_clean": r.clippy_clean if r else None,
            "similarity": round(similarity(codes[i], refs[i]), 4),
            "exact_match": norm(codes[i]) == norm(refs[i]),
        })

    cs = [it for it in items if it["in_compile_subset"]]
    report = {
        "model": args.model,
        "pairs_file": str(args.pairs),
        "n_pairs": len(pairs),
        "n_compile_subset": len(cs),
        "compile_pass_rate": (sum(it["compiles"] for it in cs) / len(cs)) if cs else None,
        "clippy_clean_rate": (sum(bool(it["clippy_clean"]) for it in cs) / len(cs)) if cs else None,
        "mean_similarity": sum(it["similarity"] for it in items) / len(items),
        "exact_match_rate": sum(it["exact_match"] for it in items) / len(items),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "items": items,
    }
    out = args.out or Path(f"eval/results/domain_{Path(args.model).name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    summary = {k: v for k, v in report.items() if k != "items"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
