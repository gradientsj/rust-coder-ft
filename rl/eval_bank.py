#!/usr/bin/env python3
"""Pass-rate eval on the held-out Rust bank (humanevalpack-rust, 158
reference-verified problems). Run with the TRAINING venv (.venv):

  .venv/bin/python rl/eval_bank.py --model export/qwen3-8b-ft   --tag pre-rl
  .venv/bin/python rl/eval_bank.py --model outputs/qwen3-8b-grpo --tag post-rl

Metric: fraction of problems whose generated solution compiles AND passes
the bank's #[cfg(test)] module (same judge as the GRPO reward — so this
measures exactly what RL optimized, on problems it never trained on).
Assembly is fair: imports from the bank's declaration + the model's code
block (models often elide use-statements that the problem context provides).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "eval"))
from compile_check.runner import check_many  # noqa: E402

RUST_BLOCK = re.compile(r"```rust\n(.*?)```", re.S)


def decl_imports(declaration: str) -> str:
    keep = [l for l in declaration.splitlines()
            if re.match(r"\s*(use |const |type |fn main\(\))", l)]
    return "\n".join(keep)


def generate(model_id: str, prompts: list[str], max_new_tokens: int,
             batch_size: int = 16) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="flash_attention_2")
    model.eval()
    outs = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        texts = []
        for p in chunk:
            msgs = [{"role": "user", "content": p}]
            try:
                texts.append(tok.apply_chat_template(
                    msgs, add_generation_prompt=True, enable_thinking=False,
                    tokenize=False))
            except TypeError:
                texts.append(tok.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=False))
        enc = tok(texts, return_tensors="pt", padding=True).to("cuda:0")
        with torch.inference_mode():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        for j in range(len(chunk)):
            outs.append(tok.decode(gen[j, enc.input_ids.shape[1]:],
                                   skip_special_tokens=True))
        print(f"  generated {min(i + batch_size, len(prompts))}/{len(prompts)}",
              file=sys.stderr)
    del model
    torch.cuda.empty_cache()
    return outs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--bank", type=Path, default=REPO / "rl/problems/eval.jsonl")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    args = ap.parse_args()

    bank = [json.loads(l) for l in args.bank.open()][: args.n]
    gens = generate(args.model, [b["prompt"] for b in bank],
                    args.max_new_tokens)

    # Pass 1: model code as-is (models usually echo the prompt's imports —
    # prepending declaration imports unconditionally causes E0252 duplicate-
    # import failures). Pass 2: only for pass-1 compile failures, retry with
    # the declaration's imports prepended (covers models that elide them).
    codes = []
    for g in gens:
        m = RUST_BLOCK.search(g)
        codes.append(m.group(1).strip() if m else None)

    idx1 = [i for i, c in enumerate(codes) if c]
    res1 = check_many([{"code": codes[i], "tests": bank[i]["tests"]}
                       for i in idx1], max_workers=32)
    results: dict[int, object] = dict(zip(idx1, res1))

    idx2 = [i for i in idx1 if not results[i].compiles]
    if idx2:
        res2 = check_many(
            [{"code": decl_imports(bank[i]["declaration"]) + "\n\n" + codes[i],
              "tests": bank[i]["tests"]} for i in idx2], max_workers=32)
        for i, r2 in zip(idx2, res2):
            if r2.compiles:
                results[i] = r2

    rows, n_pass, n_compile = [], 0, 0
    for i, b in enumerate(bank):
        if codes[i] is None:
            rows.append({"name": b["name"], "format": False,
                         "compiles": False, "passes": False})
            continue
        r = results[i]
        passes = bool(r.compiles and r.tests_passed)
        n_compile += r.compiles
        n_pass += passes
        rows.append({"name": b["name"], "format": True,
                     "compiles": r.compiles, "passes": passes,
                     "clippy_clean": r.clippy_clean})

    report = {
        "model": args.model, "tag": args.tag, "n": len(bank),
        "pass_rate": round(n_pass / len(bank), 4),
        "compile_rate": round(n_compile / len(bank), 4),
        "format_rate": round(sum(r["format"] for r in rows) / len(bank), 4),
        "clippy_clean_of_passing": round(
            sum(bool(r.get("clippy_clean")) for r in rows if r["passes"])
            / max(n_pass, 1), 4),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "items": rows,
    }
    out = REPO / f"eval/results/rustbank_{args.tag}.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "items"},
                     indent=2))


if __name__ == "__main__":
    main()
