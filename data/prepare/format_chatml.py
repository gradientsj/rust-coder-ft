#!/usr/bin/env python3
"""Final formatting + splits: Axolotl-ready jsonl under data/splits/.

Reads  data/work/mixed_chat.jsonl, data/work/mixed_completion.jsonl
Writes data/splits/
  train_chat.jsonl        {"messages": [...]}   Axolotl type: chat_template
  train_completion.jsonl  {"text": ...}         Axolotl type: completion
  val_chat.jsonl, val_completion.jsonl          same shapes, ~val-frac of data
  heldout_domain.jsonl    domain pairs held out at FILE level (a source file's
                          pairs are all-in or all-out -> no near-dup leakage
                          between train and eval), full metadata kept for the
                          cargo eval harness
  heldout_general.jsonl   anchor instruct sample, full metadata, for the
                          retention eval

We do NOT render the chat template ourselves: train records stay in messages
format and Axolotl applies the model tokenizer's own template at train time
(config: type chat_template). --validate-tokenizer renders one record through
the actual tokenizer as a smoke test of that path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

HOLDOUT_BUCKETS = 5     # of 100 -> ~5% of domain FILES held out
N_HELDOUT_GENERAL = 250          # absolute cap...
HELDOUT_GENERAL_MAX_FRAC = 0.10  # ...but never more than 10% of the anchor pool


def file_bucket(path: str) -> int:
    return int(hashlib.sha1(path.encode()).hexdigest(), 16) % 100


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workdir", type=Path, default=Path("data/work"))
    ap.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--validate-tokenizer", default=None,
                    help="HF model id; renders one sample through its chat template")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    args.splits_dir.mkdir(parents=True, exist_ok=True)

    chat = [json.loads(l) for l in (args.workdir / "mixed_chat.jsonl").open()]
    completion = [json.loads(l) for l in
                  (args.workdir / "mixed_completion.jsonl").open()]

    heldout_domain = [r for r in chat if r["origin"] == "domain"
                      and file_bucket(r["file"]) < HOLDOUT_BUCKETS]
    heldout_ids = {r["id"] for r in heldout_domain}
    remaining_chat = [r for r in chat if r["id"] not in heldout_ids]

    anchor_chat = [r for r in remaining_chat if r["origin"] != "domain"]
    rng.shuffle(anchor_chat)
    n_hg = min(N_HELDOUT_GENERAL,
               int(len(anchor_chat) * HELDOUT_GENERAL_MAX_FRAC))
    heldout_general = anchor_chat[:n_hg]
    heldout_ids |= {r["id"] for r in heldout_general}
    remaining_chat = [r for r in remaining_chat if r["id"] not in heldout_ids]

    def carve_val(rows: list[dict]) -> tuple[list[dict], list[dict]]:
        rng.shuffle(rows)
        n_val = max(int(len(rows) * args.val_frac), 1) if rows else 0
        return rows[n_val:], rows[:n_val]

    train_chat, val_chat = carve_val(remaining_chat)
    train_comp, val_comp = carve_val(completion)

    def dump(name: str, rows: list[dict], keys: tuple[str, ...] | None):
        with open(args.splits_dir / name, "w") as f:
            for r in rows:
                f.write(json.dumps(
                    {k: r[k] for k in keys if k in r} if keys else r) + "\n")

    dump("train_chat.jsonl", train_chat, ("messages",))
    dump("train_completion.jsonl", train_comp, ("text",))
    dump("val_chat.jsonl", val_chat, ("messages",))
    dump("val_completion.jsonl", val_comp, ("text",))
    dump("heldout_domain.jsonl", heldout_domain, None)   # keep metadata for eval
    dump("heldout_general.jsonl", heldout_general, None)

    counts = {
        "train_chat": len(train_chat), "train_completion": len(train_comp),
        "val_chat": len(val_chat), "val_completion": len(val_comp),
        "heldout_domain": len(heldout_domain),
        "heldout_general": len(heldout_general),
    }
    (args.splits_dir / "split_counts.json").write_text(
        json.dumps(counts, indent=2))
    print(json.dumps(counts, indent=2))

    if args.validate_tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.validate_tokenizer)
        sample = (train_chat or val_chat)[0]["messages"]
        rendered = tok.apply_chat_template(sample, tokenize=False)
        n_tokens = len(tok(rendered).input_ids)
        print(f"\n--- chat template render ({args.validate_tokenizer}, "
              f"{n_tokens} tokens) ---")
        print(rendered[:1200])


if __name__ == "__main__":
    main()
