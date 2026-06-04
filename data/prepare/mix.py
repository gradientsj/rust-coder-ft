#!/usr/bin/env python3
"""Blend domain pairs with anchor data at a configurable ratio.

Default 1:2 domain:anchor — for every domain pair, two anchor records, with
the anchor side split (configurable) between code-instruct, general-instruct
and raw-Rust completion text. If an anchor pool is smaller than requested we
take what exists and report the actual achieved ratio rather than failing.

Reads  data/work/{domain_pairs,anchor_*}.jsonl
Writes data/work/mixed_chat.jsonl        (records with "messages")
       data/work/mixed_completion.jsonl  (records with "text")
       data/work/mix_stats.json
Holdout carving happens downstream in format_chatml.py; this stage only
shuffles (seeded) and tags every record with its origin.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open()] if p.exists() else []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workdir", type=Path, default=Path("data/work"))
    ap.add_argument("--anchor-ratio", type=float, default=2.0,
                    help="anchor records per domain pair (default 1:2)")
    ap.add_argument("--anchor-weights", default="0.4,0.4,0.2",
                    help="code_instruct,general_instruct,rust_raw split")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    domain = read_jsonl(args.workdir / "domain_pairs.jsonl")
    pools = {
        "code_instruct": read_jsonl(args.workdir / "anchor_code_instruct.jsonl"),
        "general_instruct": read_jsonl(args.workdir / "anchor_general_instruct.jsonl"),
        "rust_raw": read_jsonl(args.workdir / "anchor_rust_raw.jsonl"),
    }
    w = [float(x) for x in args.anchor_weights.split(",")]
    assert len(w) == 3 and abs(sum(w) - 1.0) < 1e-6, "--anchor-weights must sum to 1"

    n_anchor_target = int(len(domain) * args.anchor_ratio)
    want = {k: int(n_anchor_target * wi)
            for k, wi in zip(("code_instruct", "general_instruct", "rust_raw"), w)}

    chat, completion = [], []
    for rec in domain:
        chat.append({"origin": "domain", "pair_type": rec["pair_type"],
                     "id": rec["id"], "crate": rec["crate"], "file": rec["file"],
                     "license": rec["license"],
                     "messages": [{"role": "user", "content": rec["prompt"]},
                                  {"role": "assistant", "content": rec["response"]}]})
    taken = {}
    for key, n_want in want.items():
        pool = pools[key]
        rng.shuffle(pool)
        sel = pool[:n_want]
        taken[key] = len(sel)
        for rec in sel:
            if "text" in rec:
                completion.append({"origin": key, "id": rec["id"],
                                   "source": rec["source"], "text": rec["text"]})
            else:
                chat.append({"origin": key, "id": rec["id"],
                             "source": rec["source"], "messages": rec["messages"]})

    rng.shuffle(chat)
    rng.shuffle(completion)
    with open(args.workdir / "mixed_chat.jsonl", "w") as f:
        for r in chat:
            f.write(json.dumps(r) + "\n")
    with open(args.workdir / "mixed_completion.jsonl", "w") as f:
        for r in completion:
            f.write(json.dumps(r) + "\n")

    n_anchor = sum(taken.values())
    stats = {
        "domain": len(domain),
        "anchor_target": n_anchor_target,
        "anchor_taken": taken,
        "achieved_ratio": round(n_anchor / len(domain), 3) if domain else None,
        "chat_records": len(chat),
        "completion_records": len(completion),
    }
    (args.workdir / "mix_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
