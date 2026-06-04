#!/usr/bin/env python3
"""Pull the HF anchor datasets that prevent catastrophic forgetting.

Slugs verified against the live Hub on 2026-06-04 (all ungated):
  ammarnasr/the-stack-rust-clean        raw Rust files (content field) —
                                        permissive-license derivative of
                                        bigcode/the-stack; used as COMPLETION
                                        text to anchor the Rust distribution
  ise-uiuc/Magicoder-OSS-Instruct-75K   (problem, solution) code instruct,
                                        multi-language — retention anchor
  allenai/tulu-3-sft-mixture            messages-format general instruct —
                                        instruction-following anchor

Writes (messages format unless noted):
  data/work/anchor_rust_raw.jsonl        {"text": ...}  (Axolotl type: completion)
  data/work/anchor_code_instruct.jsonl   {"messages": [...], "id", "source"}
  data/work/anchor_general_instruct.jsonl{"messages": [...], "id", "source"}

Streaming + take(n): nothing is fully downloaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset

MAX_CHARS = 16000  # ~4k tokens; anything longer gets truncated at format time anyway


def _id(source: str, payload: str) -> str:
    return f"{source}-{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


def dump_rust_raw(n: int, out: Path) -> int:
    ds = load_dataset("ammarnasr/the-stack-rust-clean", split="train",
                      streaming=True)
    kept = 0
    with out.open("w") as f:
        for row in ds:
            text = row["content"]
            if not (256 <= len(text) <= MAX_CHARS):
                continue
            f.write(json.dumps({
                "id": _id("stackrust", text),
                "source": "ammarnasr/the-stack-rust-clean",
                "text": text,
            }) + "\n")
            kept += 1
            if kept >= n:
                break
    return kept


def dump_code_instruct(n: int, out: Path) -> int:
    ds = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train",
                      streaming=True)
    kept = 0
    with out.open("w") as f:
        for row in ds:
            prob, sol = row["problem"].strip(), row["solution"].strip()
            if not prob or not sol or len(prob) + len(sol) > MAX_CHARS:
                continue
            f.write(json.dumps({
                "id": _id("magicoder", prob),
                "source": "ise-uiuc/Magicoder-OSS-Instruct-75K",
                "lang": row.get("lang"),
                "messages": [{"role": "user", "content": prob},
                             {"role": "assistant", "content": sol}],
            }) + "\n")
            kept += 1
            if kept >= n:
                break
    return kept


def dump_general_instruct(n: int, out: Path) -> int:
    ds = load_dataset("allenai/tulu-3-sft-mixture", split="train",
                      streaming=True)
    kept = 0
    with out.open("w") as f:
        for row in ds:
            msgs = row["messages"]
            if not msgs or msgs[0]["role"] not in ("user", "system"):
                continue
            if msgs[-1]["role"] != "assistant":
                continue
            total = sum(len(m["content"]) for m in msgs)
            if total > MAX_CHARS:
                continue
            f.write(json.dumps({
                "id": _id("tulu3", msgs[0]["content"][:512]),
                "source": "allenai/tulu-3-sft-mixture",
                "messages": [{"role": m["role"], "content": m["content"]}
                             for m in msgs],
            }) + "\n")
            kept += 1
            if kept >= n:
                break
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workdir", type=Path, default=Path("data/work"))
    ap.add_argument("--n-rust-raw", type=int, default=20000)
    ap.add_argument("--n-code-instruct", type=int, default=30000)
    ap.add_argument("--n-general-instruct", type=int, default=30000)
    args = ap.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)

    counts = {
        "rust_raw": dump_rust_raw(
            args.n_rust_raw, args.workdir / "anchor_rust_raw.jsonl"),
        "code_instruct": dump_code_instruct(
            args.n_code_instruct, args.workdir / "anchor_code_instruct.jsonl"),
        "general_instruct": dump_general_instruct(
            args.n_general_instruct, args.workdir / "anchor_general_instruct.jsonl"),
    }
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
