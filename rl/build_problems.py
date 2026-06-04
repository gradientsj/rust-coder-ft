#!/usr/bin/env python3
"""Build the GRPO problem bank (train) and Rust RL eval set (heldout).

Sources (verified on the live Hub 2026-06-04, both MIT):
  TRAIN  nuprl/MultiPL-E  config mbpp-rs   (354 problems)
  EVAL   bigcode/humanevalpack  config rust (164 problems, true #[test] style)

CONTAMINATION POLICY (documented in REPORT.md): mbpp-rs is a translation of
Python MBPP -> after RL, the Python-MBPP retention metric is tainted and
Python-HumanEval becomes the clean retention signal. humanevalpack-rust is
NEVER trained on; it is the held-out Rust eval. No HumanEval variant enters
training.

mbpp-rs format: `prompt` opens a fn and stops at `{`; `tests` closes that
brace then defines fn main(){ let candidate = <fn>; assert_eq!(...); }.
We re-shape for a chat model + cargo-test judge:
  - chat prompt: instruction + the fn signature/doc, ask for a complete
    function in a ```rust block
  - tests: main-body asserts wrapped into  #[test] fn check() { ... }
    (valid inside the judge's `mod harness_tests { use super::*; ... }`)

Every problem's REFERENCE (prompt+canonical close from MultiPL-E doctests
field is absent -> we verify the test harness instead by compiling the
prompt's own closed form where available; for mbpp-rs we verify by running
the judge on prompt+tests with the original completion-style assembly) —
practically: we keep a problem iff its transformed tests COMPILE against a
reference implementation. mbpp-rs ships no canonical solution, so the
reference check uses the FT model itself? No — we keep it honest and purely
structural: a problem is kept iff its test module PARSES+COMPILES against a
`todo!()` stub (proves the harness is well-formed; unsolvable/broken test
syntax is dropped) AND the entry fn name was extracted. Reward ceiling
reachability is then guaranteed by GRPO group filtering at train time
(all-zero-reward groups carry no gradient and are skipped).

Outputs:
  rl/problems/train.jsonl   {name, prompt, tests, entry_point, source}
  rl/problems/eval.jsonl    same fields + declaration/canonical for scoring
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from compile_check.runner import check_many  # noqa: E402

INSTR = ("Implement the following Rust function. Reply with a single "
         "```rust code block containing the complete function "
         "(signature included), no tests, no main.")


def mbpp_rs_records():
    from datasets import load_dataset
    ds = load_dataset("nuprl/MultiPL-E", "mbpp-rs", split="test")
    for row in ds:
        prompt, tests = row["prompt"], row["tests"]
        m = re.search(r"fn\s+(\w+)\s*\(", prompt)
        if not m:
            continue
        entry = m.group(1)
        # tests = "}\n\nfn main() { <body> }\n" -> extract main body
        mm = re.search(r"fn\s+main\s*\(\s*\)\s*\{(.*)\}\s*$", tests, re.S)
        if not mm:
            continue
        body = mm.group(1).rstrip()
        test_mod = f"    #[test]\n    fn check() {{\n{body}\n    }}"
        # chat prompt shows the signature + doc comments from the source prompt
        sig_block = prompt.strip()
        if sig_block.endswith("{"):
            sig_block += "\n    // your implementation\n}"
        yield {
            "name": row["name"],
            "prompt": f"{INSTR}\n\n```rust\n{sig_block}\n```",
            "tests": test_mod,
            "entry_point": entry,
            "source": "nuprl/MultiPL-E:mbpp-rs",
        }


_EXT_CRATE_IMPORT = re.compile(r"^\s*use\s+(rand|regex|md5)\b.*$", re.M)
_EXT_CRATE_USE = re.compile(r"\b(rand::|Rng\b|Regex\b|md5::|md5\()")


def humanevalpack_rust_records():
    from datasets import load_dataset
    ds = load_dataset("bigcode/humanevalpack", "rust", split="test")
    n_ext = 0
    for row in ds:
        # humanevalpack declarations blanket-import rand/regex/md5 even when
        # unused; the judge is zero-dep+offline, so strip them and drop the
        # few problems whose solution/tests genuinely need external crates
        body = row["canonical_solution"] + row["test"]
        if _EXT_CRATE_USE.search(body):
            n_ext += 1
            continue
        decl = _EXT_CRATE_IMPORT.sub("", row["declaration"]).strip()
        yield {
            "name": row["task_id"],
            "prompt": (f"{INSTR}\n\n```rust\n{decl}\n"
                       f"    // {row['docstring'].strip()[:500]}\n}}\n```"),
            "tests": row["test"],
            "tests_style": "module",   # already a #[cfg(test)] module
            "entry_point": row["entry_point"],
            "declaration": decl,
            "canonical_solution": row["canonical_solution"],
            "source": "bigcode/humanevalpack:rust",
        }


def harness_ok(records: list[dict]) -> list[bool]:
    """Train records (no canonical solution): harness is well-formed iff a
    todo!() stub + tests COMPILE. Eval records (canonical available):
    REFERENCE-VERIFIED — declaration + canonical solution must build AND
    pass its own test module."""
    items = []
    for r in records:
        if "canonical_solution" in r:
            code = r["declaration"] + "\n" + r["canonical_solution"]
            items.append({"code": code, "tests": r["tests"]})
        else:
            m = re.search(r"```rust\n(.*?)```", r["prompt"], re.S)
            stub = (m.group(1) if m else "").replace(
                "// your implementation", "todo!()")
            items.append({"code": stub, "tests": r["tests"]})
    results = check_many(items, clippy=False, max_workers=32)
    return [(res.compiles and res.tests_passed is True)
            if "canonical_solution" in r else res.compiles
            for r, res in zip(records, results)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("rl/problems"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for fname, gen in [("train.jsonl", mbpp_rs_records),
                       ("eval.jsonl", humanevalpack_rust_records)]:
        records = list(gen())
        keep = harness_ok(records)
        kept = [r for r, k in zip(records, keep) if k]
        with open(args.out / fname, "w") as f:
            for r in kept:
                f.write(json.dumps(r) + "\n")
        print(f"{fname}: {len(kept)}/{len(records)} kept "
              f"(dropped {len(records) - len(kept)} malformed harnesses)")


if __name__ == "__main__":
    main()
