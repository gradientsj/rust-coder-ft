#!/usr/bin/env python3
"""Cargo-verified reward for GRPO: reward = does the Rust actually work.

Wraps eval/compile_check/runner.py (the Phase 4 judge). Graded shaping —
defaults follow common RLVR practice for code, all knobs overridable:

  no ```rust block           -> R_FORMAT_FAIL   (-0.2)  format gate
  extracts but doesn't build -> 0.0
  builds, tests fail         -> R_COMPILE       (0.3)
  builds, all tests pass     -> R_PASS          (1.0)
  clippy-clean bonus         -> +R_CLIPPY       (+0.1, only on top of pass)

Anti-hacking notes (see REPORT.md Phase 6):
  - tests come from the PROBLEM BANK, never from the completion: a sample
    cannot weaken its own judge
  - rewards are computed against the bank's test set verbatim; the policy
    output contributes ONLY the solution code
  - empty/trivial completions fail the bank's tests -> no reward
  - bank problems are pre-verified (reference passes) so reward ceiling is
    reachable; problems where the reference fails were dropped at build time

The reward callable is thread-parallel over the whole GRPO group (cargo
verdicts ~0.2-2s each; group of 8-16 over 100+ cores is seconds per step).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from compile_check.runner import check_many  # noqa: E402

R_FORMAT_FAIL = -0.2
R_COMPILE = 0.3
R_PASS = 1.0
R_CLIPPY = 0.1

_RUST_BLOCK = re.compile(r"```rust\n(.*?)```", re.S)


def extract_rust(completion: str) -> str | None:
    m = _RUST_BLOCK.search(completion)
    return m.group(1).strip() if m else None


def score_batch(codes_tests: list[tuple[str | None, str]],
                max_workers: int = 32) -> list[float]:
    """codes_tests: [(extracted_code_or_None, tests_src)] -> rewards."""
    idx_to_check = [i for i, (c, _) in enumerate(codes_tests) if c]
    results = check_many(
        [{"code": codes_tests[i][0], "tests": codes_tests[i][1]}
         for i in idx_to_check],
        max_workers=max_workers, build_timeout=45, test_timeout=20)

    rewards = [R_FORMAT_FAIL] * len(codes_tests)
    for i, res in zip(idx_to_check, results):
        if not res.compiles:
            rewards[i] = 0.0
        elif res.tests_passed:
            rewards[i] = R_PASS + (R_CLIPPY if res.clippy_clean else 0.0)
        else:
            rewards[i] = R_COMPILE
    return rewards


def make_grpo_reward_fn(test_field: str = "tests"):
    """Returns a TRL GRPOTrainer-compatible reward function.

    TRL calls reward fns with (prompts, completions, **dataset_columns);
    extra dataset columns arrive as lists aligned with completions.
    """
    def cargo_reward(prompts, completions, **kwargs):
        tests = kwargs[test_field]
        # chat-format completions arrive as [{role, content}] lists
        texts = [c[0]["content"] if isinstance(c, list) else c
                 for c in completions]
        pairs = [(extract_rust(t), ts) for t, ts in zip(texts, tests)]
        return score_batch(pairs)

    cargo_reward.__name__ = "cargo_reward"
    return cargo_reward


if __name__ == "__main__":  # self-test
    tests = '    #[test]\n    fn t() { assert_eq!(add(2, 2), 4); }'
    cases = [
        ("```rust\npub fn add(a: i32, b: i32) -> i32 { a + b }\n```", tests, "pass+clippy"),
        ("```rust\npub fn add(a: i32, b: i32) -> i32 { a - b }\n```", tests, "compile, tests fail"),
        ("```rust\npub fn add(a: i32) -> i32 { b }\n```", tests, "no compile"),
        ("no code block at all", tests, "format fail"),
    ]
    rewards = score_batch([(extract_rust(c), t) for c, t, _ in cases])
    for (c, t, label), r in zip(cases, rewards):
        print(f"{label:22s} -> {r:+.2f}")
