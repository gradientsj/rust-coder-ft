#!/usr/bin/env python3
"""HumanEval + MBPP retention check via lm-eval (ships with axolotl).

Run BEFORE fine-tuning (baseline) and AFTER (candidate); harness.py computes
the delta. The absolute numbers are protocol-dependent (raw-completion
prompting of a chat model scores below its chat-mode ability) — that is fine,
the metric of record is the PRE/POST DELTA under the identical protocol.

Code execution is what these benchmarks do, hence HF_ALLOW_CODE_EVAL=1 and
--confirm_run_unsafe_code. 4-GPU data parallel via accelerate.

Usage:
  python eval/general_eval.py --model Qwen/Qwen3-8B --tag baseline
  python eval/general_eval.py --model outputs/qwen3-8b-fft --tag post-ft
  python eval/general_eval.py ... --limit 8        # smoke-test sized
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

TASKS = "humaneval,mbpp"


def find_results_json(out_dir: Path) -> Path | None:
    hits = sorted(out_dir.rglob("results_*.json"),
                  key=lambda p: p.stat().st_mtime)
    return hits[-1] if hits else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True, help="baseline | post-ft | ...")
    ap.add_argument("--tasks", default=TASKS)
    ap.add_argument("--limit", type=int, default=None, help="debug subset")
    ap.add_argument("--num-processes", type=int, default=4)
    args = ap.parse_args()

    out_dir = Path(f"eval/results/general_{args.tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "accelerate", "launch", "--num_processes", str(args.num_processes),
        "--mixed_precision", "bf16", "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={args.model},dtype=bfloat16",
        "--tasks", args.tasks,
        "--batch_size", "auto",
        "--output_path", str(out_dir),
        "--confirm_run_unsafe_code",
    ]
    if args.limit:
        cmd += ["--limit", str(args.limit)]

    env = {**os.environ, "HF_ALLOW_CODE_EVAL": "1",
           "TOKENIZERS_PARALLELISM": "false"}
    print(" ".join(cmd), file=sys.stderr)
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        sys.exit(f"lm_eval failed with rc={r.returncode}")

    rj = find_results_json(out_dir)
    if rj is None:
        sys.exit(f"no results json under {out_dir}")
    results = json.loads(rj.read_text())["results"]
    summary = {"model": args.model, "tag": args.tag, "source": str(rj),
               "scores": {}}
    for task, metrics in results.items():
        for k, v in metrics.items():
            # humaneval: "pass@1,create_test"; mbpp: "pass_at_1,none"
            if (k.startswith(("pass@", "pass_at")) and "_stderr" not in k
                    and isinstance(v, (int, float))):
                summary["scores"][f"{task}.{k.split(',')[0]}"] = round(v, 4)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
