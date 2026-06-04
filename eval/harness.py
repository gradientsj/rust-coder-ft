#!/usr/bin/env python3
"""Top-level eval: domain (cargo) + general (HumanEval/MBPP) -> one report.

  python eval/harness.py --model outputs/qwen3-8b-fft \
      --baseline eval/results/general_baseline/summary.json

Runs domain_eval and general_eval as subprocesses (they own GPU lifecycle),
then merges their JSON into eval/results/report_<name>.json including the
retention delta against the pre-FT baseline. Either half can be skipped or
given pre-computed results.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), file=sys.stderr)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"step failed (rc={r.returncode}): {' '.join(cmd)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", default=None, help="default: model basename")
    ap.add_argument("--baseline", type=Path,
                    default=Path("eval/results/general_baseline/summary.json"),
                    help="pre-FT general summary for the retention delta")
    ap.add_argument("--pairs", type=Path,
                    default=Path("data/splits/heldout_domain.jsonl"))
    ap.add_argument("--n-domain", type=int, default=None)
    ap.add_argument("--limit-general", type=int, default=None)
    ap.add_argument("--skip-domain", action="store_true")
    ap.add_argument("--skip-general", action="store_true")
    args = ap.parse_args()

    tag = args.tag or Path(args.model).name
    results = Path("eval/results")
    results.mkdir(parents=True, exist_ok=True)

    domain_out = results / f"domain_{tag}.json"
    if not args.skip_domain:
        cmd = [PY, "eval/domain_eval.py", "--model", args.model,
               "--pairs", str(args.pairs), "--out", str(domain_out)]
        if args.n_domain:
            cmd += ["--n", str(args.n_domain)]
        run(cmd)

    general_summary = results / f"general_{tag}/summary.json"
    if not args.skip_general:
        cmd = [PY, "eval/general_eval.py", "--model", args.model, "--tag", tag]
        if args.limit_general:
            cmd += ["--limit", str(args.limit_general)]
        run(cmd)

    report: dict = {"model": args.model, "tag": tag,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}

    if domain_out.exists():
        d = json.loads(domain_out.read_text())
        report["domain"] = {k: d[k] for k in
                            ("n_pairs", "n_compile_subset", "compile_pass_rate",
                             "clippy_clean_rate", "mean_similarity",
                             "exact_match_rate")}

    if general_summary.exists():
        g = json.loads(general_summary.read_text())
        report["general"] = g["scores"]
        if args.baseline.exists() and args.baseline != general_summary:
            base = json.loads(args.baseline.read_text())["scores"]
            report["baseline"] = base
            report["retention_delta"] = {
                k: round(report["general"][k] - base[k], 4)
                for k in report["general"] if k in base}

    out = results / f"report_{tag}.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
