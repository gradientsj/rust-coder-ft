#!/usr/bin/env python3
"""Plot training metrics from a (possibly still-running) HF Trainer run.

  python train/plot_metrics.py outputs/qwen3-8b-fft [--out eval/results/train_plots]

Reads trainer_state.json from the run dir (or its newest checkpoint-N while
the run is live), and emits:
  loss_curve.png    train loss (log-y) + eval loss points
  lr_schedule.png   learning rate over steps
  grad_norm.png     gradient norm over steps (spikes = instability)
  throughput.png    tokens/s/GPU over steps
  metrics.json      the raw series, for the report
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_state(run_dir: Path) -> Path:
    direct = run_dir / "trainer_state.json"
    if direct.exists():
        return direct
    cks = sorted(run_dir.glob("checkpoint-*/trainer_state.json"),
                 key=lambda p: int(p.parent.name.split("-")[1]))
    if not cks:
        raise SystemExit(f"no trainer_state.json under {run_dir} yet")
    return cks[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("eval/results/train_plots"))
    args = ap.parse_args()

    state_path = find_state(args.run_dir)
    state = json.loads(state_path.read_text())
    hist = state["log_history"]

    train = [h for h in hist if "loss" in h]
    evals = [h for h in hist if "eval_loss" in h]
    args.out.mkdir(parents=True, exist_ok=True)

    def series(rows, key):
        return ([r["step"] for r in rows if key in r],
                [r[key] for r in rows if key in r])

    s, loss = series(train, "loss")
    es, eloss = series(evals, "eval_loss")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(s, loss, lw=0.8, label="train loss")
    if es:
        ax.plot(es, eloss, "o-", ms=4, label="eval loss")
    ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("loss (log)")
    ax.set_title(f"{args.run_dir.name} — loss (step {state['global_step']}"
                 f"/{state['max_steps']}, epoch {state['epoch']:.2f})")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(args.out / "loss_curve.png", dpi=120)

    for key, fname, title in [
        ("learning_rate", "lr_schedule.png", "learning rate"),
        ("grad_norm", "grad_norm.png", "grad norm"),
        ("tokens/train_per_sec_per_gpu", "throughput.png", "tokens/s/GPU"),
    ]:
        xs, ys = series(train, key)
        ys = [float(y) for y in ys]
        if not xs:
            continue
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(xs, ys, lw=0.8)
        ax.set_xlabel("step"); ax.set_title(title); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(args.out / fname, dpi=120)

    (args.out / "metrics.json").write_text(json.dumps({
        "source": str(state_path), "global_step": state["global_step"],
        "max_steps": state["max_steps"], "epoch": state["epoch"],
        "final_train_loss": loss[-1] if loss else None,
        "final_eval_loss": eloss[-1] if eloss else None,
        "log_history": hist,
    }, indent=2))
    print(f"step {state['global_step']}/{state['max_steps']} | "
          f"train loss {loss[-1] if loss else '?'} | "
          f"eval loss {eloss[-1] if eloss else '?'} | plots -> {args.out}")


if __name__ == "__main__":
    main()
