#!/usr/bin/env python3
"""Dataset visuals + summary: the audit artifacts for the data pipeline.

Reads the work/ artifacts and splits/, writes to data/splits/stats/:
  license_breakdown.png   per-file license of everything that entered training
  dedup_funnel.png        harvest -> quality -> exact dedup -> near dedup
  length_hist.png         approx token length per record, by origin
  mix_composition.png     domain vs anchor composition of the final mix
  pair_types.png          sig_doc_impl vs instruction_code
  STATS.md                everything above as a table, plus split sizes
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

APPROX_CHARS_PER_TOKEN = 3.5  # code-ish heuristic, fine for a histogram


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open()] if p.exists() else []


def rec_len(r: dict) -> int:
    if "text" in r:
        chars = len(r["text"])
    else:
        chars = sum(len(m["content"]) for m in r["messages"])
    return int(chars / APPROX_CHARS_PER_TOKEN)


def bar(ax, counter: Counter, title: str):
    keys = [str(k) for k, _ in counter.most_common()]
    vals = [v for _, v in counter.most_common()]
    ax.bar(keys, vals)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workdir", type=Path, default=Path("data/work"))
    ap.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    args = ap.parse_args()
    out = args.splits_dir / "stats"
    out.mkdir(parents=True, exist_ok=True)

    prov = read_jsonl(args.workdir / "provenance.jsonl")
    filter_stats = json.loads((args.workdir / "filter_stats.json").read_text()) \
        if (args.workdir / "filter_stats.json").exists() else None
    mix_stats = json.loads((args.workdir / "mix_stats.json").read_text()) \
        if (args.workdir / "mix_stats.json").exists() else None
    chat = read_jsonl(args.workdir / "mixed_chat.jsonl")
    comp = read_jsonl(args.workdir / "mixed_completion.jsonl")
    domain_pairs = read_jsonl(args.workdir / "domain_pairs.jsonl")
    split_counts = json.loads((args.splits_dir / "split_counts.json").read_text()) \
        if (args.splits_dir / "split_counts.json").exists() else {}

    md = ["# Dataset stats\n"]

    # license breakdown (scrape side; anchors carry dataset-level licensing)
    lic = Counter(p["crate_license"] for p in prov)
    if lic:
        fig, ax = plt.subplots(figsize=(7, 4))
        bar(ax, lic, "Scraped files by crate license (post-filter)")
        fig.tight_layout(); fig.savefig(out / "license_breakdown.png", dpi=120)
        plt.close(fig)
        md.append("## License breakdown (scraped files)\n")
        md += [f"- `{k}`: {v:,}" for k, v in lic.most_common()]
        md.append("")

    # dedup funnel
    if filter_stats:
        f = filter_stats["funnel"]
        stages = ["input", "after_quality", "after_exact_dedup", "after_near_dedup"]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(stages, [f[s] for s in stages])
        for i, s in enumerate(stages):
            ax.text(i, f[s], f"{f[s]:,}", ha="center", va="bottom", fontsize=9)
        ax.set_title("Filter/dedup funnel (scraped files)")
        ax.tick_params(axis="x", rotation=15)
        fig.tight_layout(); fig.savefig(out / "dedup_funnel.png", dpi=120)
        plt.close(fig)
        md.append("## Filter funnel\n")
        md += [f"- {s}: {f[s]:,}" for s in stages]
        md.append(f"- reject reasons: `{json.dumps(filter_stats['rejects'])}`\n")

    # length histogram by origin
    if chat or comp:
        fig, ax = plt.subplots(figsize=(7, 4))
        for origin in sorted({r["origin"] for r in chat + comp}):
            lens = [rec_len(r) for r in chat + comp if r["origin"] == origin]
            ax.hist(lens, bins=40, alpha=0.55, label=f"{origin} (n={len(lens):,})")
        ax.set_xlabel("approx tokens"); ax.set_ylabel("records")
        ax.set_title("Record length by origin"); ax.legend()
        fig.tight_layout(); fig.savefig(out / "length_hist.png", dpi=120)
        plt.close(fig)

    # mix composition
    if chat or comp:
        origin_counts = Counter(r["origin"] for r in chat + comp)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.pie(origin_counts.values(),
               labels=[f"{k}\n{v:,}" for k, v in origin_counts.items()],
               autopct="%1.0f%%")
        ax.set_title("Final mix composition")
        fig.tight_layout(); fig.savefig(out / "mix_composition.png", dpi=120)
        plt.close(fig)
        md.append("## Mix composition\n")
        md += [f"- {k}: {v:,}" for k, v in origin_counts.most_common()]
        if mix_stats:
            md.append(f"- achieved domain:anchor ratio: "
                      f"1:{mix_stats['achieved_ratio']}\n")

    # pair types
    pt = Counter(p["pair_type"] for p in domain_pairs)
    if pt:
        fig, ax = plt.subplots(figsize=(5, 4))
        bar(ax, pt, "Domain pair types")
        fig.tight_layout(); fig.savefig(out / "pair_types.png", dpi=120)
        plt.close(fig)
        md.append("## Domain pair types\n")
        md += [f"- {k}: {v:,}" for k, v in pt.most_common()]
        md.append("")

    if split_counts:
        md.append("## Splits\n")
        md += [f"- {k}: {v:,}" for k, v in split_counts.items()]
        md.append("")

    (out / "STATS.md").write_text("\n".join(md))
    print(f"wrote {out}/STATS.md and {len(list(out.glob('*.png')))} plots")


if __name__ == "__main__":
    main()
