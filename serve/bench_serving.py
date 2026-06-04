#!/usr/bin/env python3
"""BF16 vs FP8 serving benchmark for the rust-coder model.

Two identical H100s, identical weights (export/qwen3-8b-ft), only precision
differs: BF16 (port 8001) vs FP8 compressed-tensors (port 8002).

  .venv/bin/python serve/bench_serving.py        # (any venv with openai pkg)

Measures, per concurrency level in --sweep:
  TTFT p50/p99        time to first token (streaming)
  e2e p50/p99         full-request latency
  out tok/s/req       per-request decode speed
  aggregate tok/s     total output tokens / wall time

Plus a QUALITY check: greedy completions for --quality-n Rust problems from
rl/problems/eval.jsonl on BOTH endpoints, judged by the cargo harness —
same-weights pass-rate delta isolates the FP8 quantization cost.

Writes eval/results/serving_bench.json.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI

REPO = Path(__file__).resolve().parent.parent
ENDPOINTS = {"bf16": "http://127.0.0.1:8001/v1",
             "fp8": "http://127.0.0.1:8002/v1"}
MODELS = {"bf16": "rc-bf16", "fp8": "rc-fp8"}


def pctl(xs, p):
    xs = sorted(xs)
    return xs[min(int(p / 100 * len(xs)), len(xs) - 1)]


async def one_request(client, model, prompt, max_tokens):
    t0 = time.perf_counter()
    ttft = None
    n_tok = 0
    stream = await client.chat.completions.create(
        model=model, max_tokens=max_tokens, temperature=0, stream=True,
        messages=[{"role": "user", "content": prompt}])
    chunks = []
    async for ch in stream:
        if ch.choices and ch.choices[0].delta.content:
            if ttft is None:
                ttft = time.perf_counter() - t0
            chunks.append(ch.choices[0].delta.content)
            n_tok += 1
    return {"ttft": ttft or 0.0, "e2e": time.perf_counter() - t0,
            "n_tok": n_tok, "text": "".join(chunks)}


async def load_test(tag, prompts, concurrency, max_tokens):
    client = AsyncOpenAI(base_url=ENDPOINTS[tag], api_key="EMPTY")
    sem = asyncio.Semaphore(concurrency)
    results = []

    async def worker(p):
        async with sem:
            results.append(await one_request(client, MODELS[tag], p, max_tokens))

    t0 = time.perf_counter()
    await asyncio.gather(*[worker(p) for p in prompts])
    wall = time.perf_counter() - t0
    total_tok = sum(r["n_tok"] for r in results)
    return {
        "concurrency": concurrency, "n_requests": len(prompts),
        "wall_s": round(wall, 2),
        "ttft_p50_ms": round(pctl([r["ttft"] for r in results], 50) * 1000, 1),
        "ttft_p99_ms": round(pctl([r["ttft"] for r in results], 99) * 1000, 1),
        "e2e_p50_s": round(pctl([r["e2e"] for r in results], 50), 2),
        "e2e_p99_s": round(pctl([r["e2e"] for r in results], 99), 2),
        "tok_per_s_per_req_p50": round(pctl(
            [r["n_tok"] / max(r["e2e"] - r["ttft"], 1e-6) for r in results], 50), 1),
        "aggregate_tok_per_s": round(total_tok / wall, 1),
    }


async def quality(tag, problems, max_tokens):
    client = AsyncOpenAI(base_url=ENDPOINTS[tag], api_key="EMPTY")
    sem = asyncio.Semaphore(16)

    async def gen(p):
        async with sem:
            r = await one_request(client, MODELS[tag], p["prompt"], max_tokens)
            return r["text"]

    texts = await asyncio.gather(*[gen(p) for p in problems])
    sys.path.insert(0, str(REPO / "eval"))
    sys.path.insert(0, str(REPO / "rl"))
    from eval_bank import RUST_BLOCK, decl_imports  # noqa: E402
    from compile_check.runner import check_many  # noqa: E402

    codes = [(RUST_BLOCK.search(t).group(1).strip()
              if RUST_BLOCK.search(t) else None) for t in texts]
    idx = [i for i, c in enumerate(codes) if c]
    res1 = check_many([{"code": codes[i], "tests": problems[i]["tests"]}
                       for i in idx], max_workers=32)
    results = dict(zip(idx, res1))
    idx2 = [i for i in idx if not results[i].compiles]
    if idx2:
        res2 = check_many(
            [{"code": decl_imports(problems[i]["declaration"]) + "\n\n" + codes[i],
              "tests": problems[i]["tests"]} for i in idx2], max_workers=32)
        for i, r in zip(idx2, res2):
            if r.compiles:
                results[i] = r
    n = len(problems)
    n_pass = sum(1 for r in results.values() if r.compiles and r.tests_passed)
    n_comp = sum(1 for r in results.values() if r.compiles)
    return {"n": n, "pass_rate": round(n_pass / n, 4),
            "compile_rate": round(n_comp / n, 4)}


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-requests", type=int, default=64)
    ap.add_argument("--sweep", default="1,8,32")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--quality-n", type=int, default=60)
    args = ap.parse_args()

    problems = [json.loads(l)
                for l in (REPO / "rl/problems/eval.jsonl").open()]
    prompts = [p["prompt"] for p in problems][: args.n_requests]

    report: dict = {"max_tokens": args.max_tokens,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "perf": {}, "quality": {}}
    for tag in ("bf16", "fp8"):
        report["perf"][tag] = []
        for c in [int(x) for x in args.sweep.split(",")]:
            r = await load_test(tag, prompts, c, args.max_tokens)
            report["perf"][tag].append(r)
            print(f"{tag} c={c}: {json.dumps(r)}")

    for tag in ("bf16", "fp8"):
        q = await quality(tag, problems[: args.quality_n], 768)
        report["quality"][tag] = q
        print(f"{tag} quality: {json.dumps(q)}")

    out = REPO / "eval/results/serving_bench.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
