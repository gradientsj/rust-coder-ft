# rust-coder-ft

Full BF16 (+ FP8 via Transformer Engine) fine-tune of a dense Qwen3-Coder ~7-8B
model, specialized for idiomatic Rust. Axolotl + FSDP full-shard across 4x H100.

## Hardware (verified Phase 0)

- 4x NVIDIA H100 80GB HBM3 (SXM), **fully NVLinked** — all pairs `NV18`
  (~900 GB/s bidirectional). FSDP `FULL_SHARD` with prefetch is the right config.
- Single NUMA node (CPUs 0-103), 885 GB RAM, 11 TB disk.
- Driver 580.105.08, CUDA toolkit 12.8 (nvcc 12.8.93), PyTorch 2.7.0 (cu128).
- Rust: rustc/cargo 1.96.0 via rustup (`source ~/.cargo/env`).

## Layout

```
configs/        qwen3coder-8b-fft.yaml (primary), qwen3coder-27b-fft.yaml (later)
data/scrape/    crates.io tarball harvest, license filter (MIT/Apache-2.0/BSD only),
                provenance logging, exact + MinHash dedup, quality filters
data/prepare/   domain pair building, HF anchor loading, mixing, ChatML formatting
data/splits/    train / val / heldout-domain / heldout-general jsonl
train/          accelerate launch wrapper + FSDP config
eval/           cargo compile/test/clippy harness + HumanEval/MBPP retention
export/         FSDP shard consolidation -> HF format, FP8 export
serve/          vLLM local endpoint
```

## Principles

- **Full fine-tune, not LoRA.** BF16 master weights, FP8 matmuls via Transformer Engine.
- **License filtering is mandatory.** Only MIT / Apache-2.0 / BSD source enters the
  corpus; license + provenance logged per file. GPL/AGPL/unlicensed are hard-excluded.
- **Forgetting guardrail.** Domain scrape blended with an HF code anchor at a
  configurable ratio (default 1:2 domain:anchor). HumanEval/MBPP run before AND
  after FT — the delta is the success/failure signal.
- **Compile-pass-rate is the primary domain metric** (cargo build/test/clippy on
  held-out Rust), not perplexity.

## Workflow

1. `data/scrape` -> `data/prepare` -> `data/splits/*.jsonl`
2. `train/train.sh configs/qwen3coder-8b-fft.yaml`
3. `eval/harness.py` (before + after)
4. `export/merge_and_save.py` -> `export/to_fp8.py` -> `serve/vllm_serve.sh`
