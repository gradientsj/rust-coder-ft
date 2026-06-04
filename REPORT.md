# rust-coder-ft — Build Report

Step-by-step record of how this repo was built, what was decided, what broke,
and why. Companion to `README.md` (which describes *what the repo is*; this
describes *how it got here*). Built phase-gated: each phase was verified by
the owner before the next began.

**Goal:** full BF16 fine-tune (not LoRA) of a dense ~8B model specialized for
idiomatic Rust, on a Lambda 4x H100 80GB node — with a license-clean data
pipeline, a compile-based eval that can't be gamed by plausible-looking code,
a catastrophic-forgetting guardrail, and an FP8 serving path.

---

## Phase 0 — Hardware diagnostics (before any code)

**Why first:** the GPU interconnect dictates FSDP strategy, and a missing
toolchain is cheaper to discover before anything depends on it.

Findings:

| Check | Result | Consequence |
|---|---|---|
| GPUs | 4x H100 80GB HBM3 (SXM), idle | 320 GB pool; full-FT 8B fits comfortably |
| `nvidia-smi topo -m` | every pair `NV18` (18 bonded NVLinks, ~900 GB/s) | FSDP `FULL_SHARD` + `reshard_after_forward` is cheap; no hybrid sharding, no NCCL tuning |
| NUMA | single node (CPUs 0–103) | safe to enable CPU affinity in accelerate |
| PyTorch (system) | 2.7.0, CUDA OK, 4 devices | later found unusable for axolotl — see Phase 1 |
| CUDA toolkit | nvcc 12.8 (driver 580 / CUDA 13) | matches torch cu128 wheels; can build kernels from source |
| Rust | **missing** → installed rustup (rustc/cargo 1.96.0) | needed for the cargo-based eval; clippy added later in Phase 4 |

## Phase 1 — Scaffold + environment

Repo skeleton (`configs/ data/ train/ eval/ export/ serve/`), `pyproject.toml`,
and a reproducible `setup_env.sh`. Two real problems surfaced and were fixed:

1. **No axolotl release supports torch 2.7.0** (the system torch). Resolver
   showed: 0.9.x pins `torch==2.6.0`, 0.13–0.16 pin `==2.8.0`, 0.17 wants
   `>=2.9.1`. **Decision:** venv-local **torch 2.8.0+cu128** + **axolotl
   0.16.1** (one minor behind latest; pins transformers 5.5.0 and
   flash-attn 2.8.3, which has mature prebuilt wheels for that torch). A
   `constraints.txt` pin keeps the resolver from ever silently replacing torch.
2. **transformer-engine failed to build**: `cudnn.h: No such file` — cuDNN
   headers live in the pip `nvidia-cudnn-cu12` package, not system paths.
   Fixed by exporting `CUDNN_PATH`/`CPLUS_INCLUDE_PATH` into the build (now in
   `setup_env.sh`).

Verification was executable, not import-only: a TE FP8 forward/backward on
GPU (`check_fp8_support()=True`) and a flash-attn BF16 causal kernel both ran.

## Phase 2 — Data pipeline (tested end-to-end on a ~50-file sample)

Design principles: **license filtering is a hard gate with per-file
provenance**, dedup before pair-building, and a domain:anchor blend to prevent
catastrophic forgetting.

Pipeline (each stage a standalone script under `data/`):

1. `scrape/github_harvest.py` — pulls **published crates.io tarballs** (not
   git HEAD) because published versions carry an authoritative SPDX license
   field per version. Curated allowlist in `sources.yaml` (rust-lang,
   tokio-rs, serde-rs, clap, rayon, hyperium). License policy is *stricter
   than SPDX OR-semantics*: every token of the expression must be allowlisted
   (`MIT OR GPL-3.0` is rejected outright), unknown/missing licenses rejected,
   per-file `SPDX-License-Identifier` headers that contradict the crate grant
   drop the file. Every kept file gets a provenance record (crate, version,
   license, sha256, source URL). Respects the crates.io 1 req/s policy.
2. `scrape/filters.py` — size/line-shape/alnum-fraction quality gates,
   generated-code markers (bindgen/prost/"DO NOT EDIT"), then **exact dedup**
   (sha256 of normalized text) and **MinHash near-dedup** (datasketch LSH,
   7-token shingles, Jaccard 0.85). Emits a funnel count at every stage.
3. `prepare/build_domain.py` — converts files to training pairs. Uses a
   character-level **code-mask scanner** (handles nested block comments, raw
   strings `r#"…"#`, byte strings, char-literal vs lifetime disambiguation) so
   brace-matching never trips on `{` inside strings/comments. Extracts
   documented functions into two pair types: `sig_doc_impl` (doc+signature+
   `todo!()` stub → full implementation) and `instruction_code` (NL
   instruction synthesized from the doc → implementation).
   *Verified:* 272/272 extracted functions structurally complete (2 flagged by
   a naive brace-counter were braces inside doc-comments, i.e. checker noise).
4. `prepare/load_hf_anchor.py` — anchors against forgetting. **All slugs
   verified against the live Hub first** (a stated project constraint):
   `ammarnasr/the-stack-rust-clean` (raw Rust, completion-format),
   `ise-uiuc/Magicoder-OSS-Instruct-75K` (code instruct),
   `allenai/tulu-3-sft-mixture` (general instruct). Gated bigcode sets were
   rejected (the-stack-v2 doesn't even ship file contents). Streaming +
   `take(n)` — nothing fully downloads.
5. `prepare/mix.py` — domain:anchor default 1:2, anchor side split
   40/40/20 across code-instruct/general-instruct/raw-Rust. Reports the
   *achieved* ratio when pools are small instead of failing.
6. `prepare/format_chatml.py` — emits Axolotl-ready jsonl. Messages stay in
   `messages` format; the model tokenizer's own ChatML template is applied by
   Axolotl at train time (no hand-rendered templates to drift). Splits:
   train/val plus `heldout_domain` (held out at **file** level so near-dupes
   of a training file can't leak into eval) and `heldout_general`
   (capped at 10% of the anchor pool — an early absolute-N bug ate the whole
   pool at tiny scale and was fixed).
7. `prepare/stats.py` — audit artifacts: license breakdown, dedup funnel,
   length histograms, mix composition, pair types → `data/splits/stats/`.

End-to-end on 3 crates: 52 files → 49 after filters → **272 pairs**; all
splits verified loadable via `datasets`; one pair rendered through the real
Qwen tokenizer (960 tokens, correct ChatML).

**Model decision forced here:** there is **no dense 7–8B Qwen3-Coder** on the
Hub (that family is MoE-only — verified live). Chosen: **`Qwen/Qwen3-8B`**
dense (newest generation; code specialization is this project's job). The
"27B" scale-up config targets `Qwen/Qwen3-32B` for the same reason.

## Phase 3 — Training configs + 10-step overfit proof

`configs/qwen3coder-8b-fft.yaml`: full FT, bf16, FSDP2 (`fsdp_version: 2`)
full-shard with activation checkpointing, `Qwen3DecoderLayer` wrap (class name
verified in installed transformers), seq 4096 + sample packing, LR 1.5e-5
warmup+cosine, effective batch 64 seqs (2 micro × 4 GPU × 8 accum),
checkpoints every 25 steps. Memory math in the config header: ~32 GB/GPU
state + activations — verified within 1 GiB by the sanity run.

`train/fsdp_config.yaml` deliberately does **not** set `distributed_type:
FSDP` — FSDP is owned by the axolotl config; defining it in both places is
the classic conflict. `train.sh` wraps `accelerate launch` and sets
`AXOLOTL_DO_NOT_TRACK=1` (also works around the 0.16.1 wheel shipping without
its telemetry whitelist file).

**Overfit-on-one-batch sanity** (8 records = exactly one global batch,
constant LR, packing off — every step trains the same batch):

| step | 1 | 2 | 3 | 4 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|---|
| loss | 0.919 | 0.424 | 0.192 | 0.059 | 0.010 | 0.0003 | **0.0001** |

Monotonic collapse, decaying grad-norms, no NaN, 32.9 GiB/GPU. The whole
stack (template → masking → FSDP2 → flash-attn → fused AdamW) is wired right.

**FP8 experiment** (same harness, `fp8: true` = axolotl's torchao-float8
path): loss curve matched BF16 step-for-step (numerically sound; one config
land-mine found: FP8+FSDP2 requires `cpu_ram_efficient_loading: false`) but
throughput **dropped 37%** — torchao float8 needs `torch.compile` to win.
**Decision:** run #1 trains BF16; a compiled-FP8 50-step shootout gates any
switch; Transformer-Engine FP8 remains a serving-side story. *FP8 was
measured, not assumed.*

## Phase 4 — Eval harness

**Core design problem:** extracted crate functions usually reference private
crate internals, so "does the generation compile standalone" would be ~0 for
any model. Solution: **reference-compile filtering** — a pair enters the
compile-pass metric only if its *reference* solution compiles standalone;
everything else is scored by edit similarity. The metric stays honest.

- `eval/compile_check/runner.py` — per-candidate temp cargo project,
  `cargo build --offline --message-format=json` (error/warning parse),
  `cargo clippy` clean-rate (the idiomatic-Rust signal), `cargo test` inside
  an `unshare -rn` **network namespace** when the kernel allows (recorded per
  result), wall-clock kill of the whole process group. ~0.2 s/verdict,
  thread-parallel. Self-test: 4/4 exact verdicts (incl. clippy catching a
  `&Vec<i32>` argument and a wrong-implementation test failure).
  Also the future GRPO reward backend (Phase 6).
- `eval/domain_eval.py` — heldout pairs → greedy non-thinking generation →
  cargo verdicts on the compile subset + similarity on all.
  Base-model smoke (12 pairs): compile-pass 1/3, clippy-clean 1/3,
  similarity 0.28, exact 0 → clear headroom for the FT to demonstrate gains.
- `eval/general_eval.py` — HumanEval + MBPP via lm-eval, 4-GPU data parallel.
  Absolute numbers are protocol-dependent; the **pre/post delta under an
  identical protocol** is the metric of record.
- `eval/harness.py` — runs both, merges into one JSON report with the
  retention delta vs. baseline.

Three stale-dependency bugs found by smoke tests and fixed: transformers 5.x
`apply_chat_template` return-type change; lm_eval 0.4.11↔datasets 4.5
incompatibility (→0.4.12); evaluate 0.4.1 breaking lm_eval's humaneval metric
module (→0.4.6). Plus one performance fix: `--batch_size auto` probed its way
to a 5.5-hour ETA; explicit batch 64 OOM'd on the KV cache; **batch 16**
landed generation in **2:26**.

**Pre-FT baseline (the "BEFORE"):**

| benchmark | pass@1 |
|---|---|
| HumanEval | **0.628** |
| MBPP | **0.658** |

## Phase 5 — Export / quantize / serve

Researched first (three parallel agents over the installed sources + live
docs), then written, then **proven live with base Qwen3-8B as a stand-in** so
the path is known-good before a real checkpoint exists.

Environment isolation (the key operational decision): three venvs, because
the tools pin mutually incompatible torch/transformers —

| venv | contents | why separate |
|---|---|---|
| `.venv` | torch 2.8.0+cu128, axolotl 0.16.1, transformers 5.5 | training |
| `.venv-quant` | llm-compressor 0.11 (pins torch 2.11, transformers 4.57) | quantization |
| `.venv-serve` | vLLM 0.22 (own torch 2.11+cu130) | serving; co-installing with llm-compressor made pip backtrack to a 2023 vllm sdist |

- `export/merge_and_save.py` — consolidates a run/checkpoint into a servable
  HF dir. Two auto-detected cases, both grounded in the installed trainer
  source: (A) `FULL_STATE_DICT` saves are already consolidated → copy, strip
  optimizer/rng debris, fix the FSDP-prefixed `architectures` field that
  axolotl only fixes on final saves; (B) DCP-sharded checkpoints → axolotl's
  `merge_fsdp_weights` (bf16 cast, 5 GB shards + index) + config/tokenizer
  from `--tokenizer-from`. Ends with a structural validation + optional GPU
  generation. *Verified on the real Qwen3-8B snapshot: 8.19 B params, clean
  reload, coherent generation.*
- `export/to_fp8.py` — llm-compressor `FP8_DYNAMIC` (per-channel RTN weights
  + dynamic per-token activations), **data-free** (no calibration set),
  `lm_head` kept bf16. *Verified: 16.4 GB → 9.44 GB, `quant_method:
  compressed-tensors` in config — which vLLM auto-detects, no flag needed.*
- `serve/vllm_serve.sh` — local-only (127.0.0.1) OpenAI-compatible endpoint;
  thinking disabled server-wide (`--default-chat-template-kwargs
  '{"enable_thinking": false}'`) to match the non-thinking SFT format;
  `/health` readiness probe; `--smoke` mode sends a Rust prompt through
  `/v1/chat/completions` and tears down. One env bug fixed: vLLM's inductor
  compile needs `ninja` (installed into `.venv-serve`).

**Serve smoke result (full chain, live):** FP8 model served on
127.0.0.1:8000, `/v1/chat/completions` asked for a case-insensitive
`is_palindrome` in Rust, the response was fed through this repo's own cargo
judge: **compiles ✓ clippy-clean ✓ tests pass ✓** (netns sandbox). Serving
stack and eval stack compose.

## Bugs found & fixed (running list)

| # | Bug | Fix |
|---|---|---|
| 1 | axolotl ✗ torch 2.7 (no release supports it) | venv torch 2.8.0 + axolotl 0.16.1 |
| 2 | TE build: `cudnn.h` not found | `CUDNN_PATH` → pip cudnn pkg |
| 3 | axolotl wheel missing telemetry whitelist → crash | `AXOLOTL_DO_NOT_TRACK=1` |
| 4 | FP8 + FSDP2 ✗ `cpu_ram_efficient_loading` | explicit error, set false |
| 5 | `heldout_general` absolute cap ate tiny anchor pool | 10%-of-pool cap |
| 6 | transformers 5.x `apply_chat_template` returns BatchEncoding | render-then-tokenize |
| 7 | lm_eval 0.4.11 ✗ datasets 4.5 (`use_auth_token`) | lm_eval 0.4.12 |
| 8 | evaluate 0.4.1 broke humaneval metric import | evaluate 0.4.6 |
| 9 | lm-eval `--batch_size auto` → 5.5 h ETA; 64 → OOM | batch 16 → 2:26 |
| 10 | pip co-install vllm+llmcompressor → resolver backtracked to vllm 0.1.3 sdist | separate venvs |
| 11 | vLLM startup: `ninja` missing for inductor compile | pip install ninja **and** put `.venv-serve/bin` on PATH (script invoked vllm by absolute path, so the engine subprocess couldn't find ninja) |
| 12 | rustup minimal profile lacks clippy | `rustup component add clippy` |
| 13 | smoke curl: backticked Rust signature inside double-quoted JSON underwent shell command substitution — model received a truncated prompt | payload via quoted heredoc |

## Current state & next steps

Done: Phases 0–5 scaffolded, tested, and pushed (github.com/gradientsj/rust-coder-ft).
Baseline retention numbers locked. Tiny-scale data pipeline proven end to end.

Next, in order:
1. **Scale the scrape** (full allowlist, no per-crate caps) → rebuild splits
   at real volume (heldout_domain populates at scale), regenerate stats.
2. **Real training run** (BF16 first; compiled-FP8 shootout before any long run).
3. **Post-FT eval**: `eval/harness.py` → compile-pass/clippy gains + retention
   delta vs. baseline 0.628/0.658.
4. **Export + serve the FT model** through the now-proven Phase 5 chain.
5. **Phase 6 (planned): GRPO RL** with the cargo runner as verifiable reward;
   LeetCode-style problem bank; owner's manual-submission calibration pass
   (~200 problems) to measure local-judge ↔ official-judge agreement.

## Instance shutdown checklist ($16/h — run before every teardown)

1. `git status` clean + `git push` — code, configs, eval JSONs, stats all live in
   the GitHub repo. (Everything else below is only needed once real artifacts exist.)
2. **Fine-tuned checkpoint** → `export/upload_hub.sh export/qwen3-8b-ft gradientsj/rust-coder-8b`
   (private HF repo; needs one-time `hf auth login`). THE irreplaceable artifact.
3. FP8 export → same script, second repo (optional — 10 min to regenerate from #2).
4. Commit `eval/results/` + `data/splits/stats/` (small JSONs/PNGs) if new.
5. Full-scale `data/work/` corpus (if the big scrape ran): either re-scrape next
   time (deterministic pipeline, ~hours) or `hf upload` it as a private dataset
   repo if you want to save the time.
6. Do NOT bother saving: `~/.cache/huggingface` (re-downloads), `.venv*`
   (`setup_env.sh` rebuilds), logs, `outputs/` intermediate checkpoints
   (superseded by the exported final).

Restore on a fresh instance: clone repo → `setup_env.sh` → `hf download` the
model repos. ~20 minutes to fully operational.
