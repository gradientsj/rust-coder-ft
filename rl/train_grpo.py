#!/usr/bin/env python3
"""GRPO on cargo-verified rewards: the Phase 6 RL trainer.

Run with the RL venv (trl 1.5.1 + vllm 0.18 + torch 2.10 — separate from the
SFT venv on purpose):

  # smoke (16 problems, 8 steps, no saves):
  .venv-rl/bin/accelerate launch --config_file rl/fsdp_grpo.yaml \
      rl/train_grpo.py --smoke
  # PoC run:
  .venv-rl/bin/accelerate launch --config_file rl/fsdp_grpo.yaml \
      rl/train_grpo.py --max-steps 120

Design (research-verified against the installed TRL source):
  - vLLM COLOCATE mode: engine in-process on each training GPU, weight sync
    via direct load_weights after FSDP full_tensor — no NCCL bridge, no
    second process group. gpu_memory_utilization=0.25 + sleep mode keeps
    policy(FSDP states ~32GB/GPU) + vLLM + activations under 80GB.
  - rewards: rl/reward.py graded cargo judge; the `tests` dataset column
    reaches the reward fn via **reward_kwargs (verified call convention).
  - rollouts forced NON-thinking via chat_template_kwargs, matching the SFT
    format and serve-time configuration.
  - LR 1e-6 (full-FT RL band), beta 0 + dapo loss (TRL 1.5 defaults).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "rl"))

from datasets import load_dataset  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402

from reward import make_grpo_reward_fn  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=str(REPO / "export/qwen3-8b-ft"))
    ap.add_argument("--problems", default=str(REPO / "rl/problems/train.jsonl"))
    ap.add_argument("--out", default=str(REPO / "outputs/qwen3-8b-grpo"))
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--smoke", action="store_true",
                    help="16 problems, 8 steps, no checkpoints")
    args = ap.parse_args()

    ds = load_dataset("json", data_files=args.problems, split="train")
    if args.smoke:
        ds = ds.select(range(16))
    # conversational format: TRL applies the tokenizer chat template
    ds = ds.map(lambda r: {
        "prompt": [{"role": "user", "content": r["prompt"]}]})

    cfg = GRPOConfig(
        output_dir=args.out,
        max_steps=8 if args.smoke else args.max_steps,
        seed=17,

        # rollouts
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.25,
        vllm_enable_sleep_mode=True,        # free vLLM VRAM during optimizer step
        vllm_max_model_length=2048,
        num_generations=8,                  # GRPO group size
        max_prompt_length=1024,
        max_completion_length=768,
        temperature=0.9,
        chat_template_kwargs={"enable_thinking": False},
        mask_truncated_completions=True,

        # optimization (full FT)
        per_device_train_batch_size=8,      # completions/device -> 4 prompts/step global
        gradient_accumulation_steps=1 if args.smoke else 4,
        learning_rate=1e-6,
        warmup_steps=0 if args.smoke else 5,
        lr_scheduler_type="constant",
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=True,
        model_init_kwargs={"dtype": "bfloat16"},

        # logging / saving
        logging_steps=1,
        save_strategy="no" if args.smoke else "steps",
        save_steps=40,
        save_total_limit=2,
        report_to=[],
        log_completions=True,
        num_completions_to_print=1,
    )

    trainer = GRPOTrainer(
        model=args.model,
        args=cfg,
        train_dataset=ds,
        reward_funcs=make_grpo_reward_fn(test_field="tests"),
    )
    trainer.train()
    if not args.smoke:
        trainer.save_model(args.out)


if __name__ == "__main__":
    main()
