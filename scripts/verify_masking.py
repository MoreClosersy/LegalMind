"""Prove that prompt tokens are excluded from the training loss.

    uv run python scripts/verify_masking.py --config configs/train_smoke_0.6b.yaml

Run in CI after the smoke training step. Completion-only loss masking is the
kind of thing that is easy to claim, easy to get wrong, and completely silent
when it breaks — the run still converges, it just wastes capacity learning to
predict its own input. A refactor that changed the dataset shape from
prompt/completion to messages would disable masking without any error, and
nothing but this check would notice.

It also verifies the boundary lands where the Qwen3 template puts it: the empty
`<think></think>` block belongs to the *prompt*, so the model is never trained
to emit it. Getting that wrong would make training and serving disagree.

Exits non-zero if masking is off or the boundary is misplaced.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from legalmind.config import load_train_config
from legalmind.train.sft import build_dataset

IGNORE_INDEX = -100


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train_smoke_0.6b.yaml"))
    args = parser.parse_args(argv)

    cfg = load_train_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    dataset = build_dataset(
        Path(cfg.data.train_path), tokenizer, enable_thinking=cfg.model.enable_thinking
    )

    model = AutoModelForCausalLM.from_pretrained(cfg.model.name, torch_dtype=torch.float32)
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir="/tmp/legalmind-verify-masking",
            max_length=cfg.data.max_seq_length,
            packing=False,
            completion_only_loss=cfg.data.completion_only_loss,
            report_to="none",
            max_steps=1,
            per_device_train_batch_size=1,
        ),
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    prepared = trainer.train_dataset
    if "completion_mask" not in prepared.column_names:
        print(
            "FAIL: no completion_mask column — TRL is computing loss over the whole "
            "sequence. Check that build_dataset still emits prompt/completion columns.",
            file=sys.stderr,
        )
        return 1

    batch = trainer.data_collator([prepared[0]])
    labels = batch["labels"][0]
    input_ids = batch["input_ids"][0]

    ignored = int((labels == IGNORE_INDEX).sum())
    total = len(labels)
    kept_positions = (labels != IGNORE_INDEX).nonzero()
    if ignored == 0:
        print(
            "FAIL: no tokens are masked — prompt tokens are contributing to loss.", file=sys.stderr
        )
        return 1
    if len(kept_positions) == 0:
        print("FAIL: every token is masked — nothing would train.", file=sys.stderr)
        return 1

    boundary = int(kept_positions[0])
    prompt_tail = tokenizer.decode(input_ids[max(0, boundary - 12) : boundary])
    completion_head = tokenizer.decode(input_ids[boundary : boundary + 12])

    print(f"masked (no loss): {ignored}/{total} tokens")
    print(f"trained (loss):   {total - ignored}/{total} tokens")
    print(f"prompt tail   -> {prompt_tail!r}")
    print(f"completion head -> {completion_head!r}")

    # The generation prefix must be on the prompt side of the boundary.
    if "<|im_start|>assistant" not in prompt_tail:
        print(
            "FAIL: the assistant generation prefix is not in the masked prompt. "
            "The loss boundary is in the wrong place.",
            file=sys.stderr,
        )
        return 1
    if cfg.model.enable_thinking is False and "</think>" not in prompt_tail:
        print(
            "FAIL: enable_thinking is off, so the empty <think></think> block should "
            "be part of the prompt. It is not — training and serving would disagree.",
            file=sys.stderr,
        )
        return 1
    if "<|im_start|>" in completion_head:
        print("FAIL: a chat control token leaked into the trained span.", file=sys.stderr)
        return 1

    print("\nOK: loss is computed on the assistant completion only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
