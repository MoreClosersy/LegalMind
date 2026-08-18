"""QLoRA supervised fine-tuning with completion-only loss masking.

    # Smoke test (CPU, ~1 min) — also what CI runs on every push.
    uv run python -m legalmind.train.sft --config configs/train_smoke_0.6b.yaml

    # Production run (g5.xlarge).
    uv run python -m legalmind.train.sft --config configs/train_qwen3_8b.yaml

Two implementation details carry most of the weight here.

**1. Loss is computed on the completion only.**
The default in most SFT setups is to compute loss over the whole sequence,
including the prompt — so the model spends capacity learning to predict its own
input. TRL 0.18 masks the prompt via `completion_only_loss`, which requires a
*prompt-completion* dataset (separate `prompt` and `completion` columns) rather
than a chat-messages dataset. `build_dataset` below produces exactly that shape.
Note the knob is not called `assistant_only_loss` at this TRL version.

**2. The chat template is rendered here, not by TRL.**
Qwen3's template behaves in a way worth being precise about, verified against
`Qwen/Qwen3-0.6B`:

    add_generation_prompt=True, enable_thinking=True
        -> '...<|im_start|>assistant\\n'
    add_generation_prompt=True, enable_thinking=False
        -> '...<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n'

    Full conversation, either flag
        -> identical; the empty <think></think> block is always inserted.

So `enable_thinking` affects *only* the generation prompt. Rendering the prompt
ourselves with `enable_thinking=False` means the empty thinking block lives in
the prompt, the model never has to emit it, and training matches serving exactly
— provided the server also sets `enable_thinking=False`. Letting TRL apply the
template would give up that control, since it does not forward the flag.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, cast

import torch
from datasets import Dataset
from dotenv import load_dotenv
from peft import LoraConfig as PeftLoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from trl import SFTConfig, SFTTrainer

from legalmind.config import TrainConfig, load_train_config
from legalmind.train.checkpoint_sync import (
    S3CheckpointCallback,
    pull_latest_checkpoint,
    s3_uri_from_env,
)

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def render_prompt(
    tokenizer: PreTrainedTokenizerBase, instruction: str, *, enable_thinking: bool
) -> str:
    """Render the user turn plus the assistant generation prefix."""
    # `apply_chat_template` is typed as returning str | list[int] | BatchEncoding
    # because it covers both the tokenizing and non-tokenizing paths. With
    # tokenize=False it is always str.
    return cast(
        str,
        tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        ),
    )


def render_completion(tokenizer: PreTrainedTokenizerBase, response: str) -> str:
    """Render the assistant content and the turn terminator.

    The terminator matters: without it the model never learns to stop, and
    generation runs to `max_tokens` on every request.
    """
    return f"{response}{tokenizer.eos_token}"


def build_dataset(
    path: Path, tokenizer: PreTrainedTokenizerBase, *, enable_thinking: bool
) -> Dataset:
    """Build a prompt-completion dataset — the shape `completion_only_loss` needs.

    Emitting plain strings rather than message lists keeps TRL's
    `maybe_apply_chat_template` from re-rendering the conversation with default
    flags, which would silently undo the `enable_thinking=False` decision.
    """
    records = read_jsonl(path)
    if not records:
        raise ValueError(f"{path} is empty")
    return Dataset.from_list(
        [
            {
                "prompt": render_prompt(
                    tokenizer, record["instruction"], enable_thinking=enable_thinking
                ),
                "completion": render_completion(tokenizer, record["response"]),
            }
            for record in records
        ]
    )


def build_quantization_config(cfg: TrainConfig) -> Any:
    """4-bit NF4 config, or None when quantization is off.

    Imported lazily: bitsandbytes has no working macOS wheel, so importing it
    unconditionally would break the CPU smoke test and CI.
    """
    if not cfg.quantization.load_in_4bit:
        return None
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.quantization.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.quantization.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=_DTYPES[cfg.quantization.bnb_4bit_compute_dtype],
    )


def train(cfg: TrainConfig) -> str:
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = build_dataset(
        Path(cfg.data.train_path), tokenizer, enable_thinking=cfg.model.enable_thinking
    )
    eval_dataset = None
    if cfg.training.get("eval_strategy", "no") != "no":
        eval_dataset = build_dataset(
            Path(cfg.data.val_path), tokenizer, enable_thinking=cfg.model.enable_thinking
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        quantization_config=build_quantization_config(cfg),
        torch_dtype=_DTYPES[cfg.model.torch_dtype],
        attn_implementation=cfg.model.attn_implementation,
    )

    peft_config = PeftLoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=list(cfg.lora.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )

    # A config that asks for W&B but finds no API key does not fail — the wandb
    # client falls back to an interactive login prompt and blocks on stdin.
    # run_on_gpu.sh is meant to run unattended in tmux; a hang there burns
    # billed GPU time silently until someone happens to check on it. Downgrading
    # to "none" here keeps that promise: preflight already warns that an unset
    # key means "trains but logs nowhere", so this is what makes that true
    # rather than "trains but logs nowhere, unless it hangs first".
    report_to = cfg.tracking.report_to
    if "wandb" in report_to and not os.environ.get("WANDB_API_KEY"):
        print(
            "WANDB_API_KEY not set; falling back to report_to='none' instead of "
            "letting wandb prompt for interactive login and hang the run.",
            file=sys.stderr,
        )
        report_to = "none"

    sft_config = SFTConfig(
        max_length=cfg.data.max_seq_length,
        packing=cfg.data.packing,
        completion_only_loss=cfg.data.completion_only_loss,
        run_name=cfg.tracking.run_name,
        report_to=report_to,
        **cfg.training,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # Log the resolved config and library versions so a W&B run is reproducible
    # from its own record rather than from whatever the working tree held.
    if trainer.args.report_to and "wandb" in trainer.args.report_to:
        import transformers
        import trl

        trainer.accelerator.init_trackers(
            project_name="legalmind",
            config={
                "resolved_config": cfg.training,
                "model": cfg.model.name,
                "lora_r": cfg.lora.r,
                "completion_only_loss": cfg.data.completion_only_loss,
                "versions": {
                    "torch": torch.__version__,
                    "transformers": transformers.__version__,
                    "trl": trl.__version__,
                },
            },
        )

    # Checkpoint mirroring is opt-in by environment variable, so the CPU smoke
    # test in CI and a laptop run are unaffected while a GPU run is protected.
    s3_uri = s3_uri_from_env()
    resume_from: str | bool = False
    if s3_uri:
        trainer.add_callback(S3CheckpointCallback(s3_uri))
        recovered = pull_latest_checkpoint(s3_uri, Path(cfg.output_dir))
        if recovered is not None:
            resume_from = str(recovered)

    trainer.train(resume_from_checkpoint=resume_from or None)
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"adapter saved to {cfg.output_dir}", file=sys.stderr)
    return cfg.output_dir


def main(argv: list[str] | None = None) -> int:
    load_dotenv(override=True)
    parser = argparse.ArgumentParser(description="QLoRA SFT for LegalMind.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)

    cfg = load_train_config(args.config)
    print(f"model={cfg.model.name} train={cfg.data.train_path}", file=sys.stderr)
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
