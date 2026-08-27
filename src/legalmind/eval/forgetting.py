"""Catastrophic-forgetting check: what did the fine-tune cost elsewhere?

    uv run python -m legalmind.eval.forgetting --out eval_results/forgetting.json

Fine-tuning on 5,509 legal instruction pairs is not free. LoRA touches every
linear projection in the model, and a model that gets better at legal exposition
can get worse at arithmetic, at following unrelated instructions, or at general
factual recall. That regression is invisible if the only thing measured is the
task the model was trained on — which is the normal way fine-tuning writeups are
presented, and the reason this file exists.

The measurement is a **delta, not a score**. Nobody cares what a 8B model scores
on MMLU; what matters is arm A minus arm C on identical items. Arm B is included
as a control: it shares arm A's weights, so if B and A differ materially the
harness is noisy and the A-to-C delta means nothing.

Two design points worth stating.

**The subjects exclude law.** A legal fine-tune improving on
`professional_law` would be a nice result, but it would be a result about the
training objective, not about forgetting. Mixing it into the average would let a
gain on the trained domain mask a loss everywhere else — which is precisely the
effect being looked for.

**The eval decontaminates its own benchmark before reporting.** The training set
was decontaminated against LegalBench only, because LegalBench was the only
benchmark in play when that ran. Quoting an MMLU delta from a model whose
training data was never checked against MMLU would repeat, in a smaller way, the
exact mistake this project was restructured to avoid. So the same 13-gram check
runs against the evaluated items on every invocation, and **refuses to write a
result file** if any overlap is found rather than footnoting it. Opting out
takes an explicit `--skip-contamination-check`, because a check that is easy to
forget to turn on is a check that does not exist.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from datasets import load_dataset

from legalmind.data.decontaminate import _shingle_hashes
from legalmind.data.filter import read_jsonl
from legalmind.eval.arms import Arm, ArmId, build_arms
from legalmind.eval.generate import (
    DEFAULT_BASE_URL,
    DEFAULT_CONCURRENCY,
    GenerationClient,
    strip_thinking_block,
)
from legalmind.eval.refusal import Rate

MMLU_REPO = "cais/mmlu"

# Deliberately non-legal, and deliberately spread across reasoning types:
# arithmetic, science recall, formal logic, and everyday commonsense. A single
# subject would measure one narrow capability and get reported as "general".
SUBJECTS: tuple[str, ...] = (
    "high_school_mathematics",
    "college_computer_science",
    "formal_logic",
    "high_school_biology",
    "moral_scenarios",
    "world_religions",
)

CHOICES = ("A", "B", "C", "D")

PROMPT_TEMPLATE = """\
{question}

A. {a}
B. {b}
C. {c}
D. {d}

Answer with a single letter (A, B, C, or D) and nothing else."""

# Same shape as the LegalBench extractor: a bare letter, or a letter leading the
# first line. Anything else is non-compliance rather than a wrong answer, so a
# formatting regression cannot masquerade as a capability regression.
_LETTER = re.compile(r"^\s*(?:answer\s*[:\-]?\s*)?\(?([ABCD])\)?(?:\W|$)", re.IGNORECASE)


@dataclass(frozen=True)
class Item:
    subject: str
    question: str
    choices: tuple[str, str, str, str]
    answer: str  # "A".."D"

    def prompt(self) -> str:
        return PROMPT_TEMPLATE.format(
            question=self.question,
            a=self.choices[0],
            b=self.choices[1],
            c=self.choices[2],
            d=self.choices[3],
        )


def extract_letter(text: str) -> str | None:
    text = strip_thinking_block(text)
    for line in text.strip().splitlines():
        if not line.strip():
            continue
        match = _LETTER.match(line.strip().lstrip("*_`# "))
        return match.group(1).upper() if match else None
    return None


def load_items(*, per_subject: int) -> list[Item]:
    """Load a deterministic prefix of each subject's test split.

    A prefix rather than a random sample: the item set has to be identical
    across arms and across reruns, or the delta is measuring which questions
    happened to be drawn rather than what the fine-tune did.
    """
    items: list[Item] = []
    for subject in SUBJECTS:
        split = load_dataset(MMLU_REPO, subject, split="test")
        for row in split.select(range(min(per_subject, len(split)))):
            choices = list(row["choices"])
            if len(choices) != 4:
                continue
            items.append(
                Item(
                    subject=subject,
                    question=str(row["question"]),
                    choices=(str(choices[0]), str(choices[1]), str(choices[2]), str(choices[3])),
                    answer=CHOICES[int(row["answer"])],
                )
            )
    return items


def check_contamination(items: list[Item], train_path: Path) -> dict[str, Any]:
    """13-gram overlap between the evaluated items and the training set.

    The same method and the same shingle size as `data/decontaminate.py`, so the
    two reports are directly comparable. Reused rather than reimplemented,
    because two subtly different contamination checks in one repository is worse
    than none.
    """
    pairs = read_jsonl(train_path)
    train_index: set[int] = set()
    for pair in pairs:
        train_index |= _shingle_hashes(f"{pair.get('instruction', '')} {pair.get('response', '')}")

    overlapping = [
        item
        for item in items
        if _shingle_hashes(f"{item.question} {' '.join(item.choices)}") & train_index
    ]
    return {
        "train_pairs_indexed": len(pairs),
        "train_ngrams": len(train_index),
        "items_checked": len(items),
        "items_overlapping": len(overlapping),
        "examples": [item.question[:160] for item in overlapping[:5]],
    }


@dataclass
class ArmScore:
    correct: int = 0
    compliant: int = 0
    total: int = 0
    # subject -> [correct, total]
    by_subject: dict[str, list[int]] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        accuracy = Rate(self.correct, self.total)
        return {
            "accuracy": accuracy.value,
            "ci95": list(accuracy.interval),
            "correct": self.correct,
            "total": self.total,
            "format_compliance": Rate(self.compliant, self.total).value,
            # Per subject as well as pooled: a fine-tune that trades arithmetic
            # for prose would show a flat average and a lopsided breakdown.
            "by_subject": {
                subject: {
                    "correct": counts[0],
                    "total": counts[1],
                    "accuracy": counts[0] / counts[1] if counts[1] else 0.0,
                }
                for subject, counts in sorted(self.by_subject.items())
            },
        }


async def run_arm(
    client: GenerationClient, arm: Arm, items: list[Item], *, model: str, max_tokens: int
) -> ArmScore:
    conversations = [arm.build_messages(item.prompt()) for item in items]
    generations = await client.complete_many(
        conversations, model=arm.adapter or model, max_tokens=max_tokens
    )
    score = ArmScore()
    for item, generation in zip(items, generations, strict=True):
        predicted = extract_letter(generation.text)
        correct = predicted == item.answer
        score.total += 1
        score.correct += correct
        score.compliant += predicted is not None
        bucket = score.by_subject.setdefault(item.subject, [0, 0])
        bucket[0] += correct
        bucket[1] += 1
    return score


def compare(base: ArmScore, tuned: ArmScore) -> dict[str, Any]:
    """The delta, with the interval that decides whether it is real."""
    base_low, base_high = Rate(base.correct, base.total).interval
    tuned_low, tuned_high = Rate(tuned.correct, tuned.total).interval
    base_acc = base.correct / base.total if base.total else 0.0
    tuned_acc = tuned.correct / tuned.total if tuned.total else 0.0
    separated = tuned_high < base_low or base_high < tuned_low
    return {
        "base_accuracy": base_acc,
        "fine_tuned_accuracy": tuned_acc,
        "delta": round(tuned_acc - base_acc, 4),
        "intervals_separated": separated,
        "verdict": (
            "measurable change in general capability"
            if separated
            else "no demonstrated change — the intervals overlap, so this delta is "
            "not evidence of forgetting in either direction"
        ),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=Path("eval_results/forgetting.json"))
    parser.add_argument("--train", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--per-subject", type=int, default=60)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--adapter", default="legalmind")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument(
        "--skip-contamination-check",
        action="store_true",
        help="report without checking MMLU against the training set. Off by default",
    )
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    items = load_items(per_subject=args.per_subject)
    print(f"{len(items)} MMLU items across {len(SUBJECTS)} non-legal subjects", file=sys.stderr)

    contamination: dict[str, Any] | None = None
    if not args.skip_contamination_check:
        if not args.train.exists():
            print(f"error: {args.train} not found; cannot verify the benchmark", file=sys.stderr)
            return 2
        contamination = check_contamination(items, args.train)
        print(
            f"contamination check: {contamination['items_overlapping']}/"
            f"{contamination['items_checked']} items share a 13-gram with training data",
            file=sys.stderr,
        )
        if contamination["items_overlapping"]:
            print(
                "REFUSING TO REPORT. The training set was decontaminated against "
                "LegalBench only, and these MMLU items overlap it. A forgetting "
                "delta measured on contaminated items is not interpretable — "
                "re-run decontamination including MMLU before quoting a number.",
                file=sys.stderr,
            )
            return 1

    arms = build_arms(args.adapter)
    scores: dict[str, ArmScore] = {}
    async with GenerationClient(base_url=args.base_url, concurrency=args.concurrency) as client:
        for arm_id in ArmId:
            arm = arms[arm_id]
            print(f"\n=== {arm.label} ===", file=sys.stderr)
            score = await run_arm(client, arm, items, model=args.model, max_tokens=args.max_tokens)
            scores[arm_id.value] = score
            print(
                f"  accuracy {score.correct}/{score.total} "
                f"({score.correct / score.total if score.total else 0:.1%}), "
                f"format compliance {score.compliant / score.total if score.total else 0:.1%}",
                file=sys.stderr,
            )

    base = scores[ArmId.BASE_ZERO_SHOT.value]
    prompted = scores[ArmId.BASE_PROMPTED.value]
    tuned = scores[ArmId.FINE_TUNED.value]

    payload: dict[str, Any] = {
        "subjects": list(SUBJECTS),
        "items_per_subject": args.per_subject,
        "n_items": len(items),
        "contamination_check": contamination,
        "arms": {name: score.summary() for name, score in scores.items()},
        "forgetting": compare(base, tuned),
        "harness_control": compare(base, prompted)
        | {
            "means": (
                "arms A and B share weights, so a separated interval here means "
                "the harness itself is noisy and the A-to-C delta above cannot be "
                "attributed to the fine-tune"
            )
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"\nforgetting: {payload['forgetting']['verdict']}", file=sys.stderr)
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_run(_parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
