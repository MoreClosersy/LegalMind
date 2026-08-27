"""Score the three arms on the frozen LegalBench subset.

    # vLLM, one process, all three arms:
    vllm serve Qwen/Qwen3-8B --enable-lora \
        --lora-modules legalmind=checkpoints/legalmind-qwen3-8b

    uv run python -m legalmind.eval.legalbench --out eval_results/legalbench.json

LegalBench is the evaluation set and the training data never touched it — that
is the claim `data/decontaminate.py` exists to evidence. This module is the
other half: turning that clean separation into numbers that mean something.

Two measurement traps sit between here and a defensible number, and both are
easy to walk into.

**Trap 1: `contained_in_output` rewards verbosity.** LegalBench's official
scoring method, used by 157 of its 162 tasks, marks a response correct when the
gold label appears anywhere inside it. 109 tasks are Yes/No. A model that
answers a Yes/No question in three paragraphs will very often contain both
labels somewhere, and scores a point whenever the gold one is present. That does
not measure legal reasoning — it measures output length, and it measures it
*unequally across the arms*, because arm C is fine-tuned to explain at length
while arm A emits a word. Reporting only the official metric here would hand the
fine-tuned arm a bonus it did not earn.

So three numbers are reported for every arm:

* `official_accuracy` — LegalBench's own method, for comparability.
* `strict_accuracy` — the label parsed out of a constrained position, matched on
  word boundaries. This is the honest one and it is what the README leads with.
* `ambiguity_rate` — how often more than one label from the answer space appears
  in the output. This is the size of the hole in the official metric. When it is
  high for an arm, that arm's official number should be disbelieved, and the
  reader can see that directly instead of taking it on trust.

**Trap 2: accuracy is not skill when the classes are unbalanced.** A task whose
test split is 70% "No" hands 70% to a model that always says "No". Every task
therefore carries its majority-class baseline from the frozen manifest, and
`balanced_accuracy` (mean per-class recall) is reported next to raw accuracy.
An arm that beats another on accuracy while losing on balanced accuracy has
learned the prior, not the task.

One deliberate departure from the published protocol: every arm gets the same
short answer-format directive appended to LegalBench's own instruction. Without
it the comparison degenerates into a test of which arm happens to emit bare
labels, which is a property of instruction tuning rather than of legal
reasoning. The consequence is that these numbers are **not comparable to the
LegalBench leaderboard**, only to each other — which is the comparison this
project is actually making. The exact template is recorded in the results file
so the departure is visible rather than buried.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

from legalmind.eval.arms import Arm, ArmId, build_arms
from legalmind.eval.generate import (
    DEFAULT_BASE_URL,
    DEFAULT_CONCURRENCY,
    GenerationClient,
    strip_thinking_block,
)
from legalmind.eval.refusal import wilson_interval

LEGALBENCH_REPO = "nguha/legalbench"
DEFAULT_MANIFEST = Path("configs/legalbench_tasks.json")

# Applied identically to all three arms. Short, neutral, and it names the legal
# labels rather than restating the task, so it cannot leak task guidance to one
# arm more than another.
ANSWER_FORMAT_DIRECTIVE = (
    "\n\nRespond with only one of the following labels, on its own line, with no "
    "explanation: {labels}"
)

_TEMPLATE_VAR = re.compile(r"\{\{(\w+)\}\}")
# Everything after the final answer marker is where a well-behaved response puts
# its label, so extraction reads from there rather than from the top.
#
# The set of markers is measured, not assumed: across the 127 selected tasks,
# LegalBench's own instructions end with "Label:" 80 times, "Answer:" 36 times,
# and "A:" 8 times. Matching only "Answer:" — which is what this started as —
# left 89 tasks falling back to scanning from the first line, where a response
# that reasons first and states its label last reads as non-compliant. That
# would have shown up as a format-compliance gap between arms rather than as a
# bug, which is exactly the kind of wrong number that never gets questioned.
_ANSWER_MARKER = re.compile(
    r"^[ \t]*(?:final\s+answer|answer|label|a)[ \t]*:", re.IGNORECASE | re.MULTILINE
)

# Markdown and punctuation a model wraps a bare label in. The last two are
# U+2013 EN DASH and U+2014 EM DASH, written by code point rather than
# literally: both are bullet characters models actually emit, and spelling them
# out keeps ruff's ambiguous-unicode check enabled for the prose in this file.
_LABEL_DECORATION = "*_`#-:.\"' " + chr(0x2013) + chr(0x2014)


@dataclass(frozen=True)
class Task:
    name: str
    instruction: str
    answer_space: tuple[str, ...]
    majority_label: str
    majority_share: float
    rows: tuple[dict[str, str], ...]


@dataclass
class TaskScore:
    task: str
    n: int = 0
    strict_correct: int = 0
    official_correct: int = 0
    compliant: int = 0
    ambiguous: int = 0
    majority_share: float = 0.0
    per_label_total: Counter[str] = field(default_factory=Counter)
    per_label_correct: Counter[str] = field(default_factory=Counter)
    completion_tokens: int = 0
    prompt_tokens: int = 0
    latency_s: float = 0.0

    @property
    def strict_accuracy(self) -> float:
        return self.strict_correct / self.n if self.n else 0.0

    @property
    def official_accuracy(self) -> float:
        return self.official_correct / self.n if self.n else 0.0

    @property
    def compliance_rate(self) -> float:
        return self.compliant / self.n if self.n else 0.0

    @property
    def ambiguity_rate(self) -> float:
        return self.ambiguous / self.n if self.n else 0.0

    @property
    def balanced_accuracy(self) -> float:
        """Mean per-class recall — the number that survives class imbalance.

        Averaging recall over classes rather than over examples means the rare
        label counts as much as the common one, so a model that has only learned
        to emit the majority label scores near 1/n_classes instead of near the
        majority share.
        """
        recalls = [
            self.per_label_correct[label] / total
            for label, total in self.per_label_total.items()
            if total
        ]
        return sum(recalls) / len(recalls) if recalls else 0.0


def render_instruction(template: str, row: dict[str, str]) -> str:
    """Fill LegalBench's `{{var}}` placeholders from a TSV row.

    A missing column would otherwise leave a literal `{{contract}}` in the
    prompt and quietly cost accuracy on that whole task, so it raises instead.
    """

    def substitute(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in row:
            raise KeyError(f"template variable {key!r} is not a column of this task")
        return row[key]

    return _TEMPLATE_VAR.sub(substitute, template)


def _label_pattern(label: str) -> re.Pattern[str]:
    """Word-boundary match, so "No" does not fire inside "Nothing" or "cannot".

    Substring containment is what makes the official metric as loose as it is;
    the strict path at least refuses to count an accidental collision.
    """
    return re.compile(rf"(?<!\w){re.escape(label)}(?!\w)", re.IGNORECASE)


def labels_present(text: str, answer_space: tuple[str, ...]) -> set[str]:
    return {label for label in answer_space if _label_pattern(label).search(text)}


def extract_label(text: str, answer_space: tuple[str, ...]) -> str | None:
    """Parse the answer from a constrained position, or return None.

    Returning None for an unparseable response is the point: those become
    non-compliance rather than being silently scored as wrong, which keeps
    "cannot follow the format" distinguishable from "knows the wrong answer".
    Collapsing the two would let a formatting regression masquerade as a
    reasoning regression.
    """
    if not text.strip():
        return None

    text = strip_thinking_block(text)
    tail = _ANSWER_MARKER.split(text)[-1]
    for line in tail.splitlines():
        candidate = line.strip().strip(_LABEL_DECORATION)
        if not candidate:
            continue

        exact = [label for label in answer_space if candidate.lower() == label.lower()]
        if exact:
            return exact[0]

        # "Yes. The statement is hearsay because ... and no exception applies."
        # is an answer of Yes with reasoning attached, not a format failure. Arm
        # C is fine-tuned to explain, so this is its normal shape, and treating
        # it as non-compliance would understate the fine-tune on every task.
        # The leading label wins over anything mentioned later in the sentence.
        leading = [
            label
            for label in answer_space
            if re.match(rf"{re.escape(label)}(?!\w)", candidate, re.IGNORECASE)
        ]
        if leading:
            return max(leading, key=len)

        found = labels_present(candidate, answer_space)
        if len(found) == 1:
            return found.pop()
        # A first non-empty line that names several labels, or none, is a
        # refusal to follow the format. Reading further would start guessing.
        return None
    return None


def score_response(
    text: str, gold: str, answer_space: tuple[str, ...]
) -> tuple[bool, bool, bool, bool]:
    """Return (strict_correct, official_correct, compliant, ambiguous)."""
    extracted = extract_label(text, answer_space)
    strict_correct = extracted is not None and extracted.lower() == gold.lower()
    # LegalBench's own method: plain containment anywhere in the output.
    official_correct = gold.lower() in text.lower()
    ambiguous = len(labels_present(text, answer_space)) > 1
    return strict_correct, official_correct, extracted is not None, ambiguous


def load_tasks(manifest_path: Path, *, limit_tasks: int | None = None) -> list[Task]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cap = int(manifest["max_examples_per_task"])
    metadata = json.loads(
        Path(hf_hub_download(LEGALBENCH_REPO, "task_metadata.json", repo_type="dataset")).read_text(
            encoding="utf-8"
        )
    )

    entries = manifest["tasks"]
    if limit_tasks is not None:
        entries = entries[:limit_tasks]

    tasks: list[Task] = []
    for entry in entries:
        name = entry["task"]
        path = hf_hub_download(LEGALBENCH_REPO, f"data/{name}/test.tsv", repo_type="dataset")
        with open(path, encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        tasks.append(
            Task(
                name=name,
                instruction=metadata[name]["instruction"],
                answer_space=tuple(entry["answer_space"]),
                majority_label=entry["majority_label"],
                majority_share=entry["majority_share"],
                # Deterministic prefix rather than a random sample: the manifest
                # is frozen, so the evaluated rows must be too.
                rows=tuple(rows[:cap]),
            )
        )
    return tasks


def build_conversations(task: Task, arm: Arm) -> list[list[dict[str, str]]]:
    directive = ANSWER_FORMAT_DIRECTIVE.format(labels=", ".join(task.answer_space))
    return [
        arm.build_messages(render_instruction(task.instruction, row) + directive)
        for row in task.rows
    ]


async def run_arm(
    client: GenerationClient,
    arm: Arm,
    tasks: list[Task],
    *,
    model: str,
    max_tokens: int,
) -> dict[str, TaskScore]:
    scores: dict[str, TaskScore] = {}
    for i, task in enumerate(tasks, start=1):
        conversations = build_conversations(task, arm)
        served_as = arm.adapter or model
        generations = await client.complete_many(
            conversations, model=served_as, max_tokens=max_tokens
        )

        score = TaskScore(task=task.name, majority_share=task.majority_share)
        for row, generation in zip(task.rows, generations, strict=True):
            gold = row["answer"]
            strict, official, compliant, ambiguous = score_response(
                generation.text, gold, task.answer_space
            )
            score.n += 1
            score.strict_correct += strict
            score.official_correct += official
            score.compliant += compliant
            score.ambiguous += ambiguous
            score.per_label_total[gold] += 1
            score.per_label_correct[gold] += strict
            score.prompt_tokens += generation.prompt_tokens
            score.completion_tokens += generation.completion_tokens
            score.latency_s += generation.latency_s
        scores[task.name] = score

        print(
            f"  [{arm.id.value}] {i}/{len(tasks)} {task.name}: "
            f"strict {score.strict_accuracy:.1%} / official {score.official_accuracy:.1%} "
            f"(majority {task.majority_share:.1%}, compliance {score.compliance_rate:.1%})",
            file=sys.stderr,
        )
    return scores


def summarise(scores: dict[str, TaskScore]) -> dict[str, Any]:
    """Aggregate over tasks, not over examples.

    Micro-averaging would let the largest tasks dominate; the per-task cap
    already limits that, but macro-averaging makes the choice explicit. The
    confidence interval is over pooled examples, which is the right unit for
    "could this gap be noise".
    """
    task_list = list(scores.values())
    n_total = sum(s.n for s in task_list)
    strict_total = sum(s.strict_correct for s in task_list)

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "n_tasks": len(task_list),
        "n_examples": n_total,
        "strict_accuracy_macro": round(mean([s.strict_accuracy for s in task_list]), 4),
        "strict_accuracy_micro": round(strict_total / n_total, 4) if n_total else 0.0,
        "strict_accuracy_ci95": [round(x, 4) for x in wilson_interval(strict_total, n_total)],
        "balanced_accuracy_macro": round(mean([s.balanced_accuracy for s in task_list]), 4),
        "official_accuracy_macro": round(mean([s.official_accuracy for s in task_list]), 4),
        "majority_baseline_macro": round(mean([s.majority_share for s in task_list]), 4),
        "compliance_rate_macro": round(mean([s.compliance_rate for s in task_list]), 4),
        "ambiguity_rate_macro": round(mean([s.ambiguity_rate for s in task_list]), 4),
        "mean_completion_tokens": (
            round(sum(s.completion_tokens for s in task_list) / n_total, 1) if n_total else 0.0
        ),
        "mean_prompt_tokens": (
            round(sum(s.prompt_tokens for s in task_list) / n_total, 1) if n_total else 0.0
        ),
        "mean_latency_s": (
            round(sum(s.latency_s for s in task_list) / n_total, 3) if n_total else 0.0
        ),
        "per_task": {
            s.task: {
                "n": s.n,
                "strict_accuracy": round(s.strict_accuracy, 4),
                "balanced_accuracy": round(s.balanced_accuracy, 4),
                "official_accuracy": round(s.official_accuracy, 4),
                "majority_share": s.majority_share,
                "compliance_rate": round(s.compliance_rate, 4),
                "ambiguity_rate": round(s.ambiguity_rate, 4),
            }
            for s in task_list
        },
    }


def render_table(summaries: dict[str, dict[str, Any]]) -> str:
    rows = [
        ("strict accuracy (macro)", "strict_accuracy_macro"),
        ("  95% CI, pooled", None),
        ("balanced accuracy (macro)", "balanced_accuracy_macro"),
        ("majority baseline", "majority_baseline_macro"),
        ("official (contained_in_output)", "official_accuracy_macro"),
        ("  ambiguity rate", "ambiguity_rate_macro"),
        ("format compliance", "compliance_rate_macro"),
        ("mean completion tokens", "mean_completion_tokens"),
        ("mean prompt tokens", "mean_prompt_tokens"),
    ]
    arms = list(summaries)
    lines = ["", f"{'':<32}" + "".join(f"{a:>22}" for a in arms)]
    for label, key in rows:
        if key is None:
            intervals = [
                f"[{s['strict_accuracy_ci95'][0]:.1%}, {s['strict_accuracy_ci95'][1]:.1%}]"
                for s in summaries.values()
            ]
            cells = "".join(f"{interval:>22}" for interval in intervals)
        elif "tokens" in key:
            cells = "".join(f"{s[key]:>22.1f}" for s in summaries.values())
        else:
            cells = "".join(f"{s[key]:>21.1%} " for s in summaries.values())
        lines.append(f"{label:<32}" + cells)

    if len(arms) >= 2:
        lines.append("")
        lines.append(
            "Read the ambiguity rate before the official row: where it is high, "
            "`contained_in_output`\nis crediting responses that merely mention the "
            "gold label. The strict row is the claim."
        )
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=Path("eval_results/legalbench.json"))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--adapter", default="legalmind")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument(
        "--arm",
        action="append",
        choices=[a.value for a in ArmId],
        default=None,
        help="restrict to specific arms; defaults to all three",
    )
    parser.add_argument(
        "--limit-tasks",
        type=int,
        default=None,
        help="smoke-test knob only. Any reported number must come from the full frozen set",
    )
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    tasks = load_tasks(args.manifest, limit_tasks=args.limit_tasks)
    n_examples = sum(len(t.rows) for t in tasks)
    print(f"{len(tasks)} tasks, {n_examples:,} examples per arm", file=sys.stderr)
    if args.limit_tasks is not None:
        print(
            "WARNING: --limit-tasks is set. These numbers are a smoke test, not a result.",
            file=sys.stderr,
        )

    all_arms = build_arms(args.adapter)
    wanted = [ArmId(value) for value in args.arm] if args.arm else list(ArmId)

    summaries: dict[str, dict[str, Any]] = {}
    async with GenerationClient(base_url=args.base_url, concurrency=args.concurrency) as client:
        for arm_id in wanted:
            arm = all_arms[arm_id]
            print(f"\n=== {arm.label} ===", file=sys.stderr)
            scores = await run_arm(client, arm, tasks, model=args.model, max_tokens=args.max_tokens)
            summaries[arm_id.value] = summarise(scores)

    payload = {
        "manifest": str(args.manifest),
        "manifest_frozen_at": json.loads(args.manifest.read_text())["frozen_at"],
        "model": args.model,
        "adapter": args.adapter,
        "max_tokens": args.max_tokens,
        "sampling": "greedy (temperature=0)",
        "answer_format_directive": ANSWER_FORMAT_DIRECTIVE,
        "not_leaderboard_comparable": (
            "Every arm receives an answer-format directive appended to LegalBench's own "
            "instruction, so that the comparison measures reasoning rather than which arm "
            "happens to emit bare labels. These numbers are comparable across arms, not to "
            "published LegalBench results."
        ),
        "smoke_test": args.limit_tasks is not None,
        "arms": summaries,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    print(render_table(summaries))
    print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_run(_parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
