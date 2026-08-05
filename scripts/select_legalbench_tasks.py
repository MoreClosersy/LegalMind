"""Freeze the LegalBench evaluation subset, before any model exists to score.

    uv run python scripts/select_legalbench_tasks.py --out configs/legalbench_tasks.json

Choosing which benchmark tasks to report *after* seeing how the model did on
them is a quiet, common, and completely invalidating form of cheating. The
defence is to make the selection mechanical and to commit it before there is a
fine-tuned model at all: this script encodes the criteria, and its output —
`configs/legalbench_tasks.json` — is checked into the repository with the commit
that created it, timestamped ahead of any training run.

The criteria are all properties of the *benchmark*, never of any model's
performance on it:

1. `eval_method == "contained_in_output"` — the method 157 of the 162 tasks use.
   Mixing scoring methods across tasks would make the aggregate meaningless.
2. A closed `answer_space` of 2-8 labels. Open-ended generation tasks
   (`citation_prediction_open`) cannot be scored without a judge, and the
   50-label task is a different problem entirely.
3. A test split of at least `--min-test-rows` examples. Below that, the
   per-task confidence interval is so wide the task cannot distinguish arms.
4. Majority-class share at most `--max-majority`. A task where always guessing
   the same label scores 92% has almost no headroom to show a difference, and
   including it inflates the aggregate for every arm equally.
5. US jurisdiction. The training data is US federal law, so a Canadian or
   Indian task measures transfer rather than the thing being trained.

Every task that passes is kept. There is no sampling step and no `--top-n`,
because any such knob is a place for discretion to leak back in.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

LEGALBENCH_REPO = "nguha/legalbench"

# Excluded by jurisdiction, by name, before any scoring. The training corpus is
# US federal (CFR, Federal Register, CourtListener opinions), so these measure
# cross-jurisdiction transfer — a different question from the one being asked.
NON_US_TASKS = frozenset(
    {
        "canada_tax_court_outcomes",
        "rule_qa",  # meta-questions about the benchmark itself, not a legal task
    }
)


def load_metadata() -> dict[str, dict[str, Any]]:
    path = hf_hub_download(LEGALBENCH_REPO, "task_metadata.json", repo_type="dataset")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_test_rows(task: str) -> list[dict[str, str]]:
    path = hf_hub_download(LEGALBENCH_REPO, f"data/{task}/test.tsv", repo_type="dataset")
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def evaluate_task(
    task: str,
    meta: dict[str, Any],
    *,
    min_test_rows: int,
    max_majority: float,
) -> dict[str, Any]:
    """Apply every criterion and record which one rejected the task.

    Rejections are recorded rather than dropped so the committed manifest shows
    the whole benchmark and why each task is in or out.
    """
    verdict: dict[str, Any] = {"task": task}

    if task in NON_US_TASKS:
        return verdict | {"selected": False, "reason": "non_us_jurisdiction"}

    if meta.get("eval_method") != "contained_in_output":
        return verdict | {"selected": False, "reason": f"eval_method={meta.get('eval_method')}"}

    answer_space = meta.get("answer_space") or []
    if not 2 <= len(answer_space) <= 8:
        return verdict | {"selected": False, "reason": f"answer_space={len(answer_space)}"}

    try:
        rows = load_test_rows(task)
    except Exception as exc:
        return verdict | {"selected": False, "reason": f"no_test_split ({type(exc).__name__})"}

    verdict["test_rows"] = len(rows)
    if len(rows) < min_test_rows:
        return verdict | {"selected": False, "reason": "test_split_too_small"}

    counts = Counter(row.get("answer", "") for row in rows)
    majority_label, majority_count = counts.most_common(1)[0]
    majority_share = majority_count / len(rows)
    verdict |= {
        "answer_space": answer_space,
        "majority_label": majority_label,
        "majority_share": round(majority_share, 4),
        "label_counts": dict(counts),
    }
    if majority_share > max_majority:
        return verdict | {"selected": False, "reason": "too_imbalanced"}

    return verdict | {"selected": True, "reason": None}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("configs/legalbench_tasks.json"))
    parser.add_argument("--min-test-rows", type=int, default=60)
    parser.add_argument("--max-majority", type=float, default=0.85)
    parser.add_argument(
        "--max-examples-per-task",
        type=int,
        default=60,
        help=(
            "cap recorded per task at eval time. Caps the cost of the sweep and "
            "stops the handful of 1000-row tasks from dominating the aggregate"
        ),
    )
    args = parser.parse_args(argv)

    metadata = load_metadata()
    print(f"{len(metadata)} tasks in LegalBench", file=sys.stderr)

    verdicts = []
    for i, (task, meta) in enumerate(sorted(metadata.items()), start=1):
        verdicts.append(
            evaluate_task(
                task, meta, min_test_rows=args.min_test_rows, max_majority=args.max_majority
            )
        )
        if i % 25 == 0:
            print(f"  screened {i}/{len(metadata)}...", file=sys.stderr)

    selected = [v for v in verdicts if v["selected"]]
    rejected = Counter(v["reason"] for v in verdicts if not v["selected"])

    manifest = {
        "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "why": (
            "Selected mechanically, before any fine-tuned model existed, so the "
            "task list cannot have been chosen to flatter a result. Criteria are "
            "properties of the benchmark only — see scripts/select_legalbench_tasks.py."
        ),
        "criteria": {
            "eval_method": "contained_in_output",
            "answer_space_size": [2, 8],
            "min_test_rows": args.min_test_rows,
            "max_majority_share": args.max_majority,
            "excluded_by_name": sorted(NON_US_TASKS),
        },
        "max_examples_per_task": args.max_examples_per_task,
        "n_considered": len(metadata),
        "n_selected": len(selected),
        "rejected_by_reason": dict(rejected.most_common()),
        "tasks": [
            {
                "task": v["task"],
                "answer_space": v["answer_space"],
                "test_rows": v["test_rows"],
                "majority_label": v["majority_label"],
                "majority_share": v["majority_share"],
                "label_counts": v["label_counts"],
            }
            for v in selected
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\nselected {len(selected)}/{len(metadata)} tasks", file=sys.stderr)
    for reason, n in rejected.most_common():
        print(f"  rejected {n:>3}  {reason}", file=sys.stderr)
    evaluated = sum(min(v["test_rows"], args.max_examples_per_task) for v in selected)
    print(f"\n{evaluated:,} examples per arm at the {args.max_examples_per_task}/task cap")
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
