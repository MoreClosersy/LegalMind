"""Compare synthesis arms — prompt variants, teacher models, sampling settings.

    uv run python scripts/compare_synthesis.py \
        --arm sonnet5=data/probe_v2.jsonl \
        --arm haiku45=data/probe_haiku.jsonl

Every synthesis decision on this project has been settled by running a small
batch and measuring it, not by reading the prompt or trusting a model's
reputation: the first probe exposed 31.6% of responses referring back to a
passage the trained model never sees, and a task type stuck at 4% share.
Neither was visible without measurement.

This runs the same comparison each time, on the metrics that actually decide
the question:

* **retention through `filter.py`** — the real yield, since rejected pairs are
  paid for and thrown away. Reported with a Wilson interval, because these
  probes are tens of examples and two arms that look different at n=72 often
  are not.
* **task-type distribution** — a teacher that will not produce
  `statutory_interpretation` is worse than its average quality suggests.
* **rejection reasons** — *how* an arm fails, which is what tells you whether a
  prompt fix or a model swap is the right response.
* **output tokens per pair** — the cost driver, at 83% of batch spend.

Cost is deliberately not compared here: batch size distorts it badly (a
24-request probe saw 42% cache reads against 94% at 294 requests), so a
per-arm cost from a small probe is misleading. Price the winner separately at
the real batch size.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from legalmind.data.filter import filter_pairs, read_jsonl
from legalmind.eval.refusal import wilson_interval


@dataclass
class ArmResult:
    name: str
    total: int
    kept: int
    reasons: Counter[str]
    task_types: Counter[str]
    sources: Counter[str]
    mean_response_chars: float

    @property
    def retention(self) -> float:
        return self.kept / self.total if self.total else 0.0

    @property
    def retention_ci(self) -> tuple[float, float]:
        return wilson_interval(self.kept, self.total)


def measure(name: str, path: Path) -> ArmResult:
    pairs = read_jsonl(path)
    kept, reasons = filter_pairs(pairs)
    responses = [len(p.get("response", "")) for p in pairs]
    return ArmResult(
        name=name,
        total=len(pairs),
        kept=len(kept),
        reasons=reasons,
        task_types=Counter(p.get("task_type", "?") for p in kept),
        sources=Counter(p.get("source", "?") for p in kept),
        mean_response_chars=sum(responses) / len(responses) if responses else 0.0,
    )


def render(results: list[ArmResult]) -> str:
    lines: list[str] = []
    width = max(len(r.name) for r in results) + 2

    lines.append("retention through filter.py (95% CI)")
    for r in results:
        low, high = r.retention_ci
        lines.append(
            f"  {r.name:<{width}} {r.kept:>4}/{r.total:<4} {r.retention:>6.1%}  "
            f"[{low:.1%}, {high:.1%}]"
        )

    # Non-overlapping intervals are the bar for calling a difference real.
    if len(results) == 2:
        a, b = results
        a_low, a_high = a.retention_ci
        b_low, b_high = b.retention_ci
        separated = a_high < b_low or b_high < a_low
        verdict = "separated" if separated else "OVERLAPPING — not a demonstrated difference"
        lines.append(f"  -> intervals {verdict}")

    lines.append("")
    lines.append("task-type share (of kept pairs)")
    all_types = sorted({t for r in results for t in r.task_types})
    for task_type in all_types:
        row = f"  {task_type:<28}"
        for r in results:
            share = r.task_types.get(task_type, 0) / r.kept if r.kept else 0.0
            row += f"  {r.name}={share:>5.1%}"
        lines.append(row)

    lines.append("")
    lines.append("rejection reasons")
    all_reasons = sorted({reason for r in results for reason in r.reasons})
    if not all_reasons:
        lines.append("  (none in any arm)")
    for reason in all_reasons:
        row = f"  {reason:<28}"
        for r in results:
            row += f"  {r.name}={r.reasons.get(reason, 0):>4}"
        lines.append(row)

    lines.append("")
    lines.append("mean response length (chars) — proxy for the cost driver")
    for r in results:
        lines.append(f"  {r.name:<{width}} {r.mean_response_chars:>7.0f}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="a labelled JSONL of raw synthesis output; repeat for each arm",
    )
    parser.add_argument("--out", type=Path, default=None, help="write JSON alongside the table")
    args = parser.parse_args(argv)

    results = []
    for spec in args.arm:
        if "=" not in spec:
            raise SystemExit(f"--arm expects NAME=PATH, got {spec!r}")
        name, _, path = spec.partition("=")
        results.append(measure(name, Path(path)))

    print(render(results))

    if args.out:
        payload: dict[str, Any] = {
            r.name: {
                "total": r.total,
                "kept": r.kept,
                "retention": r.retention,
                "retention_ci95": list(r.retention_ci),
                "rejected_by_reason": dict(r.reasons),
                "task_types": dict(r.task_types),
                "sources": dict(r.sources),
                "mean_response_chars": round(r.mean_response_chars, 1),
            }
            for r in results
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
