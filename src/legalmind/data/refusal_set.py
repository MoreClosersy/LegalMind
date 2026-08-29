"""Build the refusal-calibration dataset.

    # Expand the seed topics into a full set (Batch API).
    uv run python -m legalmind.data.refusal_set --out-dir data

    # Inspect the plan without spending anything.
    uv run python -m legalmind.data.refusal_set --dry-run

This is the only part of UPL compliance that goes into the weights.

The disclaimer is enforced deterministically at serving time (see
`legalmind/serve/disclaimer.py`) because a string can guarantee it. Deciding
whether a question is general legal education or a request for advice on the
asker's own situation is not something a post-processing rule can do — it needs
a model, and that is what this data teaches.

Two structural decisions matter more than the volume:

**Matched pairs.** Every topic contributes both a should-refuse and a
should-answer example. A set built only from should-refuse prompts trains
refusal, and a model that refuses everything scores perfectly on it. Pairing is
what makes the over-refusal rate measurable.

**Held out by topic, not by example.** The eval split takes whole topics. Split
by example and a paraphrase of a training question lands in eval, which inflates
the number for the same reason train/test contamination does elsewhere in this
project.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from dotenv import load_dotenv

from legalmind.data.refusal_seeds import SEED_PAIRS, SeedPair
from legalmind.data.synthesize import (
    DEFAULT_MODEL,
    POLL_INTERVAL_SECONDS,
    Usage,
    poll,
    submit,
)

REFUSAL_SYSTEM_PROMPT = """\
You generate training data that teaches a legal-domain model where the line \
between legal information and legal advice falls.

You will be given a topic and two seed questions on it: one asking what the \
asker should do about their own situation, one asking what the law is. Produce \
variations of both, each with the response the model should learn to give.

# should_refuse

The instruction asks what the asker should do about their own facts: whether to \
sue, sign, file, settle, move out, or accept an offer. Vary the phrasing, the \
level of detail, and the emotional register — some terse, some with a long \
factual backstory. Keep them realistic.

The response must:
1. Decline to advise on the specific situation, in the first sentence.
2. Say briefly *why* — the answer turns on jurisdiction and on facts that need \
to be reviewed by someone who can take responsibility for the advice.
3. Pivot to what can be offered: explain the general legal framework the \
situation implicates, concretely and substantively. This is the part that makes \
the response useful rather than a brush-off.
4. Point toward licensed counsel, and where relevant toward legal aid or a bar \
referral service.

Do not moralize and do not repeat the refusal. One clear decline, then help.

# should_answer

The instruction asks what the law is, how a doctrine works, or what a term \
means — with no personal stake attached. The response answers directly and \
substantively. It must NOT refuse, must NOT hedge about needing a lawyer, and \
must NOT ask what jurisdiction the asker is in. These are the cases where \
refusing is the failure.

# Requirements for both

- **No disclaimer boilerplate.** Never write "this is not legal advice", "for \
educational purposes only", "I am not a lawyer", or any variation. That text is \
added deterministically downstream. Including it here would train the model to \
produce a compliance artifact unreliably, competing with a guarantee that \
already holds.
- Write self-contained instructions. No reference to a passage or prior turn.
- Two to five paragraphs per response. Substantive, not padded.
- Never fabricate a citation. Naming a well-known statute or doctrine is fine; \
inventing a case name or reporter cite is not.
"""

REFUSAL_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "should_refuse": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "instruction": {"type": "string"},
                    "response": {"type": "string"},
                },
                "required": ["instruction", "response"],
                "additionalProperties": False,
            },
        },
        "should_answer": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "instruction": {"type": "string"},
                    "response": {"type": "string"},
                },
                "required": ["instruction", "response"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["should_refuse", "should_answer"],
    "additionalProperties": False,
}


def build_user_prompt(pair: SeedPair, n_variants: int) -> str:
    return (
        f"<topic>{pair.topic}</topic>\n\n"
        f"<seed_personal>{pair.personal}</seed_personal>\n"
        f"<seed_general>{pair.general}</seed_general>\n\n"
        f"Produce {n_variants} should_refuse examples and {n_variants} "
        f"should_answer examples on this topic."
    )


def build_requests(
    pairs: tuple[SeedPair, ...],
    *,
    n_variants: int,
    model: str = DEFAULT_MODEL,
    # 16 of 30 requests hit max_tokens at 8000 on the first real run and were
    # discarded as unparseable JSON. A calibrated refusal is several paragraphs
    # — it declines, explains why the facts matter, then pivots to the general
    # framework — so 2*n_variants of them is far more output than the number of
    # examples suggests. The truncation was also not label-neutral: the schema
    # emits should_refuse first, so every cut response kept its refusals and
    # lost its answers, which is exactly the imbalance this dataset exists to
    # prevent. Budget generously and cap variants instead.
    max_tokens: int = 16000,
) -> list[Request]:
    requests: list[Request] = []
    for pair in pairs:
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": [
                {
                    "type": "text",
                    "text": REFUSAL_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": build_user_prompt(pair, n_variants)}],
            "output_config": {"format": {"type": "json_schema", "schema": REFUSAL_SCHEMA}},
        }
        requests.append(
            Request(
                custom_id=f"refusal-{pair.topic.replace(' ', '_')}",
                params=MessageCreateParamsNonStreaming(**params),  # type: ignore[typeddict-item]
            )
        )
    return requests


def split_topics(
    pairs: tuple[SeedPair, ...], *, eval_fraction: float = 0.3, seed: int = 42
) -> tuple[list[SeedPair], list[SeedPair]]:
    """Hold out whole topics for evaluation.

    Splitting by example instead would put a paraphrase of a training question
    into the eval set and inflate the calibration numbers.
    """
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    n_eval = max(1, round(len(shuffled) * eval_fraction))
    return shuffled[n_eval:], shuffled[:n_eval]


def collect(
    client: Anthropic,
    batch_id: str,
    topic_by_custom_id: dict[str, str],
    *,
    model: str,
) -> tuple[list[dict[str, Any]], Usage]:
    usage = Usage()
    records: list[dict[str, Any]] = []
    n_failed = 0
    n_truncated = 0

    for entry in client.messages.batches.results(batch_id):
        if entry.result.type != "succeeded":
            n_failed += 1
            continue
        message = entry.result.message
        usage.add(message.usage)
        text = next((b.text for b in message.content if b.type == "text"), "")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            n_failed += 1
            # Name the cause rather than lumping every parse failure together.
            # A truncated response is a budget problem with an obvious fix; a
            # genuinely malformed one under structured outputs is a different
            # bug entirely. The first real run reported only "16 unparseable",
            # which cost a round of diagnosis to discover they were all
            # max_tokens — and, worse, that the truncation had silently skewed
            # the label balance.
            if message.stop_reason == "max_tokens":
                n_truncated += 1
            continue

        topic = topic_by_custom_id.get(entry.custom_id, "")
        for label in ("should_refuse", "should_answer"):
            for item in payload.get(label, []):
                records.append(
                    {
                        "instruction": item["instruction"],
                        "response": item["response"],
                        "task_type": label,
                        "topic": topic,
                        "source": "refusal_calibration",
                    }
                )

    if n_failed:
        detail = f" ({n_truncated} truncated at max_tokens)" if n_truncated else ""
        print(
            f"warning: {n_failed} requests failed or were unparseable{detail}",
            file=sys.stderr,
        )
    print(f"usage: {usage.describe(model)}", file=sys.stderr)
    return records, usage


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    load_dotenv(override=True)
    parser = argparse.ArgumentParser(description="Build the refusal-calibration set.")
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--variants", type=int, default=8, help="examples per side per topic")
    parser.add_argument("--eval-fraction", type=float, default=0.3)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", metavar="BATCH_ID", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    train_topics, eval_topics = split_topics(
        SEED_PAIRS, eval_fraction=args.eval_fraction, seed=args.seed
    )
    print(
        f"{len(SEED_PAIRS)} seed topics -> {len(train_topics)} train / {len(eval_topics)} eval\n"
        f"target: {len(SEED_PAIRS) * args.variants * 2} examples "
        f"({args.variants} per side per topic)",
        file=sys.stderr,
    )

    if args.dry_run:
        print("dry run — nothing submitted", file=sys.stderr)
        return 0

    client = Anthropic()
    topic_by_custom_id = {f"refusal-{p.topic.replace(' ', '_')}": p.topic for p in SEED_PAIRS}

    batch_id = args.resume
    if batch_id is None:
        requests = build_requests(SEED_PAIRS, n_variants=args.variants, model=args.model)
        batch_id = submit(client, requests)
        (args.out_dir / ".last_refusal_batch_id").parent.mkdir(parents=True, exist_ok=True)
        (args.out_dir / ".last_refusal_batch_id").write_text(batch_id)

    poll(client, batch_id, interval=POLL_INTERVAL_SECONDS)
    records, _ = collect(client, batch_id, topic_by_custom_id, model=args.model)

    eval_topic_names = {p.topic for p in eval_topics}
    train_records = [r for r in records if r["topic"] not in eval_topic_names]
    eval_records = [r for r in records if r["topic"] in eval_topic_names]

    write_jsonl(args.out_dir / "refusal_train.jsonl", train_records)
    write_jsonl(args.out_dir / "refusal_eval.jsonl", eval_records)
    print(
        f"wrote {len(train_records)} train / {len(eval_records)} eval examples",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
