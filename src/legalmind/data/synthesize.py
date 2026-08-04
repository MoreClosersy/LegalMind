"""Generate synthetic legal instruction data with the Anthropic Batch API.

Usage:

    # Cost probe first — always. 100 passages, then read the reported $/1k.
    uv run python -m legalmind.data.synthesize --limit 100 --out data/probe.jsonl

    # Full run once the probe looks sane.
    uv run python -m legalmind.data.synthesize --out data/raw_pairs.jsonl

    # Resume polling a batch that was already submitted (e.g. after a crash).
    uv run python -m legalmind.data.synthesize --resume msgbatch_01ABC... \
        --out data/raw_pairs.jsonl

Two cost levers do the heavy lifting here:

* **Batch API** — 50% off standard rates, and this workload has no latency
  requirement whatsoever.
* **Prompt caching** — the ~1k-token system prompt is identical across every
  request in the batch, so it is written once and read thereafter at ~0.1x.
  This is why nothing per-request may be interpolated into it.

The output of this module is *raw* pairs. Quality filtering, deduplication, and
decontamination against LegalBench happen downstream in filter.py and
decontaminate.py — keeping generation and filtering separate means the expensive
step never has to be re-run when a filter threshold changes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from dotenv import load_dotenv

from legalmind.data.corpus import Passage, sample_passages
from legalmind.data.prompts import (
    SYNTHESIS_SCHEMA,
    SYNTHESIS_SYSTEM_PROMPT,
    build_user_prompt,
    required_task_type_for,
)

DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_MAX_TOKENS = 4000
POLL_INTERVAL_SECONDS = 30

# USD per million tokens, standard (non-batch) rates.
# Sonnet 5 carries introductory pricing of $2/$10 through 2026-08-31; after that
# it reverts to $3/$15. The estimate below uses the standard rate so it errs
# high rather than low — an estimate that under-reports is worse than useless.
PRICING: dict[str, tuple[float, float]] = {
    "claude-sonnet-5": (3.00, 15.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-haiku-4-5": (1.00, 5.00),
}
BATCH_DISCOUNT = 0.5
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25


@dataclass
class Usage:
    """Token totals across a batch, kept separate by cache tier because they
    are billed at very different rates."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    def add(self, usage: Any) -> None:
        self.input_tokens += getattr(usage, "input_tokens", 0) or 0
        self.output_tokens += getattr(usage, "output_tokens", 0) or 0
        self.cache_creation_input_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0
        self.cache_read_input_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0

    def cost_usd(self, model: str, *, batch: bool = True) -> float:
        if model not in PRICING:
            raise KeyError(f"no pricing entry for {model!r}; add one to PRICING")
        per_mtok_in, per_mtok_out = PRICING[model]
        discount = BATCH_DISCOUNT if batch else 1.0
        cost = (
            self.input_tokens * per_mtok_in
            + self.cache_creation_input_tokens * per_mtok_in * CACHE_WRITE_MULTIPLIER
            + self.cache_read_input_tokens * per_mtok_in * CACHE_READ_MULTIPLIER
            + self.output_tokens * per_mtok_out
        ) / 1_000_000
        return cost * discount

    def describe(self, model: str) -> str:
        total_in = (
            self.input_tokens + self.cache_creation_input_tokens + self.cache_read_input_tokens
        )
        return (
            f"input {total_in:,} tok "
            f"(uncached {self.input_tokens:,} / cache-write {self.cache_creation_input_tokens:,} "
            f"/ cache-read {self.cache_read_input_tokens:,}), "
            f"output {self.output_tokens:,} tok, "
            f"cost ${self.cost_usd(model):.2f}"
        )


def build_requests(
    passages: list[Passage],
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> list[Request]:
    """One batch request per passage, asking for 2-3 instruction pairs.

    `cache_control` sits on the system block so the shared prefix — and only the
    shared prefix — is cached. The passage goes in the user turn, after the
    breakpoint, where it belongs.
    """
    requests: list[Request] = []
    for index, passage in enumerate(passages):
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": [
                {
                    "type": "text",
                    "text": SYNTHESIS_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [
                {
                    "role": "user",
                    "content": build_user_prompt(
                        passage.source, passage.text, required_task_type_for(index)
                    ),
                }
            ],
            # Guarantees parseable output. Without it, roughly a few percent of
            # responses arrive with prose wrapped around the JSON and have to be
            # thrown away — at 10k requests that is real money.
            "output_config": {
                "format": {"type": "json_schema", "schema": SYNTHESIS_SCHEMA},
            },
        }
        requests.append(
            Request(
                custom_id=passage.passage_id,
                params=MessageCreateParamsNonStreaming(**params),  # type: ignore[typeddict-item]
            )
        )
    return requests


def submit(client: Anthropic, requests: list[Request]) -> str:
    batch = client.messages.batches.create(requests=requests)
    print(f"submitted batch {batch.id} ({len(requests)} requests)", file=sys.stderr)
    return batch.id


def poll(client: Anthropic, batch_id: str, *, interval: int = POLL_INTERVAL_SECONDS) -> None:
    """Block until the batch ends. Most batches finish well inside an hour; the
    hard ceiling is 24."""
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            counts = batch.request_counts
            print(
                f"batch ended: {counts.succeeded} succeeded, {counts.errored} errored, "
                f"{counts.canceled} canceled, {counts.expired} expired",
                file=sys.stderr,
            )
            return
        print(
            f"  {batch.processing_status}: {batch.request_counts.processing} in flight...",
            file=sys.stderr,
        )
        time.sleep(interval)


def collect(
    client: Anthropic,
    batch_id: str,
    passages_by_id: dict[str, Passage],
    out_path: Path,
    *,
    model: str,
) -> tuple[int, Usage]:
    """Stream results to JSONL. Results arrive in arbitrary order, so every
    lookup is keyed by `custom_id` — never by position."""
    usage = Usage()
    n_pairs = 0
    n_failed = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
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
                # Should not happen under structured outputs, but a truncated
                # response (stop_reason == "max_tokens") can still break parsing.
                n_failed += 1
                continue

            passage = passages_by_id.get(entry.custom_id)
            for pair in payload.get("pairs", []):
                record = {
                    "instruction": pair["instruction"],
                    "response": pair["response"],
                    "task_type": pair["task_type"],
                    "passage_id": entry.custom_id,
                    "source": passage.source if passage else "",
                    "source_url": passage.url if passage else "",
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_pairs += 1

    if n_failed:
        print(f"warning: {n_failed} requests failed or were unparseable", file=sys.stderr)
    print(f"wrote {n_pairs} pairs to {out_path}", file=sys.stderr)
    print(f"usage: {usage.describe(model)}", file=sys.stderr)
    if n_pairs:
        print(
            f"unit cost: ${usage.cost_usd(model) / n_pairs * 1000:.2f} per 1,000 pairs",
            file=sys.stderr,
        )
    return n_pairs, usage


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=Path("data/raw_pairs.jsonl"))
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap total passages — use 100 for the cost probe before a full run",
    )
    parser.add_argument(
        "--per-source",
        type=int,
        default=900,
        help=(
            "passages sampled per pile-of-law source (3 sources). "
            "900 x 3 sources x ~3 pairs = ~8,100 pairs, which at the measured "
            "post-fix retention and Sonnet 5 intro pricing lands near $35"
        ),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        metavar="BATCH_ID",
        default=None,
        help="skip submission and collect results from an existing batch",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="sample passages and report sizes without calling the API",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_dotenv(override=True)
    args = _parse_args(argv)

    passages: list[Passage] = []
    for source in ("cfr", "courtlisteneropinions", "federal_register"):
        want = args.per_source
        if args.limit is not None:
            want = min(want, max(1, args.limit // 3))
        print(f"sampling {want} passages from {source}...", file=sys.stderr)
        passages.extend(sample_passages(source, want, seed=args.seed))
    if args.limit is not None:
        passages = passages[: args.limit]

    total_chars = sum(p.n_chars for p in passages)
    print(
        f"{len(passages)} passages, {total_chars:,} chars "
        f"(~{total_chars // 4:,} tokens of passage text)",
        file=sys.stderr,
    )

    if args.dry_run:
        print("dry run — nothing submitted", file=sys.stderr)
        return 0

    passages_by_id = {p.passage_id: p for p in passages}
    client = Anthropic()

    batch_id = args.resume
    if batch_id is None:
        batch_id = submit(
            client, build_requests(passages, model=args.model, max_tokens=args.max_tokens)
        )
        # Persist immediately: if polling dies, --resume picks up from here
        # rather than paying for the whole batch twice.
        Path("data/.last_batch_id").write_text(batch_id)

    poll(client, batch_id)
    collect(client, batch_id, passages_by_id, args.out, model=args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
