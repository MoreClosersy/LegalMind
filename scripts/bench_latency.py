"""Measure serving latency under load, per arm.

    uv run python scripts/bench_latency.py --concurrency 1 4 16 \
        --out eval_results/latency.json

This produces the other half of the arm-B-versus-arm-C argument. The accuracy
comparison can only say whether the fine-tune is better; this says what the
alternative costs on every single request, forever. Arm B carries a system
prompt and two exemplars into every call — roughly 800 extra input tokens — and
that shows up as prefill time and as money.

What is reported and why:

* **TTFT** (time to first token) — what a user perceives as responsiveness, and
  the thing arm B's long prompt directly inflates through prefill.
* **p50 and p95, never the mean.** A mean latency hides the tail, and the tail is
  what makes a service feel broken. Reported at each concurrency level because a
  number from a single-threaded loop describes a machine nobody is using.
* **Output tokens per second**, so a slow arm that is slow because it writes more
  can be told apart from one that is slow per token. Without it, "arm C is
  slower" and "arm C is more verbose" look identical.

The runs are interleaved rather than run arm-by-arm. Sequential blocks let
thermal state, cache warmth, and any other machine drift line up with arm
identity, which turns a measurement into an artifact of ordering.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

DEFAULT_GATEWAY = "http://localhost:8080"

# Deliberately mixed. A benchmark of one prompt shape measures one prompt shape:
# a short refusal and a full multi-paragraph explanation have very different
# output lengths, and averaging over only one of them misdescribes the service.
PROMPTS: tuple[str, ...] = (
    "What are the elements of adverse possession?",
    "My landlord kept my deposit. Should I sue?",
    "What is the difference between a motion to dismiss and a motion for summary judgment?",
    "Explain the parol evidence rule.",
    "I was fired after reporting a safety violation. Do I have a case?",
    "What standard of review applies to an agency's interpretation of its own regulation?",
)


@dataclass
class Sample:
    ttft_s: float | None
    total_s: float
    completion_tokens: int
    ok: bool


@dataclass
class ArmLatency:
    arm: str
    adapter: str | None
    concurrency: int
    samples: list[Sample] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        ok = [s for s in self.samples if s.ok]
        if not ok:
            return {"arm": self.arm, "concurrency": self.concurrency, "error": "no successful runs"}
        totals = sorted(s.total_s for s in ok)
        ttfts = sorted(s.ttft_s for s in ok if s.ttft_s is not None)
        tokens = sum(s.completion_tokens for s in ok)
        wall = sum(s.total_s for s in ok)
        return {
            "arm": self.arm,
            "adapter": self.adapter,
            "concurrency": self.concurrency,
            "n": len(ok),
            "failed": len(self.samples) - len(ok),
            "ttft_p50_ms": round(_pct(ttfts, 0.50) * 1000, 1) if ttfts else None,
            "ttft_p95_ms": round(_pct(ttfts, 0.95) * 1000, 1) if ttfts else None,
            "latency_p50_ms": round(_pct(totals, 0.50) * 1000, 1),
            "latency_p95_ms": round(_pct(totals, 0.95) * 1000, 1),
            # Kept only to show how far it sits from p95. A mean latency is the
            # number that makes a tail-latency problem invisible.
            "latency_mean_ms": round(statistics.mean(totals) * 1000, 1),
            "mean_completion_tokens": round(tokens / len(ok), 1),
            "output_tokens_per_s": round(tokens / wall, 1) if wall else 0.0,
        }


def _pct(sorted_values: list[float], q: float) -> float:
    """Nearest-rank percentile.

    Not interpolated: at the sample counts a laptop-driven benchmark produces,
    an interpolated p95 invents a value between two real measurements and reads
    as more precise than the data supports.
    """
    if not sorted_values:
        return 0.0
    # Nearest rank: the smallest value at or below which at least q of the
    # samples fall. ceil(q*N)-1 in zero-based terms.
    index = min(len(sorted_values) - 1, max(0, math.ceil(q * len(sorted_values)) - 1))
    return sorted_values[index]


async def one_request(
    client: httpx.AsyncClient, gateway: str, prompt: str, adapter: str | None, max_tokens: int
) -> Sample:
    """One streaming request, timed at the first content delta.

    The gateway's `meta` event arrives before generation starts, so timing TTFT
    from it would measure the gateway rather than the model. Only a `delta`
    counts.
    """
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "adapter": adapter,
        "max_tokens": max_tokens,
        "stream": True,
    }
    started = time.perf_counter()
    ttft: float | None = None
    tokens = 0
    try:
        async with client.stream("POST", f"{gateway}/v1/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                try:
                    chunk = json.loads(line[len("data:") :].strip())
                except json.JSONDecodeError:
                    continue
                if "text" in chunk:
                    if ttft is None:
                        ttft = time.perf_counter() - started
                    tokens += 1
    except (httpx.HTTPError, json.JSONDecodeError):
        return Sample(
            ttft_s=ttft, total_s=time.perf_counter() - started, completion_tokens=tokens, ok=False
        )
    return Sample(
        ttft_s=ttft, total_s=time.perf_counter() - started, completion_tokens=tokens, ok=True
    )


async def measure(
    client: httpx.AsyncClient,
    gateway: str,
    *,
    arm: str,
    adapter: str | None,
    concurrency: int,
    requests: int,
    max_tokens: int,
) -> ArmLatency:
    result = ArmLatency(arm=arm, adapter=adapter, concurrency=concurrency)
    semaphore = asyncio.Semaphore(concurrency)

    async def run(i: int) -> Sample:
        async with semaphore:
            return await one_request(
                client, gateway, PROMPTS[i % len(PROMPTS)], adapter, max_tokens
            )

    result.samples = list(await asyncio.gather(*(run(i) for i in range(requests))))
    return result


async def _run(args: argparse.Namespace) -> int:
    arms: list[tuple[str, str | None]] = [("base", None), ("fine_tuned", args.adapter)]
    results: list[dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        try:
            health = await client.get(f"{args.gateway}/healthz")
            health.raise_for_status()
        except httpx.HTTPError as exc:
            print(f"gateway not reachable at {args.gateway}: {exc}", file=sys.stderr)
            return 2

        print(f"warming up ({args.warmup} requests per arm)...", file=sys.stderr)
        for arm, adapter in arms:
            await measure(
                client,
                args.gateway,
                arm=arm,
                adapter=adapter,
                concurrency=1,
                requests=args.warmup,
                max_tokens=args.max_tokens,
            )

        for concurrency in args.concurrency:
            # Interleaved by concurrency level so machine drift cannot line up
            # with arm identity.
            for arm, adapter in arms:
                print(f"  {arm} @ concurrency {concurrency}...", file=sys.stderr)
                measured = await measure(
                    client,
                    args.gateway,
                    arm=arm,
                    adapter=adapter,
                    concurrency=concurrency,
                    requests=args.requests,
                    max_tokens=args.max_tokens,
                )
                summary = measured.summary()
                results.append(summary)
                print(f"    {json.dumps(summary)}", file=sys.stderr)

    payload = {
        "gateway": args.gateway,
        "requests_per_cell": args.requests,
        "max_tokens": args.max_tokens,
        "prompts": list(PROMPTS),
        "note": (
            "p50/p95 are nearest-rank, not interpolated. The mean is reported only "
            "to show how far it sits from p95. Arms are interleaved within each "
            "concurrency level so machine drift cannot align with arm identity."
        ),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gateway", default=DEFAULT_GATEWAY)
    parser.add_argument("--adapter", default="legalmind")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--requests", type=int, default=48, help="requests per arm per level")
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--out", type=Path, default=Path("eval_results/latency.json"))
    return asyncio.run(_run(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
