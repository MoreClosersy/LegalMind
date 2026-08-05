"""Shared generation client for every evaluation arm.

All three arms are served by one vLLM process started with `--enable-lora`, so
switching arms is a change of `model` in the request body — the base model name
for arms A and B, the LoRA module name for arm C. That matters for fairness as
much as for cost: the same weights, the same sampler, the same server, the same
machine, in the same session. Restarting the server between arms would let
kernel autotuning and memory fragmentation drift between measurements.

Sampling is greedy (`temperature=0`) everywhere. Benchmark accuracy measured
under sampling is a noisy estimate of a quantity nobody wants; the question is
what the model's best answer is, not what its distribution looks like.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_TIMEOUT = 180.0

# vLLM serves many requests concurrently and schedules them itself; this only
# bounds how many are outstanding client-side. Too high and the client starves
# on connection setup, too low and the GPU idles between batches.
DEFAULT_CONCURRENCY = 32


@dataclass(frozen=True)
class Generation:
    """One completion, plus what it cost to produce.

    Latency is recorded per request even during accuracy runs. It is nearly free
    to collect here, and it is the input to the arm-B-versus-arm-C cost argument:
    arm B pays for its system prompt and exemplars on every single request.
    """

    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


class GenerationClient:
    """Thin async client for a vLLM OpenAI-compatible endpoint."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        concurrency: int = DEFAULT_CONCURRENCY,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._semaphore = asyncio.Semaphore(concurrency)
        self._client = httpx.AsyncClient(timeout=timeout)

    async def __aenter__(self) -> GenerationClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._client.aclose()

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        extra_body: dict[str, object] | None = None,
    ) -> Generation:
        payload: dict[str, object] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if extra_body:
            payload |= extra_body

        async with self._semaphore:
            start = asyncio.get_running_loop().time()
            response = await self._client.post(f"{self._base_url}/chat/completions", json=payload)
            elapsed = asyncio.get_running_loop().time() - start

        response.raise_for_status()
        body = response.json()
        usage = body.get("usage") or {}
        return Generation(
            text=body["choices"][0]["message"]["content"] or "",
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            latency_s=elapsed,
        )

    async def complete_many(
        self,
        conversations: list[list[dict[str, str]]],
        *,
        model: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        extra_body: dict[str, object] | None = None,
    ) -> list[Generation]:
        """Run every conversation concurrently, preserving input order.

        `asyncio.gather` keeps results positional, which is what lets the caller
        zip generations back against gold labels. Anything that reorders here
        would silently mis-score every row.
        """
        tasks = [
            self.complete(
                conversation,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                extra_body=extra_body,
            )
            for conversation in conversations
        ]
        return await asyncio.gather(*tasks)
