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
import re
from dataclasses import dataclass

import httpx

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_TIMEOUT = 180.0

# vLLM serves many requests concurrently and schedules them itself; this only
# bounds how many are outstanding client-side. Too high and the client starves
# on connection setup, too low and the GPU idles between batches.
DEFAULT_CONCURRENCY = 32

# DOTALL so an empty or populated <think>...</think> block is matched as one
# unit regardless of the newlines Qwen3 pads it with. Anchored to the start of
# the string deliberately: a `<think>` appearing later in a response is part of
# the answer being discussed, not a wrapper around it, and must not be eaten.
_LEADING_THINK_BLOCK = re.compile(r"\A\s*<think>.*?</think>\s*", re.DOTALL)


def strip_thinking_block(text: str) -> str:
    """Drop a leading `<think>...</think>` block, if present.

    Every extractor in this package (`legalbench.extract_label`,
    `forgetting.extract_letter`, `refusal.looks_like_refusal`) scans from the
    start of the response. `GenerationClient` now asks for thinking to be
    disabled on every request — see the comment on the payload below — so this
    should rarely fire. It stays as a second line of defence rather than being
    removed, because the failure mode when it is missing is silent and total:
    a live run measured every extractor returning "unparseable" for 100% of
    responses across all three arms, with no exception and no error in the log,
    because the first non-blank line was always the literal word `<think>`.
    """
    return _LEADING_THINK_BLOCK.sub("", text, count=1)


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
            # Not a preference — see gateway.py's docstring. Training rendered
            # every prompt with thinking disabled, so any client that skips this
            # shows the model a prompt shape it never saw in training: it falls
            # back to Qwen3's default (thinking on), spends the completion
            # budget on an actual reasoning trace instead of the answer this
            # harness is scoring, and — for the fine-tuned arm, which never
            # learned to produce real reasoning content — degenerates to an
            # empty `<think>\n\n</think>\n\n` wrapper that every downstream
            # extractor chokes on. Found by measuring: this was live on a real
            # evaluation run before it was caught, and it was not visible in
            # any single number — only in completion-token counts being 60x
            # apart between arms and every arm's format-compliance rate being
            # a uniform 0%.
            "chat_template_kwargs": {"enable_thinking": False},
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
