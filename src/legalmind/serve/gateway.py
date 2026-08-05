"""FastAPI gateway in front of vLLM: compliance, streaming, and metrics.

    vllm serve Qwen/Qwen3-8B --enable-lora \
        --lora-modules legalmind=checkpoints/legalmind-qwen3-8b

    uv run uvicorn legalmind.serve.gateway:app --port 8080

The gateway exists so that the UPL disclaimer is a property of the *system*
rather than of the model. Nothing generated upstream can remove it, because the
model is not consulted about whether it appears.

Two things here are less obvious than they look.

**Streaming versus a deterministic guarantee.** Post-hoc enforcement needs the
finished text, but a streaming client is already reading. Buffering the whole
response to enforce before sending throws away the only reason to stream.
Appending the disclaimer as a final chunk works right up until the connection
drops mid-stream — and then a user has read legal text with no disclaimer, which
is the exact failure the deterministic layer was supposed to make impossible.
The resolution is to send the disclaimer *first*, as a metadata event, before
any generated token, and to append it again at the end for clients that only
render content deltas. Every byte of generated text a client can possibly
receive has then been preceded by the disclaimer, disconnects included.

**Enforcement runs on the error path too.** An upstream timeout or a truncated
generation still produces text that reaches a user, so the compliance layer runs
before anything is returned regardless of how generation ended. A guarantee with
an exception for the unhappy path is not a guarantee, and the unhappy path is
where an audit will look first.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from sse_starlette.sse import EventSourceResponse

from legalmind.serve.disclaimer import (
    DISCLAIMER_TEXT,
    DISCLAIMER_VERSION,
    enforce,
    enforce_stream_tail,
    looks_like_volunteered_disclaimer,
)
from legalmind.serve.schemas import (
    ChatRequest,
    ChatResponse,
    Disclaimer,
    StreamDone,
    StreamMeta,
    Usage,
)

DEFAULT_UPSTREAM = "http://localhost:8000/v1"
DEFAULT_MODEL = "Qwen/Qwen3-8B"

REQUESTS = Counter("legalmind_requests_total", "Chat requests", ["adapter", "streaming", "outcome"])
DISCLAIMER_ENFORCED = Counter(
    "legalmind_disclaimer_enforced_total",
    "Responses the gateway attached the disclaimer to",
    ["adapter", "added"],
)
VOLUNTEERED = Counter(
    "legalmind_model_volunteered_disclaimer_total",
    "Responses where the model produced disclaimer-shaped prose on its own. Not a "
    "gate — a rising count means training data has leaked disclaimers",
    ["adapter"],
)
LATENCY = Histogram(
    "legalmind_request_latency_seconds",
    "End-to-end gateway latency",
    ["adapter", "streaming"],
    # Buckets chosen for an 8B model on one A10G: sub-second is a short refusal,
    # the long tail is a full multi-paragraph explanation.
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0),
)
TTFT = Histogram(
    "legalmind_time_to_first_token_seconds",
    "Time to first generated token, streaming requests only",
    ["adapter"],
    buckets=(0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2),
)


@dataclass
class Settings:
    upstream: str = DEFAULT_UPSTREAM
    model: str = DEFAULT_MODEL
    # Adapters the gateway will forward. A request naming anything else is a 400
    # rather than an opaque upstream failure, and the list doubles as the
    # documentation of what this deployment can serve.
    adapters: frozenset[str] = frozenset({"legalmind"})
    timeout_s: float = 120.0

    @classmethod
    def from_env(cls) -> Settings:
        adapters = os.getenv("LEGALMIND_ADAPTERS", "legalmind")
        return cls(
            upstream=os.getenv("LEGALMIND_UPSTREAM", DEFAULT_UPSTREAM),
            model=os.getenv("LEGALMIND_MODEL", DEFAULT_MODEL),
            adapters=frozenset(a.strip() for a in adapters.split(",") if a.strip()),
            timeout_s=float(os.getenv("LEGALMIND_TIMEOUT_S", "120")),
        )


class VLLMBackend:
    """Everything that talks to vLLM, behind one seam.

    Isolated so the gateway's compliance behaviour can be tested without a GPU.
    The disclaimer guarantee is the thing most worth testing and the thing least
    worth needing an 8B model to test.
    """

    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self.settings = settings
        self._client = client

    def _payload(self, request: ChatRequest, *, stream: bool) -> dict[str, Any]:
        return {
            # A LoRA adapter is addressed as a model name by vLLM, which is what
            # makes hot-swapping an adapter a routing decision rather than a
            # redeploy: one process serves the base and every adapter at once.
            "model": request.adapter or self.settings.model,
            "messages": [m.model_dump() for m in request.messages],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": stream,
            # Training rendered prompts with thinking disabled, so serving must
            # match or the model sees a prompt shape it was never trained on.
            # See train/sft.py — this is a paired setting, not a preference.
            "chat_template_kwargs": {"enable_thinking": False},
        }

    async def complete(self, request: ChatRequest) -> tuple[str, Usage]:
        response = await self._client.post(
            f"{self.settings.upstream}/chat/completions",
            json=self._payload(request, stream=False),
            timeout=self.settings.timeout_s,
        )
        response.raise_for_status()
        body = response.json()
        usage = body.get("usage") or {}
        return (
            body["choices"][0]["message"]["content"] or "",
            Usage(
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
            ),
        )

    async def stream(self, request: ChatRequest) -> AsyncIterator[str]:
        async with self._client.stream(
            "POST",
            f"{self.settings.upstream}/chat/completions",
            json=self._payload(request, stream=True),
            timeout=self.settings.timeout_s,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    return
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = chunk["choices"][0].get("delta", {}).get("content")
                if delta:
                    yield delta


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings.from_env()
    client = httpx.AsyncClient()
    app.state.settings = settings
    app.state.backend = VLLMBackend(settings, client)
    try:
        yield
    finally:
        await client.aclose()


app = FastAPI(title="LegalMind gateway", lifespan=lifespan)


def _validate(request: ChatRequest, settings: Settings) -> None:
    if request.adapter is not None and request.adapter not in settings.adapters:
        raise HTTPException(
            status_code=400,
            detail=(
                f"unknown adapter {request.adapter!r}; this deployment serves "
                f"{sorted(settings.adapters) or ['(base model only)']}"
            ),
        )


@app.get("/healthz")
async def healthz(request: Request) -> dict[str, Any]:
    settings: Settings = request.app.state.settings
    return {
        "status": "ok",
        "model": settings.model,
        "adapters": sorted(settings.adapters),
        # Surfaced so a deployed instance can be checked against the version an
        # audited response claims to carry, without reading the source.
        "disclaimer_version": DISCLAIMER_VERSION,
    }


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest().decode(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request) -> Any:
    settings: Settings = request.app.state.settings
    backend: VLLMBackend = request.app.state.backend
    _validate(body, settings)
    label = body.adapter or "base"

    if body.stream:
        return await _stream_response(body, backend, label)

    started = time.perf_counter()
    try:
        raw, usage = await backend.complete(body)
        outcome = "ok"
    except httpx.HTTPError as exc:
        REQUESTS.labels(label, "false", "upstream_error").inc()
        raise HTTPException(status_code=502, detail=f"upstream error: {exc}") from exc

    # Enforcement is not inside the try: it must run on whatever text came back,
    # including a truncated generation.
    result = enforce(raw)
    elapsed = time.perf_counter() - started

    REQUESTS.labels(label, "false", outcome).inc()
    LATENCY.labels(label, "false").observe(elapsed)
    DISCLAIMER_ENFORCED.labels(label, str(result.disclaimer_added).lower()).inc()
    if result.model_volunteered_disclaimer:
        VOLUNTEERED.labels(label).inc()

    return ChatResponse(
        content=result.text,
        disclaimer=Disclaimer(added=result.disclaimer_added, version=result.version),
        usage=usage,
        adapter=body.adapter,
        model=settings.model,
        latency_ms=round(elapsed * 1000, 2),
        model_volunteered_disclaimer=result.model_volunteered_disclaimer,
    )


async def _stream_response(
    body: ChatRequest, backend: VLLMBackend, label: str
) -> EventSourceResponse:
    async def events() -> AsyncIterator[dict[str, str]]:
        started = time.perf_counter()
        first_token_at: float | None = None
        collected: list[str] = []

        # Before any generated token. This is what survives a disconnect.
        yield {
            "event": "meta",
            "data": StreamMeta(
                disclaimer=Disclaimer(text=DISCLAIMER_TEXT, version=DISCLAIMER_VERSION, added=True),
                model=backend.settings.model,
                adapter=body.adapter,
            ).model_dump_json(),
        }

        outcome = "ok"
        try:
            async for delta in backend.stream(body):
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                    TTFT.labels(label).observe(first_token_at - started)
                collected.append(delta)
                yield {"event": "delta", "data": json.dumps({"text": delta})}
        except httpx.HTTPError as exc:
            outcome = "upstream_error"
            yield {"event": "error", "data": json.dumps({"detail": f"upstream error: {exc}"})}

        # Runs whether the stream completed or died upstream: a partial response
        # is still a response somebody read.
        body_text = "".join(collected)
        tail = enforce_stream_tail(body_text)
        if tail:
            yield {"event": "delta", "data": json.dumps({"text": tail})}

        elapsed = time.perf_counter() - started
        REQUESTS.labels(label, "true", outcome).inc()
        LATENCY.labels(label, "true").observe(elapsed)
        DISCLAIMER_ENFORCED.labels(label, str(bool(tail)).lower()).inc()
        if looks_like_volunteered_disclaimer(body_text):
            VOLUNTEERED.labels(label).inc()

        yield {
            "event": "done",
            "data": StreamDone(
                disclaimer_appended=bool(tail),
                usage=Usage(completion_tokens=len(collected)),
                latency_ms=round(elapsed * 1000, 2),
            ).model_dump_json(),
        }

    return EventSourceResponse(events())
