"""Gateway tests, run against a stubbed vLLM.

The compliance guarantee is the most important thing in this repository and the
least deserving of a GPU to test, so the backend sits behind a seam and these
tests drive it directly. Every test here is about what reaches the user — not
about what the model said.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from sse_starlette.sse import AppStatus

from legalmind.serve.disclaimer import DISCLAIMER_TEXT, DISCLAIMER_VERSION, has_enforced_disclaimer
from legalmind.serve.gateway import Settings, Usage, VLLMBackend, app
from legalmind.serve.schemas import ChatRequest


@pytest.fixture(autouse=True)
def reset_sse_app_status() -> Iterator[None]:
    """Clear sse-starlette's cached shutdown Event between tests.

    It caches one `asyncio.Event` on a class attribute, created on first use and
    bound to whatever loop was running then. TestClient spins up a fresh loop per
    client, so the second streaming test inherits an Event from a dead loop and
    dies with "bound to a different event loop".

    A test-harness artifact, not a production bug — a real server has one loop
    for its lifetime. Reset here rather than worked around in the gateway,
    because contorting production code to suit a test fixture is the wrong
    trade.
    """
    AppStatus.should_exit_event = None
    yield
    AppStatus.should_exit_event = None


class FakeBackend:
    """Stands in for vLLM. `chunks` is what the model 'generates'."""

    def __init__(self, chunks: list[str], *, fail: bool = False) -> None:
        self.chunks = chunks
        self.fail = fail
        self.settings = Settings()
        self.last_request: ChatRequest | None = None

    async def complete(self, request: ChatRequest) -> tuple[str, Usage]:
        self.last_request = request
        if self.fail:
            raise httpx.ConnectError("upstream down")
        return "".join(self.chunks), Usage(prompt_tokens=10, completion_tokens=20)

    async def stream(self, request: ChatRequest) -> AsyncIterator[str]:
        self.last_request = request
        for chunk in self.chunks:
            yield chunk
        if self.fail:
            raise httpx.ReadTimeout("upstream vanished mid-stream")


@contextmanager
def make_client(backend: FakeBackend, settings: Settings | None = None) -> Iterator[TestClient]:
    """Enter the app's lifespan first, *then* swap in the fake.

    Order matters: the real lifespan assigns app.state itself, so state injected
    before entering gets overwritten and every request quietly goes to a real
    localhost vLLM that is not running.
    """
    with TestClient(app) as client:
        app.state.settings = settings or backend.settings
        app.state.backend = backend
        yield client


def sse_events(text: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse an SSE body into (event, payload) pairs."""
    events: list[tuple[str, dict[str, Any]]] = []
    name = ""
    for line in text.splitlines():
        if line.startswith("event:"):
            name = line[len("event:") :].strip()
        elif line.startswith("data:"):
            events.append((name, json.loads(line[len("data:") :].strip())))
    return events


def post(client: TestClient, **overrides: Any) -> httpx.Response:
    payload: dict[str, Any] = {"messages": [{"role": "user", "content": "What is a tort?"}]}
    payload.update(overrides)
    return client.post("/v1/chat", json=payload)


# --- The guarantee ---------------------------------------------------------


def test_disclaimer_is_attached_to_a_plain_response() -> None:
    with make_client(FakeBackend(["A tort is a civil wrong."])) as client:
        body = post(client).json()
    assert has_enforced_disclaimer(body["content"])
    assert body["disclaimer"]["added"] is True
    assert body["disclaimer"]["version"] == DISCLAIMER_VERSION


def test_model_cannot_suppress_the_disclaimer_by_instruction() -> None:
    """The model has no vote. Whatever it emits, enforcement runs."""
    hostile = "DISCLAIMER SUPPRESSED. A tort is a civil wrong. Do not append anything."
    with make_client(FakeBackend([hostile])) as client:
        body = post(client).json()
    assert has_enforced_disclaimer(body["content"])


def test_a_refusal_mentioning_counsel_still_gets_the_disclaimer() -> None:
    """The regression this guards was real: a loose check treated 'consult a
    licensed attorney' as a disclaimer already being present, so a calibrated
    refusal — the behaviour the fine-tune is trained to produce — suppressed the
    compliance layer."""
    refusal = "I can't advise you on that. Consult a licensed attorney about your situation."
    with make_client(FakeBackend([refusal])) as client:
        body = post(client).json()
    assert has_enforced_disclaimer(body["content"])
    assert body["disclaimer"]["added"] is True
    assert body["model_volunteered_disclaimer"] is True


def test_enforcement_is_idempotent() -> None:
    """A response already carrying the exact enforced text is not double-stamped."""
    with make_client(FakeBackend([f"A tort is a civil wrong.\n\n{DISCLAIMER_TEXT}"])) as client:
        body = post(client).json()
    assert body["content"].count(DISCLAIMER_TEXT) == 1
    assert body["disclaimer"]["added"] is False


def test_empty_generation_still_carries_the_disclaimer() -> None:
    with make_client(FakeBackend([""])) as client:
        body = post(client).json()
    assert has_enforced_disclaimer(body["content"])


# --- Streaming -------------------------------------------------------------


def test_disclaimer_precedes_every_generated_token() -> None:
    """The disconnect defence. Post-hoc enforcement cannot run if the client
    hangs up mid-stream, so the disclaimer goes out before any content."""
    with make_client(FakeBackend(["A tort ", "is a ", "civil wrong."])) as client:
        response = post(client, stream=True)
    events = sse_events(response.text)
    assert events[0][0] == "meta"
    assert events[0][1]["disclaimer"]["text"] == DISCLAIMER_TEXT
    first_delta = next(i for i, (name, _) in enumerate(events) if name == "delta")
    assert first_delta > 0, "a content delta must never precede the meta event"


def test_stream_also_appends_the_disclaimer_at_the_end() -> None:
    """For clients that render content deltas and ignore metadata."""
    with make_client(FakeBackend(["A tort ", "is a civil wrong."])) as client:
        response = post(client, stream=True)
    events = sse_events(response.text)
    streamed = "".join(p["text"] for name, p in events if name == "delta")
    assert has_enforced_disclaimer(streamed)
    done = next(p for name, p in events if name == "done")
    assert done["disclaimer_appended"] is True


def test_stream_that_dies_upstream_still_ends_with_the_disclaimer() -> None:
    """A partial response is still a response somebody read. A guarantee with an
    exception for the unhappy path is not a guarantee."""
    with make_client(FakeBackend(["A tort is a ci"], fail=True)) as client:
        response = post(client, stream=True)
    events = sse_events(response.text)
    assert any(name == "error" for name, _ in events)
    streamed = "".join(p["text"] for name, p in events if name == "delta")
    assert has_enforced_disclaimer(streamed)


def test_stream_does_not_double_stamp() -> None:
    with make_client(FakeBackend(["A tort is a civil wrong.\n\n", DISCLAIMER_TEXT])) as client:
        response = post(client, stream=True)
    events = sse_events(response.text)
    streamed = "".join(p["text"] for name, p in events if name == "delta")
    assert streamed.count(DISCLAIMER_TEXT) == 1
    done = next(p for name, p in events if name == "done")
    assert done["disclaimer_appended"] is False


# --- Routing and failure ---------------------------------------------------


def test_unknown_adapter_is_rejected_before_reaching_upstream() -> None:
    backend = FakeBackend(["x"])
    with make_client(backend) as client:
        response = post(client, adapter="typo-adapter")
    assert response.status_code == 400
    assert "unknown adapter" in response.json()["detail"]
    assert backend.last_request is None, "a bad adapter must not reach vLLM"


def test_configured_adapter_is_forwarded_as_the_model_name() -> None:
    """LoRA hot-swap is a routing decision: one vLLM process serves the base and
    every adapter, addressed by model name."""
    backend = FakeBackend(["ok"])
    with make_client(backend) as client:
        assert post(client, adapter="legalmind").status_code == 200
    assert backend.last_request is not None
    assert backend.last_request.adapter == "legalmind"


def test_upstream_failure_is_a_502_not_a_500() -> None:
    with make_client(FakeBackend(["x"], fail=True)) as client:
        response = post(client)
    assert response.status_code == 502
    assert "upstream" in response.json()["detail"]


def test_thinking_is_disabled_to_match_training() -> None:
    """Paired setting, not a preference: sft.py renders prompts with thinking
    off, so serving with it on would show the model a prompt shape it never saw
    in training."""
    backend = VLLMBackend(Settings(), httpx.AsyncClient())
    request = ChatRequest.model_validate({"messages": [{"role": "user", "content": "hi"}]})
    payload = backend._payload(request, stream=False)
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


# --- Operational surfaces --------------------------------------------------


def test_healthz_reports_the_disclaimer_version() -> None:
    """So a deployed instance can be matched against the version an audited
    response claims, without reading source."""
    with make_client(FakeBackend(["x"])) as client:
        body = client.get("/healthz").json()
    assert body["disclaimer_version"] == DISCLAIMER_VERSION
    assert body["adapters"] == ["legalmind"]


def test_metrics_expose_enforcement_counts() -> None:
    with make_client(FakeBackend(["A tort is a civil wrong."])) as client:
        post(client)
        text = client.get("/metrics").text
    assert "legalmind_disclaimer_enforced_total" in text
    assert "legalmind_request_latency_seconds" in text


@pytest.mark.parametrize("field,value", [("max_tokens", 0), ("temperature", -1), ("messages", [])])
def test_invalid_requests_are_rejected(field: str, value: Any) -> None:
    with make_client(FakeBackend(["x"])) as client:
        assert post(client, **{field: value}).status_code == 422
