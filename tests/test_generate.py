"""Tests for the shared generation client and its thinking-block guard.

The guard exists because of a real incident: a live evaluation run measured
every extractor in the package returning "unparseable" for 100% of responses,
across all three arms, with no exception anywhere. The root cause was one
missing field in one request payload — `GenerationClient` talks to vLLM
directly and never asked for thinking to be disabled, so every response opened
with a literal `<think>...</think>` line that no extractor was built to skip.
These tests pin both the fix and the regression it fixes.
"""

from __future__ import annotations

from typing import Any

import pytest

from legalmind.eval.generate import GenerationClient, strip_thinking_block


def test_empty_think_block_is_stripped() -> None:
    """The exact shape Qwen3 emits when told not to think: an empty block."""
    assert strip_thinking_block("<think>\n\n</think>\n\nNo") == "No"


def test_populated_think_block_is_stripped() -> None:
    assert strip_thinking_block("<think>\nweighing both readings...\n</think>\n\nYes") == "Yes"


def test_text_with_no_think_block_is_unchanged() -> None:
    assert strip_thinking_block("No") == "No"
    assert strip_thinking_block("Label: No") == "Label: No"


def test_a_think_tag_later_in_the_text_is_not_eaten() -> None:
    """Anchored to the start on purpose. A response that discusses `<think>`
    tags, or a malformed response with a stray tag mid-answer, is not the
    wrapper this function exists to remove."""
    text = "Yes, and note the model's own <think> tag is discussed in the brief."
    assert strip_thinking_block(text) == text


def test_leading_whitespace_before_the_block_is_absorbed() -> None:
    assert strip_thinking_block("  \n<think>\n\n</think>\nYes") == "Yes"


def test_empty_string_is_safe() -> None:
    assert strip_thinking_block("") == ""


class _FakeResponse:
    def __init__(self, body: dict[str, Any]) -> None:
        self._body = body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._body


@pytest.mark.asyncio
async def test_complete_sends_thinking_disabled_on_every_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The actual regression test. This is what a live run was missing: the
    client talks to vLLM directly (never through the gateway, which is where
    this setting used to live alone), so it has to ask for this itself on
    every request rather than relying on a caller to remember `extra_body`."""
    captured: dict[str, Any] = {}

    async def fake_post(self: Any, url: str, *, json: dict[str, Any]) -> _FakeResponse:
        captured["url"] = url
        captured["json"] = json
        return _FakeResponse({"choices": [{"message": {"content": "No"}}], "usage": {}})

    monkeypatch.setattr("httpx.AsyncClient.post", fake_post)

    async with GenerationClient() as client:
        await client.complete([{"role": "user", "content": "hi"}], model="m")

    assert captured["json"]["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.asyncio
async def test_extra_body_can_override_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """A caller with a genuine reason to want thinking on must still be able
    to ask for it — the default is a starting point, not a lock."""
    captured: dict[str, Any] = {}

    async def fake_post(self: Any, url: str, *, json: dict[str, Any]) -> _FakeResponse:
        captured["json"] = json
        return _FakeResponse({"choices": [{"message": {"content": "x"}}], "usage": {}})

    monkeypatch.setattr("httpx.AsyncClient.post", fake_post)

    async with GenerationClient() as client:
        await client.complete(
            [{"role": "user", "content": "hi"}],
            model="m",
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )

    assert captured["json"]["chat_template_kwargs"] == {"enable_thinking": True}
