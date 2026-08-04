"""Tests for the synthesis prompts.

Both properties asserted here were found by measuring a real 99-passage probe,
not by reading the prompt. Prose asking for variety did not produce variety, and
a rule about self-contained instructions did nothing about self-contained
responses.
"""

from __future__ import annotations

import re

from legalmind.data.prompts import (
    SYNTHESIS_SYSTEM_PROMPT,
    TASK_TYPES,
    build_user_prompt,
    required_task_type_for,
)


def test_required_type_rotates_evenly() -> None:
    """Measured on the first probe with the type left free: statutory
    interpretation came out at 4% of pairs against ~32% for each of the other
    three, consistently across all three sources. Rotation makes the
    distribution a property of the batch rather than a hope."""
    assigned = [required_task_type_for(i) for i in range(len(TASK_TYPES) * 25)]
    counts = {t: assigned.count(t) for t in TASK_TYPES}
    assert set(counts.values()) == {25}, counts


def test_user_prompt_carries_the_required_type() -> None:
    prompt = build_user_prompt("cfr", "Some legal text.", "statutory_interpretation")
    assert "statutory_interpretation" in prompt
    assert "Some legal text." in prompt


def test_unknown_task_type_is_rejected() -> None:
    try:
        build_user_prompt("cfr", "text", "freeform")
    except ValueError as exc:
        assert "freeform" in str(exc)
    else:
        raise AssertionError("expected ValueError for an unknown task type")


def test_system_prompt_forbids_dangling_references_in_responses() -> None:
    """31.6% of probe responses referred back to the source passage. The rule
    covering only instructions caught none of them: the model writes a
    self-contained question and then reaches back to the passage in the answer."""
    lowered = SYNTHESIS_SYSTEM_PROMPT.lower()
    assert "self-contained" in lowered
    assert "response is where it goes wrong" in lowered
    # The concrete bad phrasings must be named, not just described.
    assert "the passage does not commit" in lowered
    assert "the text specifies" in lowered


def test_system_prompt_forbids_disclaimer_boilerplate() -> None:
    lowered = SYNTHESIS_SYSTEM_PROMPT.lower()
    assert "no disclaimers" in lowered
    assert "not legal advice" in lowered


def test_system_prompt_forbids_fabricated_citations() -> None:
    assert "never fabricate a citation" in SYNTHESIS_SYSTEM_PROMPT.lower()


def test_system_prompt_holds_no_per_request_placeholders() -> None:
    """The system prompt is the cached prefix for the whole batch. A single
    varying byte here invalidates the cache for every request."""
    assert not re.search(r"\{[a-z_]+\}", SYNTHESIS_SYSTEM_PROMPT)
