"""Tests for the refusal-calibration set.

The properties asserted here are the ones that make the calibration numbers in
Phase 3 mean something. Each is a way the set could look fine and measure
nothing.
"""

from __future__ import annotations

import re

from legalmind.data.refusal_seeds import SEED_PAIRS, general_questions, personal_questions
from legalmind.data.refusal_set import REFUSAL_SYSTEM_PROMPT, build_user_prompt, split_topics

# Same patterns the serving layer treats as disclaimer text. Training data must
# not contain them — that guarantee lives in the gateway, not the weights.
_DISCLAIMER = re.compile(
    r"(not legal advice|educational purposes only|i am not a lawyer)", re.IGNORECASE
)


def test_every_topic_is_a_matched_pair() -> None:
    """A refusal set without paired should-answer prompts cannot detect
    over-refusal, and a model that refuses everything would score perfectly."""
    for pair in SEED_PAIRS:
        assert pair.personal and pair.general
        assert pair.personal != pair.general
    assert len(personal_questions()) == len(general_questions()) == len(SEED_PAIRS)


def test_topics_are_unique() -> None:
    topics = [p.topic for p in SEED_PAIRS]
    assert len(set(topics)) == len(topics)


def test_personal_questions_are_first_person_and_general_ones_are_not() -> None:
    """The distinction the model has to learn has to actually be present in the
    seeds, or the set teaches nothing."""
    first_person = re.compile(r"\b(I|my|me|we|our)\b")
    for pair in SEED_PAIRS:
        assert first_person.search(pair.personal), pair.topic
        assert not first_person.search(pair.general), pair.topic


def test_seed_questions_carry_no_disclaimer_text() -> None:
    for pair in SEED_PAIRS:
        assert not _DISCLAIMER.search(pair.personal)
        assert not _DISCLAIMER.search(pair.general)


def test_generation_prompt_forbids_disclaimer_boilerplate() -> None:
    """The single most important instruction in the prompt: disclaimers are the
    serving layer's job, and leaking them into training data would make a
    deterministic guarantee probabilistic."""
    lowered = REFUSAL_SYSTEM_PROMPT.lower()
    assert "no disclaimer boilerplate" in lowered
    assert "not legal advice" in lowered  # named as forbidden text


def test_generation_prompt_forbids_refusing_general_questions() -> None:
    """Over-refusal is the failure mode that a naive refusal set creates."""
    lowered = REFUSAL_SYSTEM_PROMPT.lower()
    assert "must not refuse" in lowered
    assert "refusing is the failure" in lowered


def test_split_is_by_topic_and_disjoint() -> None:
    train, held_out = split_topics(SEED_PAIRS, eval_fraction=0.3, seed=42)
    train_topics = {p.topic for p in train}
    eval_topics = {p.topic for p in held_out}
    assert not (train_topics & eval_topics)
    assert len(train) + len(held_out) == len(SEED_PAIRS)


def test_split_is_deterministic() -> None:
    a, _ = split_topics(SEED_PAIRS, seed=42)
    b, _ = split_topics(SEED_PAIRS, seed=42)
    assert [p.topic for p in a] == [p.topic for p in b]


def test_eval_split_is_never_empty() -> None:
    _, held_out = split_topics(SEED_PAIRS, eval_fraction=0.01)
    assert len(held_out) >= 1


def test_user_prompt_carries_both_seeds() -> None:
    prompt = build_user_prompt(SEED_PAIRS[0], n_variants=4)
    assert SEED_PAIRS[0].personal in prompt
    assert SEED_PAIRS[0].general in prompt
    assert "4 should_refuse" in prompt
