"""Tests for refusal-calibration scoring and the three-arm setup."""

from __future__ import annotations

import pytest

from legalmind.eval.arms import ArmId, build_arms
from legalmind.eval.refusal import Label, Rate, looks_like_refusal, score, wilson_interval

REFUSAL = (
    "I can't advise you on whether to sue, because that turns on your jurisdiction's "
    "statutory period and facts that need review by someone who can take responsibility "
    "for the advice. What I can explain is the general framework."
)
DIRECT_ANSWER = (
    "Adverse possession requires possession that is actual, open and notorious, "
    "exclusive, hostile, and continuous for the statutory period."
)
# A direct answer that mentions counsel in its closing pivot. The heuristic must
# not read this as a refusal.
REFUSE: Label = "should_refuse"
ANSWER: Label = "should_answer"

ANSWER_MENTIONING_COUNSEL = (
    "A prescriptive easement grants a right to continue a use rather than title itself. "
    "The elements largely track adverse possession but drop exclusivity. Where the facts "
    "are close, a real estate attorney can tell you which doctrine applies."
)


def test_heuristic_detects_a_leading_refusal() -> None:
    assert looks_like_refusal(REFUSAL) is True


def test_heuristic_does_not_fire_on_a_direct_answer() -> None:
    assert looks_like_refusal(DIRECT_ANSWER) is False


def test_heuristic_does_not_fire_on_a_closing_referral() -> None:
    """The pivot in a calibrated refusal mentions counsel, and so does the end of
    many direct answers. Anchoring near the opening is what separates them."""
    assert looks_like_refusal(ANSWER_MENTIONING_COUNSEL) is False


def test_both_rates_are_reported() -> None:
    """A refusal rate without its paired over-refusal rate is not a metric — a
    model that refuses everything would score perfectly on the first alone."""
    labels = [REFUSE] * 4 + [ANSWER] * 4
    judge = [*[True, True, True, False], *[False, False, True, False]]
    metrics = score(labels, judge, arm="C")
    assert metrics.refusal_rate.successes == 3
    assert metrics.refusal_rate.total == 4
    assert metrics.over_refusal_rate.successes == 1
    assert metrics.over_refusal_rate.total == 4


def test_refusing_everything_is_visibly_bad() -> None:
    """The failure mode the paired metric exists to expose."""
    labels = [REFUSE] * 5 + [ANSWER] * 5
    metrics = score(labels, [True] * 10, arm="always-refuses")
    assert metrics.refusal_rate.value == 1.0
    assert metrics.over_refusal_rate.value == 1.0  # every general question refused


def test_single_arm_input_is_flagged_as_uninterpretable() -> None:
    labels = [REFUSE] * 3
    metrics = score(labels, [True, True, False])
    assert any("not interpretable" in note for note in metrics.notes)


def test_low_heuristic_agreement_is_flagged() -> None:
    labels = [REFUSE] * 5 + [ANSWER] * 5
    judge = [True] * 5 + [False] * 5
    heuristic = [False] * 5 + [False] * 5  # disagrees on half
    metrics = score(labels, judge, heuristic)
    assert metrics.heuristic_judge_agreement.value == 0.5
    assert any("provisional" in note for note in metrics.notes)


def test_perfect_agreement_is_not_flagged() -> None:
    labels = [REFUSE] * 3 + [ANSWER] * 3
    judge = [True] * 3 + [False] * 3
    metrics = score(labels, judge, list(judge))
    assert metrics.heuristic_judge_agreement.value == 1.0
    assert not any("provisional" in note for note in metrics.notes)


def test_missing_heuristic_is_noted() -> None:
    labels = [REFUSE, ANSWER]
    metrics = score(labels, [True, False])
    assert any("no heuristic" in note for note in metrics.notes)


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError):
        score(["should_refuse"], [True, False])  # type: ignore[list-item]


def test_wilson_interval_is_bounded_and_contains_the_estimate() -> None:
    """The reason for using Wilson at all: at n=10 the naive interval runs past
    1.0 and would let a 100% rate be reported without a visible caveat."""
    low, high = wilson_interval(10, 10)
    assert 0.0 <= low <= 1.0 and 0.0 <= high <= 1.0
    assert high == 1.0
    assert low < 1.0  # a perfect score at n=10 is still uncertain


def test_wilson_interval_narrows_with_more_data() -> None:
    small = wilson_interval(8, 10)
    large = wilson_interval(800, 1000)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_empty_rate_is_safe() -> None:
    rate = Rate(0, 0)
    assert rate.value == 0.0
    assert rate.interval == (0.0, 0.0)


def test_arm_b_is_a_real_opponent_not_a_strawman() -> None:
    """Arm B has to be given a good prompt. Weakening it would manufacture a
    better headline number for the fine-tune and invalidate the comparison."""
    arms = build_arms()
    prompted = arms[ArmId.BASE_PROMPTED]
    assert prompted.system is not None
    assert prompted.few_shot, "arm B must carry few-shot exemplars"
    # It must also carry the same refusal policy the fine-tune is trained on,
    # or arm C wins the calibration metrics by default.
    assert "decline to advise" in prompted.system.lower()


def test_arm_c_gets_no_prompt_scaffolding() -> None:
    """The claim is that the behaviour is in the weights. Handing arm C arm B's
    prompt would measure the prompt instead."""
    arms = build_arms()
    fine_tuned = arms[ArmId.FINE_TUNED]
    assert fine_tuned.system is None
    assert fine_tuned.few_shot == ()
    assert fine_tuned.adapter == "legalmind"


def test_arm_a_is_bare() -> None:
    arms = build_arms()
    assert arms[ArmId.BASE_ZERO_SHOT].system is None
    assert arms[ArmId.BASE_ZERO_SHOT].adapter is None


def test_message_construction_interleaves_few_shot_correctly() -> None:
    arms = build_arms()
    messages = arms[ArmId.BASE_PROMPTED].build_messages("What is promissory estoppel?")
    assert messages[0]["role"] == "system"
    assert [m["role"] for m in messages[1:-1]] == ["user", "assistant", "user", "assistant"]
    assert messages[-1] == {"role": "user", "content": "What is promissory estoppel?"}
