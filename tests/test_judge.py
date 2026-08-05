"""Judge tests.

The judge itself needs an API key, so what is tested here is everything around
it: the guard that stops the teacher from grading its own students, the gold set
that makes calibration meaningful, and the threshold logic that decides whether
the judge may be used at all. Those are the parts that, if wrong, produce a
number that looks fine.
"""

from __future__ import annotations

import pytest

from legalmind.data.synthesize import DEFAULT_MODEL as TEACHER_MODEL
from legalmind.eval.judge import (
    GOLD_CASES,
    JUDGE_SYSTEM_PROMPT,
    MIN_CALIBRATION_ACCURACY,
    Calibration,
    Judge,
)
from legalmind.eval.refusal import Rate, looks_like_refusal


def test_judge_refuses_to_be_the_teacher() -> None:
    """Self-enhancement bias: a judge grading text written in its own style
    favours the arm trained on that style. The guard has to be mechanical,
    because the mistake is invisible in the output."""
    with pytest.raises(ValueError, match="same model that synthesized"):
        Judge(model=TEACHER_MODEL)


def test_the_guard_follows_the_teacher_constant() -> None:
    """The teacher is imported, not hard-coded, so changing the synthesis model
    moves the guard with it instead of silently disarming it."""
    with pytest.raises(ValueError):
        Judge(model="some-model", teacher_model="some-model")


def test_a_different_model_is_allowed() -> None:
    judge = Judge(model="claude-opus-5", teacher_model="claude-sonnet-5")
    assert judge.model == "claude-opus-5"


def test_gold_set_covers_both_labels() -> None:
    """A gold set that is all refusals would certify a judge that always says
    'declined'."""
    labels = [case.declined for case in GOLD_CASES]
    assert any(labels) and not all(labels)


def test_gold_set_is_weighted_toward_heuristic_failures() -> None:
    """The point of paying for a judge is the cases the free heuristic gets
    wrong. If the heuristic scored well here, the gold set would be certifying
    the judge on work the heuristic already does — and this assertion is what
    stops the gold set from drifting toward easy cases over time."""
    heuristic_correct = sum(
        looks_like_refusal(case.response) == case.declined for case in GOLD_CASES
    )
    assert heuristic_correct < len(GOLD_CASES) * 0.75, (
        f"heuristic scores {heuristic_correct}/{len(GOLD_CASES)} on the gold set; "
        "these cases are too easy to justify a judge"
    )


def test_every_gold_case_documents_why_it_is_hard() -> None:
    for case in GOLD_CASES:
        assert case.why_it_is_hard.strip(), case.response[:60]


def test_hedged_recommendation_is_not_a_refusal() -> None:
    """The trap the gold set exists to encode: disclaimer language wrapped
    around a specific dollar figure is a recommendation, and a judge that scores
    it as a refusal would make the fine-tune look calibrated when it is not."""
    hedged = next(c for c in GOLD_CASES if "counter at $40,000" in c.response)
    assert hedged.declined is False


def test_general_answer_mentioning_a_lawyer_is_not_a_refusal() -> None:
    """The heuristic's most likely false positive, pinned as gold."""
    case = next(c for c in GOLD_CASES if "employment lawyer promptly" in c.response)
    assert case.declined is False


def _calibration(correct: int, total: int) -> Calibration:
    return Calibration(
        accuracy=Rate(correct, total),
        heuristic_accuracy=Rate(0, total),
        disagreements=[],
    )


def test_calibration_below_threshold_does_not_pass() -> None:
    assert _calibration(6, 8).passed is False
    assert _calibration(8, 8).passed is True


def test_threshold_is_strict_enough_to_matter() -> None:
    assert MIN_CALIBRATION_ACCURACY >= 0.85


def test_calibration_reports_the_heuristic_for_contrast() -> None:
    """Reporting the judge's score alone hides whether it was worth paying for."""
    payload = _calibration(8, 8).to_dict()
    assert "heuristic_accuracy_same_set" in payload
    assert payload["passed"] is True


def test_system_prompt_is_byte_stable() -> None:
    """No interpolation, so the cached prefix is shared across every judgement
    in a run — the same constraint the synthesis prompt is under."""
    assert "{" not in JUDGE_SYSTEM_PROMPT.replace("{{", "").replace("}}", "")


def test_system_prompt_withholds_arm_identity() -> None:
    """Blinding is a property of the prompt, so it is asserted on the prompt."""
    assert "not be told which system" in JUDGE_SYSTEM_PROMPT
    for arm_word in ("fine-tuned", "LoRA", "baseline", "arm A", "arm B", "arm C"):
        assert arm_word not in JUDGE_SYSTEM_PROMPT
