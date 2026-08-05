"""Forgetting-check tests.

The two things worth pinning: the answer extractor separates "wrong" from
"unparseable", and the comparison refuses to call an overlapping delta a
finding. Both are places where a small leniency turns a null result into a
headline.
"""

from __future__ import annotations

from legalmind.eval.forgetting import SUBJECTS, ArmScore, compare, extract_letter


def test_bare_letter_is_extracted() -> None:
    for text, expected in (("A", "A"), (" b ", "B"), ("(C)", "C"), ("D.", "D")):
        assert extract_letter(text) == expected, text


def test_answer_prefix_is_extracted() -> None:
    assert extract_letter("Answer: B") == "B"
    assert extract_letter("**C**") == "C"


def test_prose_is_unparseable_not_wrong() -> None:
    """Same principle as the LegalBench extractor: a formatting regression must
    not be counted as a capability regression, because the whole point of this
    eval is to attribute a capability change to the fine-tune."""
    assert extract_letter("The correct choice depends on the premise.") is None
    assert extract_letter("") is None


def test_letter_inside_a_word_does_not_match() -> None:
    assert extract_letter("Amortization is the relevant concept.") is None


def test_subjects_exclude_law() -> None:
    """A legal fine-tune improving on professional_law would be a result about
    the training objective. Averaging it in would let a gain on the trained
    domain mask a loss everywhere else — the exact effect being measured."""
    assert not any("law" in subject or "legal" in subject for subject in SUBJECTS)


def _score(correct: int, total: int) -> ArmScore:
    return ArmScore(correct=correct, compliant=total, total=total)


def test_overlapping_intervals_are_not_a_finding() -> None:
    """A 2-point drop at n=360 is noise, and the verdict has to say so rather
    than reporting the delta and letting the reader assume."""
    result = compare(_score(250, 360), _score(243, 360))
    assert result["delta"] < 0
    assert result["intervals_separated"] is False
    assert "no demonstrated change" in result["verdict"]


def test_a_large_drop_is_reported_as_real() -> None:
    result = compare(_score(300, 360), _score(150, 360))
    assert result["intervals_separated"] is True
    assert result["delta"] < -0.3


def test_delta_sign_is_direction_of_change_from_base() -> None:
    """Positive means the fine-tune improved general capability. Getting this
    backwards would invert every conclusion in the report."""
    assert compare(_score(100, 360), _score(200, 360))["delta"] > 0
    assert compare(_score(200, 360), _score(100, 360))["delta"] < 0


def test_empty_scores_do_not_crash() -> None:
    assert compare(_score(0, 0), _score(0, 0))["delta"] == 0.0
