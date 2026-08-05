"""Scoring tests for the LegalBench harness.

Everything here is a pure function over a response string. That is deliberate:
a bug in extraction or scoring does not crash, it just produces a wrong number
that looks plausible and ends up in the README. These are the tests that stop
that, so they are written against the failure modes the arms actually exhibit
rather than against the happy path.
"""

from __future__ import annotations

from legalmind.eval.legalbench import (
    TaskScore,
    extract_label,
    labels_present,
    render_instruction,
    score_response,
    summarise,
)

YES_NO = ("Yes", "No")
ABERCROMBIE = ("generic", "descriptive", "suggestive", "arbitrary", "fanciful")


def test_bare_label_is_extracted() -> None:
    assert extract_label("Yes", YES_NO) == "Yes"
    assert extract_label("  no  ", YES_NO) == "No"


def test_label_after_answer_marker_is_extracted() -> None:
    """The marker is anchored to the last occurrence because LegalBench's own
    instructions end with the word "Answer:"."""
    assert extract_label("Answer: Yes", YES_NO) == "Yes"
    assert extract_label("Some reasoning.\n\nAnswer:\nNo", YES_NO) == "No"


def test_markdown_decoration_is_stripped() -> None:
    for decorated in ("**Yes**", "`Yes`", "- Yes", "Yes.", '"Yes"', "## Yes"):
        assert extract_label(decorated, YES_NO) == "Yes", decorated


def test_prose_answer_is_non_compliant_not_wrong() -> None:
    """The fine-tuned arm is trained to explain, so it will sometimes open with a
    sentence instead of a label. That must register as a format failure, not as a
    wrong answer — collapsing the two would let a formatting regression look like
    a reasoning regression."""
    text = "This turns on whether the statement was offered for its truth."
    assert extract_label(text, YES_NO) is None


def test_word_boundary_prevents_false_positives() -> None:
    """ "No" inside "Nothing" and "cannot" is the single most likely way to
    silently corrupt a Yes/No task, and there are 109 of them."""
    assert labels_present("Nothing in the statute is dispositive.", YES_NO) == set()
    assert labels_present("The court cannot reach that question.", YES_NO) == set()
    assert labels_present("Yesterday the rule changed.", YES_NO) == set()
    assert labels_present("No.", YES_NO) == {"No"}


def test_official_metric_credits_a_verbose_wrong_answer() -> None:
    """This is the trap the harness exists to expose, asserted rather than
    described: a response that argues its way to "Yes" still contains the word
    "no" somewhere, and LegalBench's containment method scores it correct
    against a gold "No"."""
    verbose = (
        "Yes. The statement is hearsay because it was offered to prove the truth "
        "of the matter asserted, and no exception applies on these facts."
    )
    strict, official, compliant, ambiguous = score_response(verbose, "No", YES_NO)
    assert strict is False, "strict scoring must read the actual answer"
    assert official is True, "the official method credits the incidental 'no'"
    assert compliant is True
    assert ambiguous is True, "both labels appear — the official number is unreliable here"


def test_ambiguity_is_zero_for_a_bare_label() -> None:
    _, _, _, ambiguous = score_response("No", "No", YES_NO)
    assert ambiguous is False


def test_multiclass_extraction() -> None:
    assert extract_label("Answer: suggestive", ABERCROMBIE) == "suggestive"
    # Naming two labels in the answer line is a refusal to choose, not a choice.
    assert extract_label("It is either suggestive or arbitrary.", ABERCROMBIE) is None


def test_empty_response_is_non_compliant() -> None:
    assert extract_label("", YES_NO) is None
    assert extract_label("   \n  ", YES_NO) is None


def test_render_instruction_substitutes_columns() -> None:
    rendered = render_instruction("Mark: {{text}} What type?", {"text": "Ivory Soap"})
    assert rendered == "Mark: Ivory Soap What type?"


def test_render_instruction_raises_on_missing_column() -> None:
    """A silent miss would leave a literal {{contract}} in the prompt and cost
    accuracy across a whole task without ever surfacing."""
    try:
        render_instruction("See {{contract}}.", {"text": "x"})
    except KeyError as exc:
        assert "contract" in str(exc)
    else:  # pragma: no cover - the assertion below reports the failure
        raise AssertionError("expected a KeyError for the missing column")


def _score_from(pairs: list[tuple[str, bool]], majority_share: float) -> TaskScore:
    score = TaskScore(task="t", majority_share=majority_share)
    for gold, correct in pairs:
        score.n += 1
        score.strict_correct += correct
        score.compliant += 1
        score.per_label_total[gold] += 1
        score.per_label_correct[gold] += correct
    return score


def test_balanced_accuracy_exposes_a_majority_class_guesser() -> None:
    """A model that always answers "No" on a 80/20 split scores 80% accuracy and
    50% balanced accuracy. Reporting only the first would make it look skilled."""
    pairs = [("No", True)] * 80 + [("Yes", False)] * 20
    score = _score_from(pairs, majority_share=0.8)
    assert score.strict_accuracy == 0.8
    assert score.balanced_accuracy == 0.5


def test_summary_reports_the_majority_baseline_alongside_accuracy() -> None:
    """Accuracy without its baseline is not interpretable, so the summary must
    always carry both."""
    summary = summarise({"t": _score_from([("No", True)] * 80 + [("Yes", False)] * 20, 0.8)})
    assert summary["strict_accuracy_macro"] == 0.8
    assert summary["majority_baseline_macro"] == 0.8
    assert summary["balanced_accuracy_macro"] == 0.5
    # 80/100 is not "80% and certainly better than chance" at this n.
    low, high = summary["strict_accuracy_ci95"]
    assert low < 0.8 < high


def test_summary_survives_an_empty_task() -> None:
    assert summarise({})["n_tasks"] == 0
