"""Quality-filter tests.

The disclaimer-leak and personal-advice gates are the load-bearing ones: both
protect design decisions the project is explicitly making, so a regression in
either is a correctness bug, not a tuning issue.
"""

from __future__ import annotations

from typing import Any

from legalmind.data.filter import FilterConfig, check_pair, filter_pairs

CFG = FilterConfig()


def make_pair(**overrides: Any) -> dict[str, Any]:
    pair = {
        "instruction": "What are the elements of adverse possession at common law?",
        "response": (
            "Adverse possession requires possession that is actual, open and notorious, "
            "exclusive, hostile, and continuous for the statutory period. Each element "
            "carries independent significance and the claimant bears the burden on all "
            "of them. Actual possession means physical use consistent with the property's "
            "character rather than mere assertion of a claim."
        ),
        "task_type": "rule_statement",
        "passage_id": "cfr-000001-00",
        "source": "cfr",
    }
    pair.update(overrides)
    return pair


def test_clean_pair_passes() -> None:
    assert check_pair(make_pair(), CFG) is None


def test_disclaimer_in_response_is_rejected() -> None:
    """Disclaimers belong to the serving layer. Training on them would make a
    deterministic guarantee probabilistic."""
    for leak in (
        "This is not legal advice.",
        "You should consult a licensed attorney about your situation.",
        "Provided for educational purposes only.",
        "I am not a lawyer, but generally speaking...",
    ):
        pair = make_pair(response=make_pair()["response"] + " " + leak)
        assert check_pair(pair, CFG) == "disclaimer_leaked", leak


def test_non_self_contained_instruction_is_rejected() -> None:
    """The trained model never sees the source passage."""
    for bad in (
        "According to the passage, what is the filing deadline?",
        "Summarize the above section in two sentences.",
        "Based on this excerpt, who bears the burden of proof?",
    ):
        assert check_pair(make_pair(instruction=bad), CFG) == "not_self_contained", bad


def test_personal_advice_instruction_is_rejected() -> None:
    """Specific-case advice is the held-out behaviour, taught by the refusal set
    rather than the general instruction mixture."""
    for bad in (
        "My landlord is evicting me, what should I do?",
        "Can I sue my employer for this?",
        "Do I have a case against the contractor?",
    ):
        assert check_pair(make_pair(instruction=bad), CFG) == "personal_advice", bad


def test_dangling_reference_in_response_is_rejected() -> None:
    """Found by measuring, not by reading the code: 31.6% of responses in the
    first probe pointed back at the source passage the trained model never sees.
    The instruction-only check caught none of them, because the model writes a
    self-contained question and then reaches back in the answer."""
    for bad in (
        "The passage does not commit the agency to preparing such an analysis.",
        "Based on the text, this is a genuinely close question.",
        "The text specifies that the opportunity to be heard may be limited.",
        "As stated above, the burden falls on the agency.",
        "This excerpt treats occupied and potential acreage as distinct.",
    ):
        pair = make_pair(response=make_pair()["response"] + " " + bad)
        assert check_pair(pair, CFG) == "response_not_self_contained", bad


def test_naming_the_authority_is_not_a_dangling_reference() -> None:
    """The fix the prompt asks for — name the provision instead of pointing at
    the passage — must survive the filter, or the filter defeats the fix."""
    for good in (
        "Section 207 does not commit the agency to preparing such an analysis.",
        "Section 970.207(a) specifies that the review is conducted by two bodies.",
        "The statutory text of Title VII controls this question.",
        "Courts reading this provision have reached differing conclusions.",
    ):
        pair = make_pair(response=make_pair()["response"] + " " + good)
        assert check_pair(pair, CFG) is None, good


def test_length_bounds() -> None:
    assert check_pair(make_pair(instruction="Too short?"), CFG) == "instruction_length"
    assert check_pair(make_pair(response="Short."), CFG) == "response_length"


def test_bad_task_type_is_rejected() -> None:
    assert check_pair(make_pair(task_type="freeform"), CFG) == "bad_task_type"


LONG_INSTRUCTION = (
    "What are the required elements that a claimant must establish in order to "
    "prevail on a claim of adverse possession under the common law rule as "
    "applied by most American jurisdictions today?"
)


def test_exact_duplicates_are_dropped_despite_cosmetic_differences() -> None:
    """The duplicate that actually occurs here: the same question generated from
    two overlapping source passages, differing only in case or punctuation."""
    base = make_pair()
    twin = make_pair(instruction="  what are the ELEMENTS of adverse possession at common law  ")
    kept, reasons = filter_pairs([base, twin])
    assert len(kept) == 1
    assert reasons["exact_duplicate"] == 1


def test_substantially_overlapping_instructions_are_dropped() -> None:
    base = make_pair(instruction=LONG_INSTRUCTION)
    twin = make_pair(instruction=LONG_INSTRUCTION.replace("today?", "currently?"))
    kept, reasons = filter_pairs([base, twin])
    assert len(kept) == 1
    assert reasons["near_duplicate"] == 1


def test_one_word_paraphrase_of_a_short_question_survives() -> None:
    """Documented limitation, not an oversight. Inserting a word into a 10-word
    question puts Jaccard near 0.5; a threshold low enough to catch that would
    also merge genuinely distinct questions. Keeping both is the safer error."""
    base = make_pair()
    twin = make_pair(instruction=base["instruction"].replace("common law", "the common law"))
    kept, _ = filter_pairs([base, twin])
    assert len(kept) == 2


def test_distinct_instructions_both_survive() -> None:
    a = make_pair()
    b = make_pair(
        instruction="What is the standard of review for an agency's statutory interpretation?",
        task_type="statutory_interpretation",
    )
    kept, reasons = filter_pairs([a, b])
    assert len(kept) == 2
    assert reasons["near_duplicate"] == 0


def test_reasons_are_counted_for_reporting() -> None:
    pairs = [make_pair(), make_pair(task_type="nope"), make_pair(response="x")]
    kept, reasons = filter_pairs(pairs)
    assert len(kept) == 1
    assert reasons["bad_task_type"] == 1
    assert reasons["response_length"] == 1
