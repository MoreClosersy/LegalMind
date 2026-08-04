"""Decontamination tests. No network — the LegalBench index is stubbed."""

from __future__ import annotations

from legalmind.data.decontaminate import (
    SHINGLE_SIZE,
    _shingle_hashes,
    decontaminate,
    split_train_val,
)

LEAKED = (
    "the mark salt for packages of sodium chloride is generic because it names "
    "the product itself rather than identifying its commercial source clearly"
)


def test_shingles_are_deterministic_across_calls() -> None:
    """The report must be reproducible; Python's salted hash() would not be."""
    assert _shingle_hashes(LEAKED) == _shingle_hashes(LEAKED)


def test_text_shorter_than_shingle_yields_nothing() -> None:
    short = " ".join(["word"] * (SHINGLE_SIZE - 1))
    assert _shingle_hashes(short) == set()


def test_shingling_is_case_insensitive() -> None:
    assert _shingle_hashes(LEAKED.upper()) == _shingle_hashes(LEAKED.lower())


def test_contaminated_pair_is_dropped() -> None:
    index = _shingle_hashes(LEAKED)
    pairs = [
        {"instruction": "Is the mark generic?", "response": LEAKED},
        {"instruction": "What is a fanciful mark?", "response": "A coined term with no meaning."},
    ]
    clean, contaminated = decontaminate(pairs, index)
    assert len(contaminated) == 1
    assert contaminated[0]["response"] == LEAKED
    assert len(clean) == 1


def test_overlap_is_checked_across_instruction_and_response() -> None:
    """A leak split across the two fields still has to be caught — the check
    runs on the concatenation, not each field alone."""
    index = _shingle_hashes(LEAKED)
    words = LEAKED.split()
    pair = {"instruction": " ".join(words[:7]), "response": " ".join(words[7:])}
    _, contaminated = decontaminate([pair], index)
    assert len(contaminated) == 1


def test_clean_corpus_drops_nothing() -> None:
    index = _shingle_hashes(LEAKED)
    pairs = [
        {"instruction": f"Question {i}?", "response": f"Answer number {i}."} for i in range(20)
    ]
    clean, contaminated = decontaminate(pairs, index)
    assert contaminated == []
    assert len(clean) == 20


def test_split_is_deterministic_and_disjoint() -> None:
    pairs = [{"instruction": str(i), "response": "r"} for i in range(1000)]
    train_a, val_a = split_train_val(pairs, val_size=100, seed=42)
    train_b, val_b = split_train_val(pairs, val_size=100, seed=42)
    assert [p["instruction"] for p in val_a] == [p["instruction"] for p in val_b]
    assert len(train_a) == 900 and len(val_a) == 100
    assert not ({p["instruction"] for p in train_a} & {p["instruction"] for p in val_a})


def test_val_size_is_capped_at_ten_percent() -> None:
    """Guards against a tiny corpus donating most of itself to validation."""
    pairs = [{"instruction": str(i), "response": "r"} for i in range(50)]
    train, val = split_train_val(pairs, val_size=500)
    assert len(val) == 5
    assert len(train) == 45
