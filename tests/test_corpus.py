"""Chunker tests. Network-free by construction — CI must not depend on the Hub."""

from __future__ import annotations

from legalmind.data.corpus import SOURCE_SHARDS, chunk_record

# Prose-shaped filler: long lines with sentence terminators, so it survives the
# substantive-content filter.
SENTENCE = (
    "The agency shall determine whether the applicant has satisfied each of the "
    "enumerated conditions before issuing a permit under this subpart. "
)


def _prose(n_lines: int) -> str:
    return "\n".join(SENTENCE * 2 for _ in range(n_lines))


def test_chunks_respect_size_bounds() -> None:
    chunks = chunk_record(_prose(200), min_chars=2000, max_chars=8000)
    assert chunks, "expected at least one chunk from 200 lines of prose"
    for chunk in chunks:
        assert len(chunk) >= 2000
        # The greedy packer may overshoot by at most one line.
        assert len(chunk) <= 8000 + len(SENTENCE) * 2 + 1


def test_short_record_yields_nothing() -> None:
    """A record below `min_chars` produces no chunk rather than a runt one."""
    assert chunk_record(SENTENCE, min_chars=2000) == []


def test_table_of_contents_is_rejected() -> None:
    """TOC blocks are abundant in CFR and would produce useless training pairs."""
    toc = "\n".join(f"Sec. {i}.1 Scope" for i in range(400))
    assert chunk_record(toc, min_chars=2000) == []


def test_indentation_is_normalized() -> None:
    """Source text is heavily indented; chunks should carry clean lines."""
    indented = "\n".join("        " + SENTENCE * 2 for _ in range(100))
    chunks = chunk_record(indented, min_chars=2000)
    assert chunks
    assert not any(line.startswith(" ") for line in chunks[0].split("\n"))


def test_section_boundary_starts_a_new_chunk() -> None:
    """Once a chunk is big enough, a section marker should end it — so a rule
    doesn't get split across two passages."""
    body = _prose(60)
    record = f"{body}\n§ 200.1 Definitions.\n{body}"
    chunks = chunk_record(record, min_chars=2000, max_chars=100_000)
    assert len(chunks) >= 2
    assert chunks[1].startswith("§ 200.1")


def test_shard_names_match_repo_layout() -> None:
    """`courtlisteneropinions` has no underscore in the repo, unlike its
    siblings. Getting this wrong is a 404 at sampling time."""
    assert SOURCE_SHARDS["courtlisteneropinions"] == ("data/train.courtlisteneropinions.0.jsonl.xz")
    assert set(SOURCE_SHARDS) == {"cfr", "courtlisteneropinions", "federal_register"}
