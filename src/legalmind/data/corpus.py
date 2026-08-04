"""Sampling passages of authentic legal text from `pile-of-law`.

Three notes on why this does not use `datasets.load_dataset`:

1. `pile-of-law` ships a loading script, and `datasets` >= 3.0 dropped support
   for script-based datasets. `load_dataset("pile-of-law/pile-of-law", "cfr")`
   fails outright on the pinned version.
2. The shards are large — `train.courtlisteneropinions.0.jsonl.xz` alone is 1 GB
   compressed. We need a few thousand passages, not the corpus, so we stream the
   HTTP response through an incremental LZMA decompressor and stop early. Peak
   disk and memory stay in the low tens of MB.
3. One record is not one passage. A single CFR record is an entire Title —
   the first one is 2.8 million characters. Records must be chunked before they
   are usable as synthesis inputs, which is most of what this module does.

Shard names are taken from the repo listing verbatim; note `courtlisteneropinions`
has no underscore, unlike the other two.
"""

from __future__ import annotations

import json
import lzma
import random
import re
from collections.abc import Iterator
from dataclasses import dataclass

import requests
from huggingface_hub import hf_hub_url

REPO_ID = "pile-of-law/pile-of-law"

# Shard per source. Court opinions are split into 16 shards; one is far more
# than we need, and taking a single shard keeps the sample from one era/court
# mix rather than none at all.
SOURCE_SHARDS: dict[str, str] = {
    "cfr": "data/train.cfr.jsonl.xz",
    "courtlisteneropinions": "data/train.courtlisteneropinions.0.jsonl.xz",
    "federal_register": "data/train.federal_register.jsonl.xz",
}

# A line that starts a new logical unit. Breaking chunks here keeps a regulation
# section or a numbered subsection intact instead of splitting mid-rule.
_SECTION_BOUNDARY = re.compile(
    r"^(§+\s*\d|Sec\.\s*\d|PART\s+\d|Subpart\s+[A-Z]|\(\w{1,3}\)\s)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Passage:
    """One chunk of legal text, ready to be handed to the synthesis model."""

    source: str
    passage_id: str
    text: str
    url: str

    @property
    def n_chars(self) -> int:
        return len(self.text)


def _stream_jsonl_xz(filename: str, timeout: int = 120) -> Iterator[dict]:
    """Yield records from an `.xz`-compressed JSONL file on the Hub without
    downloading it in full."""
    url = hf_hub_url(REPO_ID, filename, repo_type="dataset")
    decompressor = lzma.LZMADecompressor(format=lzma.FORMAT_XZ)
    buffer = b""
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        for compressed in response.iter_content(chunk_size=1 << 20):
            if not compressed:
                continue
            buffer += decompressor.decompress(compressed)
            *complete_lines, buffer = buffer.split(b"\n")
            for line in complete_lines:
                if line.strip():
                    yield json.loads(line)
            if decompressor.eof:
                break


def _clean_lines(text: str) -> list[str]:
    """Collapse the source's heavy indentation into plain non-empty lines."""
    lines = []
    for raw in text.split("\n"):
        # \u00a0 is deliberate: government legal text is littered with
        # non-breaking spaces, which otherwise survive into the training data
        # and create spurious token boundaries.
        line = re.sub(r"[ \t\u00a0]+", " ", raw).strip()
        if line:
            lines.append(line)
    return lines


def _looks_substantive(chunk: str, min_avg_line_chars: int) -> bool:
    """Reject tables of contents, index pages, and citation dumps.

    These are abundant in CFR and Federal Register records and produce useless
    instruction pairs — the synthesis model will dutifully write questions about
    a list of section headings. Two cheap signals separate them from prose:
    TOC lines are short, and TOC blocks contain few sentence terminators.
    """
    lines = chunk.split("\n")
    if not lines:
        return False
    avg_line = sum(len(line) for line in lines) / len(lines)
    if avg_line < min_avg_line_chars:
        return False
    # At least a few full sentences. A heading block has almost none.
    return chunk.count(". ") >= 3


def chunk_record(
    text: str,
    *,
    min_chars: int = 2000,
    max_chars: int = 8000,
    min_avg_line_chars: int = 40,
) -> list[str]:
    """Split one record's text into substantive, roughly section-aligned chunks.

    Bounds are in characters rather than tokens on purpose: this runs before any
    tokenizer is loaded, and the bound only needs to be approximate. At roughly
    4 characters per token the default window is about 500-2000 tokens.
    """
    lines = _clean_lines(text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if current_len >= min_chars:
            candidate = "\n".join(current)
            if _looks_substantive(candidate, min_avg_line_chars):
                chunks.append(candidate)
        current = []
        current_len = 0

    for line in lines:
        # Prefer to start a new chunk at a section boundary, but only once the
        # current one is already big enough to stand on its own.
        if current_len >= min_chars and _SECTION_BOUNDARY.match(line):
            flush()
        current.append(line)
        current_len += len(line) + 1
        if current_len >= max_chars:
            flush()

    flush()
    return chunks


def sample_passages(
    source: str,
    n_passages: int,
    *,
    max_per_record: int = 8,
    min_chars: int = 2000,
    max_chars: int = 8000,
    seed: int = 42,
) -> list[Passage]:
    """Collect `n_passages` chunks from one pile-of-law source.

    `max_per_record` bounds how much of the sample any single document can
    contribute. Without it, one 2.8M-character CFR Title would supply hundreds of
    consecutive chunks and the "corpus" would really be one document.
    """
    if source not in SOURCE_SHARDS:
        raise KeyError(f"unknown source {source!r}; known: {sorted(SOURCE_SHARDS)}")

    rng = random.Random(seed)
    collected: list[Passage] = []

    for record_index, record in enumerate(_stream_jsonl_xz(SOURCE_SHARDS[source])):
        chunks = chunk_record(record.get("text", ""), min_chars=min_chars, max_chars=max_chars)
        if not chunks:
            continue
        # Spread the picks across the document rather than taking the first N,
        # which for CFR would be nothing but front matter.
        picks = chunks if len(chunks) <= max_per_record else rng.sample(chunks, max_per_record)
        for chunk_index, chunk in enumerate(picks):
            collected.append(
                Passage(
                    source=source,
                    passage_id=f"{source}-{record_index:06d}-{chunk_index:02d}",
                    text=chunk,
                    url=record.get("url", ""),
                )
            )
            if len(collected) >= n_passages:
                return collected

    return collected
