"""Quality filtering and near-duplicate removal for synthesized pairs.

    uv run python -m legalmind.data.filter \
        --in data/raw_pairs.jsonl --out data/filtered_pairs.jsonl

Deliberately separate from generation: filter thresholds get tuned several times,
and re-running a $30 batch to change a length bound would be absurd.

Every rejection is counted and reported by reason. A filter that silently drops
40% of its input is a bug you want to notice, and the counts are what tell you
whether a threshold is doing useful work or just deleting data.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasketch import MinHash, MinHashLSH

# The generator is instructed not to produce these (see prompts.py rule 4), but
# instructions are probabilistic and this check is not. Disclaimer text in
# training data would teach the model to emit compliance boilerplate unreliably,
# competing with the deterministic serving-layer enforcement.
_DISCLAIMER_PATTERNS = re.compile(
    r"(not\s+legal\s+advice"
    r"|consult\s+(?:with\s+)?(?:a|an|your)\s+(?:licensed\s+)?(?:attorney|lawyer)"
    r"|educational\s+purposes\s+only"
    r"|i\s+am\s+not\s+a\s+lawyer"
    r"|\bIANAL\b"
    r"|seek\s+professional\s+legal\s+counsel)",
    re.IGNORECASE,
)

# Instructions that only make sense next to the source passage. The model being
# trained never sees that passage, so these are unanswerable at training time
# and teach it to hallucinate context.
_NON_SELF_CONTAINED = re.compile(
    r"(the\s+(?:above|following|preceding)\s+(?:passage|text|excerpt|section)"
    r"|according\s+to\s+(?:the|this)\s+(?:passage|text|excerpt)"
    r"|in\s+the\s+passage\s+(?:above|provided|given)"
    r"|based\s+on\s+(?:the|this)\s+(?:passage|excerpt)"
    r"|as\s+(?:stated|described)\s+(?:above|in\s+this\s+section))",
    re.IGNORECASE,
)

# Dangling references in the *response*. Measured at 31.6% on a 99-passage probe
# before the generation prompt was tightened — a response saying "the passage
# does not commit the agency to..." teaches the model to cite a source it was
# never given, which is hallucination taught directly.
#
# Checking only the instruction (the rule above) missed all of it: the model
# writes self-contained questions and then reaches back to the passage in the
# answer. The generation prompt now forbids this explicitly; this is the backstop
# that measures whether the prompt is holding.
_DANGLING_REFERENCE = re.compile(
    r"\b(?:the|this)\s+(?:passage|excerpt)\b"
    r"|\b(?:the|this)\s+text\s+(?:does|specifies|states|expressly|provides|contemplates|requires|says)"
    r"|based\s+on\s+(?:the|this)\s+(?:text|passage|excerpt)"
    r"|as\s+(?:stated|described|noted|written)\s+above"
    r"|in\s+the\s+(?:passage|excerpt)\s+(?:above|provided|given)?",
    re.IGNORECASE,
)

# First- or second-person requests for advice about the asker's own situation.
# This behaviour is deliberately held out of the SFT mixture and handled by the
# refusal-calibration set instead — letting it leak in here would train the model
# to do the exact thing UPL compliance forbids.
_PERSONAL_ADVICE = re.compile(
    r"(what\s+should\s+i\s+do"
    r"|can\s+i\s+sue"
    r"|should\s+i\s+(?:sue|file|appeal|settle|sign)"
    r"|my\s+(?:landlord|employer|spouse|case|lawsuit|contract)\b"
    r"|do\s+i\s+have\s+a\s+(?:case|claim))",
    re.IGNORECASE,
)

VALID_TASK_TYPES = {
    "issue_spotting",
    "rule_statement",
    "clause_explanation",
    "statutory_interpretation",
}


@dataclass(frozen=True)
class FilterConfig:
    min_instruction_chars: int = 40
    max_instruction_chars: int = 2000
    min_response_chars: int = 200
    max_response_chars: int = 8000
    # Jaccard threshold for near-duplicate instructions.
    dedup_threshold: float = 0.8
    # 3-word shingles, not 5. Instructions are short (10-25 words), and at k=5 a
    # 12-word question yields only 8 shingles — editing one word destroys five of
    # them and drops Jaccard to ~0.4. k=3 gives usable resolution at this length.
    #
    # Even so, MinHash cannot catch a single-word paraphrase of a short sentence
    # at any sane threshold: inserting one word into a 10-word question puts
    # Jaccard near 0.5, and a threshold low enough to catch that would merge
    # genuinely distinct questions. That gap is covered by the exact-match pass
    # in `filter_pairs`, which handles the duplicates that actually occur here
    # (the same question generated from two overlapping passages).
    shingle_size: int = 3


def _shingles(text: str, k: int) -> set[str]:
    words = re.findall(r"\w+", text.lower())
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def _minhash(text: str, k: int, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for shingle in _shingles(text, k):
        m.update(shingle.encode("utf-8"))
    return m


def _normalize(text: str) -> str:
    """Casefold, strip punctuation, collapse whitespace — for exact-duplicate
    detection that is insensitive to cosmetic differences."""
    return " ".join(re.findall(r"\w+", text.lower()))


def check_pair(pair: dict[str, Any], cfg: FilterConfig) -> str | None:
    """Return a rejection reason, or None if the pair passes.

    Gate order is load-bearing for the report, not just cosmetic. The content
    gates (disclaimer leakage, non-self-contained, personal advice) run *before*
    the length gates, because each of them measures how well the generation
    prompt is holding. A short personal-advice instruction checked in the other
    order would be filed under `instruction_length`, and the report would
    understate how much out-of-policy content the generator produced — which is
    exactly the number worth watching.
    """
    instruction = (pair.get("instruction") or "").strip()
    response = (pair.get("response") or "").strip()

    if not instruction or not response:
        return "empty_field"
    if pair.get("task_type") not in VALID_TASK_TYPES:
        return "bad_task_type"

    # Content gates first — see docstring.
    if _DISCLAIMER_PATTERNS.search(response):
        return "disclaimer_leaked"
    if _NON_SELF_CONTAINED.search(instruction):
        return "not_self_contained"
    if _DANGLING_REFERENCE.search(response):
        return "response_not_self_contained"
    if _PERSONAL_ADVICE.search(instruction):
        return "personal_advice"

    if not (cfg.min_instruction_chars <= len(instruction) <= cfg.max_instruction_chars):
        return "instruction_length"
    if not (cfg.min_response_chars <= len(response) <= cfg.max_response_chars):
        return "response_length"
    # A response that merely restates the instruction is filler.
    if response.lower().startswith(instruction.lower()[:60]):
        return "response_echoes_instruction"
    return None


def filter_pairs(
    pairs: list[dict[str, Any]], cfg: FilterConfig | None = None
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Apply quality gates, then drop near-duplicate instructions.

    Deduplication runs second so the reject counts for quality reasons are not
    inflated by duplicates that would have been dropped anyway.
    """
    cfg = cfg or FilterConfig()
    reasons: Counter[str] = Counter()

    kept: list[dict[str, Any]] = []
    for pair in pairs:
        reason = check_pair(pair, cfg)
        if reason:
            reasons[reason] += 1
        else:
            kept.append(pair)

    # Two passes, because they catch different things. Exact-normalized match
    # handles the duplicates that actually occur in this pipeline — the same
    # question generated from two overlapping passages — with zero false
    # positives. MinHash then catches substantially-overlapping instructions
    # that are not character-identical. Neither pass alone is sufficient: see
    # the note on `shingle_size` for why MinHash misses one-word paraphrases.
    seen_exact: set[str] = set()
    lsh = MinHashLSH(threshold=cfg.dedup_threshold, num_perm=128)
    deduped: list[dict[str, Any]] = []
    for index, pair in enumerate(kept):
        normalized = _normalize(pair["instruction"])
        if normalized in seen_exact:
            reasons["exact_duplicate"] += 1
            continue
        signature = _minhash(pair["instruction"], cfg.shingle_size)
        if lsh.query(signature):
            reasons["near_duplicate"] += 1
            continue
        seen_exact.add(normalized)
        lsh.insert(str(index), signature)
        deduped.append(pair)

    return deduped, reasons


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Filter and deduplicate synthesized pairs.")
    parser.add_argument("--in", dest="in_path", type=Path, default=Path("data/raw_pairs.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/filtered_pairs.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("eval_results/filter_report.json"))
    args = parser.parse_args(argv)

    pairs = read_jsonl(args.in_path)
    kept, reasons = filter_pairs(pairs)

    write_jsonl(args.out, kept)

    report = {
        "input_pairs": len(pairs),
        "kept_pairs": len(kept),
        "retention_rate": round(len(kept) / len(pairs), 4) if pairs else 0.0,
        "rejected_by_reason": dict(reasons),
        "by_task_type": dict(Counter(p["task_type"] for p in kept)),
        "by_source": dict(Counter(p.get("source", "") for p in kept)),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps(report, indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
