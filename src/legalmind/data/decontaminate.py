"""Decontaminate the training set against LegalBench, then split train/val.

    uv run python -m legalmind.data.decontaminate \
        --in data/filtered_pairs.jsonl --report eval_results/decontamination.json

This is the script that makes the headline evaluation claim defensible.

LegalBench is a *benchmark*. Training on it and then reporting held-out accuracy
on it is train/test contamination, and it is the first thing a careful reader
checks. This project's training data comes from a different source entirely
(raw `pile-of-law` text, synthesized into instructions), so contamination should
be near zero by construction — but "should be" is not evidence. This script
produces the evidence: an n-gram overlap check against all 162 LegalBench tasks,
both their train and test splits, with the drop count written to
`eval_results/decontamination.json` and cited in the README.

Method: 13-word shingles, matching the convention used by GPT-3 and the Llama
reports. Any training pair sharing a single 13-gram with any LegalBench text is
dropped. That is aggressive — 13 consecutive words of legal boilerplate can
collide innocently — but the cost of over-dropping a few hundred synthetic pairs
is nothing next to the cost of an inflated benchmark number.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download, list_repo_files

from legalmind.data.filter import read_jsonl, write_jsonl

LEGALBENCH_REPO = "nguha/legalbench"
SHINGLE_SIZE = 13

# Columns that carry labels or row ids rather than prose. Everything else in a
# LegalBench TSV is text the model could have memorized.
_NON_TEXT_COLUMNS = {"index", "answer", "label"}


def _shingle_hashes(text: str, k: int = SHINGLE_SIZE) -> set[int]:
    """Deterministic 64-bit hashes of every k-word shingle.

    Hashed rather than stored as strings to keep the LegalBench index (millions
    of shingles) in a few hundred MB. blake2b rather than `hash()` because
    `hash()` is salted per process, and a decontamination report that changes
    between runs is not a report.
    """
    words = text.lower().split()
    if len(words) < k:
        return set()
    out: set[int] = set()
    for i in range(len(words) - k + 1):
        shingle = " ".join(words[i : i + k]).encode("utf-8")
        out.add(int.from_bytes(hashlib.blake2b(shingle, digest_size=8).digest(), "big"))
    return out


def iter_legalbench_texts(cache_dir: str | None = None) -> Iterator[str]:
    """Yield every prose field from every LegalBench task, both splits.

    Column names vary across the 162 tasks (`text`, `question`, `contract`,
    `clause`, ...), so this takes everything that is not an id or a label rather
    than hard-coding a schema that would silently miss most tasks.
    """
    tsv_files = [
        f for f in list_repo_files(LEGALBENCH_REPO, repo_type="dataset") if f.endswith(".tsv")
    ]
    for filename in tsv_files:
        local = hf_hub_download(LEGALBENCH_REPO, filename, repo_type="dataset", cache_dir=cache_dir)
        with open(local, encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                for column, value in row.items():
                    if column and column.lower() not in _NON_TEXT_COLUMNS and value:
                        yield value


def build_legalbench_index(cache_dir: str | None = None) -> set[int]:
    index: set[int] = set()
    n_texts = 0
    for text in iter_legalbench_texts(cache_dir):
        index |= _shingle_hashes(text)
        n_texts += 1
        if n_texts % 20_000 == 0:
            print(f"  indexed {n_texts:,} LegalBench fields...", file=sys.stderr)
    print(
        f"LegalBench index: {n_texts:,} text fields, {len(index):,} distinct {SHINGLE_SIZE}-grams",
        file=sys.stderr,
    )
    return index


def decontaminate(
    pairs: list[dict[str, Any]], index: set[int]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split pairs into (clean, contaminated)."""
    clean: list[dict[str, Any]] = []
    contaminated: list[dict[str, Any]] = []
    for pair in pairs:
        combined = f"{pair.get('instruction', '')} {pair.get('response', '')}"
        if _shingle_hashes(combined) & index:
            contaminated.append(pair)
        else:
            clean.append(pair)
    return clean, contaminated


def split_train_val(
    pairs: list[dict[str, Any]], *, val_size: int, seed: int = 42
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Hold out a validation set for loss monitoring only.

    This is *not* an evaluation set — it comes from the same synthetic
    distribution as training data, so a good loss on it says nothing about
    legal-task performance. LegalBench is the evaluation set. Conflating the two
    is the exact mistake this module exists to prevent.
    """
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    val_size = min(val_size, len(shuffled) // 10)
    return shuffled[val_size:], shuffled[:val_size]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Decontaminate against LegalBench and split.")
    parser.add_argument(
        "--in", dest="in_path", type=Path, default=Path("data/filtered_pairs.jsonl")
    )
    parser.add_argument("--train-out", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--val-out", type=Path, default=Path("data/val.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("eval_results/decontamination.json"))
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    pairs = read_jsonl(args.in_path)
    print(f"loaded {len(pairs):,} filtered pairs", file=sys.stderr)

    index = build_legalbench_index()
    clean, contaminated = decontaminate(pairs, index)
    train, val = split_train_val(clean, val_size=args.val_size, seed=args.seed)

    write_jsonl(args.train_out, train)
    write_jsonl(args.val_out, val)

    report = {
        "shingle_size": SHINGLE_SIZE,
        "legalbench_ngrams": len(index),
        "input_pairs": len(pairs),
        "contaminated_dropped": len(contaminated),
        "contamination_rate": round(len(contaminated) / len(pairs), 6) if pairs else 0.0,
        "clean_pairs": len(clean),
        "train_pairs": len(train),
        "val_pairs": len(val),
        # A handful of dropped instructions, so a reader can judge whether the
        # 13-gram threshold is catching real leakage or innocent boilerplate.
        "dropped_examples": [p["instruction"][:200] for p in contaminated[:5]],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
