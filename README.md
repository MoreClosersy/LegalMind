# LegalMind

> Domain-adapted legal LLM: QLoRA fine-tuning of Qwen3-8B, with a deterministic
> UPL compliance layer at serving time rather than a probabilistic one in the weights.

> **Status: in progress.** Sections marked `<!-- PENDING -->` are placeholders.
> No number appears in this README until a committed file under `eval_results/`
> backs it and a command in this README reproduces it.

## What it does

```
Input:  "What are the elements of a Rule 12(b)(6) motion?"
        ↓
        Qwen3-8B + LegalMind LoRA adapter  (served by vLLM, streamed over SSE)
        ↓
        Gateway compliance layer (deterministic, unit-tested)
        ↓
Output: A general legal-education answer, with an enforced UPL disclaimer.

Input:  "My landlord is evicting me — should I sue?"
        ↓
Output: A calibrated refusal — specific-case advice is unauthorized practice
        of law — plus a pointer to licensed counsel.
```

The second behaviour is what the fine-tune actually buys. The disclaimer itself
is not learned; see *Design decisions* below.

## Architecture

<!-- PENDING: docs/architecture.png -->

| Layer | Component | Responsibility |
| --- | --- | --- |
| Data | `src/legalmind/data/` | Synthesize instruction pairs from raw legal corpora; filter, dedupe, decontaminate against the eval set |
| Training | `src/legalmind/train/` | QLoRA SFT with completion-only loss masking; S3 checkpointing for Spot resilience |
| Evaluation | `src/legalmind/eval/` | Three-arm comparison, refusal calibration, red-team suite, catastrophic-forgetting check |
| Serving | `src/legalmind/serve/` | vLLM backend with hot-swappable LoRA; FastAPI gateway; deterministic UPL enforcement |

## Design decisions

These are the trade-offs the project is actually about.

**1. Train and evaluate on disjoint sources.** LegalBench is a *benchmark*, not a
training set. Fine-tuning on it and then reporting held-out accuracy on it is
train/test contamination. Training data here is synthesized from raw legal text
(`pile-of-law`: CFR, CourtListener opinions, Federal Register); LegalBench is
touched only at evaluation time. `src/legalmind/data/decontaminate.py` enforces
this with 13-gram overlap detection and reports how many samples it dropped.

**2. The UPL disclaimer is enforced at the serving layer, not learned.** Teaching
a model to emit boilerplate via SFT gives a probabilistic guarantee that degrades
under distribution shift, cannot be audited, and requires a retrain to change one
word of legal text. The gateway enforces it deterministically and the enforcement
is unit-tested. What the fine-tune *does* teach is **refusal calibration** —
when to decline specific-case advice versus answer a general legal question —
which is genuinely not solvable with a post-processing rule.

**3. The comparison includes a real opponent.** Base zero-shot versus fine-tuned
is a rigged comparison. The harness runs three arms: base zero-shot, base with a
carefully written system prompt and few-shot examples, and fine-tuned. If the
fine-tune only narrowly beats a good prompt, that is what gets reported, along
with the conditions under which the fine-tuning cost is justified anyway
(latency, per-request token cost, private deployment).

**4. ~10k training samples, not 90k.** Instruction tuning teaches format and
behaviour, not knowledge (the LIMA argument). A smaller, filtered, deduplicated
set trains in one Spot window and avoids paying for noise.

**5. g5.xlarge over g4dn.xlarge.** Both draw on the same 4-vCPU G-instance quota.
The A10G (Ampere) supports bf16 and FlashAttention-2 and runs vLLM properly; the
T4 (Turing) does none of these.

## Results

<!-- PENDING: three-arm comparison table. Populated from eval_results/*.json. -->

## Performance

<!-- PENDING: measured p50/p95/TTFT/tokens-per-second under load. See docs/latency.md. -->

## Quick start

### Prerequisites

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- An Anthropic API key (synthetic data generation and LLM-as-judge)
- For the full 8B path: an NVIDIA GPU with ≥24GB (A10G / L4 / A100)

### Local setup

```bash
git clone https://github.com/MoreClosersy/LegalMind.git
cd LegalMind

uv sync --extra train --extra serve

cp .env.example .env
# Fill in ANTHROPIC_API_KEY at minimum.
```

### Smoke-test the training pipeline (CPU, ~1 minute)

```bash
uv run python -m legalmind.train.sft --config configs/train_smoke_0.6b.yaml
```

Runs 10 steps of Qwen3-0.6B QLoRA on committed fixtures. No GPU, no API key, and
no network beyond the model download. This is also what CI runs on every push.

### Verify that prompt tokens are excluded from the loss

```bash
uv run python scripts/verify_masking.py --config configs/train_smoke_0.6b.yaml
```

Prints the exact loss boundary. On the smoke fixtures, 34 of 119 tokens are
masked; the last untrained span ends with the assistant generation prefix and the
first trained token is the start of the answer.

### Build the training data

```bash
uv run python -m legalmind.data.synthesize --limit 100 --out data/probe.jsonl
```

Cost probe — run this before the full batch and read the reported cost per 1,000
pairs. Then:

```bash
uv run python -m legalmind.data.synthesize --out data/raw_pairs.jsonl
uv run python -m legalmind.data.filter --in data/raw_pairs.jsonl --out data/filtered_pairs.jsonl
uv run python -m legalmind.data.decontaminate --in data/filtered_pairs.jsonl
```

The last two steps write reports to `eval_results/`. The decontamination report
is the evidence behind the no-contamination claim above.

<!-- PENDING: evaluation and serving quick-start steps. -->

## What I'd do next

- **Real citation verification.** The current eval only detects malformed or
  self-evidently fabricated citations. Verifying that a cited case exists and
  says what the model claims requires grounding against the CourtListener API —
  worth doing, and out of scope for a two-week project.
- **Preference optimization (DPO/KTO)** on the refusal boundary, using the
  red-team set to mine hard negatives.
- **Multi-turn.** Everything here is single-turn; real legal Q&A is not.
- **Larger eval.** Per-task confidence intervals on LegalBench are wide at the
  sample sizes used here.

## License

MIT
