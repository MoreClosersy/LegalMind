"""Prompts for synthetic instruction generation.

Kept in their own module so the test suite can assert on them directly — in
particular that the generator is told *not* to emit disclaimer boilerplate.
Disclaimers are the serving layer's job; baking them into training data would
teach the model to produce a compliance artifact probabilistically, which is
exactly the design mistake this project avoids.

The system prompt is deliberately long and byte-stable: it is the cached prefix
for every request in the batch (`cache_control` is set on its last block), so
the per-request cost is dominated by the passage in the user turn. Sonnet 5's
minimum cacheable prefix is 1024 tokens; this prompt clears that comfortably.
Do not interpolate anything per-request into it — a single varying byte here
invalidates the cache for the entire batch.
"""

from __future__ import annotations

TASK_TYPES = (
    "issue_spotting",
    "rule_statement",
    "clause_explanation",
    "statutory_interpretation",
)

SYNTHESIS_SYSTEM_PROMPT = """\
You generate training data for a legal-domain instruction-following model. You \
will be given a passage of authentic United States legal text — a section of the \
Code of Federal Regulations, a judicial opinion, or a Federal Register notice. \
Your job is to write instruction/response pairs that a law student or paralegal \
might plausibly produce while studying that passage.

# Task types

Write pairs of these four kinds. Vary them; do not produce four of the same kind.

- `issue_spotting`: the instruction presents a short hypothetical governed by the \
passage and asks which legal issues it raises. The response identifies the issues \
and ties each one to the operative language.
- `rule_statement`: the instruction asks what rule the passage establishes. The \
response states the rule and its elements precisely, in the passage's own terms.
- `clause_explanation`: the instruction quotes or points at a specific clause and \
asks what it means. The response explains it, including any terms of art.
- `statutory_interpretation`: the instruction asks how the text applies to a fact \
pattern at the edge of its language. The response reasons from the text, names the \
interpretive question, and is honest about genuine ambiguity.

# Requirements

1. **Ground every response in the passage.** Do not introduce cases, statutes, or \
holdings that are not in the text you were given. If you cannot answer a question \
from the passage alone, do not ask that question.
2. **Never fabricate a citation.** Cite only what the passage itself cites, and \
only in the form the passage uses. A response with an invented case name or \
reporter cite is worse than no response.
3. **Both the instruction and the response must be self-contained.** The model \
being trained never sees this passage. Anything that points at it is a dangling \
reference that teaches the model to invent a source it was not given.

   This applies to the response just as much as the instruction, and the \
response is where it goes wrong. Do not write "the passage does not commit the \
agency to...", "based on the text, this is a close question", "the text \
specifies that...", "as stated above", or "in the excerpt". Name the authority \
instead — "Section 207 does not commit the agency to...", "§ 970.207(a) \
specifies that..." — or, where the wording itself is what matters, quote it \
inline in the instruction so the response can refer to something the reader \
actually has.

   Read each response back as if you had never seen the passage. If any \
sentence would leave a reader asking "which text?", rewrite it.
4. **No disclaimers, no hedging boilerplate.** Do not write "consult an attorney", \
"this is not legal advice", "for educational purposes only", or any variation. \
That text is added deterministically downstream; including it here would teach the \
model to produce it unreliably instead. Answer the legal question directly.
5. **General education only.** Every instruction must be a question about what the \
law *is*. Never write an instruction that asks what a specific person should do \
about their own situation — that is a separate, deliberately-held-out behaviour.
6. **Respond at working length.** Two to five paragraphs for most pairs; longer \
where the passage genuinely requires it. Do not pad.

# Output

Return 2 to 3 pairs. Prefer 2 high-quality pairs over 3 that stretch the passage.
"""

# A required task type is assigned per request rather than left to the model.
#
# Measured on a 99-passage probe with the type left free: `statutory_interpretation`
# came out at 4% of pairs while the other three sat near 32% each — consistent
# across all three sources, so it is the model's preference, not the corpus.
# Asking for variety in prose did not produce it. Rotating a required type across
# requests makes the distribution a property of the batch instead of a hope.
SYNTHESIS_USER_TEMPLATE = """\
<source>{source}</source>

<passage>
{passage}
</passage>

Write 2-3 instruction/response pairs grounded in this passage.

At least one pair must be of type `{required_task_type}`. If this passage genuinely \
cannot support that type, use the closest type it does support rather than \
forcing it — a strained pair is worse than an unbalanced distribution.

Before returning, reread each response with the passage covered. Rewrite any \
sentence that refers to "the passage", "the text", or "the excerpt"."""

# Structured-output schema. Note the deliberate absence of `minItems`/`maxItems`:
# the API's structured-output subset does not support array-length constraints,
# so the 2-3 bound is stated in the prompt and enforced client-side in filter.py.
SYNTHESIS_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "task_type": {"type": "string", "enum": list(TASK_TYPES)},
                    "instruction": {"type": "string"},
                    "response": {"type": "string"},
                },
                "required": ["task_type", "instruction", "response"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["pairs"],
    "additionalProperties": False,
}


def build_user_prompt(source: str, passage: str, required_task_type: str) -> str:
    """Render the per-request user turn. Everything variable lives here, after
    the cached system prefix."""
    if required_task_type not in TASK_TYPES:
        raise ValueError(f"unknown task type {required_task_type!r}; known: {TASK_TYPES}")
    return SYNTHESIS_USER_TEMPLATE.format(
        source=source, passage=passage, required_task_type=required_task_type
    )


def required_task_type_for(index: int) -> str:
    """Rotate the required type across a batch so coverage is deterministic."""
    return TASK_TYPES[index % len(TASK_TYPES)]
