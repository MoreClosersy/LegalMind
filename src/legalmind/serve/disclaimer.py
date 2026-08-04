"""Deterministic UPL (Unauthorized Practice of Law) enforcement.

This module is the project's compliance guarantee. It is deliberately *not* a
model behaviour.

The tempting design is to teach the model to emit a disclaimer during
fine-tuning. That looks elegant and fails on four counts:

* it is probabilistic — the model drops the disclaimer under distribution shift,
  adversarial prompting, or simply an unusual question shape;
* it is unauditable — you cannot point a compliance reviewer at a weight;
* it is unversionable — changing one word of legal text means retraining;
* it spends model capacity on text a string concatenation can produce.

So the guarantee lives here: a pure function, unit-tested, that runs on every
response before it reaches a user. It cannot be prompted away, because the model
is not involved in the decision.

The fine-tune is still doing compliance work — but the part that genuinely
requires a model: **refusal calibration**, deciding whether a question is general
legal education (answer it) or a request for advice on the asker's own situation
(decline it). No post-processing rule can make that judgement. That division —
rules for what rules can guarantee, model for what only a model can judge — is
the actual design position.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Bump on any edit to DISCLAIMER_TEXT. Responses are logged with this version so
# an auditor can tell which text a given historical response carried — the whole
# point of keeping this out of the weights.
DISCLAIMER_VERSION = "1.0.0"

DISCLAIMER_TEXT = (
    "This response is general legal information for educational purposes only. "
    "It is not legal advice, does not create an attorney-client relationship, "
    "and may not reflect the law of your jurisdiction. Consult a licensed "
    "attorney about your specific situation."
)

# Recognizes a disclaimer that is *already* present so enforcement is idempotent.
# Intentionally loose: a false positive costs a missing duplicate, while a false
# negative costs a duplicated paragraph — both cosmetic, neither a compliance
# failure. The failure mode this module exists to prevent is *absence*, and
# absence is unambiguous.
_DISCLAIMER_PRESENT = re.compile(
    r"(not\s+legal\s+advice"
    r"|educational\s+purposes\s+only"
    r"|consult\s+a\s+licensed\s+attorney)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EnforcementResult:
    """Outcome of running the compliance layer over one response.

    `disclaimer_added` is the metric worth alerting on. A rate near 100% is the
    expected steady state — the model is not trained to produce this text, so the
    layer should be doing the work on essentially every response. A rate that
    drifts toward zero means the model has started emitting disclaimer-like text
    on its own, which is a training-data leak worth investigating.
    """

    text: str
    disclaimer_added: bool
    already_present: bool
    version: str = DISCLAIMER_VERSION


def has_disclaimer(text: str) -> bool:
    return bool(_DISCLAIMER_PRESENT.search(text))


def enforce(response: str) -> EnforcementResult:
    """Guarantee that `response` carries a UPL disclaimer.

    Idempotent: enforcing an already-compliant response is a no-op. Safe on
    empty input — an empty response still gets the disclaimer rather than being
    passed through unlabelled.
    """
    body = response.rstrip()

    if has_disclaimer(body):
        return EnforcementResult(text=body, disclaimer_added=False, already_present=True)

    separator = "\n\n" if body else ""
    return EnforcementResult(
        text=f"{body}{separator}{DISCLAIMER_TEXT}",
        disclaimer_added=True,
        already_present=False,
    )


def enforce_stream_tail(streamed_body: str) -> str:
    """Return the text to append after an SSE stream completes.

    Streaming and post-hoc enforcement are in tension: the disclaimer cannot be
    checked until generation finishes, but the user is already reading. The
    resolution is to stream the body verbatim and emit the disclaimer as a final
    chunk — the user sees tokens immediately and the guarantee still holds at the
    moment the response is complete. Returns an empty string when the model
    already produced acceptable text.
    """
    result = enforce(streamed_body)
    return f"\n\n{DISCLAIMER_TEXT}" if result.disclaimer_added else ""
