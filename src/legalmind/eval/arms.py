"""The three comparison arms.

Base-zero-shot versus fine-tuned is a rigged comparison. It is the one almost
every fine-tuning writeup reports, and it measures mostly whether the base model
was prompted competently — which, at zero-shot with a bare instruction, it was
not. The number it produces is large and means very little.

So there are three arms, and the middle one is the real opponent:

    A  base, zero-shot, bare instruction
    B  base, with a carefully written system prompt and few-shot examples
    C  fine-tuned (LoRA adapter)

If C only narrowly beats B, that is the finding, and it gets reported. The
honest follow-up is not "the fine-tune failed" but "here is when the fine-tuning
cost is justified anyway": B pays for its system prompt and exemplars on every
single request, forever, in both latency and input tokens, and it cannot be
deployed anywhere the prompt cannot travel with it.

Arm B is deliberately given a good prompt rather than a strawman. Weakening it
would manufacture a better headline number and invalidate the comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ArmId(str, Enum):
    BASE_ZERO_SHOT = "A_base_zero_shot"
    BASE_PROMPTED = "B_base_prompted"
    FINE_TUNED = "C_fine_tuned"


# Arm B's system prompt. This is the thing the fine-tune has to beat, so it is
# written to be genuinely good: it states the role, the expected shape of an
# answer, and — critically — the same refusal policy the fine-tune is trained on.
# Omitting the refusal policy here would let arm C win the calibration metrics by
# default, which would be a rigged result of a different kind.
PROMPTED_SYSTEM = """\
You are a legal research assistant answering questions about United States law.

Answer questions about what the law is — doctrines, elements, standards, \
statutory frameworks, procedural rules — directly and substantively. Give the \
rule, its elements, and the reasoning that connects them. Two to five \
paragraphs. Do not ask which jurisdiction the reader is in before answering a \
general question.

When a question asks what the asker should do about their own situation — \
whether to sue, sign, file, settle, or accept an offer — decline to advise on \
those specific facts. Say so in the first sentence, explain briefly that the \
answer turns on jurisdiction and on facts that need review by someone who can \
take responsibility for the advice, then pivot to explaining the general legal \
framework the situation implicates. Point toward licensed counsel.

Never invent a case name or citation. Naming a well-known statute or doctrine \
is fine; fabricating a reporter cite is not.
"""

# Few-shot exemplars for arm B: one of each behaviour, so the shape of both a
# direct answer and a calibrated refusal is demonstrated rather than described.
FEW_SHOT: tuple[tuple[str, str], ...] = (
    (
        "What are the elements of adverse possession?",
        "Adverse possession requires possession that is actual, open and notorious, "
        "exclusive, hostile, and continuous for the statutory period. Each element is "
        "independent and the claimant bears the burden on all of them.\n\n"
        "Actual possession means using the land in a way consistent with its character "
        "rather than merely asserting a claim to it. Open and notorious means the use "
        "would put a reasonably attentive owner on notice. Exclusive means the claimant "
        "is not sharing possession with the true owner or the general public. Hostile "
        "means without the owner's permission — in most jurisdictions this is an "
        "objective inquiry that does not depend on the claimant's state of mind. "
        "Continuous means uninterrupted for the full statutory period, which varies by "
        "state and is commonly between five and twenty years.",
    ),
    (
        "My neighbor has been using a strip of my yard for years. Should I sue him?",
        "I can't advise you on whether to sue, because that turns on your jurisdiction's "
        "statutory period, the specific history of the use, and facts about your deed and "
        "survey that would need to be reviewed by someone who can take responsibility for "
        "the advice.\n\n"
        "What I can explain is the framework your situation implicates. A neighbor's "
        "long-running use of your land raises two distinct doctrines that are often "
        "confused. Adverse possession transfers title outright and requires possession "
        "that is actual, open, exclusive, hostile, and continuous for the statutory "
        "period. A prescriptive easement transfers only a right to continue the use, does "
        "not require exclusivity, and is the more common outcome where a neighbor has been "
        "crossing or encroaching rather than occupying.\n\n"
        "Both clocks generally stop if the owner grants permission, which is why the "
        "history of any conversations between you matters. A real estate attorney or your "
        "local bar's referral service can tell you where your facts fall.",
    ),
)


@dataclass(frozen=True)
class Arm:
    """One comparison arm.

    `adapter` is the vLLM LoRA module name for arm C and None for the base arms,
    so all three can be served by one vLLM process with `--enable-lora`.
    """

    id: ArmId
    label: str
    system: str | None = None
    few_shot: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    adapter: str | None = None

    def build_messages(self, instruction: str) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        for user_turn, assistant_turn in self.few_shot:
            messages.append({"role": "user", "content": user_turn})
            messages.append({"role": "assistant", "content": assistant_turn})
        messages.append({"role": "user", "content": instruction})
        return messages


def build_arms(adapter_name: str = "legalmind") -> dict[ArmId, Arm]:
    return {
        ArmId.BASE_ZERO_SHOT: Arm(
            id=ArmId.BASE_ZERO_SHOT,
            label="Base, zero-shot",
        ),
        ArmId.BASE_PROMPTED: Arm(
            id=ArmId.BASE_PROMPTED,
            label="Base + system prompt + few-shot",
            system=PROMPTED_SYSTEM,
            few_shot=FEW_SHOT,
        ),
        ArmId.FINE_TUNED: Arm(
            id=ArmId.FINE_TUNED,
            label="Fine-tuned (LoRA)",
            # No system prompt and no exemplars by design: the point of the
            # fine-tune is that the behaviour is in the weights. Giving arm C
            # arm B's prompt would measure the prompt, not the training.
            adapter=adapter_name,
        ),
    }
