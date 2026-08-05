"""Request and response models for the gateway.

The disclaimer is a first-class field rather than a string glued onto the end of
the content. A client that renders `content` and ignores everything else still
sees the disclaimer, because the gateway also appends it to the text — but a
client that wants to render it distinctly, or log which version was served, can
do that without parsing prose. `serve/disclaimer.py` keeps a version number
precisely so it can be reported here and matched against a stored response
during an audit.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from legalmind.serve.disclaimer import DISCLAIMER_TEXT, DISCLAIMER_VERSION


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1)
    # None means the base model. Named adapters are validated against the set
    # the gateway was configured with, so a typo returns 400 rather than being
    # forwarded to vLLM and coming back as an opaque upstream error.
    adapter: str | None = None
    max_tokens: int = Field(default=1024, ge=1, le=4096)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    stream: bool = False


class Disclaimer(BaseModel):
    text: str = DISCLAIMER_TEXT
    version: str = DISCLAIMER_VERSION
    # True when the gateway added it, False when the response already carried
    # the exact enforced text. A rate drifting away from ~100% means the model
    # has started emitting disclaimer-shaped prose, which is a training-data
    # leak worth investigating rather than a convenience.
    added: bool = True


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0


class ChatResponse(BaseModel):
    content: str
    disclaimer: Disclaimer
    usage: Usage
    adapter: str | None = None
    model: str
    latency_ms: float
    # Observability only, never gates enforcement. See disclaimer.py.
    model_volunteered_disclaimer: bool = False


class StreamMeta(BaseModel):
    """First SSE event, emitted before any generated token.

    Sending the disclaimer up front is what makes the guarantee survive a
    client disconnect. Post-hoc enforcement cannot run if the connection dies
    mid-stream, and buffering the whole response to enforce before sending would
    throw away the reason to stream at all. Emitting it as metadata first costs
    nothing and means every byte of generated text a client ever sees has
    already been preceded by the disclaimer.
    """

    disclaimer: Disclaimer
    model: str
    adapter: str | None = None


class StreamDone(BaseModel):
    disclaimer_appended: bool
    usage: Usage
    latency_ms: float
