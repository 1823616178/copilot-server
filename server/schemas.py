"""Pydantic request models for the OpenAI-compatible endpoints."""

from typing import Any, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

from .config import MODEL_NAME


class ChatMessage(BaseModel):
    role: str
    # content is a plain string, or OpenAI "content parts" (list of dicts), or
    # null for some tool/assistant messages.
    content: Optional[Union[str, List[Any]]] = None


class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = MODEL_NAME
    stream: bool = False
    # Copilot's own conversation id (returned in earlier responses). Pass it back
    # to continue that thread; omit it to start a fresh conversation. Outside
    # OpenAI's schema, but standard clients can set it via extra_body.
    conversation_id: Optional[str] = None
    # Thinking control (see server/reasoning.py). OpenAI style:
    #   "reasoning_effort": "none"|"minimal"|"low"|"medium"|"high"
    #   "reasoning": {"effort": ...}   (Responses-API style)
    # Anthropic / DeepSeek style:  "thinking": {"type": "enabled"|"disabled"|"auto"}
    # Direct:  "copilot_mode": "auto"|"quick"|"deep"
    reasoning_effort: Optional[str] = None
    reasoning: Optional[Any] = None
    thinking: Optional[Any] = None
    copilot_mode: Optional[str] = None
    # Return the thinking summary as reasoning_content (default: env
    # COPILOT_INCLUDE_REASONING, on).
    include_reasoning: Optional[bool] = None
    # Any other OpenAI fields (temperature, max_tokens, ...) are accepted and
    # ignored — Copilot's consumer protocol doesn't expose those knobs.


class ImageGenerationRequest(BaseModel):
    """Subset of OpenAI's POST /v1/images/generations body."""

    prompt: str = Field(min_length=1)
    model: Optional[str] = None
    n: int = Field(default=1, ge=1, le=10)
    size: Optional[str] = None
    response_format: Literal["url", "b64_json"] = "url"

    @field_validator("prompt")
    @classmethod
    def nonblank_prompt(cls, value):
        if not value.strip():
            raise ValueError("prompt must not be blank")
        return value
