"""Map OpenAI-style reasoning knobs onto Copilot's response modes.

Copilot (copilot.com / M365 consumer chat hub) exposes exactly three modes via
the chat frame's ``tone`` field — see copilot/protocol.py:

    mode    tone         UI label
    auto    Magic        自动 (decides how long to think)
    quick   Chat         快速响应 (no thinking)
    deep    Reasoning    深度思考 (think deeper)

There is no finer-grained effort/budget control, so OpenAI's five
``reasoning_effort`` levels and Anthropic-style ``thinking`` collapse onto these.

Precedence (first one set wins): ``copilot_mode`` > ``thinking`` >
``reasoning_effort`` (or ``reasoning.effort``) > model-name alias >
``COPILOT_DEFAULT_MODE`` env (default ``auto``).
"""

import os
from typing import Any, Optional

from copilot.protocol import TONE_AUTO, TONE_DEEP, TONE_QUICK

MODE_TONES = {"auto": TONE_AUTO, "quick": TONE_QUICK, "deep": TONE_DEEP}

_MODE_ALIASES = {
    "auto": "auto", "magic": "auto", "default": "auto",
    "quick": "quick", "fast": "quick", "chat": "quick", "off": "quick", "none": "quick",
    "disabled": "quick", "false": "quick",
    "deep": "deep", "think": "deep", "thinking": "deep", "reasoning": "deep",
    "on": "deep", "enabled": "deep", "true": "deep",
}

EFFORT_MODES = {
    "none": "quick", "minimal": "quick",
    "low": "auto", "medium": "auto",
    "high": "deep", "xhigh": "deep", "max": "deep",
}

# Model ids advertised on /v1/models (the bare id keeps the env default).
MODEL_MODES = {
    "copilot": None,
    "copilot-auto": "auto",
    "copilot-quick": "quick",
    "copilot-fast": "quick",
    "copilot-think": "deep",
    "copilot-deep": "deep",
}


def _norm_mode(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bool):
        return "deep" if value else "quick"
    return _MODE_ALIASES.get(str(value).strip().lower())


def default_mode() -> str:
    return _norm_mode(os.environ.get("COPILOT_DEFAULT_MODE", "auto")) or "auto"


def resolve_mode(model: Optional[str] = None, reasoning_effort: Any = None,
                 thinking: Any = None, copilot_mode: Any = None,
                 reasoning: Any = None) -> str:
    """Return ``auto`` / ``quick`` / ``deep``. Raises ValueError on bad values."""
    if copilot_mode is not None:
        m = _norm_mode(copilot_mode)
        if m is None:
            raise ValueError(f"copilot_mode must be one of auto, quick, deep (got {copilot_mode!r})")
        return m
    if thinking is not None:
        kind = thinking.get("type") if isinstance(thinking, dict) else thinking
        m = _norm_mode(kind)
        if m is None:
            raise ValueError("thinking.type must be 'enabled', 'disabled' or 'auto'")
        return m
    if reasoning_effort is None and isinstance(reasoning, dict):
        reasoning_effort = reasoning.get("effort")
    if reasoning_effort is not None:
        m = EFFORT_MODES.get(str(reasoning_effort).strip().lower())
        if m is None:
            raise ValueError("reasoning_effort must be one of none, minimal, low, medium, high")
        return m
    if model:
        m = MODEL_MODES.get(model.strip().lower())
        if m:
            return m
    return default_mode()


def resolve_tone(**kw) -> str:
    return MODE_TONES[resolve_mode(**kw)]
