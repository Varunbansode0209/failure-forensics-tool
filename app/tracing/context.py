from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.llm.client import LLMResult


_current_llm_result: ContextVar[Any | None] = ContextVar(
    "current_llm_result",
    default=None,
)


def set_llm_result(result: LLMResult) -> None:
    _current_llm_result.set(result)


def get_llm_result() -> LLMResult | None:
    return _current_llm_result.get()


def clear_llm_result() -> None:
    _current_llm_result.set(None)