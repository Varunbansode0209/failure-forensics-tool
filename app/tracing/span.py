from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class Span(BaseModel):
    span_id: str
    trace_id: str
    step_name: str

    input: dict[str, Any]
    output: Optional[dict[str, Any]] = None

    prompt: Optional[str] = None
    raw_response: Optional[str] = None

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    latency_ms: float = 0.0

    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0
    )

    error: Optional[str] = None

    started_at: datetime
    ended_at: datetime

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary representation of the span."""
        return self.model_dump(mode="json")