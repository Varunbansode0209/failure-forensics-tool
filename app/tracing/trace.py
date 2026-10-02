from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.tracing.span import Span


class Trace(BaseModel):
    trace_id: str
    started_at: datetime
    ended_at: datetime | None = None

    spans: list[Span] = Field(default_factory=list)

    status: str = "running"
    error: str | None = None

    def add_span(self, span: Span) -> None:
        """Add a completed span to this trace."""
        self.spans.append(span)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary representation of the trace."""
        return self.model_dump(mode="json")