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
        self.spans.append(span)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")