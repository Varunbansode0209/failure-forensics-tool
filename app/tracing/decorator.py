import time
import uuid
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable
from contextvars import ContextVar
from app.tracing.context import get_llm_result, clear_llm_result

from app.tracing.span import Span
from app.tracing.trace import Trace


_current_trace: ContextVar[Trace | None] = ContextVar(
    "current_trace",
    default=None,
)


def _serialize(value: Any) -> dict[str, Any]:
    """Convert a Pydantic model into JSON-safe dictionary data."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")

    if isinstance(value, dict):
        return value

    return {"value": value}

def _extract_confidence(output: Any) -> float | None:
    """Extract confidence from a pipeline output if available."""
    value = getattr(output, "confidence", None)

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)

    return None

def start_trace() -> Trace:
    trace = Trace(
        trace_id=str(uuid.uuid4()),
        started_at=datetime.now(timezone.utc),
    )
    _current_trace.set(trace)
    return trace


def get_current_trace() -> Trace | None:
    return _current_trace.get()


def end_trace() -> Trace | None:
    trace = _current_trace.get()

    if trace is not None:
        trace.ended_at = datetime.now(timezone.utc)
        trace.status = "completed"

    _current_trace.set(None)
    return trace


def traced(func: Callable) -> Callable:
    """
    Trace the execution of a pipeline function.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        trace = get_current_trace()

        span_id = str(uuid.uuid4())
        started_at = datetime.now(timezone.utc)
        start_time = time.perf_counter()

        input_data = {}

        if args:
            input_data = _serialize(args[0])
        elif kwargs:
            input_data = {
                key: _serialize(value)
                for key, value in kwargs.items()
            }

        try:
            result = func(*args, **kwargs)

            ended_at = datetime.now(timezone.utc)
            latency_ms = (time.perf_counter() - start_time) * 1000
            llm_result = get_llm_result()

            span = Span(
                span_id=span_id,
                trace_id=trace.trace_id if trace else "pending",
                step_name=func.__name__,
                input=input_data,
                output=_serialize(result),
                confidence=_extract_confidence(result),
                started_at=started_at,
                ended_at=ended_at,
                latency_ms=latency_ms,
                model=llm_result.model if llm_result else None,
                prompt=llm_result.prompt if llm_result else None,
                raw_response=llm_result.text if llm_result else None,
                prompt_tokens=llm_result.prompt_tokens or 0 if llm_result else 0,
                completion_tokens=llm_result.completion_tokens or 0 if llm_result else 0,
                thought_tokens=llm_result.thought_tokens if llm_result else None,
                total_tokens=llm_result.total_tokens or 0 if llm_result else 0,
                            )

            if trace:
                trace.add_span(span)

            return result

        except Exception as exc:
            ended_at = datetime.now(timezone.utc)
            latency_ms = (time.perf_counter() - start_time) * 1000

            span = Span(
                span_id=span_id,
                trace_id=trace.trace_id if trace else "pending",
                step_name=func.__name__,
                input=input_data,
                output=None,
                started_at=started_at,
                ended_at=ended_at,
                latency_ms=latency_ms,
                error=str(exc),
            )

            if trace:
                trace.add_span(span)

            raise

    return wrapper