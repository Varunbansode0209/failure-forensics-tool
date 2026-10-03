import json
from pathlib import Path

from app.tracing.trace import Trace


TRACE_DIR = Path("data/traces")


def save_trace(trace: Trace) -> Path:
    TRACE_DIR.mkdir(parents=True, exist_ok=True)

    file_path = TRACE_DIR / f"{trace.trace_id}.json"

    with file_path.open("w", encoding="utf-8") as file:
        json.dump(trace.to_dict(), file, indent=2, ensure_ascii=False)

    return file_path