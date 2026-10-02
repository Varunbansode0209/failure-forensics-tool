import os
import time



from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel


load_dotenv()
DEFAULT_MODEL = "gemini-3.1-flash-lite"
DEFAULT_THINKING_LEVEL = "low"
TIMEOUT_MS = int(os.getenv("LLM_TIMEOUT_MS", "60000"))  # milliseconds

_client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options={
        "timeout": 60000,
        "retry_options": {
            "attempts": 1
        }
    }
)

def _get_client():
    # Created on first use, so GEMINI_API_KEY is read after .env has loaded
    global _client
    if _client is None:
        _client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY"),
            
        )
    return _client


class LLMError(Exception):
    """Raised when the Gemini call fails, times out, or returns nothing."""


class LLMResult(BaseModel):
    text: str
    prompt: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    thought_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: float


def call_llm(prompt: str, model: str = DEFAULT_MODEL,
             thinking_level: str = DEFAULT_THINKING_LEVEL) -> LLMResult:
    start = time.perf_counter()
    try:
        interaction = _get_client().interactions.create(
            model=model,
            input=prompt,
            generation_config={"thinking_level": thinking_level},
            store=False,
        )
    except Exception as e:
        raise LLMError(f"Gemini call failed: {e}") from e
    latency_ms = (time.perf_counter() - start) * 1000

    status = getattr(interaction, "status", None)
    status = getattr(status, "value", status)  # works for enum or plain string
    if status not in (None, "completed"):
        raise LLMError(f"Interaction ended with status: {status}")

    text = interaction.output_text
    if not text:
        raise LLMError("Gemini returned an empty response")

    usage = getattr(interaction, "usage", None)
    return LLMResult(
        text=text,
        prompt=prompt,
        model=model,
        prompt_tokens=getattr(usage, "total_input_tokens", None),
        completion_tokens=getattr(usage, "total_output_tokens", None),
        thought_tokens=getattr(usage, "total_thought_tokens", None),
        total_tokens=getattr(usage, "total_tokens", None),
        latency_ms=latency_ms,
    )