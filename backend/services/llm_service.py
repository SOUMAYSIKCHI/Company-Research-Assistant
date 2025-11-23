# backend/services/llm_service.py

import os
from typing import Generator, Optional

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set. Please configure it in your environment or .env file.")

client = Groq(api_key=GROQ_API_KEY)


def call_llm(
    system_prompt: str,
    user_prompt: str,
    *,
    model: Optional[str] = None,
    max_tokens: int = 2000,
    temperature: float = 0.2,
) -> str:
    """Synchronous (non-streaming) Groq chat completion with basic error safety."""
    m = model or GROQ_MODEL
    try:
        completion = client.chat.completions.create(
            model=m,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = completion.choices[0].message.content or ""
    except Exception as e:
        content = f"(llm_error: {e})"
    return content


def stream_llm(
    system_prompt: str,
    user_prompt: str,
    *,
    model: Optional[str] = None,
    max_tokens: int = 2000,
    temperature: float = 0.2,
) -> Generator[str, None, None]:
    """Streaming generator yielding Groq response chunks (plain text)."""
    m = model or GROQ_MODEL
    try:
        stream = client.chat.completions.create(
            model=m,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
    except Exception as e:
        yield f"(llm_stream_error: {e})"
        return

    for chunk in stream:
        delta = getattr(chunk.choices[0], "delta", None)
        if delta and getattr(delta, "content", None):
            yield delta.content
