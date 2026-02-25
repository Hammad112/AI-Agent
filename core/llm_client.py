"""
core/llm_client.py
------------------
Unified LLM caller with Gemini as primary and OpenAI as fallback.
Uses the new google-genai SDK (google.genai).

Rate-limit aware: uses exponential backoff and a global retry loop
so the free Gemini tier (15 req/min) auto-recovers without crashing.
"""

import os
import time

# Gemini models to try in order
GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-2.0-flash-001",
]


def llm_call(prompt: str, temperature: float = 0.3, max_tokens: int = 4096) -> str:
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")

    if gemini_key and gemini_key not in ("your_gemini_api_key_here", ""):
        for cycle in range(3):
            result, exhausted = _try_gemini_cycle(prompt, temperature, max_tokens)
            if result is not None:
                return result
            if not exhausted:
                break
            wait_secs = 60
            print(f"[LLM] Gemini quota exhausted. Waiting {wait_secs}s for rate limit reset… (attempt {cycle+1}/3)")
            time.sleep(wait_secs)
        print("[LLM] Gemini unavailable after retries, falling back to OpenAI…")

    if openai_key and openai_key not in ("your_openai_api_key_here", ""):
        return _openai_call(prompt, temperature, max_tokens)

    raise RuntimeError(
        "No valid LLM API key found. "
        "Open .env and set GEMINI_API_KEY or OPENAI_API_KEY."
    )


def _try_gemini_cycle(
    prompt: str, temperature: float, max_tokens: int
) -> tuple[str | None, bool]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    any_rate_limited = False

    for model in GEMINI_MODELS:
        wait = 5
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    ),
                )
                return response.text, False
            except Exception as e:
                err = str(e)
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    any_rate_limited = True
                    if attempt < 2:
                        print(f"[LLM] {model} rate-limited, waiting {wait}s…")
                        time.sleep(wait)
                        wait = min(wait * 2, 30)
                    else:
                        print(f"[LLM] {model} still rate-limited, trying next model…")
                elif "404" in err or "NOT_FOUND" in err:
                    break
                else:
                    print(f"[LLM] {model} error: {type(e).__name__}: {err[:80]}")
                    return None, False

    return None, any_rate_limited


def _openai_call(prompt: str, temperature: float, max_tokens: int) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content
