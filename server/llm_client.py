"""
Thin LLM client wrapper. Deliberately provider-agnostic and *optional*:
if no API key is configured, every caller in this project has a
deterministic heuristic fallback, so the whole application runs and
demos correctly with zero external credentials (see README - Demo Mode).

Primary provider: Google Gemini (REST, no SDK dependency needed).
Set GEMINI_API_KEY to enable real LLM calls.
"""
from __future__ import annotations
import os
import json
import time
import requests

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
)

LLM_ENABLED = bool(GEMINI_API_KEY)


class LLMFailure(Exception):
    """Raised on network/parsing failure so callers can fall back gracefully."""


def call_llm_json(system_prompt: str, user_prompt: str, timeout: float = 20.0, simulate_failure: bool = False) -> dict:
    """
    Calls Gemini and asks for strict JSON back. Returns a parsed dict.
    Raises LLMFailure on any problem (timeout, bad JSON, network error) so
    that callers can apply their documented fallback behavior instead of
    silently guessing - this is the project's literal failure-recovery path.
    """
    if simulate_failure:
        # used by the "Break It" debug endpoint to demonstrate live recovery
        raise LLMFailure("Simulated LLM timeout (injected via /debug/inject_failure)")

    if not LLM_ENABLED:
        raise LLMFailure("No GEMINI_API_KEY configured - running in heuristic/demo mode")

    payload = {
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "systemInstruction": {"parts": [{"text": system_prompt + "\nRespond with STRICT JSON only, no markdown fences, no commentary."}]},
        "generationConfig": {"temperature": 0.4, "responseMimeType": "application/json"},
    }
    try:
        resp = requests.post(
            GEMINI_URL,
            params={"key": GEMINI_API_KEY},
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(cleaned)
    except requests.exceptions.RequestException as e:
        raise LLMFailure(f"LLM network error: {e}")
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        raise LLMFailure(f"LLM returned unparseable output: {e}")
