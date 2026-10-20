"""Talk to a free AI API (Gemini or Groq) with one simple function: ask(prompt).

Setup (once, in the same terminal where you run the app):
    Windows (cmd):   set GEMINI_API_KEY=your_key_here
    Mac/Linux:       export GEMINI_API_KEY=your_key_here
or use GROQ_API_KEY for Groq. NEVER write the key inside a file you upload to GitHub.

Model names change over time, so by default we ask the API which models exist and
pick a suitable one. To force a model, set GEMINI_MODEL or GROQ_MODEL.
"""
import json
import os
import re
import time

import requests

TIMEOUT = 60
TEMPORARY = (500, 502, 503, 504)   # server-side trouble, usually gone in seconds
MAX_ATTEMPTS = 3
RETRY_WAIT = 3                      # seconds; grows each attempt (3s, then 6s)
_sleep = time.sleep
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
GROQ_BASE = "https://api.groq.com/openai/v1"


class LLMUnavailable(Exception):
    """No key, no internet, quota used up, or the API said no. The app should degrade
    gracefully (show the rule-based report) instead of crashing.
    retryable=True means it is worth trying again in a few seconds (server busy)."""

    def __init__(self, message, retryable=False):
        super().__init__(message)
        self.retryable = retryable


def redact(text: str) -> str:
    """Remove emails, phone numbers and profile links before text leaves your computer."""
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "[email]", text)
    # Phone numbers: any run of digits (with spaces, dashes, brackets) that has 10+ digits.
    # Year ranges like 2022-2026 and scores like 7.48 have fewer digits, so they stay.
    def _mask(m):
        digits = sum(c.isdigit() for c in m.group())
        only_years = re.fullmatch(r"(?:(?:19|20)\d{2}[\s-]*)+", m.group().strip())
        return "[phone]" if digits >= 10 and not only_years else m.group()
    text = re.sub(r"\+?\d[\d\s().-]{8,}\d", _mask, text)
    text = re.sub(r"(?:https?://)?(?:www\.)?(?:linkedin\.com|github\.com)/\S+", "[profile link]", text)
    return text


def _get_key(name: str):
    key = os.environ.get(name)
    if key:
        return key.strip()
    try:  # Streamlit Cloud / Hugging Face store secrets here (used on Day 7)
        import streamlit as st
        return str(st.secrets[name]).strip()
    except Exception:
        return None


def provider() -> str:
    """Which provider has a key set. Returns 'gemini', 'groq' or ''."""
    if _get_key("GEMINI_API_KEY"):
        return "gemini"
    if _get_key("GROQ_API_KEY"):
        return "groq"
    return ""


def _api_reason(response) -> str:
    """The API usually says exactly what is wrong. Show it (with any key removed)."""
    try:
        message = str(response.json()["error"]["message"])
    except Exception:
        return ""
    message = re.sub(r"AIza[0-9A-Za-z_\-]{20,}|gsk_[0-9A-Za-z]{20,}", "[key]", message)
    return " Reason from the API: " + message[:200]


def _raise_for(response):
    if response.status_code == 429:
        raise LLMUnavailable("Free quota reached. Wait a minute and try again." + _api_reason(response))
    if response.status_code in (400, 401, 403):
        raise LLMUnavailable(f"The API rejected the request ({response.status_code})." + _api_reason(response))
    if response.status_code in TEMPORARY:
        raise LLMUnavailable(f"The AI service is busy right now ({response.status_code}). "
                             "This is usually temporary: wait a minute and try again." + _api_reason(response),
                             retryable=True)
    if not response.ok:
        raise LLMUnavailable(f"The API returned an error ({response.status_code})." + _api_reason(response))


_working_gemini_model = None   # remembered after the first success, so later calls are fast


def _version_of(name: str) -> float:
    """gemini-3.8-flash -> 3.8. The '-latest' alias counts as newest of all."""
    if "latest" in name:
        return 999.0
    match = re.search(r"(\d+(?:\.\d+)?)", name)
    return float(match.group(1)) if match else 0.0


def _gemini_candidates(key: str) -> list:
    """Usable Gemini models, newest first. The list from Google still contains retired
    models, so we try them in order and move on if one is refused."""
    response = requests.get(f"{GEMINI_BASE}/models", params={"key": key, "pageSize": 200}, timeout=TIMEOUT)
    _raise_for(response)
    names = [m["name"].split("/", 1)[1] for m in response.json().get("models", [])
             if "generateContent" in m.get("supportedGenerationMethods", [])]
    # Skip image, audio, tts, embedding and experimental models.
    skip = ("image", "audio", "tts", "live", "embedding", "vision", "thinking", "exp", "preview", "lite")
    usable = [n for n in names if not any(x in n for x in skip)]
    flash = [n for n in usable if "flash" in n]
    pool = flash or [n for n in usable if "gemini" in n] or usable
    if not pool:
        raise LLMUnavailable("No usable Gemini model found for this key.")
    return sorted(pool, key=lambda n: (-_version_of(n), len(n), n))


def _model_refused(response) -> bool:
    """True when the model itself is the problem (retired, not found), so another may work."""
    if response.status_code == 404:
        return True
    reason = _api_reason(response).lower()
    return response.status_code == 400 and ("no longer available" in reason or "not found" in reason)


def _ask_gemini(prompt: str, system: str, want_json: bool) -> str:
    global _working_gemini_model
    key = _get_key("GEMINI_API_KEY")
    forced = os.environ.get("GEMINI_MODEL")
    candidates = [forced] if forced else ([_working_gemini_model] if _working_gemini_model else _gemini_candidates(key)[:5])
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "systemInstruction": {"parts": [{"text": system}]},
        "generationConfig": {"temperature": 0.3},
    }
    if want_json:
        body["generationConfig"]["responseMimeType"] = "application/json"

    last_problem = None
    for model in candidates:
        response = requests.post(f"{GEMINI_BASE}/models/{model}:generateContent",
                                 params={"key": key}, json=body, timeout=TIMEOUT)
        if (_model_refused(response) or response.status_code in TEMPORARY) and not forced:
            last_problem, _working_gemini_model = response, None
            continue
        _raise_for(response)
        _working_gemini_model = model
        try:
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            raise LLMUnavailable("The AI returned an empty answer (it may have been blocked). Try again.")
    _raise_for(last_problem)
    raise LLMUnavailable("None of the available Gemini models accepted the request.")


def _groq_model(key: str) -> str:
    chosen = os.environ.get("GROQ_MODEL")
    if chosen:
        return chosen
    response = requests.get(f"{GROQ_BASE}/models", headers={"Authorization": f"Bearer {key}"}, timeout=TIMEOUT)
    _raise_for(response)
    names = [m["id"] for m in response.json().get("data", [])]
    skip = ("whisper", "tts", "guard", "embed", "vision", "prompt")
    chat = [n for n in names if not any(s in n for s in skip)]
    # Prefer a large general-purpose model, then any chat model.
    ranked = sorted(chat, key=lambda n: (("versatile" not in n and "70b" not in n), len(n), n))
    if not ranked:
        raise LLMUnavailable("No usable Groq model found for this key.")
    return ranked[0]


def _ask_groq(prompt: str, system: str, want_json: bool) -> str:
    key = _get_key("GROQ_API_KEY")
    body = {
        "model": _groq_model(key),
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "temperature": 0.3,
    }
    if want_json:
        body["response_format"] = {"type": "json_object"}
    response = requests.post(f"{GROQ_BASE}/chat/completions", headers={"Authorization": f"Bearer {key}"},
                             json=body, timeout=TIMEOUT)
    _raise_for(response)
    try:
        return response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        raise LLMUnavailable("The AI returned an empty answer. Try again.")


def ask(prompt: str, system: str = "You are a helpful assistant.", want_json: bool = False) -> str:
    """Send a prompt to whichever provider has a key. Raises LLMUnavailable on any problem.
    If the service is only busy (503 and friends), wait a few seconds and try again."""
    which = provider()
    if not which:
        raise LLMUnavailable("No API key found. Set GEMINI_API_KEY or GROQ_API_KEY to turn on AI suggestions.")
    send = _ask_gemini if which == "gemini" else _ask_groq
    for attempt in range(MAX_ATTEMPTS):
        try:
            return send(prompt, system, want_json)
        except LLMUnavailable as error:
            if error.retryable and attempt < MAX_ATTEMPTS - 1:
                _sleep(RETRY_WAIT * (attempt + 1))
                continue
            raise
        except requests.RequestException as error:
            raise LLMUnavailable(f"Could not reach the AI service ({type(error).__name__}). Check your internet.")


def parse_json(text: str) -> dict:
    """AI models sometimes wrap JSON in ```json fences or add a sentence. Handle that."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise
