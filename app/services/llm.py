from __future__ import annotations

from functools import lru_cache
from typing import Any

from app.config import (
    ENABLE_LLM,
    LLM_API_KEY,
    LLM_APP_NAME,
    LLM_APP_URL,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_PROVIDER,
)

_LAST_LLM_ERROR: str | None = None


def _as_text(content: Any) -> str | None:
    if content is None:
        return None
    if isinstance(content, str):
        text = content.strip()
        return text or None
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        joined = "\n".join(part for part in parts if part.strip()).strip()
        return joined or None
    text = str(content).strip()
    return text or None


def _to_langchain_messages(messages: list[dict[str, str]]):
    return [
        ((message.get("role") or "user").lower(), message.get("content", ""))
        for message in messages
    ]


@lru_cache(maxsize=1)
def _get_langchain_chat_model():
    if not ENABLE_LLM or not LLM_API_KEY:
        return None

    try:
        from langchain_openai import ChatOpenAI
    except Exception:
        return None

    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=0.0,
    )


@lru_cache(maxsize=1)
def _get_openai_client():
    if not ENABLE_LLM or not LLM_API_KEY:
        return None

    try:
        from openai import OpenAI
    except Exception:
        return None

    return OpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        default_headers={
            "HTTP-Referer": LLM_APP_URL,
            "X-Title": LLM_APP_NAME,
        },
    )


def llm_enabled() -> bool:
    if not ENABLE_LLM or not LLM_API_KEY:
        return False
    return _get_langchain_chat_model() is not None or _get_openai_client() is not None


def llm_last_error() -> str | None:
    return _LAST_LLM_ERROR


def chat_completion(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.2,
    max_tokens: int = 600,
) -> str | None:
    global _LAST_LLM_ERROR
    _LAST_LLM_ERROR = None

    if not ENABLE_LLM or not LLM_API_KEY:
        _LAST_LLM_ERROR = "LLM is disabled or LLM_API_KEY is missing."
        return None

    provider = LLM_PROVIDER
    if provider in {"langchain", "langchain_openai", "lc"}:
        model = _get_langchain_chat_model()
        if model is None:
            provider = "openai"
        else:
            try:
                runnable = model.bind(max_tokens=max_tokens, temperature=temperature)
                response = runnable.invoke(_to_langchain_messages(messages))
            except Exception as exc:
                _LAST_LLM_ERROR = f"LangChain request failed: {type(exc).__name__}: {exc}"
                return None

            content = _as_text(getattr(response, "content", response))
            return content

    client = _get_openai_client()
    if client is None:
        _LAST_LLM_ERROR = "OpenAI client could not be created."
        return None

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        _LAST_LLM_ERROR = f"OpenAI request failed: {type(exc).__name__}: {exc}"
        return None

    choice = response.choices[0] if response.choices else None
    content = choice.message.content if choice and choice.message else None
    return content.strip() if isinstance(content, str) else None
