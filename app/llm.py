"""Groq access through the OpenAI-compatible surface."""

from functools import lru_cache

from openai import OpenAI

from app.config import get_settings


@lru_cache
def get_client() -> OpenAI:
    settings = get_settings()
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set")
    return OpenAI(api_key=settings.groq_api_key, base_url=settings.groq_base_url)
