from __future__ import annotations
from typing import Any, Dict
from openai import OpenAI

class LLMClient:
    """Simple wrapper around OpenAI client for OpenAI-compatible endpoints."""
    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        if base_url:
            self._client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self._client = OpenAI(api_key=api_key)

    def chat(self, **kwargs: Any):
        return self._client.chat.completions.create(**kwargs)
