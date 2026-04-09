from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from groq import Groq

from utils.config import settings


class GroqLLM:
    def __init__(self):
        self.client: Optional[Groq] = None
        if settings.groq_api_key:
            self.client = Groq(api_key=settings.groq_api_key)

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def chat_json(self, system_prompt: str, user_payload: Dict[str, Any], temperature: float = 0.3) -> str:
        if self.client is None:
            raise ValueError("GROQ_API_KEY not configured.")
        response = self.client.chat.completions.create(
            model=settings.groq_model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
        )
        return (response.choices[0].message.content or "").strip()

    def chat_text(self, system_prompt: str, user_text: str, temperature: float = 0.3) -> str:
        if self.client is None:
            raise ValueError("GROQ_API_KEY not configured.")
        response = self.client.chat.completions.create(
            model=settings.groq_model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
        )
        return (response.choices[0].message.content or "").strip()

    def health_check(self) -> Dict[str, Any]:
        """
        Lightweight connectivity test with latency.
        """
        if self.client is None:
            return {"ok": False, "error": "GROQ_API_KEY not configured.", "latency_ms": None}
        start = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=settings.groq_model,
                temperature=0,
                max_tokens=16,
                messages=[
                    {"role": "system", "content": "You are a health check assistant."},
                    {"role": "user", "content": "Respond with: ok"},
                ],
            )
            latency_ms = int((time.perf_counter() - start) * 1000)
            text = (response.choices[0].message.content or "").strip()
            return {"ok": True, "latency_ms": latency_ms, "response": text}
        except Exception as exc:
            latency_ms = int((time.perf_counter() - start) * 1000)
            return {"ok": False, "latency_ms": latency_ms, "error": str(exc)}
