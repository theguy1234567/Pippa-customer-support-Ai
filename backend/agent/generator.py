"""Grounded response generation with a deterministic no-LLM fallback."""

from __future__ import annotations

import json
from urllib.request import Request, urlopen

from .config import settings
from .models import EvidenceItem

SAFE_ESCALATION = "I’m not confident I have enough information from the available support history to give you a reliable answer. This should be reviewed by a support specialist."


def generate_response(message: str, intent: str, confidence: float, evidence: list[EvidenceItem], escalated: bool = False) -> dict:
    if not evidence or not any(item.support_response.strip() for item in evidence):
        return {"reply": SAFE_ESCALATION, "evidence": evidence, "grounded": False, "confidence": 0.0, "generation_method": "retrieval_fallback", "error": "No historical support response was available."}
    strongest = max(evidence, key=lambda item: item.similarity)
    if strongest.similarity <= 0.0:
        return {"reply": SAFE_ESCALATION, "evidence": evidence, "grounded": False, "confidence": 0.0, "generation_method": "retrieval_fallback", "error": "Historical similarity was insufficient."}
    if settings.llm_api_key and settings.llm_model:
        try:
            evidence_text = "\n".join(f"[{item.tweet_id}] {item.support_response}" for item in evidence if item.support_response.strip())
            payload = json.dumps({"model": settings.llm_model, "messages": [
                {"role": "system", "content": "Use only the current message, predicted intent, and retrieved historical support responses. Do not invent policies, account actions, prices, dates, or guarantees. If evidence is insufficient, recommend human review."},
                {"role": "user", "content": f"Current message: {message}\nPredicted intent: {intent}\nHistorical evidence:\n{evidence_text}"},
            ], "temperature": 0}).encode("utf-8")
            request = Request("https://api.openai.com/v1/chat/completions", data=payload, headers={"Authorization": f"Bearer {settings.llm_api_key}", "Content-Type": "application/json"}, method="POST")
            with urlopen(request, timeout=20) as response:
                reply = json.loads(response.read().decode("utf-8"))["choices"][0]["message"]["content"].strip()
            return {"reply": reply, "evidence": evidence, "grounded": True, "confidence": min(1.0, max(0.0, strongest.similarity)), "generation_method": "llm", "error": None}
        except Exception as error:
            fallback_error = f"LLM generation failed; deterministic fallback used: {type(error).__name__}"
    else:
        fallback_error = None
    reply = strongest.support_response.strip()
    if escalated:
        reply = f"A support specialist should review this case. A relevant historical support response was: {reply}"
    return {"reply": reply, "evidence": evidence, "grounded": True, "confidence": min(1.0, max(0.0, strongest.similarity)), "generation_method": "retrieval_fallback", "error": fallback_error}
