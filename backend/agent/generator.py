"""Grounded response generation using hosted/local LLMs with a safe fallback."""

from __future__ import annotations

import json
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import settings
from .models import EvidenceItem

SAFE_ESCALATION = "I’m not confident I have enough information from the available support history to give you a reliable answer. This should be reviewed by a support specialist."
HF_DEFAULT_PROVIDER = "featherless-ai"


def _ollama_model() -> str | None:
    if settings.ollama_model:
        return settings.ollama_model
    try:
        with urlopen(f"{settings.ollama_base_url.rstrip('/')}/api/tags", timeout=2) as response:
            models = json.loads(response.read().decode("utf-8")).get("models", [])
        return models[0].get("name") if models else None
    except Exception:
        return None


def llm_available() -> bool:
    return bool(settings.hf_token and settings.hf_model) or _ollama_model() is not None or bool(settings.llm_api_key and settings.llm_model)


def _clean(text: str) -> str:
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _retrieval_fallback(evidence: list[EvidenceItem]) -> str | None:
    if not evidence:
        return None
    answer = _clean(evidence[0].support_response)
    return answer or None


def _prompt(current_message: str, contextual_query: str, intent: str, evidence: list[EvidenceItem]) -> tuple[str, str]:
    evidence_text = "\n\n".join(
        f"Customer case: {item.customer_message}\nAppleSupport response: {_clean(item.support_response)}"
        for item in evidence[:6]
    )
    system = (
        "You are Pippa, an Apple customer-support assistant. Answer naturally and concisely. "
        "The historical cases below are the only factual support source. Do not invent Apple policies, "
        "settings, troubleshooting steps, guarantees, prices, dates, or account actions. "
        "Use the historical cases to answer the CURRENT user question, not merely repeat a historical reply. "
        "Never mention Twitter, tweet IDs, historical evidence, retrieval, or the model. "
        "Never address the customer with a social-media handle. "
        "If the cases do not contain enough information to answer reliably, output exactly INSUFFICIENT_EVIDENCE. "
        "Otherwise return only the customer-facing answer in 1-3 short sentences."
    )
    user = f"Current user message: {current_message}\nContextual problem: {contextual_query}\nIntent: {intent}\n\nHistorical support cases:\n{evidence_text}"
    return system, user


def _hf_model_id() -> str:
    model = settings.hf_model.strip()
    if ":" in model:
        return model
    return f"{model}:{HF_DEFAULT_PROVIDER}"


def _call_huggingface(current_message: str, contextual_query: str, intent: str, evidence: list[EvidenceItem]) -> str | None:
    if not settings.hf_token or not settings.hf_model:
        return None
    system, user = _prompt(current_message, contextual_query, intent, evidence)
    payload = json.dumps({
        "model": _hf_model_id(),
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0.1,
        "max_tokens": 180,
    }).encode("utf-8")
    request = Request(
        "https://router.huggingface.co/v1/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {settings.hf_token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=45) as response:
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:240]
        raise RuntimeError(f"Hugging Face HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise RuntimeError(f"Hugging Face connection failed: {error.reason}") from error
    answer = str(body.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()
    if not answer or "INSUFFICIENT_EVIDENCE" in answer.upper():
        return None
    return _clean(answer)


def _call_ollama(current_message: str, contextual_query: str, intent: str, evidence: list[EvidenceItem]) -> str | None:
    model = _ollama_model()
    if not model:
        return None
    system, user = _prompt(current_message, contextual_query, intent, evidence)
    payload = json.dumps({"model": model, "stream": False, "options": {"temperature": 0.1}, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}).encode("utf-8")
    request = Request(f"{settings.ollama_base_url.rstrip('/')}/api/chat", data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=45) as response:
        body = json.loads(response.read().decode("utf-8"))
    answer = str(body.get("message", {}).get("content", "")).strip()
    if not answer or "INSUFFICIENT_EVIDENCE" in answer.upper():
        return None
    return _clean(answer)


def _call_openai_compatible(current_message: str, contextual_query: str, intent: str, evidence: list[EvidenceItem]) -> str | None:
    if not settings.llm_api_key or not settings.llm_model:
        return None
    system, user = _prompt(current_message, contextual_query, intent, evidence)
    payload = json.dumps({"model": settings.llm_model, "temperature": 0.1, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}).encode("utf-8")
    request = Request("https://api.openai.com/v1/chat/completions", data=payload, headers={"Authorization": f"Bearer {settings.llm_api_key}", "Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=30) as response:
        body = json.loads(response.read().decode("utf-8"))
    answer = str(body["choices"][0]["message"]["content"]).strip()
    if not answer or "INSUFFICIENT_EVIDENCE" in answer.upper():
        return None
    return _clean(answer)


def generate_response(current_message: str, contextual_query: str, intent: str, confidence: float, evidence: list[EvidenceItem]) -> dict:
    if not evidence:
        return {"reply": SAFE_ESCALATION, "safe_reply": SAFE_ESCALATION, "evidence": [], "grounded": False, "confidence": 0.0, "generation_method": "safe_escalation", "error": "No sufficiently relevant historical evidence was found."}
    errors: list[str] = []
    configured = bool(settings.hf_token or settings.ollama_model or settings.llm_api_key)
    for method, call in (("huggingface_qwen", _call_huggingface), ("ollama_grounded", _call_ollama), ("llm", _call_openai_compatible)):
        try:
            answer = call(current_message, contextual_query, intent, evidence)
            if answer:
                return {"reply": answer, "safe_reply": SAFE_ESCALATION, "evidence": evidence, "grounded": True, "confidence": min(1.0, max(0.0, confidence)), "generation_method": method, "error": None}
        except Exception as error:
            errors.append(f"{method} unavailable: {type(error).__name__}: {error}")
    if configured:
        return {"reply": SAFE_ESCALATION, "safe_reply": SAFE_ESCALATION, "evidence": evidence, "grounded": False, "confidence": 0.0, "generation_method": "safe_escalation", "error": "; ".join(errors) or "Configured LLM did not return a usable answer."}
    answer = _retrieval_fallback(evidence)
    if answer:
        return {"reply": answer, "safe_reply": SAFE_ESCALATION, "evidence": evidence, "grounded": True, "confidence": min(1.0, max(0.0, confidence)), "generation_method": "retrieval_fallback", "error": "No LLM configured; used strongest grounded historical response."}
    return {"reply": SAFE_ESCALATION, "safe_reply": SAFE_ESCALATION, "evidence": evidence, "grounded": False, "confidence": 0.0, "generation_method": "safe_escalation", "error": "No usable grounded response exists."}
