"""LLM judge contract; intentionally returns unavailable until an LLM provider is configured."""


def judge_response(message: str, response: str, evidence: list[dict]) -> dict:
    return {"available": False, "reason": "LLM judge is not configured; no scores were fabricated.", "scores": None}
