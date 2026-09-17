from __future__ import annotations


def evaluate_responses(*args, **kwargs) -> dict:
    return {"status": "PENDING_HUMAN_OR_LLM_JUDGMENTS", "message": "Response quality metrics are unavailable without judgments."}
