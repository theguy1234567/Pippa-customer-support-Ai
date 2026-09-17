from __future__ import annotations


def evaluate_retrieval(*args, **kwargs) -> dict:
    return {"status": "PENDING_HUMAN_LABELS", "message": "Retrieval correctness cannot be measured without labeled relevance judgments."}
