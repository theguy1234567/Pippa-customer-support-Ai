"""Agreement utilities for independently human-rated response subsets."""

from __future__ import annotations

from sklearn.metrics import cohen_kappa_score


def calculate_agreement(human_a: list[str], human_b: list[str]) -> dict:
    if len(human_a) != len(human_b):
        raise ValueError("agreement raters must have equal-length ratings")
    if not human_a:
        return {"status": "PENDING_HUMAN_RATINGS", "agreement_percentage": None, "cohen_kappa": None}
    agreement = sum(left == right for left, right in zip(human_a, human_b)) / len(human_a)
    return {
        "status": "MEASURED",
        "agreement_percentage": agreement * 100,
        "cohen_kappa": float(cohen_kappa_score(human_a, human_b)),
    }
