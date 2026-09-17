"""Explain accuracy limitations from measured class distributions only."""

from __future__ import annotations

from collections import Counter


def analyze_accuracy_limitations(labels: list[str]) -> dict:
    if not labels:
        return {"status": "PENDING_GOLD_LABELS", "class_distribution": {}, "message": "No measured labels are available; accuracy analysis is pending."}
    counts = Counter(labels)
    majority_share = max(counts.values()) / len(labels)
    return {
        "status": "MEASURED",
        "class_distribution": dict(counts),
        "majority_class_share": majority_share,
        "message": "Compare accuracy with macro F1 and per-class F1 because the measured class distribution may be imbalanced.",
    }
