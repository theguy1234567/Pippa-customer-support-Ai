from __future__ import annotations


def analyze_failures(predictions: list[dict]) -> dict:
    failures = [item for item in predictions if item.get("expected") != item.get("predicted")]
    return {"failure_count": len(failures), "categories": {}, "records": failures, "status": "MEASURED" if predictions else "PENDING_GOLD_LABELS"}
