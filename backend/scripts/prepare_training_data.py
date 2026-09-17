"""Convert actual human labels into training data; blank review labels are rejected."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TAXONOMY = PROCESSED / "final_taxonomy.json"
REVIEW = PROCESSED / "label_review_sample.json"
OUTPUT = PROCESSED / "training_data.jsonl"
REPORT = PROCESSED / "training_data_report.json"
GOLD = PROCESSED / "gold_review_candidates.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> int:
    taxonomy = json.loads(TAXONOMY.read_text(encoding="utf-8"))
    names = {item["intent_name"] for item in taxonomy["intents"]}
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    records = review.get("records", [])
    training = []
    invalid = 0
    excluded = 0
    for record in records:
        label = record.get("human_label")
        if not label:
            excluded += 1
            continue
        if label not in names:
            invalid += 1
            continue
        training.append({"tweet_id": record["tweet_id"], "text": record["text"], "intent": label, "source": REVIEW.name})
    ids = [record["tweet_id"] for record in training]
    duplicate_count = len(ids) - len(set(ids))
    if duplicate_count:
        raise ValueError("Conflicting or duplicate human-labeled tweet IDs detected")
    with OUTPUT.open("w", encoding="utf-8") as handle:
        for record in training:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    report = {
        "source": REVIEW.name,
        "labeled_records": len(training),
        "counts_by_intent": dict(Counter(record["intent"] for record in training)),
        "duplicate_count": duplicate_count,
        "invalid_label_count": invalid,
        "excluded_blank_labels": excluded,
        "gold_contamination_check": not any(record["tweet_id"] in {item["tweet_id"] for item in json.loads(GOLD.read_text(encoding='utf-8')).get('records', [])} for record in training),
        "source_sha256": digest(REVIEW),
        "status": "READY" if training else "UNAVAILABLE_HUMAN_LABELS",
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
