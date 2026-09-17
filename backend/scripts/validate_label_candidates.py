"""Validate label-candidate, review-sample, and gold-review artifacts."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
CASES_PATH = PROCESSED / "apple_cases.jsonl"
TAXONOMY_PATH = PROCESSED / "final_taxonomy.json"
CANDIDATES_PATH = PROCESSED / "label_candidates.jsonl"
GOLD_PATH = PROCESSED / "gold_review_candidates.json"
REVIEW_SAMPLE_PATH = PROCESSED / "label_review_sample.json"
REPORT_PATH = PROCESSED / "label_candidate_report.json"
RAW_PATH = ROOT / "data" / "raw" / "twcs.csv"
VALID_STATUSES = {"HIGH_CONFIDENCE", "AMBIGUOUS", "UNLABELED"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stat(path: Path) -> tuple[int, int]:
    value = path.stat()
    return value.st_size, value.st_mtime_ns


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _load_source(path: Path) -> tuple[dict[str, dict[str, str]], list[dict[str, str]]]:
    messages: dict[str, dict[str, str]] = {}
    ordered: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            case = json.loads(line)
            for message in case.get("conversation", []):
                if message.get("role") != "customer":
                    continue
                record = {
                    "tweet_id": str(message["tweet_id"]),
                    "text": message["text"],
                    "case_id": str(case["case_id"]),
                }
                if record["tweet_id"] in messages:
                    raise ValueError(f"Duplicate source tweet ID: {record['tweet_id']}")
                messages[record["tweet_id"]] = record
                ordered.append(record)
    return messages, ordered


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"JSONL record is not an object at {path}:{line_number}")
            records.append(record)
    return records


def validate() -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    taxonomy = _load_json(TAXONOMY_PATH)
    source_by_id, source_records = _load_source(CASES_PATH)
    taxonomy_names = [item["intent_name"] for item in taxonomy["intents"]]
    candidates = _load_jsonl(CANDIDATES_PATH)
    gold = _load_json(GOLD_PATH)
    review = _load_json(REVIEW_SAMPLE_PATH)
    report = _load_json(REPORT_PATH)
    candidate_ids = [record.get("tweet_id") for record in candidates]
    gold_records = gold.get("records", [])
    gold_ids = [record.get("tweet_id") for record in gold_records]
    review_records = review.get("records", [])

    if len(candidate_ids) != len(set(candidate_ids)):
        errors.append("label candidate tweet IDs are not unique")
    if len(gold_ids) != len(set(gold_ids)):
        errors.append("gold-review tweet IDs are not unique")
    if set(candidate_ids) & set(gold_ids):
        errors.append("gold-review candidates appear in label_candidates.jsonl")
    if len(gold_records) != 200:
        errors.append(f"expected 200 gold-review candidates, found {len(gold_records)}")
    if len(review_records) < 900 or len(review_records) > 1000:
        errors.append(f"review sample size should be approximately 1000, found {len(review_records)}")

    seen_texts: set[str] = set()
    gold_texts = {record.get("text") for record in gold_records}
    for record in candidates:
        tweet_id = record.get("tweet_id")
        source = source_by_id.get(tweet_id)
        if source is None:
            errors.append(f"candidate tweet ID is absent from source: {tweet_id}")
            continue
        if record.get("text") != source["text"]:
            errors.append(f"candidate text changed for tweet ID: {tweet_id}")
        if record.get("text") in seen_texts:
            errors.append(f"duplicate candidate message text: {tweet_id}")
        if record.get("text") in gold_texts:
            errors.append(f"gold-review message text appears in candidates: {tweet_id}")
        seen_texts.add(record.get("text"))
        status = record.get("candidate_status")
        if status not in VALID_STATUSES:
            errors.append(f"invalid candidate status for {tweet_id}: {status}")
        candidate_intent = record.get("candidate_intent")
        if candidate_intent is not None and candidate_intent not in taxonomy_names:
            errors.append(f"invalid candidate intent for {tweet_id}: {candidate_intent}")
        if status in {"AMBIGUOUS", "UNLABELED"} and candidate_intent is not None:
            errors.append(f"uncertain candidate has a forced intent for {tweet_id}")
        if status == "HIGH_CONFIDENCE" and candidate_intent is None:
            errors.append(f"high-confidence candidate has no intent for {tweet_id}")
        if not isinstance(record.get("reason"), str) or not record["reason"].strip():
            errors.append(f"missing candidate reason for {tweet_id}")
        if not isinstance(record.get("possible_intents"), list) or any(intent not in taxonomy_names for intent in record["possible_intents"]):
            errors.append(f"invalid possible_intents for {tweet_id}")

    for record in gold_records:
        tweet_id = record.get("tweet_id")
        source = source_by_id.get(tweet_id)
        if source is None:
            errors.append(f"gold tweet ID is absent from source: {tweet_id}")
            continue
        if record.get("text") != source["text"] or record.get("case_id") != source["case_id"]:
            errors.append(f"gold source data changed for tweet ID: {tweet_id}")
        if record.get("human_label") is not None or record.get("notes") is not None:
            errors.append(f"gold review fields are not blank for tweet ID: {tweet_id}")

    review_ids = [record.get("tweet_id") for record in review_records]
    review_texts = [record.get("text") for record in review_records]
    if len(review_ids) != len(set(review_ids)):
        errors.append("review sample tweet IDs are not unique")
    if len(review_texts) != len(set(review_texts)):
        errors.append("review sample message texts are not unique")
    if set(review_ids) & set(gold_ids):
        errors.append("gold-review candidates appear in the high-confidence review sample")
    if set(review_texts) & gold_texts:
        errors.append("gold-review message text appears in the high-confidence review sample")
    for record in review_records:
        source = source_by_id.get(record.get("tweet_id"))
        if source is None or record.get("text") != source["text"]:
            errors.append(f"review sample text is absent or changed: {record.get('tweet_id')}")
        if record.get("candidate_status") != "HIGH_CONFIDENCE":
            errors.append(f"review sample contains non-high-confidence record: {record.get('tweet_id')}")
        if record.get("candidate_intent") not in taxonomy_names:
            errors.append(f"review sample has invalid intent: {record.get('tweet_id')}")

    expected_status_counts = report.get("assessment_counts", {})
    if set(expected_status_counts) != VALID_STATUSES:
        errors.append("report does not contain all assessment status counts")
    if sum(expected_status_counts.values()) != len(source_records):
        errors.append("assessment status counts do not sum to the source dataset size")
    if report.get("dataset_size") != len(source_records):
        errors.append("report dataset_size does not match source")
    if report.get("gold_review_candidates_count") != len(gold_records):
        errors.append("report gold count does not match gold artifact")
    if report.get("review_sample_size") != len(review_records):
        errors.append("report review sample size does not match review artifact")
    if report.get("candidate_file_records") != len(candidates):
        errors.append("report candidate file count does not match JSONL")
    for item in report.get("quality_by_intent", []):
        if item.get("intent") not in taxonomy_names:
            errors.append(f"quality report contains unknown intent: {item.get('intent')}")
        if not isinstance(item.get("high_confidence_count"), int) or item["high_confidence_count"] < 0:
            errors.append(f"invalid quality count: {item.get('intent')}")
        for example in item.get("candidate_examples", []):
            if example not in source_by_id.values() and example not in {value["text"] for value in source_by_id.values()}:
                errors.append(f"quality example absent from source: {item.get('intent')}")
    if report.get("human_labels_created") is not False or report.get("classifier_created") is not False:
        errors.append("report incorrectly claims human labels or classifier creation")

    summary = {
        "source_messages": len(source_records),
        "candidate_file_records": len(candidates),
        "status_counts": dict(Counter(record.get("candidate_status") for record in candidates)),
        "gold_review_candidates": len(gold_records),
        "review_sample": len(review_records),
    }
    return errors, summary


def main() -> int:
    required = (CASES_PATH, TAXONOMY_PATH, CANDIDATES_PATH, GOLD_PATH, REVIEW_SAMPLE_PATH, REPORT_PATH, RAW_PATH)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("Missing required files:\n- " + "\n- ".join(missing), file=sys.stderr)
        return 1
    protected = {path: (_sha256(path), _stat(path)) for path in (CASES_PATH, TAXONOMY_PATH, RAW_PATH)}
    try:
        errors, summary = validate()
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"Label candidate validation failed: {error}", file=sys.stderr)
        return 1
    for path, (digest, stat) in protected.items():
        if _sha256(path) != digest or _stat(path) != stat:
            errors.append(f"protected source changed: {path}")

    print("Label Candidate Validation")
    print("--------------------------")
    print(f"Source customer messages: {summary['source_messages']}")
    print(f"Candidate file records: {summary['candidate_file_records']}")
    print(f"HIGH_CONFIDENCE: {summary['status_counts'].get('HIGH_CONFIDENCE', 0)}")
    print(f"AMBIGUOUS: {summary['status_counts'].get('AMBIGUOUS', 0)}")
    print(f"UNLABELED: {summary['status_counts'].get('UNLABELED', 0)}")
    print(f"Gold review candidates: {summary['gold_review_candidates']}")
    print(f"Review sample: {summary['review_sample']}")
    if errors:
        print("VALIDATION: FAIL", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("Source integrity: PASS")
    print("Candidate text and ID integrity: PASS")
    print("Gold exclusion: PASS")
    print("VALIDATION: PASS")
    print("Training-label candidates prepared and validated. Classifier has NOT been implemented.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
