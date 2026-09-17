"""Prepare transparent label candidates and an unlabeled gold-review reserve."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
CASES_PATH = PROCESSED / "apple_cases.jsonl"
TAXONOMY_PATH = PROCESSED / "final_taxonomy.json"
MANUAL_REVIEW_PATH = PROCESSED / "intent_manual_review.json"
CANDIDATES_PATH = PROCESSED / "label_candidates.jsonl"
GOLD_PATH = PROCESSED / "gold_review_candidates.json"
REVIEW_SAMPLE_PATH = PROCESSED / "label_review_sample.json"
REPORT_PATH = PROCESSED / "label_candidate_report.json"
RAW_PATH = ROOT / "data" / "raw" / "twcs.csv"

VALID_STATUSES = {"HIGH_CONFIDENCE", "AMBIGUOUS", "UNLABELED"}
TAXONOMY_NAMES = [
    "ios_update_issue",
    "keyboard_text_input_issue",
    "app_store_app_issue",
    "device_hardware_issue",
    "device_functionality_issue",
    "apple_service_issue",
    "support_dm_request",
    "general_support_request",
    "non_actionable_other",
]

KEYBOARD = re.compile(
    r"\b(?:keyboard|typing|typed|type|autocorrect|spell(?:ing)?|letter|question\s+mark|emoji|text\s+input|input)\b|i[\ufe0f\u200d]*[’']?s",
    re.IGNORECASE,
)
KEYBOARD_CONCRETE = re.compile(
    r"\b(?:keyboard|typing|typed|autocorrect|spelling|question\s+marks?|symbol|letter)\b|(?:changing|appearing|showing|turning|replacing).{0,40}\bi[\ufe0f\u200d]*\b",
    re.IGNORECASE,
)
APP = re.compile(r"\b(?:app\s*store|appstore|apps?|application)\b", re.IGNORECASE)
APP_ACTION = re.compile(
    r"\b(?:downloa\w*|install\w*|open\w*|launch\w*|crash\w*|down|won['’]?t|can['’]?t|cannot|not\s+work|doesn['’]?t\s+work|update\w*|fail\w*|bug\w*)\b",
    re.IGNORECASE,
)
HARDWARE = re.compile(
    r"\b(?:battery|batteries|charg(?:e|ing|er)|screen|display|camera|speaker|microphone|overheat\w*|crack\w*|home\s+button|hardware)\b",
    re.IGNORECASE,
)
HARDWARE_ACTION = re.compile(
    r"\b(?:lasts?|poor|short|dies?|drain\w*|stop\w*|stopped|not\s+working|doesn['’]?t\s+work|won['’]?t\s+work|broken|crash\w*|freez\w*|slow|worse|worst|fail\w*)\b",
    re.IGNORECASE,
)
SERVICE = re.compile(
    r"\b(?:apple\s+id|icloud|itunes|apple\s+music|apple\s+pay|facetime|imessage|apple\s+watch)\b",
    re.IGNORECASE,
)
SERVICE_ACTION = re.compile(
    r"\b(?:help|problem|issue|wrong|cannot|can['’]?t|lost|missing|greyed|disabled|not\s+working|doesn['’]?t\s+work|fail\w*|charge\w*|code|account)\b",
    re.IGNORECASE,
)
UPDATE = re.compile(r"\b(?:ios\s*[-.]?\s*\d|ios\b|update\w*|upgrad\w*|firmware)\b", re.IGNORECASE)
UPDATE_ACTION = re.compile(
    r"\b(?:can['’]?t|cannot|won['’]?t|fail\w*|problem|issue|wrong|after|since|now|slow\w*|barely|crash\w*|broken|battery|work\w*)\b",
    re.IGNORECASE,
)
DEVICE = re.compile(r"\b(?:iphone|ipad|phone|device|macbook|apple\s+watch|watch|apple\s+tv)\b", re.IGNORECASE)
DEVICE_ACTION = re.compile(
    r"\b(?:not\s+working|isn['’]?t\s+working|doesn['’]?t\s+work|won['’]?t\s+work|can['’]?t|cannot|crash\w*|freez\w*|stuck|slow|bug\w*|glitch\w*|stutter\w*|broken|problem|issue|wrong|keeps?)\b",
    re.IGNORECASE,
)
DM = re.compile(r"\b(?:dm|direct\s+message|private\s+message|send\s+(?:me\s+)?(?:a\s+)?message|message\s+me)\b", re.IGNORECASE)
GENERAL = re.compile(r"\b(?:help|support|fix\w*|problem\w*|issue\w*|trouble|question|need\w*|why|please|wrong|annoying)\b", re.IGNORECASE)
ACKNOWLEDGEMENT = re.compile(
    r"^(?:thanks?|thank\s+you|done|okay|ok|worked|fixed|yes|no|both|will\s+do|sent\s+(?:a\s+)?dm)[!.?,\s\w'’😊😘👍🏽🤗]*$",
    re.IGNORECASE,
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _load_messages(path: Path) -> list[dict[str, str]]:
    messages = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            case = json.loads(line)
            for message in case.get("conversation", []):
                if message.get("role") == "customer" and isinstance(message.get("text"), str) and message["text"].strip():
                    messages.append(
                        {
                            "tweet_id": str(message["tweet_id"]),
                            "text": message["text"],
                            "case_id": str(case["case_id"]),
                        }
                    )
    if not messages:
        raise ValueError(f"No customer messages found: {path}")
    return messages


def _source_clusters(path: Path) -> dict[str, list[int]]:
    review = _load_json(path)
    mapping: defaultdict[str, list[int]] = defaultdict(list)
    for cluster in review.get("clusters", []):
        for example in cluster.get("examples", []):
            mapping[example].append(int(cluster["cluster_id"]))
    return dict(mapping)


def _specific_candidates(text: str) -> list[str]:
    candidates = []
    if KEYBOARD_CONCRETE.search(text):
        candidates.append("keyboard_text_input_issue")
    if APP.search(text) and APP_ACTION.search(text):
        candidates.append("app_store_app_issue")
    if HARDWARE.search(text) and HARDWARE_ACTION.search(text):
        candidates.append("device_hardware_issue")
    if SERVICE.search(text) and SERVICE_ACTION.search(text):
        candidates.append("apple_service_issue")
    return candidates


def _assess_message(message: dict[str, str], source_clusters: dict[str, list[int]]) -> dict[str, Any]:
    text = message["text"]
    specific = _specific_candidates(text)
    update_match = bool(UPDATE.search(text))
    update_action = update_match and bool(UPDATE_ACTION.search(text))
    device_match = bool(DEVICE.search(text))
    device_action = device_match and bool(DEVICE_ACTION.search(text))
    dm_match = bool(DM.search(text))
    general_match = bool(GENERAL.search(text))
    stripped = text.strip()

    if ACKNOWLEDGEMENT.fullmatch(stripped) or (device_match and not device_action and not specific and not update_action):
        return {
            "candidate_intent": "non_actionable_other",
            "possible_intents": [],
            "candidate_status": "HIGH_CONFIDENCE",
            "reason": "Acknowledgement or device/entity-only message with no identifiable actionable problem.",
            "source_cluster": source_clusters.get(text, []),
        }

    if len(specific) > 1:
        return {
            "candidate_intent": None,
            "possible_intents": specific,
            "candidate_status": "AMBIGUOUS",
            "reason": "Multiple specific issue objects matched and the message does not deterministically establish one primary reason.",
            "source_cluster": source_clusters.get(text, []),
        }
    if specific:
        intent = specific[0]
        return {
            "candidate_intent": intent,
            "possible_intents": [intent],
            "candidate_status": "HIGH_CONFIDENCE",
            "reason": f"Concrete {intent} evidence matched and was more specific than generic support language.",
            "source_cluster": source_clusters.get(text, []),
        }
    if update_action:
        return {
            "candidate_intent": "ios_update_issue",
            "possible_intents": ["ios_update_issue"],
            "candidate_status": "HIGH_CONFIDENCE",
            "reason": "Update/version language is paired with an explicit failure, symptom, or post-update effect.",
            "source_cluster": source_clusters.get(text, []),
        }
    if device_action:
        return {
            "candidate_intent": "device_functionality_issue",
            "possible_intents": ["device_functionality_issue"],
            "candidate_status": "HIGH_CONFIDENCE",
            "reason": "A concrete device malfunction is present without a more specific issue object.",
            "source_cluster": source_clusters.get(text, []),
        }
    if dm_match and not (specific or update_action or device_action):
        return {
            "candidate_intent": "support_dm_request",
            "possible_intents": ["support_dm_request"],
            "candidate_status": "HIGH_CONFIDENCE",
            "reason": "The primary actionable request is to use or confirm a direct-message support channel.",
            "source_cluster": source_clusters.get(text, []),
        }
    if general_match and len(stripped) >= 20 and not (specific or update_match or device_match):
        return {
            "candidate_intent": "general_support_request",
            "possible_intents": ["general_support_request"],
            "candidate_status": "HIGH_CONFIDENCE",
            "reason": "An actionable support request is present but no specific issue object is identifiable.",
            "source_cluster": source_clusters.get(text, []),
        }
    if specific or update_match or device_match or dm_match or general_match:
        possible = specific or (["ios_update_issue"] if update_match else []) or (["device_functionality_issue"] if device_match else []) or (["support_dm_request"] if dm_match else []) or (["general_support_request"] if general_match else [])
        return {
            "candidate_intent": None,
            "possible_intents": possible,
            "candidate_status": "AMBIGUOUS",
            "reason": "A weak or overlapping rule match exists, but the message is not specific enough for a high-confidence candidate.",
            "source_cluster": source_clusters.get(text, []),
        }
    return {
        "candidate_intent": None,
        "possible_intents": [],
        "candidate_status": "UNLABELED",
        "reason": "No sufficiently specific taxonomy evidence was found; message remains unlabeled.",
        "source_cluster": source_clusters.get(text, []),
    }


def _gold_sort_key(tweet_id: str) -> str:
    return hashlib.sha256(f"gold-review-v1:{tweet_id}".encode("utf-8")).hexdigest()


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _make_report(messages: list[dict[str, str]], assessments: list[dict[str, Any]], gold_ids: set[str], review_sample: list[dict[str, Any]]) -> dict[str, Any]:
    high = [a for a in assessments if a["candidate_status"] == "HIGH_CONFIDENCE"]
    ambiguous = [a for a in assessments if a["candidate_status"] == "AMBIGUOUS"]
    unlabeled = [a for a in assessments if a["candidate_status"] == "UNLABELED"]
    quality = []
    for intent in TAXONOMY_NAMES:
        candidates = [a for a in high if a.get("candidate_intent") == intent]
        ambiguous_matches = [a for a in ambiguous if intent in a.get("possible_intents", [])]
        quality.append(
            {
                "intent": intent,
                "high_confidence_count": len(candidates),
                "percentage_of_all_messages": round(len(candidates) / len(messages) * 100, 4),
                "candidate_examples": [a["text"] for a in candidates[:10]],
                "obvious_false_positives": [a["text"] for a in ambiguous_matches[:10]],
                "obvious_ambiguities": [a["reason"] for a in ambiguous_matches[:10]],
                "flag": "FEW_HIGH_CONFIDENCE_CANDIDATES" if len(candidates) < 20 else None,
            }
        )
    return {
        "dataset_size": len(messages),
        "gold_review_candidates_count": len(gold_ids),
        "assessment_counts": {
            "HIGH_CONFIDENCE": len(high),
            "AMBIGUOUS": len(ambiguous),
            "UNLABELED": len(unlabeled),
        },
        "candidate_file_records": len(messages) - len(gold_ids),
        "review_sample_size": len(review_sample),
        "review_sample_counts_by_intent": dict(Counter(record["candidate_intent"] for record in review_sample)),
        "quality_by_intent": quality,
        "interpretation": "These are transparent candidate assessments, not human labels or ground truth. Gold-review candidates have blank human labels and notes and are excluded from label_candidates.jsonl.",
    }


def prepare() -> dict[str, Any]:
    taxonomy = _load_json(TAXONOMY_PATH)
    if taxonomy.get("intent_count") != 9 or [item["intent_name"] for item in taxonomy.get("intents", [])] != TAXONOMY_NAMES:
        raise ValueError("final_taxonomy.json does not match the expected validated taxonomy")
    messages = _load_messages(CASES_PATH)
    source_clusters = _source_clusters(MANUAL_REVIEW_PATH)
    assessments = []
    for message in messages:
        assessment = _assess_message(message, source_clusters)
        assessments.append({**message, **assessment})

    gold_messages = []
    gold_texts: set[str] = set()
    for message in sorted(messages, key=lambda item: _gold_sort_key(item["tweet_id"])):
        if message["text"] in gold_texts:
            continue
        gold_messages.append(message)
        gold_texts.add(message["text"])
        if len(gold_messages) == 200:
            break
    gold_ids = {item["tweet_id"] for item in gold_messages}
    gold = {
        "source": CASES_PATH.name,
        "selection": "Deterministic SHA-256 order using gold-review-v1; no automatic labels assigned.",
        "count": len(gold_messages),
        "records": [
            {"tweet_id": item["tweet_id"], "text": item["text"], "case_id": item["case_id"], "human_label": None, "notes": None}
            for item in gold_messages
        ],
    }
    candidate_records = []
    candidate_texts: set[str] = set()
    for assessment in assessments:
        if assessment["tweet_id"] in gold_ids or assessment["text"] in gold_texts or assessment["text"] in candidate_texts:
            continue
        candidate_texts.add(assessment["text"])
        candidate_records.append(
            {
                "tweet_id": assessment["tweet_id"],
                "text": assessment["text"],
                "candidate_intent": assessment["candidate_intent"],
                "candidate_status": assessment["candidate_status"],
                "reason": assessment["reason"],
                "possible_intents": assessment["possible_intents"],
                "source_cluster": assessment["source_cluster"],
                "case_id": assessment["case_id"],
            }
        )
    high_available = [
        a
        for a in assessments
        if a["candidate_status"] == "HIGH_CONFIDENCE"
        and a["tweet_id"] not in gold_ids
        and a["text"] not in gold_texts
    ]
    review_sample = []
    review_texts: set[str] = set()
    per_intent = max(1, 1000 // len(TAXONOMY_NAMES))
    for intent in TAXONOMY_NAMES:
        available = [a for a in high_available if a.get("candidate_intent") == intent]
        available.sort(key=lambda item: _gold_sort_key(item["tweet_id"]))
        for item in available:
            if item["text"] in review_texts:
                continue
            review_texts.add(item["text"])
            review_sample.append(
                {
                    "tweet_id": item["tweet_id"],
                    "text": item["text"],
                    "case_id": item["case_id"],
                    "candidate_intent": item["candidate_intent"],
                    "candidate_status": item["candidate_status"],
                    "reason": item["reason"],
                    "source_cluster": item["source_cluster"],
                }
            )
            if sum(record["candidate_intent"] == intent for record in review_sample) >= per_intent:
                break
    remaining = [a for a in high_available if a["tweet_id"] not in {r["tweet_id"] for r in review_sample}]
    remaining.sort(key=lambda item: _gold_sort_key(item["tweet_id"]))
    for item in remaining[: max(0, 1000 - len(review_sample))]:
        if item["text"] in review_texts:
            continue
        review_texts.add(item["text"])
        review_sample.append(
            {
                "tweet_id": item["tweet_id"],
                "text": item["text"],
                "case_id": item["case_id"],
                "candidate_intent": item["candidate_intent"],
                "candidate_status": item["candidate_status"],
                "reason": item["reason"],
                "source_cluster": item["source_cluster"],
            }
        )

    report = _make_report(messages, assessments, gold_ids, review_sample)
    report["candidate_file_records"] = len(candidate_records)
    report["source_files"] = [TAXONOMY_PATH.name, CASES_PATH.name, MANUAL_REVIEW_PATH.name]
    report["human_labels_created"] = False
    report["classifier_created"] = False
    _write_jsonl(CANDIDATES_PATH, candidate_records)
    GOLD_PATH.write_text(json.dumps(gold, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    REVIEW_SAMPLE_PATH.write_text(json.dumps({"count": len(review_sample), "records": review_sample}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    try:
        report = prepare()
    except (OSError, ValueError, json.JSONDecodeError, KeyError) as error:
        print(f"Label candidate preparation failed: {error}", file=sys.stderr)
        return 1
    print("Training candidate preparation complete")
    print(f"HIGH_CONFIDENCE: {report['assessment_counts']['HIGH_CONFIDENCE']}")
    print(f"AMBIGUOUS: {report['assessment_counts']['AMBIGUOUS']}")
    print(f"UNLABELED: {report['assessment_counts']['UNLABELED']}")
    print(f"Gold review candidates: {report['gold_review_candidates_count']}")
    print(f"Review sample: {report['review_sample_size']}")
    print(f"Outputs: {CANDIDATES_PATH}, {GOLD_PATH}, {REPORT_PATH}")
    print("Candidate assessments are not human labels or ground truth.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
