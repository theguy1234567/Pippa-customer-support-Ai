"""Validate a proposed taxonomy using transparent candidate retrieval only."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
CASES_PATH = PROCESSED / "apple_cases.jsonl"
MANUAL_REVIEW_PATH = PROCESSED / "intent_manual_review.json"
OUTPUT_PATH = PROCESSED / "taxonomy_validation.json"
RAW_PATH = ROOT / "data" / "raw" / "twcs.csv"

TAXONOMY = [
    "ios_update_issue",
    "keyboard_text_input_issue",
    "app_store_app_issue",
    "device_hardware_issue",
    "device_functionality_issue",
    "apple_service_issue",
    "support_dm_request",
    "general_support_request",
    "other_unclear",
]

RULES = {
    "ios_update_issue": re.compile(
        r"\b(?:ios\s*[-.]?\s*\d|ios\b|update\w*|upgrad\w*|firmware)\b", re.IGNORECASE
    ),
    "keyboard_text_input_issue": re.compile(
        r"\b(?:keyboard|typing|typed|type|autocorrect|spell(?:ing)?|letter|question\s+mark|emoji|text\s+input|input)\b|i[\ufe0f\u200d]*[’']?s",
        re.IGNORECASE,
    ),
    "app_store_app_issue": re.compile(
        r"\b(?:app\s*store|appstore|apps?|application|download\w*|install\w*|launch\w*|open\w*)\b",
        re.IGNORECASE,
    ),
    "device_hardware_issue": re.compile(
        r"\b(?:battery|batteries|charg(?:e|ing|er)|screen|display|camera|speaker|microphone|overheat\w*|crack\w*|button|macbook|iphone\s*6s|hardware)\b",
        re.IGNORECASE,
    ),
    "device_functionality_issue": re.compile(
        r"\b(?:not\s+working|isn['’]?t\s+working|doesn['’]?t\s+work|won['’]?t\s+work|can['’]?t|cannot|crash\w*|crashing|freez\w*|stuck|slow|bug\w*|glitch\w*|iphone|ipad|phone|device|watch|macbook)\b",
        re.IGNORECASE,
    ),
    "apple_service_issue": re.compile(
        r"\b(?:apple\s+music|apple\s+id|icloud|itunes|apple\s+pay|facetime|imessage|apple\s+watch|apple\s+store)\b",
        re.IGNORECASE,
    ),
    "support_dm_request": re.compile(
        r"\b(?:dm|direct\s+message|private\s+message|send\s+(?:me\s+)?(?:a\s+)?message|message\s+me)\b",
        re.IGNORECASE,
    ),
    "general_support_request": re.compile(
        r"\b(?:help|support|fix\w*|problem\w*|issue\w*|trouble|question|need\w*|why|please|wrong|annoying)\b",
        re.IGNORECASE,
    ),
}

FOLLOW_UP_PATTERN = re.compile(
    r"\b(?:thanks?|thank\s+you|done|worked|fixed|okay|ok|sent\s+dm|will\s+do|same\s+issue|having\s+the\s+same)\b",
    re.IGNORECASE,
)
DEVICE_PATTERN = re.compile(r"\b(?:iphone|ipad|phone|device|macbook|apple\s+watch|watch)\b", re.IGNORECASE)
PROBLEM_PATTERN = re.compile(
    r"\b(?:problem|issue|trouble|wrong|broken|not\s+working|doesn['’]?t\s+work|won['’]?t\s+work|help|fix|crash|bug|glitch|slow|stuck|battery|screen|charging)\b",
    re.IGNORECASE,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stat_fingerprint(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def _load_messages(path: Path) -> list[str]:
    messages: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                case = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path} line {line_number}: {error}") from error
            for message in case.get("conversation", []):
                if message.get("role") == "customer" and isinstance(message.get("text"), str) and message["text"].strip():
                    messages.append(message["text"])
    if not messages:
        raise ValueError(f"No customer messages found in {path}")
    return messages


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read valid JSON from {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _candidate_messages(messages: list[str], pattern: re.Pattern[str]) -> list[str]:
    return [message for message in messages if pattern.search(message)]


def _unique_examples(messages: list[str], limit: int = 10) -> list[str]:
    return list(dict.fromkeys(messages))[:limit]


def _is_obvious_false_positive(message: str, intent: str) -> bool:
    stripped = message.strip()
    if re.fullmatch(r"(?:thanks?|thank you|done|fixed|worked|ok|okay|yes|no|both|help|please help)[!.?\s]*", stripped, re.IGNORECASE):
        return True
    if intent in {"device_hardware_issue", "device_functionality_issue"} and DEVICE_PATTERN.search(stripped) and not PROBLEM_PATTERN.search(stripped):
        return True
    if intent == "ios_update_issue" and re.fullmatch(r"(?:ios\s*[-.]?\s*\d+(?:\.\d+)?|update)[!.?\s]*", stripped, re.IGNORECASE):
        return True
    return len(stripped) <= 18 and intent not in {"support_dm_request", "other_unclear"}


def _false_positive_examples(candidates: list[str], intent: str) -> list[str]:
    return _unique_examples([message for message in candidates if _is_obvious_false_positive(message, intent)])


def _follow_up_patterns(messages: list[str]) -> dict[str, Any]:
    candidates = _candidate_messages(messages, FOLLOW_UP_PATTERN)
    return {
        "count": len(candidates),
        "percentage": round(len(candidates) / len(messages) * 100, 4),
        "examples": _unique_examples(candidates),
        "notes": "These are message-level pattern matches, not human-assigned labels.",
    }


def _device_without_problem(messages: list[str]) -> dict[str, Any]:
    candidates = [message for message in messages if DEVICE_PATTERN.search(message) and not PROBLEM_PATTERN.search(message)]
    return {
        "count": len(candidates),
        "percentage": round(len(candidates) / len(messages) * 100, 4),
        "examples": _unique_examples(candidates),
    }


def _multiple_possible_issues(messages: list[str]) -> dict[str, Any]:
    candidates = []
    for message in messages:
        matched_intents = [name for name, pattern in RULES.items() if pattern.search(message)]
        if len(matched_intents) >= 2:
            candidates.append(message)
    return {
        "count": len(candidates),
        "percentage": round(len(candidates) / len(messages) * 100, 4),
        "examples": _unique_examples(candidates),
        "notes": "A message can match multiple retrieval rules; this does not assign a single label.",
    }


def _cluster_investigation(manual_review: dict[str, Any]) -> list[dict[str, Any]]:
    observations = {
        0: "Examples and terms are mostly generic reports about something happening, with limited issue specificity.",
        2: "Examples are largely same-issue or having-problem follow-ups without a stable issue object.",
        4: "Examples are short requests for help and may be useful as broad support-request evidence.",
        7: "Terms span phone, update, and new-device language, making the cluster broad and overlapping.",
        8: "This is the largest cluster and contains many short contextual replies such as acknowledgements or answers.",
        10: "Terms and examples span Apple products and services, with mixed service and device references.",
        16: "Examples are generic not-working follow-ups and do not identify a consistent issue type.",
        18: "Examples repeat not-working language and may overlap with broad device-functionality retrieval.",
        21: "Examples and terms are broad conversational follow-ups involving phone, update, and general context.",
        12: "Examples are predominantly thanks or confirmation after support interaction.",
        17: "Examples are predominantly short gratitude or acknowledgement messages.",
    }
    by_id = {cluster["cluster_id"]: cluster for cluster in manual_review.get("clusters", [])}
    return [
        {
            "cluster_id": cluster_id,
            "top_terms": by_id[cluster_id]["top_terms"],
            "observation": observations[cluster_id],
        }
        for cluster_id in observations
        if cluster_id in by_id
    ]


def _uncovered_patterns(messages: list[str], matched_any: set[int]) -> list[dict[str, Any]]:
    uncovered = [message for index, message in enumerate(messages) if index not in matched_any]
    patterns = Counter()
    examples: dict[str, list[str]] = {}
    for message in uncovered:
        normalized = re.sub(r"[^a-z0-9 ]+", " ", message.lower())
        tokens = normalized.split()
        pattern = " ".join(tokens[:3]) if tokens else "empty"
        patterns[pattern] += 1
        examples.setdefault(pattern, [])
        if len(examples[pattern]) < 5:
            examples[pattern].append(message)
    return [
        {"pattern": pattern, "count": count, "examples": examples[pattern]}
        for pattern, count in patterns.most_common(20)
    ]


def _validate_report(report: dict[str, Any], messages: list[str]) -> list[str]:
    errors: list[str] = []
    expected = set(TAXONOMY)
    intents = report.get("intents")
    if report.get("taxonomy") != TAXONOMY:
        errors.append("taxonomy does not exactly match the nine proposed intents")
    if report.get("dataset_size") != len(messages):
        errors.append("dataset_size does not match the processed customer-message source")
    if not isinstance(intents, list) or {item.get("name") for item in intents if isinstance(item, dict)} != expected:
        errors.append("all proposed intents are not present exactly once")
    if isinstance(intents, list) and len(intents) != len(TAXONOMY):
        errors.append("intent count is not 9")

    message_set = set(messages)
    for intent in intents if isinstance(intents, list) else []:
        if not isinstance(intent, dict):
            errors.append("intent entry is not an object")
            continue
        name = intent.get("name", "<missing>")
        count = intent.get("candidate_count")
        percentage = intent.get("percentage")
        if not isinstance(count, int) or count < 0:
            errors.append(f"{name}: candidate_count is invalid")
        if not isinstance(percentage, (int, float)) or not 0 <= percentage <= 100:
            errors.append(f"{name}: percentage is invalid")
        for field in ("examples", "false_positive_examples"):
            values = intent.get(field)
            if not isinstance(values, list):
                errors.append(f"{name}: {field} is not a list")
                continue
            if any(not isinstance(value, str) or not value.strip() for value in values):
                errors.append(f"{name}: {field} contains an empty or non-string value")
            if any(value not in message_set for value in values):
                errors.append(f"{name}: {field} contains a message absent from the source")
        if not isinstance(intent.get("missing_pattern_notes"), str):
            errors.append(f"{name}: missing_pattern_notes is not a string")

    interpretation = report.get("interpretation", {})
    for field in ("candidate_retrieval", "human_assigned_labels", "uncovered_messages"):
        if not isinstance(interpretation.get(field), str) or not interpretation[field].strip():
            errors.append(f"interpretation.{field} must clearly be documented")
    return errors


def build_report(messages: list[str], manual_review: dict[str, Any]) -> dict[str, Any]:
    matched_any: set[int] = set()
    intents = []
    for name in TAXONOMY[:-1]:
        pattern = RULES[name]
        candidates = []
        for index, message in enumerate(messages):
            if pattern.search(message):
                candidates.append(message)
                matched_any.add(index)
        intents.append(
            {
                "name": name,
                "candidate_count": len(candidates),
                "percentage": round(len(candidates) / len(messages) * 100, 4),
                "examples": _unique_examples(candidates),
                "false_positive_examples": _false_positive_examples(candidates, name),
                "missing_pattern_notes": "Rule-based retrieval is a candidate screen only; implicit, multilingual, and context-dependent cases may be missed.",
            }
        )

    unclear = [message for index, message in enumerate(messages) if index not in matched_any]
    intents.append(
        {
            "name": "other_unclear",
            "candidate_count": len(unclear),
            "percentage": round(len(unclear) / len(messages) * 100, 4),
            "examples": _unique_examples(unclear),
            "false_positive_examples": [],
            "missing_pattern_notes": "This is the residual set with no configured retrieval-rule match; it is not a human-assigned label.",
        }
    )
    return {
        "taxonomy": TAXONOMY,
        "dataset_size": len(messages),
        "intents": intents,
        "uncovered_patterns": _uncovered_patterns(messages, matched_any),
        "follow_up_patterns": _follow_up_patterns(messages),
        "device_specific_without_problem": _device_without_problem(messages),
        "multiple_possible_issues": _multiple_possible_issues(messages),
        "cluster_investigation": _cluster_investigation(manual_review),
        "recommendations": [
            "Treat candidate counts as transparent retrieval coverage, not ground-truth labels.",
            "Manually review the broad and generic clusters before deciding whether the proposed categories are sufficiently distinct.",
            "Review follow-up and acknowledgement messages separately from issue-bearing messages.",
            "Review multi-match messages individually because the rules intentionally allow overlap.",
            "Do not train a classifier until human-assigned labels and taxonomy boundaries exist.",
        ],
        "interpretation": {
            "candidate_retrieval": "Counts and examples were produced by transparent keyword/rule matching over the full customer-message source; candidates may overlap across intents.",
            "human_assigned_labels": "None. No customer message was assigned a ground-truth or final taxonomy label by this report.",
            "uncovered_messages": "Messages without a configured rule match are reported as uncovered/residual observations, not automatically assigned to a final intent.",
        },
    }


def main() -> int:
    required = (CASES_PATH, MANUAL_REVIEW_PATH, RAW_PATH)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("Missing required input files:\n- " + "\n- ".join(missing), file=sys.stderr)
        return 1
    cases_digest = _sha256(CASES_PATH)
    protected_stats = {path: _stat_fingerprint(path) for path in (CASES_PATH, MANUAL_REVIEW_PATH, RAW_PATH)}
    try:
        messages = _load_messages(CASES_PATH)
        manual_review = _load_json(MANUAL_REVIEW_PATH)
        report = build_report(messages, manual_review)
        OUTPUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written = _load_json(OUTPUT_PATH)
        errors = _validate_report(written, messages)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Taxonomy validation failed: {error}", file=sys.stderr)
        return 1

    for path, before in protected_stats.items():
        if _stat_fingerprint(path) != before:
            errors.append(f"protected source changed: {path}")
    if _sha256(CASES_PATH) != cases_digest:
        errors.append("processed customer-message source content changed")

    if errors:
        print("Taxonomy validation: FAIL", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("Taxonomy validation: PASS")
    print(f"Customer messages: {len(messages)}")
    print("Candidate retrieval only: PASS")
    print("Human-assigned labels created: NO")
    print(f"Report: {OUTPUT_PATH}")
    print("\nCandidate counts:")
    for intent in written["intents"]:
        print(f"- {intent['name']}: {intent['candidate_count']} ({intent['percentage']}%)")
    print("\nUncovered patterns:")
    for pattern in written["uncovered_patterns"][:10]:
        print(f"- {pattern['pattern']}: {pattern['count']}")
    print("\nTaxonomy validation complete. Final classifier labels have NOT been created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
