"""Create a human-review adjudication report for the proposed taxonomy."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from validate_taxonomy import CASES_PATH, MANUAL_REVIEW_PATH, RULES, TAXONOMY, _load_json, _load_messages

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
VALIDATION_PATH = PROCESSED / "taxonomy_validation.json"
OUTPUT_PATH = PROCESSED / "taxonomy_adjudication.json"
RAW_PATH = ROOT / "data" / "raw" / "twcs.csv"

ASSESSMENTS = {
    "ios_update_issue": "Clearly actionable for messages describing update failures or post-update effects, but too broad when it captures version-only or status-only messages. It overlaps with device functionality and hardware symptoms.",
    "keyboard_text_input_issue": "Potentially actionable when a message describes typing, keyboard, letter, or text rendering behavior. It is noisy because glyph-only messages and generic text references do not always identify a support issue.",
    "app_store_app_issue": "Actionable for installation, download, App Store, or app failures. It is broad where a message merely mentions an app or asks which app is being used, and it can overlap with Apple service references.",
    "device_hardware_issue": "Actionable for concrete battery, charging, screen, camera, speaker, or physical-device symptoms. Device names alone are entities rather than intents, and hardware terms overlap with update and functionality symptoms.",
    "device_functionality_issue": "Potentially useful but too broad: device names and generic not-working language dominate the retrieval rule. It overlaps heavily with hardware, update, app, and general-support candidates.",
    "apple_service_issue": "Actionable for some Apple service or account problems, but broad because product and service mentions are mixed with commentary and entity-only messages. The actual service problem must be human-reviewed.",
    "support_dm_request": "Primarily a support-contact or conversational pattern rather than a problem intent. It can be operationally useful, but a DM request does not identify the underlying customer issue.",
    "general_support_request": "Too generic to stand alone as a reliable issue intent. Help, fix, question, and problem language often co-occurs with a more specific issue and overlaps nearly every other candidate category.",
    "other_unclear": "A residual retrieval bucket, not a substantive intent. It contains short, multilingual, contextual, entity-only, and otherwise unmatched messages that require human interpretation.",
}

MISSING_PATTERNS = {
    "ios_update_issue": [
        "Version-only messages and update announcements do not establish a customer problem.",
        "Post-update symptoms may also match hardware or device-functionality retrieval.",
        "Implicit and multilingual update complaints may be missed by the English keyword rule.",
    ],
    "keyboard_text_input_issue": [
        "Glyph-only or rendering messages may not contain an explicit keyboard or typing keyword.",
        "Generic text references can match without identifying the affected behavior.",
        "Some text-input problems are expressed through screenshots or symbols rather than words.",
    ],
    "app_store_app_issue": [
        "App mentions, app-selection questions, and actual app failures are not separated by the candidate rule.",
        "Service-specific app problems may also match Apple service retrieval.",
    ],
    "device_hardware_issue": [
        "Device names such as iPhone 6S or MacBook Pro are entities unless a concrete symptom is present.",
        "Battery, screen, and charging symptoms can be consequences of an update or general functionality problem.",
    ],
    "device_functionality_issue": [
        "Generic not-working statements often omit the affected feature or device behavior.",
        "Device mentions without a problem are retrieved even though they are entity information only.",
        "The broad rule does not distinguish software behavior from physical hardware symptoms.",
    ],
    "apple_service_issue": [
        "Product and service mentions are mixed with account, security, commentary, and entity-only messages.",
        "The rule may miss service problems expressed without the service name.",
    ],
    "support_dm_request": [
        "A DM request usually omits the underlying issue and should be reviewed alongside the surrounding conversation.",
        "Messages saying a DM was sent are follow-ups rather than new support problems.",
    ],
    "general_support_request": [
        "Generic help or fix wording does not identify the issue being requested.",
        "Specific issue messages often also contain help language, creating substantial overlap.",
    ],
    "other_unclear": [
        "Residual messages include short acknowledgements, multilingual text, entity-only messages, and context-dependent replies.",
        "A residual match does not explain why the customer contacted support.",
    ],
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stat(path: Path) -> tuple[int, int]:
    value = path.stat()
    return value.st_size, value.st_mtime_ns


def _candidate_sets(messages: list[str]) -> dict[str, set[int]]:
    return {
        name: {index for index, message in enumerate(messages) if pattern.search(message)}
        for name, pattern in RULES.items()
    }


def _overlapping_intents(name: str, candidates: dict[str, set[int]]) -> list[str]:
    current = candidates[name]
    scored = []
    for other, values in candidates.items():
        if other == name or not current:
            continue
        shared = len(current & values)
        ratio = shared / min(len(current), len(values)) if values else 0.0
        if ratio >= 0.05:
            scored.append((shared, ratio, other))
    return [other for _, _, other in sorted(scored, reverse=True)[:4]]


def _build_report(source: dict[str, Any], messages: list[str], manual_review: dict[str, Any]) -> dict[str, Any]:
    candidate_sets = _candidate_sets(messages)
    source_by_name = {item["name"]: item for item in source["intents"]}
    proposed_intents = []
    for name in TAXONOMY:
        source_item = source_by_name[name]
        expected_count = source_item["candidate_count"]
        actual_count = len(candidate_sets[name]) if name in candidate_sets else len(messages) - len(set().union(*candidate_sets.values()))
        if actual_count != expected_count:
            raise ValueError(f"Rule count mismatch for {name}: source={expected_count}, recomputed={actual_count}")
        proposed_intents.append(
            {
                "name": name,
                "candidate_count": expected_count,
                "examples": source_item["examples"],
                "false_positives": source_item["false_positive_examples"],
                "overlapping_intents": _overlapping_intents(name, candidate_sets) if name in candidate_sets else [],
                "missing_patterns": MISSING_PATTERNS[name],
                "assessment": ASSESSMENTS[name],
            }
        )

    return {
        "source": VALIDATION_PATH.name,
        "dataset_size": source["dataset_size"],
        "proposed_intents": proposed_intents,
        "cross_cutting_patterns": {
            "follow_up": source["follow_up_patterns"],
            "device_only": source["device_specific_without_problem"],
            "multi_intent": source["multiple_possible_issues"],
        },
        "review_status": "Candidate evidence and adjudication guidance only. No final labels or training labels were created.",
        "cluster_investigation": source.get("cluster_investigation", []),
    }


def _validate_report(source: dict[str, Any], report: dict[str, Any], messages: list[str]) -> list[str]:
    errors: list[str] = []
    source_intents = {item["name"]: item for item in source.get("intents", [])}
    output_intents = report.get("proposed_intents")
    if report.get("source") != VALIDATION_PATH.name:
        errors.append("source does not identify taxonomy_validation.json")
    if report.get("dataset_size") != len(messages):
        errors.append("dataset_size does not match the processed customer-message source")
    if not isinstance(output_intents, list) or len(output_intents) != len(TAXONOMY):
        errors.append("output does not contain exactly nine proposed intents")
        output_intents = output_intents if isinstance(output_intents, list) else []
    output_names = [item.get("name") for item in output_intents if isinstance(item, dict)]
    if output_names != TAXONOMY or len(set(output_names)) != len(TAXONOMY):
        errors.append("proposed intent names are missing, reordered, or duplicated")

    message_set = set(messages)
    for item in output_intents:
        if not isinstance(item, dict):
            errors.append("proposed intent entry is not an object")
            continue
        name = item.get("name", "<missing>")
        source_item = source_intents.get(name)
        if source_item is None:
            errors.append(f"unexpected intent: {name}")
            continue
        if item.get("candidate_count") != source_item["candidate_count"]:
            errors.append(f"{name}: candidate count changed")
        for field in ("examples", "false_positives"):
            values = item.get(field)
            source_field = "examples" if field == "examples" else "false_positive_examples"
            if values != source_item[source_field]:
                errors.append(f"{name}: {field} differ from taxonomy_validation.json")
            if not isinstance(values, list) or any(not isinstance(value, str) or not value.strip() for value in values):
                errors.append(f"{name}: {field} contains invalid text")
            if isinstance(values, list) and any(value not in message_set for value in values):
                errors.append(f"{name}: {field} contains text absent from the real dataset")
        if not isinstance(item.get("overlapping_intents"), list):
            errors.append(f"{name}: overlapping_intents is not a list")
        if not isinstance(item.get("missing_patterns"), list) or not item["missing_patterns"]:
            errors.append(f"{name}: missing_patterns is empty")
        if not isinstance(item.get("assessment"), str) or not item["assessment"].strip():
            errors.append(f"{name}: assessment is missing")

    expected_cross = {
        "follow_up": source["follow_up_patterns"],
        "device_only": source["device_specific_without_problem"],
        "multi_intent": source["multiple_possible_issues"],
    }
    if report.get("cross_cutting_patterns") != expected_cross:
        errors.append("cross-cutting pattern evidence differs from taxonomy_validation.json")
    if "final_labels" in report or "training_labels" in report:
        errors.append("report contains prohibited final or training labels")
    return errors


def main() -> int:
    required = (VALIDATION_PATH, CASES_PATH, MANUAL_REVIEW_PATH, RAW_PATH)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("Missing required files:\n- " + "\n- ".join(missing), file=sys.stderr)
        return 1

    protected = {path: _stat(path) for path in (VALIDATION_PATH, CASES_PATH, MANUAL_REVIEW_PATH, RAW_PATH)}
    validation_digest = _sha256(VALIDATION_PATH)
    try:
        source = _load_json(VALIDATION_PATH)
        manual_review = _load_json(MANUAL_REVIEW_PATH)
        messages = _load_messages(CASES_PATH)
        report = _build_report(source, messages, manual_review)
        OUTPUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written = _load_json(OUTPUT_PATH)
        errors = _validate_report(source, written, messages)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Taxonomy adjudication failed: {error}", file=sys.stderr)
        return 1

    for path, before in protected.items():
        if _stat(path) != before:
            errors.append(f"protected source changed: {path}")
    if _sha256(VALIDATION_PATH) != validation_digest:
        errors.append("taxonomy_validation.json content changed")

    if errors:
        print("Taxonomy adjudication validation: FAIL", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("Taxonomy adjudication validation: PASS")
    print(f"Dataset messages: {len(messages)}")
    print(f"Proposed intents: {len(written['proposed_intents'])}")
    print("Human labels created: NO")
    print(f"Report: {OUTPUT_PATH}")
    print("Taxonomy adjudication complete. Final classifier taxonomy has NOT been created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
