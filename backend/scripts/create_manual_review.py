"""Create and validate a non-destructive manual-review artifact."""

from __future__ import annotations

import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
SOURCE_PATH = PROCESSED / "intent_discovery_v2.json"
OUTPUT_PATH = PROCESSED / "intent_manual_review.json"
RAW_PATH = ROOT / "data" / "raw" / "twcs.csv"
CASES_PATH = PROCESSED / "apple_cases.jsonl"
PERCENTAGE_TOLERANCE = 0.01
EXPECTED_REVIEW_FIELDS = {
    "proposed_theme": None,
    "confidence": None,
    "merge_with": [],
    "keep_as_intent": None,
    "notes": None,
}


def _file_fingerprint(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def _source_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Required JSON file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def build_manual_review(source: dict[str, Any]) -> dict[str, Any]:
    source_clusters = source.get("clusters")
    if not isinstance(source_clusters, list):
        raise ValueError("Source discovery JSON has no valid clusters list")
    clusters = []
    for source_cluster in source_clusters:
        clusters.append(
            {
                "cluster_id": source_cluster["cluster"],
                "count": source_cluster["count"],
                "percentage": source_cluster["percentage"],
                "top_terms": source_cluster["top_terms"],
                "examples": source_cluster["examples"],
                "review_fields": {
                    "proposed_theme": None,
                    "confidence": None,
                    "merge_with": [],
                    "keep_as_intent": None,
                    "notes": None,
                },
            }
        )
    return {
        "source": SOURCE_PATH.name,
        "sample_size": source["sample_size"],
        "n_clusters": source["n_clusters"],
        "clusters": clusters,
    }


def validate_manual_review(source: dict[str, Any], output: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    source_clusters = source.get("clusters")
    output_clusters = output.get("clusters")
    if not isinstance(source_clusters, list):
        return ["source clusters are not a list"]
    if not isinstance(output_clusters, list):
        return ["output clusters are not a list"]

    if output.get("source") != SOURCE_PATH.name:
        errors.append(f"source must be {SOURCE_PATH.name!r}")
    if output.get("sample_size") != source.get("sample_size"):
        errors.append("sample_size does not match the source")
    if output.get("n_clusters") != source.get("n_clusters"):
        errors.append("n_clusters does not match the source")
    if len(output_clusters) != len(source_clusters):
        errors.append(f"expected {len(source_clusters)} clusters, found {len(output_clusters)}")

    expected_ids = [cluster.get("cluster") for cluster in source_clusters]
    actual_ids = [cluster.get("cluster_id") for cluster in output_clusters if isinstance(cluster, dict)]
    if len(actual_ids) != len(set(actual_ids)):
        errors.append("output cluster IDs are not unique")
    if set(actual_ids) != set(expected_ids):
        errors.append(f"cluster IDs differ: expected {sorted(expected_ids)}, found {sorted(actual_ids)}")

    source_by_id = {cluster.get("cluster"): cluster for cluster in source_clusters}
    output_by_id = {cluster.get("cluster_id"): cluster for cluster in output_clusters if isinstance(cluster, dict)}
    total_examples = 0
    all_source_examples: list[str] = []
    all_output_examples: list[str] = []
    for cluster_id in expected_ids:
        expected = source_by_id[cluster_id]
        actual = output_by_id.get(cluster_id)
        if actual is None:
            errors.append(f"cluster {cluster_id} is missing")
            continue
        for field in ("count", "percentage", "top_terms"):
            if actual.get(field) != expected.get(field):
                errors.append(f"cluster {cluster_id} field {field} differs from source")
        expected_examples = expected.get("examples")
        actual_examples = actual.get("examples")
        if not isinstance(actual_examples, list) or len(actual_examples) != 5:
            errors.append(f"cluster {cluster_id} must contain exactly 5 examples")
        elif actual_examples != expected_examples:
            errors.append(f"cluster {cluster_id} examples differ from source or changed order")
        if isinstance(actual_examples, list):
            total_examples += len(actual_examples)
            all_output_examples.extend(actual_examples)
        if isinstance(expected_examples, list):
            all_source_examples.extend(expected_examples)
        if len(actual_examples) != len(set(actual_examples)) if isinstance(actual_examples, list) else False:
            errors.append(f"cluster {cluster_id} contains duplicate examples")
        if isinstance(actual_examples, list) and any(not isinstance(example, str) or not example for example in actual_examples):
            errors.append(f"cluster {cluster_id} contains an empty or non-string example")

        if actual.get("review_fields") != EXPECTED_REVIEW_FIELDS:
            errors.append(f"cluster {cluster_id} review_fields are not empty")

    if total_examples != 110:
        errors.append(f"expected 110 total examples, found {total_examples}")
    if all_output_examples != all_source_examples:
        errors.append("global example order/content differs from the source")
    if sum(cluster.get("count", 0) for cluster in source_clusters) != source.get("sample_size"):
        errors.append("source cluster counts do not sum to source sample_size")
    percentage_sum = sum(cluster.get("percentage", 0.0) for cluster in source_clusters)
    if abs(percentage_sum - 100.0) > PERCENTAGE_TOLERANCE:
        errors.append(f"source percentages sum to {percentage_sum}, outside tolerance")
    return errors


def _shared_term_relationships(clusters: list[dict[str, Any]]) -> list[str]:
    relationships: list[str] = []
    for left, right in combinations(clusters, 2):
        shared = [term for term in left["top_terms"] if term in right["top_terms"]]
        if len(shared) >= 3:
            relationships.append(
                f"Clusters {left['cluster_id']} and {right['cluster_id']} appear related because both contain: "
                + ", ".join(shared[:5])
                + "."
            )
    return relationships


def print_summary(source: dict[str, Any], review: dict[str, Any]) -> None:
    source_analysis = source.get("overlap_analysis", {})
    broad = source_analysis.get("overly_broad_clusters", [])
    noisy = source_analysis.get("generic_or_noisy_clusters", [])
    excluded = set(broad) | set(noisy)
    clear = [cluster["cluster_id"] for cluster in review["clusters"] if cluster["cluster_id"] not in excluded]
    relationships = _shared_term_relationships(review["clusters"])

    print("\nExploratory Cluster Review")
    print("--------------------------")
    print("1. Clear thematic clusters")
    print(f"Clusters {clear}; these are only candidates for human review based on more specific terms.")
    print("2. Broad clusters")
    print(f"Clusters {broad}; these have comparatively large sample coverage.")
    print("3. Generic/noisy clusters")
    print(f"Clusters {noisy}; these contain generic language or low-specificity terms.")
    print("4. Likely duplicate/related clusters")
    if relationships:
        for relationship in relationships:
            print(relationship)
    else:
        print("No likely relationships were identified from shared top terms.")
    print("These observations are exploratory only. Final taxonomy decisions remain manual.")


def main() -> int:
    if not SOURCE_PATH.exists():
        print(f"Required source does not exist: {SOURCE_PATH}", file=sys.stderr)
        return 1
    if SOURCE_PATH.resolve() == OUTPUT_PATH.resolve():
        print("Source and output paths must be different.", file=sys.stderr)
        return 1

    source_digest_before = _source_digest(SOURCE_PATH)
    protected_before = {
        path: _file_fingerprint(path)
        for path in (SOURCE_PATH, RAW_PATH, CASES_PATH)
        if path.exists()
    }
    try:
        source = _load_json(SOURCE_PATH)
        review = build_manual_review(source)
        OUTPUT_PATH.write_text(json.dumps(review, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        output = _load_json(OUTPUT_PATH)
        errors = validate_manual_review(source, output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Manual review generation failed: {error}", file=sys.stderr)
        return 1

    protected_after = {
        path: _file_fingerprint(path)
        for path in protected_before
    }
    if _source_digest(SOURCE_PATH) != source_digest_before:
        errors.append("source discovery JSON changed during generation")
    if protected_after != protected_before:
        errors.append("a protected source, raw, or reconstruction file changed during generation")

    print("Manual Review Validation")
    print("------------------------")
    print(f"Source clusters: {len(source.get('clusters', []))}")
    print(f"Output clusters: {len(output.get('clusters', []))}")
    print("Expected examples: 110")
    print(f"Output examples: {sum(len(cluster.get('examples', [])) for cluster in output.get('clusters', []))}")
    checks = [
        "Metadata integrity",
        "Example integrity",
        "Review fields",
        "Cluster IDs",
        "Sample-size consistency",
        "Percentage consistency",
        "Source immutability",
    ]
    for check in checks:
        print(f"{check}: {'FAIL' if errors else 'PASS'}")
    if errors:
        print("\nVALIDATION: FAIL")
        for error in errors:
            print(f"- {error}")
        return 1

    print("\nVALIDATION: PASS")
    print("\nManual review dataset created:")
    print(OUTPUT_PATH)
    print("Clusters: 22")
    print("Examples: 110")
    print("Sample size: 30,000")
    print("Validation: PASS")
    print_summary(source, output)
    print("Manual review dataset complete. Final intent taxonomy has NOT been created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
