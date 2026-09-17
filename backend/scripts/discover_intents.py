"""Explore recurring topics in real AppleSupport customer messages."""

from __future__ import annotations

import html
import json
import logging
import re
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "data" / "processed" / "apple_cases.jsonl"
OUTPUT_PATH = ROOT / "data" / "processed" / "intent_discovery_v2.json"
RANDOM_SEED = 42
SAMPLE_SIZE = 30_000
N_CLUSTERS = 22
URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
MENTION_PATTERN = re.compile(r"(?<!\w)@[A-Za-z0-9_]+")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_discovery_text(text: str) -> str:
    """Remove routing noise while leaving the original message untouched."""
    cleaned = html.unescape(text)
    cleaned = URL_PATTERN.sub(" ", cleaned)
    cleaned = MENTION_PATTERN.sub(" ", cleaned)
    return WHITESPACE_PATTERN.sub(" ", cleaned).strip()


def load_customer_messages(cases_path: Path) -> list[str]:
    """Load individual inbound/customer messages from reconstructed cases."""
    messages: list[str] = []
    with cases_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                case = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {cases_path} at line {line_number}: {error}") from error
            for message in case.get("conversation", []):
                if message.get("role") != "customer":
                    continue
                text = message.get("text")
                if isinstance(text, str) and text.strip():
                    messages.append(text)
    if not messages:
        raise ValueError(f"No non-empty customer messages found in {cases_path}.")
    return messages


def _representative_examples(
    sample_originals: list[str],
    labels: np.ndarray,
    distances: np.ndarray,
    cluster_id: int,
    used_examples: set[str],
) -> list[str]:
    positions = np.flatnonzero(labels == cluster_id)
    ordered_positions = positions[np.argsort(distances[positions])]
    examples: list[str] = []
    for position in ordered_positions:
        original = sample_originals[int(position)]
        if original in used_examples:
            continue
        examples.append(original)
        used_examples.add(original)
        if len(examples) == 5:
            break
    return examples


def _build_overlap_analysis(model: KMeans, counts: np.ndarray, clusters: list[dict]) -> dict:
    centroids = model.cluster_centers_
    centroid_norms = np.linalg.norm(centroids, axis=1)
    similarities = (centroids @ centroids.T) / np.outer(centroid_norms, centroid_norms)
    related_pairs = [
        {
            "cluster_a": left,
            "cluster_b": right,
            "centroid_cosine_similarity": round(float(similarities[left, right]), 4),
        }
        for left, right in combinations(range(len(clusters)), 2)
        if similarities[left, right] >= 0.55
    ]
    mean_count = float(np.mean(counts))
    broad = [int(cluster["cluster"]) for cluster in clusters if cluster["count"] >= mean_count * 1.75]
    generic = [
        int(cluster["cluster"])
        for cluster in clusters
        if sum(len(term.split()) == 1 and len(term) <= 4 for term in cluster["top_terms"][:6]) >= 3
    ]
    useful = [
        int(cluster["cluster"])
        for cluster in clusters
        if int(cluster["cluster"]) not in generic and int(cluster["cluster"]) not in broad
    ]
    return {
        "potentially_related_cluster_pairs": related_pairs,
        "overly_broad_clusters": broad,
        "generic_or_noisy_clusters": generic,
        "clusters_worth_manual_theme_review": useful,
        "notes": "These are inspection flags only; clusters were not merged or assigned final intent names.",
    }


def validate_result(result: dict, sampled_originals: list[str]) -> list[str]:
    errors: list[str] = []
    required_fields = {"method", "sample_size", "n_clusters", "quality", "clusters"}
    errors.extend(f"missing top-level field: {field}" for field in sorted(required_fields - result.keys()))
    if errors:
        return errors

    sample_size = result["sample_size"]
    n_clusters = result["n_clusters"]
    if not isinstance(sample_size, int) or sample_size <= 0:
        errors.append("sample_size must be a positive integer")
    if not isinstance(n_clusters, int) or n_clusters <= 0:
        errors.append("n_clusters must be a positive integer")
    clusters = result["clusters"]
    if not isinstance(clusters, list) or len(clusters) != n_clusters:
        errors.append("n_clusters does not match the actual cluster count")
        clusters = clusters if isinstance(clusters, list) else []

    cluster_ids: list[int] = []
    all_examples: list[str] = []
    total_count = 0
    total_percentage = 0.0
    sampled_set = set(sampled_originals)
    for index, cluster in enumerate(clusters):
        if not isinstance(cluster, dict):
            errors.append(f"cluster {index} is not an object")
            continue
        missing = {"cluster", "count", "percentage", "top_terms", "examples"} - cluster.keys()
        errors.extend(f"cluster {index} missing field: {field}" for field in sorted(missing))
        if missing:
            continue
        cluster_id = cluster["cluster"]
        cluster_ids.append(cluster_id)
        total_count += cluster["count"] if isinstance(cluster["count"], int) else 0
        total_percentage += cluster["percentage"] if isinstance(cluster["percentage"], (int, float)) else 0.0
        examples = cluster["examples"]
        if not isinstance(examples, list) or len(examples) != 5:
            errors.append(f"cluster {cluster_id} must contain exactly 5 examples")
            continue
        if any(not isinstance(example, str) or not example.strip() for example in examples):
            errors.append(f"cluster {cluster_id} contains an empty or non-string example")
        for example in examples:
            if example not in sampled_set:
                errors.append(f"cluster {cluster_id} contains an example not present in sampled source data")
            all_examples.append(example)

    if len(cluster_ids) != len(set(cluster_ids)):
        errors.append("cluster IDs are not unique")
    if isinstance(sample_size, int) and total_count != sample_size:
        errors.append(f"cluster counts sum to {total_count}, expected {sample_size}")
    if abs(total_percentage - 100.0) > 0.05:
        errors.append(f"cluster percentages sum to {total_percentage:.4f}, expected approximately 100")
    if len(all_examples) != len(set(all_examples)):
        errors.append("duplicate examples appear across clusters")

    quality = result["quality"]
    silhouette = quality.get("silhouette_score") if isinstance(quality, dict) else None
    if silhouette is not None and (not isinstance(silhouette, (int, float)) or not -1.0 <= silhouette <= 1.0):
        errors.append("silhouette_score must be between -1 and 1 when present")
    return errors


def discover(cases_path: Path = CASES_PATH, output_path: Path = OUTPUT_PATH) -> dict:
    started_at = time.perf_counter()
    all_messages = load_customer_messages(cases_path)
    cleaned_messages = [(original, clean_discovery_text(original)) for original in all_messages]
    cleaned_messages = [(original, cleaned) for original, cleaned in cleaned_messages if cleaned]
    sample_size = min(SAMPLE_SIZE, len(cleaned_messages))
    rng = np.random.default_rng(RANDOM_SEED)
    sample_indices = rng.choice(len(cleaned_messages), size=sample_size, replace=False)
    sample_originals = [cleaned_messages[int(index)][0] for index in sample_indices]
    sample_cleaned = [cleaned_messages[int(index)][1] for index in sample_indices]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=8,
        max_df=0.85,
        sublinear_tf=True,
        norm="l2",
        max_features=50_000,
    )
    matrix = vectorizer.fit_transform(sample_cleaned)
    model = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED, n_init=10)
    labels = model.fit_predict(matrix)
    distances = model.transform(matrix)
    terms = vectorizer.get_feature_names_out()
    used_examples: set[str] = set()
    clusters: list[dict] = []
    for cluster_id in range(N_CLUSTERS):
        positions = np.flatnonzero(labels == cluster_id)
        centroid = model.cluster_centers_[cluster_id]
        top_term_indices = centroid.argsort()[-12:][::-1]
        clusters.append(
            {
                "cluster": cluster_id,
                "count": int(len(positions)),
                "percentage": round(float(len(positions) / len(labels) * 100), 4),
                "top_terms": [terms[index] for index in top_term_indices],
                "examples": _representative_examples(
                    sample_originals, labels, distances[:, cluster_id], cluster_id, used_examples
                ),
            }
        )

    term_presence = np.asarray((matrix > 0).sum(axis=0)).ravel()
    common_terms = [terms[index] for index in term_presence.argsort()[::-1] if term_presence[index] > 0][:25]
    quality = {
        "silhouette_score": round(
            float(
                silhouette_score(
                    matrix,
                    labels,
                    metric="cosine",
                    sample_size=min(10_000, len(labels)),
                    random_state=RANDOM_SEED,
                )
            ),
            6,
        ),
        "cluster_size_min": int(min(cluster["count"] for cluster in clusters)),
        "cluster_size_max": int(max(cluster["count"] for cluster in clusters)),
        "cluster_size_mean": round(float(np.mean([cluster["count"] for cluster in clusters])), 3),
    }
    result = {
        "method": "Cleaned customer-message TF-IDF (unigram/bigram) followed by KMeans",
        "source": str(cases_path),
        "customer_messages_available": len(all_messages),
        "sample_size": len(sample_originals),
        "n_clusters": N_CLUSTERS,
        "random_seed": RANDOM_SEED,
        "quality": quality,
        "common_meaningful_terms": common_terms,
        "clusters": clusters,
    }
    result["overlap_analysis"] = _build_overlap_analysis(
        model, np.array([cluster["count"] for cluster in clusters]), clusters
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        readable_result = json.loads(output_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Generated output is not valid JSON: {error}") from error
    errors = validate_result(readable_result, sample_originals)
    if errors:
        output_path.unlink(missing_ok=True)
        raise ValueError("Intent discovery validation failed:\n- " + "\n- ".join(errors))
    readable_result["execution_time_seconds"] = round(time.perf_counter() - started_at, 3)
    output_path.write_text(json.dumps(readable_result, indent=2, ensure_ascii=False), encoding="utf-8")
    return readable_result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        result = discover()
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("Validator: PASS")
    print("Intent discovery complete. Final intent taxonomy has NOT been created.")
