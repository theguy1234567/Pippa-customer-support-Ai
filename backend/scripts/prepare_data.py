"""Extract AppleSupport conversations from twcs.csv without copying the raw dataset."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "twcs.csv"
PROCESSED_PATH = ROOT / "data" / "processed"
CSV_COLUMNS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]
CHUNK_SIZE = 100_000


def _parse_ids(value: object) -> list[str]:
    """Parse scalar, comma-separated, or list-like relationship fields."""
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []
    if text[0] in "[(" and text[-1] in ")]":
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, (list, tuple, set)):
            return [str(item).strip() for item in parsed if str(item).strip()]
        if isinstance(parsed, str):
            text = parsed
    return [item.strip().strip("'\"") for item in text.split(",") if item.strip()]


def _references(row: dict[str, str]) -> set[str]:
    references: set[str] = set()
    for field in ("response_tweet_id", "in_response_to_tweet_id"):
        references.update(_parse_ids(row.get(field, "")))
    return references


def _read_rows(
    path: Path,
    wanted: set[str] | None = None,
) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for frame in pd.read_csv(path, usecols=CSV_COLUMNS, chunksize=CHUNK_SIZE, keep_default_na=False, dtype=str):
        frame = frame.astype(str)
        if wanted is not None:
            frame = frame[frame["tweet_id"].isin(wanted)]
        for row in frame.to_dict(orient="records"):
            rows[row["tweet_id"]] = row
    return rows


def _build_relationship_index(
    path: Path,
) -> tuple[set[str], dict[str, str], defaultdict[str, list[str]], float]:
    """Build relationship lookups in one CSV pass without retaining all message text."""
    apple_ids: set[str] = set()
    parent_by_tweet: dict[str, str] = {}
    children_by_parent: defaultdict[str, list[str]] = defaultdict(list)
    seen_children: defaultdict[str, set[str]] = defaultdict(set)
    filtering_seconds = 0.0

    for frame in pd.read_csv(path, usecols=CSV_COLUMNS, chunksize=CHUNK_SIZE, keep_default_na=False, dtype=str):
        filtering_started_at = time.perf_counter()
        apple_ids.update(frame.loc[frame["author_id"].eq("AppleSupport"), "tweet_id"].tolist())
        filtering_seconds += time.perf_counter() - filtering_started_at
        for row in frame.to_dict(orient="records"):
            tweet_id = row["tweet_id"]
            response_ids = _parse_ids(row["response_tweet_id"])
            parent_ids = _parse_ids(row["in_response_to_tweet_id"])
            for parent_id in parent_ids:
                if tweet_id not in seen_children[parent_id]:
                    children_by_parent[parent_id].append(tweet_id)
                    seen_children[parent_id].add(tweet_id)
                parent_by_tweet.setdefault(tweet_id, parent_id)
            for child_id in response_ids:
                if child_id not in seen_children[tweet_id]:
                    children_by_parent[tweet_id].append(child_id)
                    seen_children[tweet_id].add(child_id)
                parent_by_tweet.setdefault(child_id, tweet_id)

    if not apple_ids:
        raise RuntimeError("No author_id=AppleSupport rows were found in twcs.csv.")
    return apple_ids, parent_by_tweet, children_by_parent, filtering_seconds


def _connected_ids(
    seeds: set[str],
    parent_by_tweet: dict[str, str],
    children_by_parent: defaultdict[str, list[str]],
) -> set[str]:
    """Traverse the indexed reply graph once from every AppleSupport seed."""
    connected: set[str] = set()
    pending = list(seeds)
    while pending:
        tweet_id = pending.pop()
        if tweet_id in connected:
            continue
        connected.add(tweet_id)
        parent_id = parent_by_tweet.get(tweet_id)
        if parent_id is not None and parent_id not in connected:
            pending.append(parent_id)
        pending.extend(child_id for child_id in children_by_parent.get(tweet_id, ()) if child_id not in connected)
    return connected


def _components(rows: dict[str, dict[str, str]]) -> list[list[dict[str, str]]]:
    parent = {tweet_id: tweet_id for tweet_id in rows}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for tweet_id, row in rows.items():
        for reference in _references(row):
            if reference in rows:
                union(tweet_id, reference)
    grouped: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for tweet_id, row in rows.items():
        grouped[find(tweet_id)].append(row)
    return list(grouped.values())


def _case(component: list[dict[str, str]]) -> dict:
    messages = [
        {
            "tweet_id": row["tweet_id"],
            "role": "support" if row["author_id"] == "AppleSupport" else "customer",
            "text": row["text"],
            "timestamp": row["created_at"],
        }
        for row in sorted(component, key=lambda item: (item["created_at"], item["tweet_id"]))
    ]
    ids = sorted(message["tweet_id"] for message in messages)
    case_id = "case_" + hashlib.sha1("|".join(ids).encode("utf-8")).hexdigest()[:16]
    customer_messages = [message["text"] for message in messages if message["role"] == "customer"]
    support_messages = [message["text"] for message in messages if message["role"] == "support"]
    return {
        "case_id": case_id,
        "customer_message": "\n".join(customer_messages),
        "support_response": "\n".join(support_messages),
        "resolution": None,
        "conversation": messages,
        "metadata": {
            "message_count": len(messages),
            "customer_message_count": len(customer_messages),
            "support_message_count": len(support_messages),
            "complete_customer_support_pair": bool(customer_messages and support_messages),
        },
    }


def prepare(path: Path = RAW_PATH, output_dir: Path = PROCESSED_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    started_at = time.perf_counter()
    LOGGER.info("Loading dataset relationships from %s", path)
    index_started_at = time.perf_counter()
    apple_ids, parent_by_tweet, children_by_parent, filtering_time = _build_relationship_index(path)
    index_time = time.perf_counter() - index_started_at
    LOGGER.info(
        "Indexed %d AppleSupport seeds and %d parent links in %.2fs",
        len(apple_ids),
        len(parent_by_tweet),
        index_time,
    )

    reconstruction_started_at = time.perf_counter()
    connected_ids = _connected_ids(apple_ids, parent_by_tweet, children_by_parent)
    rows = _read_rows(path, connected_ids)
    reconstruction_time = time.perf_counter() - reconstruction_started_at
    LOGGER.info("Reconstructed graph closure with %d rows in %.2fs", len(rows), reconstruction_time)
    if len(rows) != len(connected_ids):
        missing = len(connected_ids) - len(rows)
        LOGGER.warning("%d relationship targets were not present as rows in the CSV", missing)

    cases = [_case(component) for component in _components(rows)]
    cases = [case for case in cases if case["metadata"]["complete_customer_support_pair"]]
    component_time = time.perf_counter() - reconstruction_started_at - reconstruction_time
    output_dir.mkdir(parents=True, exist_ok=True)
    writing_started_at = time.perf_counter()
    cases_path = output_dir / "apple_cases.jsonl"
    with cases_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")

    customer_messages = [case["customer_message"] for case in cases]
    analysis = {
        "source": str(path),
        "apple_support_rows": len(apple_ids),
        "linked_rows": len(rows),
        "conversations_with_customer_and_support": len(cases),
        "conversation_length": Counter(len(case["conversation"]) for case in cases),
        "inbound_customer_messages": sum(case["metadata"]["customer_message_count"] for case in cases),
        "outbound_support_messages": sum(case["metadata"]["support_message_count"] for case in cases),
        "customer_message_characters": sum(len(message) for message in customer_messages),
        "timings_seconds": {
            "dataset_loading": round(index_time - filtering_time, 3),
            "apple_support_filtering": round(filtering_time, 3),
            "relationship_index": round(index_time, 3),
            "conversation_reconstruction": round(reconstruction_time + component_time, 3),
            "output_writing": 0.0,
            "total": 0.0,
        },
    }
    analysis["conversation_length"] = dict(sorted(analysis["conversation_length"].items()))
    analysis["timings_seconds"]["output_writing"] = round(time.perf_counter() - writing_started_at, 3)
    analysis["timings_seconds"]["total"] = round(time.perf_counter() - started_at, 3)
    (output_dir / "analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    LOGGER.info("Wrote %d cases to %s in %.2fs total", len(cases), cases_path, analysis["timings_seconds"]["total"])
    return analysis


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=RAW_PATH)
    parser.add_argument("--output", type=Path, default=PROCESSED_PATH)
    args = parser.parse_args()
    print(json.dumps(prepare(args.input, args.output), indent=2))