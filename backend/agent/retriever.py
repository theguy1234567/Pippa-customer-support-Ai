"""Historical AppleSupport retrieval with persisted artifacts and safe fallback."""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np

from .config import settings
from .models import EvidenceItem


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


class HistoricalRetriever:
    def __init__(self, cases_path: Path | None = None, artifact_path: Path | None = None, metadata_path: Path | None = None):
        self.cases_path = cases_path or settings.cases_path
        self.artifact_path = artifact_path or settings.retrieval_artifact
        self.metadata_path = metadata_path or settings.retrieval_metadata
        self.records: list[dict[str, Any]] = []
        self.index: Any = None
        self.vectorizer: Any = None
        self.loaded = False
        if self.artifact_path.exists() and self.metadata_path.exists():
            with self.artifact_path.open("rb") as handle:
                artifact = pickle.load(handle)
            self.vectorizer = artifact.get("vectorizer")
            self.index = artifact.get("matrix")
            self.records = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            self.loaded = True
        elif self.cases_path.exists():
            self._load_runtime_corpus()

    def _load_runtime_corpus(self) -> None:
        with self.cases_path.open(encoding="utf-8") as handle:
            for line in handle:
                case = json.loads(line)
                messages = case.get("conversation", [])
                customer = [message for message in messages if message.get("role") == "customer"]
                support = [message for message in messages if message.get("role") == "support"]
                if not customer:
                    continue
                for message in customer:
                    self.records.append({
                        "tweet_id": str(message["tweet_id"]),
                        "conversation_id": str(case["case_id"]),
                        "case_id": str(case["case_id"]),
                        "role": "customer",
                        "text": message["text"],
                        "created_at": message.get("timestamp"),
                        "support_response": support[0]["text"] if support else "",
                        "support_tweet_id": str(support[0]["tweet_id"]) if support else None,
                        "source": "twcs",
                    })

    @property
    def health(self) -> bool:
        return bool(self.records)

    @property
    def method(self) -> str:
        return "persisted_tfidf" if self.loaded else "deterministic_token_overlap"

    def retrieve(self, query: str, top_k: int = 5, exclude_tweet_id: str | None = None) -> list[EvidenceItem]:
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        candidates = [record for record in self.records if record["tweet_id"] != exclude_tweet_id]
        if self.loaded and self.vectorizer is not None and self.index is not None:
            query_vector = self.vectorizer.transform([query])
            similarities = (self.index @ query_vector.T).toarray().ravel()
            ranked = sorted(zip(similarities, self.records), key=lambda value: (float(value[0]), value[1]["tweet_id"]), reverse=True)
            candidates = [(float(score), record) for score, record in ranked if record["tweet_id"] != exclude_tweet_id]
        else:
            candidates = [(None, record) for record in candidates]
        query_tokens = _tokens(query)
        scored = []
        for persisted_score, record in candidates:
            if persisted_score is not None:
                scored.append((persisted_score, record))
                continue
            overlap = query_tokens & _tokens(record["text"])
            union = query_tokens | _tokens(record["text"])
            score = len(overlap) / len(union) if union else 0.0
            scored.append((score, record))
        results: list[EvidenceItem] = []
        seen_cases: set[str] = set()
        for score, record in sorted(scored, key=lambda value: (value[0], value[1]["tweet_id"]), reverse=True):
            if record["conversation_id"] in seen_cases:
                continue
            seen_cases.add(record["conversation_id"])
            results.append(EvidenceItem(
                case_id=record["case_id"], tweet_id=record["tweet_id"], conversation_id=record["conversation_id"],
                role="customer", timestamp=record.get("created_at"), similarity=max(-1.0, min(1.0, float(score))),
                customer_message=record["text"], support_response=record.get("support_response", ""), resolution=None, source="twcs",
            ))
            if len(results) >= top_k:
                break
        return results
