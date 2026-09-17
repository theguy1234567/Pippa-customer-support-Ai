"""Two-stage historical retrieval with generalized problem-domain reranking."""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any

from .config import settings
from .models import EvidenceItem

GENERIC = {"apple", "iphone", "ipad", "phone", "device", "ios", "issue", "problem", "help", "please", "support", "work", "working", "thing", "things", "one", "just", "really", "need", "want", "my"}
DOMAIN_TERMS = {
    "wifi": {"wifi", "wi-fi", "wireless", "network", "router", "internet", "connect", "connection", "disconnect", "connected"},
    "bluetooth": {"bluetooth", "airpods", "headphones", "earbuds", "pair", "pairing", "disconnect"},
    "power": {"battery", "charge", "charging", "charger", "power", "drain", "draining"},
    "screen": {"screen", "display", "touch", "touchscreen", "unresponsive"},
    "keyboard": {"keyboard", "typing", "type", "autocorrect", "key", "keys", "letter", "symbol"},
    "app_store": {"app", "apps", "appstore", "store", "download", "install", "purchase"},
    "safari": {"safari", "browser", "webpage", "website"},
    "update": {"update", "updating", "upgrade", "upgrading", "firmware", "ios"},
    "apple_service": {"icloud", "appleid", "music", "facetime", "imessage", "itunes", "pay"},
}
GENERIC_REPLY_PATTERNS = (r"^we'?re here to help\b", r"^we want to help\b", r"^let'?s look into", r"\bdm us\b", r"\bwhich software version\b", r"\bwhat version\b")


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower().replace("wi-fi", "wifi")))


def _meaningful_tokens(text: str) -> set[str]:
    return {token for token in _tokens(text) if token not in GENERIC and len(token) > 2}


def _domains(text: str) -> set[str]:
    tokens = _tokens(text)
    return {domain for domain, terms in DOMAIN_TERMS.items() if tokens & terms}


def _generic_response(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    return any(re.search(pattern, normalized, re.I) for pattern in GENERIC_REPLY_PATTERNS)


def _domain_compatibility(query: str, customer: str, response: str) -> tuple[float, str | None]:
    q_domains = _domains(query)
    c_domains = _domains(customer)
    r_domains = _domains(response)
    if not q_domains:
        return 0.5, None
    if not (q_domains & c_domains):
        return 0.0, "historical customer problem is from a different domain"
    if r_domains and not (r_domains & q_domains):
        return 0.0, "historical support response addresses a different domain"
    return min(1.0, 0.55 + 0.2 * len(q_domains & c_domains)), None


def _keyword_relevance(query: str, customer: str, response: str) -> float:
    q_tokens = _meaningful_tokens(query)
    if not q_tokens:
        return 0.0
    c_tokens = _meaningful_tokens(customer)
    r_tokens = _meaningful_tokens(response)
    customer_overlap = len(q_tokens & c_tokens) / len(q_tokens)
    response_overlap = len(q_tokens & r_tokens) / len(q_tokens)
    return min(1.0, 0.75 * customer_overlap + 0.25 * response_overlap)


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
                for index, message in enumerate(messages):
                    if message.get("role") != "customer":
                        continue
                    next_support = next((m for m in messages[index + 1:] if m.get("role") == "support"), None)
                    if not next_support:
                        continue
                    self.records.append({
                        "tweet_id": str(message["tweet_id"]), "conversation_id": str(case["case_id"]), "case_id": str(case["case_id"]),
                        "role": "customer", "text": message["text"], "created_at": message.get("timestamp"),
                        "support_response": next_support["text"], "support_tweet_id": str(next_support["tweet_id"]), "source": "twcs",
                    })

    @property
    def health(self) -> bool:
        return bool(self.records)

    @property
    def method(self) -> str:
        return "persisted_tfidf_reranked" if self.loaded else "deterministic_reranked"

    def retrieve(self, query: str, top_k: int = 5, exclude_tweet_id: str | None = None) -> list[EvidenceItem]:
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        candidate_count = min(len(self.records), max(80, top_k * 12))
        if self.loaded and self.vectorizer is not None and self.index is not None:
            query_vector = self.vectorizer.transform([query])
            similarities = (self.index @ query_vector.T).toarray().ravel()
            ranked_indices = sorted(range(len(self.records)), key=lambda i: (float(similarities[i]), self.records[i]["tweet_id"]), reverse=True)[:candidate_count]
            candidates = [(float(similarities[i]), self.records[i]) for i in ranked_indices]
        else:
            query_tokens = _meaningful_tokens(query)
            scored = []
            for record in self.records:
                if record["tweet_id"] == exclude_tweet_id:
                    continue
                tokens = _meaningful_tokens(record["text"])
                union = query_tokens | tokens
                score = len(query_tokens & tokens) / len(union) if union else 0.0
                scored.append((score, record))
            candidates = sorted(scored, key=lambda x: (x[0], x[1]["tweet_id"]), reverse=True)[:candidate_count]

        q_tokens = _meaningful_tokens(query)
        accepted: list[tuple[float, dict[str, Any]]] = []
        seen_cases: set[str] = set()
        q_domains = _domains(query)
        for semantic, record in candidates:
            if record["tweet_id"] == exclude_tweet_id or record["conversation_id"] in seen_cases:
                continue
            response = record.get("support_response", "").strip()
            if not response:
                continue
            overlap = len(q_tokens & _meaningful_tokens(record["text"])) / max(1, len(q_tokens))
            keyword_score = _keyword_relevance(query, record["text"], response)
            domain_score, rejection = _domain_compatibility(query, record["text"], response)
            if _generic_response(response) and (overlap < 0.25 or domain_score <= 0):
                rejection = rejection or "generic historical support response is not sufficiently specific"
            # Prefer exact problem vocabulary over generic TF-IDF similarity.
            final_score = 0.40 * max(0.0, semantic) + 0.35 * keyword_score + 0.25 * domain_score
            if q_domains and domain_score > 0:
                final_score += 0.15
            if rejection or domain_score <= 0 or final_score < 0.30:
                continue
            accepted.append((min(1.0, final_score), record))
            seen_cases.add(record["conversation_id"])

        accepted.sort(key=lambda x: (x[0], x[1]["tweet_id"]), reverse=True)
        return [EvidenceItem(
            case_id=record["case_id"], tweet_id=record["tweet_id"], conversation_id=record["conversation_id"], role="customer",
            timestamp=record.get("created_at"), similarity=max(-1.0, min(1.0, float(score))), customer_message=record["text"],
            support_response=record.get("support_response", ""), resolution=None, source="twcs", relevance_score=float(score), accepted=True,
        ) for score, record in accepted[:top_k]]
