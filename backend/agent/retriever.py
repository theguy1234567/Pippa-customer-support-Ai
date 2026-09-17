"""Robust historical retrieval for the AppleSupport corpus."""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any

from .config import settings
from .models import EvidenceItem

STOP = {
    "the", "a", "an", "and", "or", "to", "of", "is", "am", "are", "my", "i", "it", "this",
    "that", "on", "in", "for", "with", "be", "me", "please", "can", "could", "would", "do", "does",
    "not", "no", "im", "ive", "you", "your", "we", "us", "help", "support", "issue", "problem",
}
DOMAIN_TERMS = {
    "wifi": {"wifi", "wi-fi", "wireless", "network", "router", "internet", "connect", "connecting", "connection", "disconnect", "disconnected", "connected", "online", "offline"},
    "bluetooth": {"bluetooth", "airpods", "headphones", "earbuds", "pair", "pairing", "disconnect", "disconnected"},
    "power": {"battery", "charge", "charging", "charger", "power", "drain", "draining"},
    "screen": {"screen", "display", "touch", "touchscreen", "unresponsive"},
    "keyboard": {"keyboard", "typing", "typed", "type", "autocorrect", "key", "keys", "letter", "symbol", "text"},
    "app_store": {"app", "apps", "appstore", "store", "download", "install", "purchase"},
    "safari": {"safari", "browser", "webpage", "website"},
    "update": {"update", "updating", "upgrade", "upgrading", "firmware", "ios"},
    "apple_service": {"icloud", "appleid", "music", "facetime", "imessage", "itunes", "pay"},
}
SYNONYMS = {
    "connecting": "connect", "connected": "connect", "connection": "connect", "disconnecting": "disconnect", "disconnected": "disconnect",
    "charging": "charge", "charged": "charge", "draining": "drain", "typing": "type", "typed": "type", "updating": "update", "upgraded": "upgrade",
}
GENERIC_PATTERNS = (
    r"\bdm us\b", r"\bwe'?re here to help\b", r"\bwe want to help\b",
    r"\bwhich software version\b", r"\bwhat version\b", r"^please .*\b(?:dm|message)\b",
)


def _tokens(text: str) -> set[str]:
    text = text.lower().replace("wi-fi", "wifi")
    raw = {t for t in re.findall(r"[a-z0-9']+", text) if len(t) > 2 and t not in STOP}
    return {SYNONYMS.get(t, t) for t in raw}


def _domains(text: str) -> set[str]:
    tokens = _tokens(text)
    return {domain for domain, terms in DOMAIN_TERMS.items() if tokens & {SYNONYMS.get(t, t) for t in terms}}


def _generic(text: str) -> bool:
    value = " ".join(text.lower().split())
    return any(re.search(pattern, value, re.I) for pattern in GENERIC_PATTERNS)


def _overlap(a: set[str], b: set[str]) -> float:
    return len(a & b) / max(1, len(a))


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
                for idx, message in enumerate(messages):
                    if message.get("role") != "customer":
                        continue
                    support = next((m for m in messages[idx + 1:] if m.get("role") == "support"), None)
                    if not support:
                        continue
                    self.records.append({
                        "tweet_id": str(message["tweet_id"]), "conversation_id": str(case["case_id"]), "case_id": str(case["case_id"]),
                        "role": "customer", "text": message["text"], "created_at": message.get("timestamp"),
                        "support_response": support["text"], "support_tweet_id": str(support["tweet_id"]), "source": "twcs",
                    })

    @property
    def health(self) -> bool:
        return bool(self.records)

    @property
    def method(self) -> str:
        return "persisted_tfidf_hybrid" if self.loaded else "deterministic_hybrid"

    def _semantic_candidates(self, query: str, limit: int) -> list[tuple[float, dict[str, Any]]]:
        if self.loaded and self.vectorizer is not None and self.index is not None:
            vector = self.vectorizer.transform([query])
            sims = (self.index @ vector.T).toarray().ravel()
            order = sorted(range(len(self.records)), key=lambda i: (float(sims[i]), self.records[i]["tweet_id"]), reverse=True)[:limit]
            return [(float(sims[i]), self.records[i]) for i in order]
        q = _tokens(query)
        scored = [(_overlap(q, _tokens(record["text"])), record) for record in self.records]
        return sorted(scored, key=lambda x: (x[0], x[1]["tweet_id"]), reverse=True)[:limit]

    def retrieve(self, query: str, top_k: int = 5, exclude_tweet_id: str | None = None) -> list[EvidenceItem]:
        if not query.strip():
            raise ValueError("query must not be empty")
        q_tokens = _tokens(query)
        q_domains = _domains(query)
        candidate_count = min(len(self.records), max(300, top_k * 60))
        candidates = self._semantic_candidates(query, candidate_count)
        ranked: list[tuple[float, dict[str, Any]]] = []
        seen_cases: set[str] = set()

        for semantic, record in candidates:
            if record["tweet_id"] == exclude_tweet_id:
                continue
            customer = record["text"]
            response = record.get("support_response", "").strip()
            if not response:
                continue
            c_tokens = _tokens(customer)
            r_tokens = _tokens(response)
            c_overlap = _overlap(q_tokens, c_tokens)
            r_overlap = _overlap(q_tokens, r_tokens)
            c_domains = _domains(customer)
            r_domains = _domains(response)
            domain_match = len(q_domains & c_domains) / max(1, len(q_domains)) if q_domains else 0.5
            response_domain_match = 1.0 if not q_domains or not r_domains or (q_domains & r_domains) else 0.0
            generic = _generic(response)

            # A query domain must be present in the historical customer case. This prevents
            # semantically similar but operationally unrelated replies from leaking through.
            if q_domains and not (q_domains & c_domains):
                continue
            if q_domains and r_domains and not (q_domains & r_domains):
                continue
            if generic and "dm us" in response.lower():
                continue
            if generic and q_domains and not (q_domains & r_domains):
                continue

            score = (
                0.42 * max(0.0, semantic)
                + 0.25 * c_overlap
                + 0.13 * r_overlap
                + 0.15 * domain_match
                + 0.05 * response_domain_match
            )
            if q_domains and domain_match > 0:
                score += 0.10
            elif not q_domains and c_overlap >= 0.25:
                score += 0.05
            if generic:
                score -= 0.08

            # Domain matches need only modest lexical overlap because short customer messages
            # such as "internet not working" are common in support data.
            acceptable = (
                (q_domains and domain_match > 0 and (c_overlap >= 0.08 or semantic >= 0.12))
                or (not q_domains and (c_overlap >= 0.20 or semantic >= 0.24))
            )
            if not acceptable or score < 0.18:
                continue
            if record["conversation_id"] in seen_cases:
                continue
            ranked.append((min(1.0, score), record))
            seen_cases.add(record["conversation_id"])

        ranked.sort(key=lambda x: (x[0], x[1]["tweet_id"]), reverse=True)
        return [EvidenceItem(
            case_id=record["case_id"], tweet_id=record["tweet_id"], conversation_id=record["conversation_id"], role="customer",
            timestamp=record.get("created_at"), similarity=max(-1.0, min(1.0, float(score))),
            customer_message=record["text"], support_response=record.get("support_response", ""), resolution=None,
            source="twcs", relevance_score=float(score), accepted=True,
        ) for score, record in ranked[:top_k]]
