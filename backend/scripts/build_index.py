"""Build a persisted TF-IDF retrieval artifact from real AppleSupport cases.

FAISS is preferred when installed; this lightweight artifact remains usable in
minimal environments and preserves the same real-data metadata contract.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "data" / "processed" / "apple_cases.jsonl"
ARTIFACT = ROOT / "artifacts" / "retrieval" / "index.pkl"
METADATA = ROOT / "artifacts" / "retrieval" / "metadata.json"
FAISS_PATH = ROOT / "artifacts" / "retrieval" / "apple_support.faiss"


def main() -> int:
    records = []
    with CASES.open(encoding="utf-8") as handle:
        for line in handle:
            case = json.loads(line)
            customer = [message for message in case.get("conversation", []) if message.get("role") == "customer"]
            support = [message for message in case.get("conversation", []) if message.get("role") == "support"]
            for message in customer:
                records.append({"tweet_id": str(message["tweet_id"]), "conversation_id": str(case["case_id"]), "case_id": str(case["case_id"]), "role": "customer", "text": message["text"], "created_at": message.get("timestamp"), "support_response": support[0]["text"] if support else "", "support_tweet_id": str(support[0]["tweet_id"]) if support else None, "source": "twcs"})
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True)
    matrix = vectorizer.fit_transform([record["text"] for record in records])
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    with ARTIFACT.open("wb") as handle:
        pickle.dump({"vectorizer": vectorizer, "matrix": matrix}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    METADATA.write_text(json.dumps(records, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        import faiss

        dense = matrix.toarray().astype("float32")
        index = faiss.IndexFlatIP(dense.shape[1])
        index.add(dense)
        faiss.write_index(index, str(FAISS_PATH))
        print(f"FAISS index written: {FAISS_PATH}")
    except ImportError:
        print("FAISS unavailable; persisted TF-IDF fallback artifact written instead.")
    print(f"Retrieval artifact written with {len(records)} real customer messages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
