"""Build a persisted TF-IDF artifact from real AppleSupport cases.

Each customer message is paired with the next AppleSupport message in the
same reconstructed conversation instead of always using the first response.
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
            messages = case.get("conversation", [])
            for index, message in enumerate(messages):
                if message.get("role") != "customer":
                    continue
                support = next((m for m in messages[index + 1:] if m.get("role") == "support"), None)
                if not support:
                    continue
                records.append({
                    "tweet_id": str(message["tweet_id"]),
                    "conversation_id": str(case["case_id"]),
                    "case_id": str(case["case_id"]),
                    "role": "customer",
                    "text": message["text"],
                    "created_at": message.get("timestamp"),
                    "support_response": support["text"],
                    "support_tweet_id": str(support["tweet_id"]),
                    "source": "twcs",
                })
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
    except ImportError:
        pass
    print(f"Retrieval artifact written with {len(records)} real customer messages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
