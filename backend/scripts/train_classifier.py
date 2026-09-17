"""Train the production classifier only from actual human labels."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.agent.classifier import SentenceTransformerClassifier
from backend.agent.config import settings

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TRAINING = PROCESSED / "training_data.jsonl"
ARTIFACT = ROOT / "artifacts" / "classifier" / "model.pkl"
METADATA = ROOT / "artifacts" / "classifier" / "metadata.json"


def main() -> int:
    if not TRAINING.exists():
        print("Training data unavailable. Run prepare_training_data.py after human labels are supplied.")
        return 0
    records = [json.loads(line) for line in TRAINING.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records or len({record["intent"] for record in records}) < 2:
        print("Training blocked: at least two human-labeled intents are required.")
        return 0
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(settings.embedding_model)
    embeddings = encoder.encode([record["text"] for record in records], normalize_embeddings=True)
    classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
    classifier.fit(embeddings, [record["intent"] for record in records])
    model = SentenceTransformerClassifier(settings.embedding_model, classifier)
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    with ARTIFACT.open("wb") as handle:
        pickle.dump(model, handle)
    METADATA.write_text(json.dumps({"method": "sentence_transformer_logistic_regression", "embedding_model": settings.embedding_model, "records": len(records), "warning": "Human labels only."}, indent=2) + "\n", encoding="utf-8")
    print(f"Classifier artifact written: {ARTIFACT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
