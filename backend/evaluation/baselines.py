from __future__ import annotations

from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def majority_predict(train_labels: list[str], texts: list[str]) -> list[str]:
    if not train_labels:
        raise ValueError("human training labels unavailable")
    majority = Counter(train_labels).most_common(1)[0][0]
    return [majority for _ in texts]


def tfidf_logistic(train_texts: list[str], train_labels: list[str], test_texts: list[str]) -> list[str]:
    if len(set(train_labels)) < 2:
        raise ValueError("at least two human-labeled classes are required")
    model = Pipeline([("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))), ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    model.fit(train_texts, train_labels)
    return [str(value) for value in model.predict(test_texts)]
