from __future__ import annotations

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score


def classification_metrics(expected: list[str], predicted: list[str], labels: list[str]) -> dict:
    return {
        "accuracy": float(accuracy_score(expected, predicted)),
        "macro_precision": float(precision_score(expected, predicted, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(expected, predicted, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(expected, predicted, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(expected, predicted, labels=labels, average="weighted", zero_division=0)),
        "per_class": classification_report(expected, predicted, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(expected, predicted, labels=labels).tolist(),
        "labels": labels,
    }
