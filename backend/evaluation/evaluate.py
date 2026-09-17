"""Run measured baselines and classifier evaluation when labels exist."""

from __future__ import annotations

import json
from pathlib import Path

from .agreement import calculate_agreement
from .baselines import majority_predict, tfidf_logistic
from .failure_analysis import analyze_failures
from .judge import judge_response
from .metrics import classification_metrics
from .misleading_metrics import analyze_accuracy_limitations

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts" / "evaluation"
GOLD = PROCESSED / "gold_review_candidates.json"
TRAINING = PROCESSED / "training_data.jsonl"
TAXONOMY = PROCESSED / "final_taxonomy.json"


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _human_records(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [record for record in data.get("records", []) if record.get("human_label")]


def run_evaluation() -> dict:
    taxonomy = json.loads(TAXONOMY.read_text(encoding="utf-8"))
    labels = [item["intent_name"] for item in taxonomy["intents"]]
    gold_all = json.loads(GOLD.read_text(encoding="utf-8"))
    gold = _human_records(GOLD)
    training = _records(TRAINING) if TRAINING.exists() else []
    contamination = not ({record["tweet_id"] for record in gold_all.get("records", [])} & {record["tweet_id"] for record in training})
    report: dict = {
        "status": "MEASURED" if gold and training else "BLOCKED",
        "message": "Measured evaluation completed." if gold and training else "Evaluation blocked: gold labels unavailable.",
        "gold_records": len(gold),
        "training_records": len(training),
        "contamination_check": "PASS" if contamination else "FAIL",
        "metrics": None,
        "models": {},
        "retrieval": {"status": "PENDING_HUMAN_RELEVANCE_LABELS"},
        "response_quality": {"status": "PENDING_HUMAN_OR_LLM_JUDGMENTS"},
        "agreement": calculate_agreement([], []),
        "failure_analysis": {"status": "PENDING_GOLD_LABELS"},
        "accuracy_analysis": analyze_accuracy_limitations([]),
    }
    if gold and training and contamination:
        train_texts = [record["text"] for record in training]
        train_labels = [record["intent"] for record in training]
        test_texts = [record["text"] for record in gold]
        expected = [record["human_label"] for record in gold]
        predictions = {
            "majority": majority_predict(train_labels, test_texts),
            "tfidf_logistic": tfidf_logistic(train_texts, train_labels, test_texts),
        }
        report["models"] = {name: classification_metrics(expected, predicted, labels) for name, predicted in predictions.items()}
        report["metrics"] = report["models"]
        measured_predictions = [{"tweet_id": record["tweet_id"], "input": record["text"], "expected": expected[index], "predicted": predictions["tfidf_logistic"][index], "failure_type": "false_intent_prediction"} for index, record in enumerate(gold)]
        report["failure_analysis"] = analyze_failures(measured_predictions)
        report["accuracy_analysis"] = analyze_accuracy_limitations(expected)
        report["response_quality"] = judge_response("", "", [])
        report["status"] = "MEASURED"
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "evaluation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown = ["# Evaluation Report", "", f"**Status:** {report['status']}", "", report["message"], "", f"**Gold records:** {report['gold_records']}", f"**Training records:** {report['training_records']}", f"**Contamination check:** {report['contamination_check']}"]
    if report["status"] == "BLOCKED":
        markdown.extend(["", "No classifier metrics were fabricated."])
    else:
        markdown.extend(["", "## Metrics", "", "```json", json.dumps(report["models"], indent=2), "```"])
    (ARTIFACTS / "evaluation_report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run_evaluation(), indent=2))
