from backend.evaluation.agreement import calculate_agreement
from backend.evaluation.baselines import majority_predict
from backend.evaluation.metrics import classification_metrics
from backend.evaluation.misleading_metrics import analyze_accuracy_limitations


def test_baseline_and_metrics():
    expected = ['a', 'b', 'a']
    predicted = majority_predict(['a', 'a', 'b'], expected)
    metrics = classification_metrics(expected, predicted, ['a', 'b'])
    assert metrics['accuracy'] == 2 / 3
    assert len(metrics['confusion_matrix']) == 2


def test_pending_agreement_and_distribution():
    assert calculate_agreement([], [])['status'] == 'PENDING_HUMAN_RATINGS'
    assert analyze_accuracy_limitations([])['status'] == 'PENDING_GOLD_LABELS'
