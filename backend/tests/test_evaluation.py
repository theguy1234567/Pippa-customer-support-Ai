import json
from pathlib import Path

from backend.evaluation.evaluate import run_evaluation


def test_missing_gold_labels_are_blocked():
    report = run_evaluation()
    assert report['status'] == 'BLOCKED'
    assert report['metrics'] is None
    assert 'gold labels unavailable' in report['message'].lower()
