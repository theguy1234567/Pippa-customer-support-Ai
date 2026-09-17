import pytest

from backend.agent.classifier import IntentClassifier, TAXONOMY


def test_classifier_fallback_output():
    result = IntentClassifier().predict("my keyboard is not working")
    assert result.name in TAXONOMY
    assert 0 <= result.confidence <= 1
    assert result.method


def test_classifier_rejects_empty():
    with pytest.raises(ValueError):
        IntentClassifier().predict(" ")
