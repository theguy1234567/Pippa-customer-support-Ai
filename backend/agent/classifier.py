"""Intent prediction with an optional trained model and deterministic fallback."""

from __future__ import annotations

import html
import pickle
import re
from pathlib import Path
from typing import Any

from .config import settings
from .models import IntentResult

TAXONOMY = [
    "ios_update_issue", "keyboard_text_input_issue", "app_store_app_issue",
    "device_hardware_issue", "device_functionality_issue", "apple_service_issue",
    "support_dm_request", "general_support_request", "non_actionable_other",
]
_RULES = {
    "keyboard_text_input_issue": re.compile(r"\b(keyboard|typing|typed|autocorrect|spelling|letter|question mark|symbol)\b", re.I),
    "app_store_app_issue": re.compile(r"\b(app store|appstore|apps?|application)\b", re.I),
    "device_hardware_issue": re.compile(r"\b(battery|charging|charger|screen|display|camera|speaker|microphone|home button|hardware)\b", re.I),
    "apple_service_issue": re.compile(r"\b(apple id|icloud|itunes|apple music|apple pay|facetime|imessage|apple watch)\b", re.I),
    "ios_update_issue": re.compile(r"\b(ios\s*[-.]?\s*\d|ios\b|update\w*|upgrad\w*|firmware)\b", re.I),
    "support_dm_request": re.compile(r"\b(dm|direct message|private message|send .*message|message me)\b", re.I),
    "device_functionality_issue": re.compile(r"\b(not working|isn't working|doesn't work|won't work|cannot|can't|crash\w*|freez\w*|stuck|slow|bug\w*|glitch\w*|problem|issue)\b", re.I),
    "general_support_request": re.compile(r"\b(help|support|fix\w*|problem|issue|trouble|question|need|why|please|wrong)\b", re.I),
}
_ACK = re.compile(r"^(thanks?|thank you|done|okay|ok|worked|fixed|yes|no|both|will do)[!.?,\s\w'’😊😘👍🏽🤗]*$", re.I)
_DEVICE = re.compile(r"\b(iphone|ipad|phone|device|macbook|apple watch|watch|apple tv)\b", re.I)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


class IntentClassifier:
    def __init__(self, artifact_path: Path | None = None, taxonomy_path: Path | None = None):
        self.artifact_path = artifact_path or settings.classifier_artifact
        self.taxonomy_path = taxonomy_path or settings.taxonomy_path
        self.model: Any = None
        self.loaded = False
        if self.artifact_path.exists():
            with self.artifact_path.open("rb") as handle:
                self.model = pickle.load(handle)
            self.loaded = True

    @property
    def method(self) -> str:
        return "sentence_transformer_logistic_regression" if self.loaded else "deterministic_taxonomy_fallback"

    def _fallback(self, text: str) -> IntentResult:
        normalized = normalize_text(text)
        if not normalized:
            raise ValueError("message must not be empty")
        if _ACK.fullmatch(normalized) or (_DEVICE.search(normalized) and not any(rule.search(normalized) for rule in _RULES.values())):
            return IntentResult(name="non_actionable_other", confidence=0.98, probabilities={"non_actionable_other": 0.98}, is_actionable=False, method=self.method)
        specific = [name for name in ("keyboard_text_input_issue", "app_store_app_issue", "device_hardware_issue", "apple_service_issue") if _RULES[name].search(normalized)]
        update = bool(_RULES["ios_update_issue"].search(normalized))
        functionality = bool(_RULES["device_functionality_issue"].search(normalized))
        dm = bool(_RULES["support_dm_request"].search(normalized))
        general = bool(_RULES["general_support_request"].search(normalized))
        if len(specific) == 1:
            name = specific[0]
        elif len(specific) > 1:
            name = specific[0]
        elif update and re.search(r"\b(can't|cannot|won't|fail|problem|issue|after|since|slow|broken|work)\b", normalized, re.I):
            name = "ios_update_issue"
        elif functionality:
            name = "device_functionality_issue"
        elif dm:
            name = "support_dm_request"
        elif general and len(normalized) >= 20:
            name = "general_support_request"
        else:
            name = "non_actionable_other"
        confidence = 0.82 if name != "non_actionable_other" else 0.72
        if len(specific) > 1:
            confidence = 0.52
        return IntentResult(name=name, confidence=confidence, probabilities={name: confidence}, is_actionable=name != "non_actionable_other", method=self.method)

    def predict(self, text: str) -> IntentResult:
        if not text or not text.strip():
            raise ValueError("message must not be empty")
        if not self.loaded:
            return self._fallback(text)
        normalized = normalize_text(text)
        prediction = self.model.predict([normalized])[0]
        probabilities = self.model.predict_proba([normalized])[0]
        probability_map = {str(label): float(value) for label, value in zip(self.model.classes_, probabilities)}
        confidence = max(probability_map.values())
        return IntentResult(name=str(prediction), confidence=confidence, probabilities=probability_map, is_actionable=str(prediction) != "non_actionable_other", method=self.method)

    def predict_batch(self, texts: list[str]) -> list[IntentResult]:
        return [self.predict(text) for text in texts]

    def health(self) -> bool:
        return self.loaded


class SentenceTransformerClassifier:
    """Persistable embedding classifier used when human labels are available."""

    def __init__(self, embedding_model: str, classifier: Any):
        from sentence_transformers import SentenceTransformer

        self.encoder = SentenceTransformer(embedding_model)
        self.classifier = classifier

    def _encode(self, texts: list[str]):
        return self.encoder.encode([normalize_text(text) for text in texts], normalize_embeddings=True)

    def predict(self, texts: list[str]):
        return self.classifier.predict(self._encode(texts))

    def predict_proba(self, texts: list[str]):
        return self.classifier.predict_proba(self._encode(texts))
