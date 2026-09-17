"""Deterministic escalation policy; thresholds are initial configurable baselines."""

from __future__ import annotations

from .config import settings
from .models import DecisionResult, EvidenceItem, IntentResult


def decide(intent: IntentResult, evidence: list[EvidenceItem], grounded: bool = False) -> DecisionResult:
    codes: list[str] = []
    strongest = max((item.similarity for item in evidence), default=0.0)
    if not intent.is_actionable:
        codes.append("NON_ACTIONABLE")
    if intent.confidence < settings.classifier_confidence_threshold:
        codes.append("LOW_INTENT_CONFIDENCE")
    else:
        codes.append("HIGH_INTENT_CONFIDENCE")
    if strongest < settings.retrieval_similarity_threshold:
        codes.append("LOW_RETRIEVAL_SIMILARITY")
    elif evidence:
        codes.append("STRONG_HISTORICAL_MATCH")
    if not evidence or not any(item.support_response.strip() for item in evidence):
        codes.append("INSUFFICIENT_EVIDENCE")
    if not grounded:
        codes.append("GENERATION_NOT_GROUNDED")
    auto = bool(intent.is_actionable and intent.confidence >= settings.classifier_confidence_threshold and strongest >= settings.retrieval_similarity_threshold and grounded)
    if auto:
        return DecisionResult(action="AUTO_HANDLE", confidence=min(intent.confidence, strongest), reason="Initial deterministic thresholds found a confident intent, sufficiently similar historical evidence, and a grounded response.", reason_codes=codes)
    return DecisionResult(action="HUMAN_ESCALATION", confidence=max(0.0, min(1.0, 1.0 - min(intent.confidence, strongest))), reason="The initial deterministic policy found insufficient confidence, evidence, or groundedness for automatic handling.", reason_codes=codes)
