"""End-to-end support-agent orchestration."""

from __future__ import annotations

import logging

from .classifier import IntentClassifier
from .escalation import decide
from .generator import generate_response
from .models import SupportRequest, SupportResponse
from .retriever import HistoricalRetriever

LOGGER = logging.getLogger(__name__)


class SupportAgent:
    def __init__(self, classifier: IntentClassifier | None = None, retriever: HistoricalRetriever | None = None):
        self.classifier = classifier or IntentClassifier()
        self.retriever = retriever or HistoricalRetriever()

    def run(self, request: SupportRequest) -> SupportResponse:
        LOGGER.info("support request received")
        intent = self.classifier.predict(request.message)
        LOGGER.info("intent prediction: %s", intent.name)
        evidence = self.retriever.retrieve(request.message, top_k=5)
        LOGGER.info("retrieval completed: %d evidence items", len(evidence))
        generated = generate_response(request.message, intent.name, intent.confidence, evidence, False)
        decision = decide(intent, evidence, generated["grounded"])
        if decision.action == "HUMAN_ESCALATION" and generated["grounded"]:
            generated = generate_response(request.message, intent.name, intent.confidence, evidence, True)
        LOGGER.info("escalation decision: %s", decision.action)
        LOGGER.info("generation completed: %s", generated["generation_method"])
        return SupportResponse(
            message=request.message,
            intent=intent,
            evidence=generated["evidence"],
            response=generated["reply"],
            reply=generated["reply"],
            generation_error=generated.get("error"),
            decision=decision,
            grounded=generated["grounded"],
            retrieval_method=self.retriever.method,
            generation_method=generated["generation_method"],
        )


def run_agent(message: str, agent: SupportAgent | None = None) -> SupportResponse:
    return (agent or SupportAgent()).run(SupportRequest(message=message))
