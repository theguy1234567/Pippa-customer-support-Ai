"""End-to-end support-agent orchestration with conversation-aware grounding."""

from __future__ import annotations

import logging

from .classifier import IntentClassifier
from .conversation import build_contextual_query
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
        contextual_query = build_contextual_query(request.message, request.conversation)
        intent = self.classifier.predict(contextual_query)
        evidence = self.retriever.retrieve(contextual_query, top_k=10)
        generated = generate_response(
            current_message=request.message,
            contextual_query=contextual_query,
            intent=intent.name,
            confidence=intent.confidence,
            evidence=evidence,
        )
        decision = decide(intent, generated["evidence"], generated["grounded"])

        # Customer-facing safety contract: escalations never leak historical replies.
        if decision.action == "HUMAN_ESCALATION" or not generated["grounded"]:
            generated["reply"] = generated["safe_reply"]
            generated["grounded"] = False

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
            context_query=contextual_query,
        )


def run_agent(message: str, conversation=None, agent: SupportAgent | None = None) -> SupportResponse:
    return (agent or SupportAgent()).run(SupportRequest(message=message, conversation=conversation or []))
