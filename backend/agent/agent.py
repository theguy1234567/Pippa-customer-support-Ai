"""End-to-end conversational support pipeline."""

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
        # Context resolution is deliberately separate from retrieval and generation.
        # Only customer turns are allowed to influence the contextual query.
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
        grounded = bool(generated["grounded"] and decision.action == "AUTO_HANDLE")
        reply = generated["reply"] if grounded else generated["safe_reply"]

        return SupportResponse(
            message=request.message,
            intent=intent,
            evidence=generated["evidence"],
            response=reply,
            reply=reply,
            generation_error=generated.get("error"),
            decision=decision,
            grounded=grounded,
            retrieval_method=self.retriever.method,
            generation_method=generated["generation_method"] if grounded else "safe_escalation",
            context_query=contextual_query,
        )


def run_agent(message: str, conversation=None, agent: SupportAgent | None = None) -> SupportResponse:
    return (agent or SupportAgent()).run(SupportRequest(message=message, conversation=conversation or []))
