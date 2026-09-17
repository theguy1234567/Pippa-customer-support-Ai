from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from ..agent.agent import SupportAgent
from ..agent.generator import llm_available
from ..agent.models import SupportRequest, SupportResponse

LOGGER = logging.getLogger(__name__)
router = APIRouter()


def get_agent(request: Request) -> SupportAgent:
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        agent = SupportAgent()
        request.app.state.agent = agent
    return agent


@router.get("/health")
def health(request: Request) -> dict[str, bool | str]:
    agent = getattr(request.app.state, "agent", None)
    return {
        "status": "ok",
        "classifier_loaded": bool(agent),
        "retriever_loaded": bool(agent and agent.retriever.health),
        "llm_available": llm_available(),
    }


@router.post("/api/support/respond", response_model=SupportResponse)
def respond(payload: SupportRequest, request: Request) -> SupportResponse:
    if not payload.message.strip():
        raise HTTPException(status_code=422, detail="message must not be empty")
    try:
        return get_agent(request).run(payload)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        LOGGER.exception("support request failed")
        raise HTTPException(status_code=500, detail="support agent failed to process request") from error
