from typing import Literal

from pydantic import BaseModel, Field


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class SupportRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation: list[ConversationTurn] = Field(default_factory=list)


class IntentResult(BaseModel):
    name: str
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float] = Field(default_factory=dict)
    is_actionable: bool = True
    method: str = "unknown"


class EvidenceItem(BaseModel):
    case_id: str
    tweet_id: str
    conversation_id: str
    role: Literal["customer", "support"]
    timestamp: str | None = None
    similarity: float = Field(ge=-1.0, le=1.0)
    customer_message: str
    support_response: str
    resolution: str | None = None
    source: str = "twcs"
    relevance_score: float | None = None
    accepted: bool | None = None
    rejection_reason: str | None = None


class DecisionResult(BaseModel):
    action: Literal["AUTO_HANDLE", "HUMAN_ESCALATION"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    reason_codes: list[str] = Field(default_factory=list)


class SupportResponse(BaseModel):
    message: str
    intent: IntentResult
    evidence: list[EvidenceItem]
    response: str | None
    reply: str | None = None
    generation_error: str | None = None
    decision: DecisionResult
    grounded: bool = False
    retrieval_method: str = "none"
    generation_method: str = "none"
    context_query: str | None = None
