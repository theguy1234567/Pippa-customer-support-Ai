from backend.agent.escalation import decide
from backend.agent.models import IntentResult


def test_low_confidence_escalates():
    result = decide(IntentResult(name='general_support_request', confidence=0.1, is_actionable=True), [], grounded=False)
    assert result.action == 'HUMAN_ESCALATION'
    assert 'LOW_INTENT_CONFIDENCE' in result.reason_codes
