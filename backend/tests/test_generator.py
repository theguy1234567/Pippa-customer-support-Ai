from backend.agent.generator import SAFE_ESCALATION, generate_response


def test_generator_without_evidence_is_not_grounded():
    result = generate_response('unknown issue', 'non_actionable_other', 0.2, [])
    assert result['grounded'] is False
    assert result['reply'] == SAFE_ESCALATION
