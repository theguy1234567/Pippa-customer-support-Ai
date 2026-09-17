from backend.agent.conversation import build_contextual_query
from backend.agent.models import ConversationTurn
from backend.agent.generator import SAFE_ESCALATION


def test_follow_up_keeps_user_problem_and_ignores_assistant_text():
    history = [
        ConversationTurn(role="user", content="My iPhone won't connect to Wi-Fi at home."),
        ConversationTurn(role="assistant", content="Which software version are you using?"),
    ]
    query = build_contextual_query("What should I try first?", history)
    assert "won't connect to Wi-Fi" in query
    assert "software version" not in query


def test_explicit_new_problem_switches_topic():
    history = [ConversationTurn(role="user", content="My iPhone won't connect to Wi-Fi at home.")]
    query = build_contextual_query("Also, my App Store won't open.", history)
    assert query == "Also, my App Store won't open."
    assert "Wi-Fi" not in query


def test_network_reset_question_keeps_context():
    history = [ConversationTurn(role="user", content="My iPhone won't connect to Wi-Fi at home.")]
    query = build_contextual_query("Will resetting the network delete my photos?", history)
    assert "Wi-Fi" in query
    assert "resetting the network" in query


def test_safe_escalation_text_is_not_historical_evidence():
    assert "historical support response" not in SAFE_ESCALATION.lower()
