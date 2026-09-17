from backend.agent.conversation import build_contextual_query
from backend.agent.models import ConversationTurn
from backend.agent.generator import SAFE_ESCALATION
from backend.agent.retriever import HistoricalRetriever


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


def test_numeric_message_does_not_inherit_previous_problem():
    history = [ConversationTurn(role="user", content="My iPhone won't connect to Wi-Fi." )]
    assert build_contextual_query("91", history) == "91"


def test_safe_escalation_text_is_not_historical_evidence():
    assert "historical support response" not in SAFE_ESCALATION.lower()


def test_retriever_accepts_internet_as_wifi_domain():
    retriever = HistoricalRetriever()
    retriever.loaded = False
    retriever.records = [
        {
            "tweet_id": "1", "conversation_id": "c1", "case_id": "c1", "text": "My iPhone cannot connect to the internet at home",
            "support_response": "Let's make sure your iPhone stays connected to Wi-Fi. What happens when you try to connect?",
            "created_at": None,
        },
        {
            "tweet_id": "2", "conversation_id": "c2", "case_id": "c2", "text": "My iPhone battery drains very quickly",
            "support_response": "We can look into the battery issue.", "created_at": None,
        },
    ]
    evidence = retriever.retrieve("my iphone is not connecting to the internet", top_k=1)
    assert len(evidence) == 1
    assert evidence[0].conversation_id == "c1"


def test_retriever_rejects_generic_dm_reply():
    retriever = HistoricalRetriever()
    retriever.loaded = False
    retriever.records = [
        {
            "tweet_id": "1", "conversation_id": "c1", "case_id": "c1", "text": "My iPhone won't connect to Wi-Fi",
            "support_response": "We're here to help. DM us the country you're located in and we'll go from there.", "created_at": None,
        },
    ]
    assert retriever.retrieve("my iphone won't connect to wifi", top_k=3) == []
