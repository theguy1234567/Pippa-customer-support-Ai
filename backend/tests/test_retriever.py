import json
from pathlib import Path

from backend.agent.retriever import HistoricalRetriever


def test_retriever_uses_real_corpus():
    message = None
    for line in Path('backend/data/processed/apple_cases.jsonl').open(encoding='utf-8'):
        case = json.loads(line)
        message = next((item['text'] for item in case['conversation'] if item.get('role') == 'customer'), None)
        if message:
            break
    results = HistoricalRetriever().retrieve(message, top_k=2)
    assert len(results) <= 2
    assert all(item.source == 'twcs' and item.tweet_id and item.conversation_id for item in results)
