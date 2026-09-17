import json
from pathlib import Path


def test_gold_is_excluded_from_candidates():
    processed = Path('backend/data/processed')
    gold = json.loads((processed / 'gold_review_candidates.json').read_text(encoding='utf-8'))
    gold_ids = {record['tweet_id'] for record in gold['records']}
    candidate_ids = {json.loads(line)['tweet_id'] for line in (processed / 'label_candidates.jsonl').open(encoding='utf-8')}
    assert not gold_ids & candidate_ids
