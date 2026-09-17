import json
from pathlib import Path


def test_taxonomy_shape():
    data = json.loads(Path('backend/data/processed/final_taxonomy.json').read_text(encoding='utf-8'))
    names = [item['intent_name'] for item in data['intents']]
    assert len(names) == 9
    assert len(names) == len(set(names))
    assert data['multi_intent_policy']['ordered_rules']
    assert data['non_actionable_policy']['intent_name'] == 'non_actionable_other'
