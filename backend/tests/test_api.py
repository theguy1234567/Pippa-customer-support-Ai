import json
from pathlib import Path

from fastapi.testclient import TestClient

from backend.main import app


def test_api_health_and_response():
    line = next(Path('backend/data/processed/apple_cases.jsonl').open(encoding='utf-8'))
    case = json.loads(line)
    message = next(item['text'] for item in case['conversation'] if item.get('role') == 'customer')
    with TestClient(app) as client:
        assert client.get('/health').status_code == 200
        response = client.post('/api/support/respond', json={'message': message})
        assert response.status_code == 200
        assert response.json()['evidence'][0]['source'] == 'twcs'
        assert client.post('/api/support/respond', json={'message': ''}).status_code == 422
