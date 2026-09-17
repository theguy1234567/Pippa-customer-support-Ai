"""Repository-wide backend integrity and fallback smoke validation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
PROCESSED = ROOT / "data" / "processed"


def main() -> int:
    failures: list[str] = []
    required = [
        ROOT / "agent" / "classifier.py", ROOT / "agent" / "retriever.py", ROOT / "agent" / "generator.py",
        ROOT / "agent" / "escalation.py", ROOT / "agent" / "agent.py", ROOT / "api" / "routes.py",
        ROOT / "main.py", PROCESSED / "final_taxonomy.json", PROCESSED / "apple_cases.jsonl",
        PROCESSED / "label_candidates.jsonl", PROCESSED / "gold_review_candidates.json",
    ]
    for path in required:
        if not path.exists():
            failures.append(f"missing required file: {path}")
    try:
        taxonomy = json.loads((PROCESSED / "final_taxonomy.json").read_text(encoding="utf-8"))
        names = [item["intent_name"] for item in taxonomy["intents"]]
        if len(names) != 9 or len(names) != len(set(names)):
            failures.append("taxonomy is not exactly nine unique intents")
        source_ids: set[str] = set()
        first_message = None
        for line in (PROCESSED / "apple_cases.jsonl").open(encoding="utf-8"):
            case = json.loads(line)
            for message in case.get("conversation", []):
                if message.get("role") == "customer":
                    tweet_id = str(message["tweet_id"])
                    if tweet_id in source_ids:
                        failures.append(f"duplicate source tweet ID: {tweet_id}")
                    source_ids.add(tweet_id)
                    first_message = first_message or message["text"]
        gold = json.loads((PROCESSED / "gold_review_candidates.json").read_text(encoding="utf-8"))
        gold_ids = {record["tweet_id"] for record in gold["records"]}
        candidate_ids = {json.loads(line)["tweet_id"] for line in (PROCESSED / "label_candidates.jsonl").open(encoding="utf-8")}
        if gold_ids & candidate_ids:
            failures.append("gold candidates contaminate label candidates")
        if not first_message:
            failures.append("no real customer smoke message found")
        from backend.main import app
        from fastapi.testclient import TestClient
        with TestClient(app) as client:
            if client.get("/health").status_code != 200:
                failures.append("health endpoint failed")
            response = client.post("/api/support/respond", json={"message": first_message})
            if response.status_code != 200:
                failures.append(f"real-message API smoke failed: {response.status_code}")
            else:
                payload = response.json()
                if any(item.get("source") != "twcs" for item in payload.get("evidence", [])):
                    failures.append("evidence provenance is invalid")
                if any(item.get("tweet_id") in gold_ids for item in payload.get("evidence", [])):
                    failures.append("gold record appeared in runtime evidence")
    except Exception as error:
        failures.append(f"backend smoke/import failure: {error}")
    if failures:
        print("BACKEND VALIDATION: FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("BACKEND VALIDATION: PASS")
    print("Fallback runtime, source integrity, candidate provenance, and gold separation passed.")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT))
    raise SystemExit(main())
