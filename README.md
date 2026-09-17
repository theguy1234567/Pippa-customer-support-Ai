# Hiver Support Agent

A dataset-grounded AppleSupport customer-support agent built from the real Customer Support on Twitter dataset. The backend preserves historical evidence and does not fabricate labels, replies, policies, or evaluation metrics.

## Backend Status

Implemented and validated:

- AppleSupport conversation reconstruction
- Nine-intent final taxonomy
- Candidate-label and gold-review infrastructure
- Deterministic classifier fallback
- Persisted TF-IDF historical retrieval fallback
- Evidence-provenance contract
- Deterministic grounded response fallback
- Explainable escalation policy
- FastAPI integration
- Evaluation, judge, failure-analysis, and traceability scaffolding

Pending external inputs:

- Human labels for the 1,000 review candidates
- Human labels for the 200 gold-review candidates
- Optional LLM credentials
- Optional `faiss-cpu` installation for the FAISS artifact path

Candidate labels are not ground truth. Gold-review records remain evaluation-only.

## Architecture

```text
Next.js frontend
        |
        v
FastAPI /api/support/respond
        |
        v
SupportAgent
  |       |       |
Classifier Retriever Generator
        \   |   /
          Escalation
```

Every evidence item includes its source tweet ID, conversation ID, role, timestamp when available, original text, and `source: twcs`.

## Prerequisites

- Python 3.11+
- Node.js for the existing frontend
- `backend/data/raw/twcs.csv` locally; the raw dataset is not committed

## Setup

```bash
cd "D:/ALL programming/Agentic_projects/hiver project"
python -m venv .venv
.venv\\Scripts\\activate
pip install -r backend/requirements.txt
```

Copy `backend/.env.example` to `backend/.env` only when configuration is needed. Never commit real credentials.

## Data and Candidate Preparation

The validated processed data already exists. To regenerate candidate assessments:

```bash
python backend/scripts/prepare_label_candidates.py
python backend/scripts/validate_label_candidates.py
```

After a human reviewer fills `human_label` values in `backend/data/processed/label_review_sample.json`:

```bash
python backend/scripts/prepare_training_data.py
python backend/scripts/train_classifier.py
```

The training command refuses to train when labels are blank. It uses Sentence Transformer embeddings plus Logistic Regression once real labels exist.

The 200 records in `gold_review_candidates.json` must be labeled independently and must never be used for training or retrieval indexing.

## Retrieval Artifact

Build the real-data retrieval artifact:

```bash
python backend/scripts/build_index.py
```

The command writes a persisted TF-IDF artifact and metadata. If `faiss-cpu` is installed, it also writes `backend/artifacts/retrieval/apple_support.faiss`; otherwise it reports the explicit fallback.

## Run FastAPI

From the repository root:

```bash
uvicorn backend.main:app --reload --port 8000
```

Endpoints:

- `GET http://localhost:8000/`
- `GET http://localhost:8000/health`
- `POST http://localhost:8000/api/support/respond`

Example:

```bash
curl -X POST http://localhost:8000/api/support/respond \
  -H "Content-Type: application/json" \
  -d "{\"message\":\"I can't update my iPhone\"}"
```

The backend works without an LLM key. If evidence is insufficient, it returns a controlled escalation-oriented response and marks it `grounded: false`.

## Evaluation

Run the current gold-aware evaluation:

```bash
python backend/scripts/run_evaluation.py
```

Until the 200 gold records receive human labels, the result is explicitly:

```text
Evaluation blocked: gold labels unavailable.
```

No evaluation score is fabricated. Evaluation artifacts are written under `backend/artifacts/evaluation/`.

## Validation

```bash
python -m compileall -q backend
pytest -q
python backend/scripts/validate_label_candidates.py
python backend/scripts/validate_backend.py
```

A real processed customer message is used by the API tests and backend validator smoke path.

## Project Structure

- `backend/agent/`: classifier, retrieval, generation, escalation, schemas, and orchestration
- `backend/api/`: FastAPI routes
- `backend/data/raw/`: local raw TWCS dataset
- `backend/data/processed/`: validated conversations, taxonomy, candidate artifacts, and gold-review reserve
- `backend/evaluation/`: metrics, baselines, evaluation gates, judge and failure-analysis infrastructure
- `backend/scripts/`: preparation, training, indexing, evaluation, and validation commands
- `backend/tests/`: focused backend tests
- `frontend/`: existing Next.js application, unchanged by backend work

## Limitations

The current candidate labels are automatically generated review candidates, not human labels. The classifier artifact is therefore not trained. Retrieval is real-data-backed and deterministic; FAISS and LLM paths are optional and report their availability explicitly. Final supervised metrics, response judgments, agreement, and failure analysis remain pending human-labeled evaluation data.
