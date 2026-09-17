# Requirements Traceability

| Requirement                | Implementation                                         | Validation                          | Output artifact                              |
| -------------------------- | ------------------------------------------------------ | ----------------------------------- | -------------------------------------------- |
| Intent classification      | `backend/agent/classifier.py`                          | classifier/API tests                | optional classifier artifact                 |
| Historical retrieval       | `backend/agent/retriever.py`, `scripts/build_index.py` | retriever tests                     | retrieval artifacts when built               |
| Response generation        | `backend/agent/generator.py`                           | generator tests                     | API response                                 |
| Escalation                 | `backend/agent/escalation.py`                          | escalation tests                    | API response                                 |
| Gold set infrastructure    | `gold_review_candidates.json`                          | contamination checks                | 200 blank-label records                      |
| Baselines                  | `evaluation/baselines.py`                              | evaluation tests                    | PENDING — NOT FABRICATED without labels      |
| Automated metrics          | `evaluation/metrics.py`                                | evaluation tests                    | PENDING — NOT FABRICATED without labels      |
| LLM judge                  | `evaluation/judge.py`                                  | unavailable-state test              | PENDING — NOT FABRICATED without credentials |
| Human/LLM agreement        | evaluation infrastructure                              | pending human ratings               | PENDING — NOT FABRICATED                     |
| Failure analysis           | `evaluation/failure_analysis.py`                       | schema tests                        | pending labeled predictions                  |
| Misleading metric analysis | evaluation report                                      | pending measured class distribution | PENDING — NOT FABRICATED                     |
| LLM fallback               | `agent/generator.py`                                   | generator/API tests                | deterministic fallback response              |
| FAISS retrieval            | `scripts/build_index.py`                               | optional dependency gate            | PENDING — FAISS unavailable locally          |
| Design decisions           | `evaluation/design_decisions.md`                       | file check                          | design decisions artifact                    |
| Reproducibility            | README commands                                        | backend validation                  | README                                       |
