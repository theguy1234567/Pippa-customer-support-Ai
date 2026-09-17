# Design Decisions

1. AppleSupport is isolated using the validated real dataset reconstruction.
2. The taxonomy is read from `final_taxonomy.json` and is not recreated in runtime code.
3. Candidate assessments remain separate from human labels.
4. The 200 gold-review records are excluded from candidate training artifacts.
5. One primary intent is required for deterministic API output.
6. Device-only messages map to `non_actionable_other`.
7. Sentence Transformers remain the configurable production embedding choice.
8. TF-IDF is retained as a lightweight baseline and fallback.
9. Retrieval evidence preserves real tweet and conversation provenance.
10. Low evidence produces escalation rather than an invented answer.
11. The deterministic fallback works without LLM credentials.
12. Evaluation reports blocked status when gold labels are unavailable.
13. Accuracy is not reported without measured gold predictions; macro metrics are planned.
14. Source integrity and gold contamination are validated explicitly.
15. Human labels are required before production classifier training.
