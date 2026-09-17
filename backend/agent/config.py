from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    classifier_confidence_threshold: float = float(os.getenv("CLASSIFIER_CONFIDENCE_THRESHOLD", "0.65"))
    retrieval_similarity_threshold: float = float(os.getenv("RETRIEVAL_SIMILARITY_THRESHOLD", "0.34"))
    top_k_retrieval: int = int(os.getenv("TOP_K_RETRIEVAL", "5"))
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "")
    hf_token: str = os.getenv("HF_TOKEN", "")
    hf_model: str = os.getenv("HF_MODEL", "Qwen/Qwen2.5-3B-Instruct")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "")
    cases_path: Path = ROOT / "data" / "processed" / "apple_cases.jsonl"
    taxonomy_path: Path = ROOT / "data" / "processed" / "final_taxonomy.json"
    classifier_artifact: Path = ROOT / "artifacts" / "classifier" / "model.pkl"
    retrieval_artifact: Path = ROOT / "artifacts" / "retrieval" / "index.pkl"
    retrieval_metadata: Path = ROOT / "artifacts" / "retrieval" / "metadata.json"


settings = Settings()
