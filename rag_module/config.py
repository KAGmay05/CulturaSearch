import os
from dataclasses import dataclass


@dataclass
class RAGConfig:
    """Configuración del pipeline RAG."""
    model_name: str = os.getenv("OLLAMA_MODEL", "neural-chat")
    top_k: int = int(os.getenv("RAG_TOP_K", "4"))
    max_tokens: int = int(os.getenv("RAG_MAX_TOKENS", "512"))
    temperature: float = float(os.getenv("RAG_TEMPERATURE", "0.7"))

