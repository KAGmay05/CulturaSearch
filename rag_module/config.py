import os
from dataclasses import dataclass


@dataclass
class RAGConfig:
    """Configuración del pipeline RAG."""
    model_name: str = os.getenv("OLLAMA_MODEL", "neural-chat")
    top_k: int = int(os.getenv("RAG_TOP_K", "4"))  # 3 documentos para reducir contexto
    max_tokens: int = int(os.getenv("RAG_MAX_TOKENS", "400"))  # Respuestas más concisas
    temperature: float = float(os.getenv("RAG_TEMPERATURE", "0.5"))  # Más determinista
    rewrite_query: bool = True
    rewrite_max_tokens: int = int(os.getenv("RAG_REWRITE_MAX_TOKENS", "64"))

