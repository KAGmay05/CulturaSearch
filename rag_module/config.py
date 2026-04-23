from dataclasses import dataclass


@dataclass
class RAGConfig:
    """Configuración del pipeline RAG."""
    model_name: str = "neural-chat"
    top_k: int = 4
    max_tokens: int = 512
    temperature: float = 0.7 

