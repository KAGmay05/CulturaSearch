"""
Módulo RAG - Retrieval-Augmented Generation

Exporta las clases y funciones principales.
"""

from rag_module.config import RAGConfig
from rag_module.generator import OllamaGenerator
from rag_module.pipeline import RAGPipeline
from rag_module.retriever_wrapper import RetrieverWrapped

__all__ = [
    "RAGConfig",
    "OllamaGenerator",
    "RAGPipeline",
    "RetrieverWrapped",
]
