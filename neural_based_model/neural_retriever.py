import json
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_DATASET_PATH = "data/movies.json"
DEFAULT_VECTOR_DB_PATH = "bd/movies_vectors.pkl"


@dataclass
class SearchResult:
    rank: int
    score: float
    url: str
    title: str
    media_type: str
    plot: str


class NeuralRetriever:
    """Neural retriever for semantic search over CultureSearch data."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        dataset_path: str = DEFAULT_DATASET_PATH,
        vector_db_path: str = DEFAULT_VECTOR_DB_PATH,
    ) -> None:
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.vector_db_path = vector_db_path
        self.model = SentenceTransformer(self.model_name)

        self.documents: List[Dict] = []
        self.urls: List[str] = []
        self.embeddings: np.ndarray | None = None

    def _build_text(self, movie: Dict) -> str:
        title = str(movie.get("title", ""))
        plot = str(movie.get("plot", ""))

        genres = movie.get("genres", [])
        if isinstance(genres, list):
            genres_text = " ".join(str(g) for g in genres)
        else:
            genres_text = str(genres)

        actors = movie.get("actors", [])
        if isinstance(actors, list):
            actors_text = " ".join(str(a) for a in actors)
        else:
            actors_text = str(actors)

        return f"{title} {genres_text} {plot} {actors_text}".strip()

    def _load_dataset(self) -> List[Dict]:
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                f"No se encontro el dataset en: {self.dataset_path}. "
                "Primero ejecuta tu scraper para generar data/movies.json."
            )

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        if not isinstance(dataset, list) or not dataset:
            raise ValueError("El archivo data/movies.json esta vacio o no tiene el formato esperado.")

        return dataset

    def build_vector_db(self, force_rebuild: bool = False) -> None:
        if os.path.exists(self.vector_db_path) and not force_rebuild:
            self.load_vector_db()
            return

        self.documents = self._load_dataset()
        self.urls = [str(doc.get("url", "")) for doc in self.documents]

        corpus = [self._build_text(doc) for doc in self.documents]
        vectors = self.model.encode(corpus, show_progress_bar=True)

        self.embeddings = np.array(vectors)

        payload = {
            "urls": self.urls,
            "embeddings": self.embeddings,
            "documents": self.documents,
            "model_name": self.model_name,
        }

        os.makedirs(os.path.dirname(self.vector_db_path), exist_ok=True)
        with open(self.vector_db_path, "wb") as f:
            pickle.dump(payload, f)

    def load_vector_db(self) -> None:
        if not os.path.exists(self.vector_db_path):
            raise FileNotFoundError(
                f"No se encontro la BD vectorial en: {self.vector_db_path}. "
                "Ejecuta build_vector_db() primero."
            )

        with open(self.vector_db_path, "rb") as f:
            payload = pickle.load(f)

        self.urls = payload.get("urls", [])
        self.embeddings = np.array(payload.get("embeddings", []))

        stored_docs = payload.get("documents")
        if isinstance(stored_docs, list) and stored_docs:
            self.documents = stored_docs
        else:
            # Backward compatibility with old pkl files that only stored URLs + embeddings.
            self.documents = self._load_dataset()

        if self.embeddings.size == 0:
            raise ValueError("La matriz de embeddings esta vacia.")

    def ensure_ready(self, force_rebuild: bool = False) -> None:
        self.build_vector_db(force_rebuild=force_rebuild)

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        if not query or not query.strip():
            raise ValueError("La consulta no puede estar vacia.")

        if self.embeddings is None:
            self.ensure_ready()

        query_vector = self.model.encode([query])
        similarities = cosine_similarity(query_vector, self.embeddings)[0]

        top_k = max(1, min(top_k, len(similarities)))
        sorted_indices = np.argsort(similarities)[::-1][:top_k]

        results: List[SearchResult] = []
        for rank, idx in enumerate(sorted_indices, start=1):
            doc = self.documents[idx]
            results.append(
                SearchResult(
                    rank=rank,
                    score=float(similarities[idx]),
                    url=str(doc.get("url", "")),
                    title=str(doc.get("title", "Sin titulo")),
                    media_type=str(doc.get("type", "desconocido")),
                    plot=str(doc.get("plot", "")),
                )
            )

        return results
