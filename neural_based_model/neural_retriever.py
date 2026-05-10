import json
import math
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Sequence

import faiss
import numpy as np
from collections import Counter
from sentence_transformers import CrossEncoder, SentenceTransformer

from bd import vectorizer as bd_vectorizer
from index import indexer
from web_search.web_expander import WebExpander
from positioning_module.ranking import compute_lexical_scores, compute_hybrid_fusion, compute_positioning_score, compute_rerank_fusion, rank_by_positioning, sigmoid


DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
# Modelo de re-ranking pequeño (~500MB, no 14.5GB)
DEFAULT_RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_DATASET_PATH = "data/dataset.json"
DEFAULT_VECTOR_DB_PATH = "bd/movies_vectors.pkl"
DEFAULT_VECTOR_INDEX_PATH = "bd/movies_vectors.index"
DEFAULT_VECTOR_METADATA_PATH = "bd/metadata.pkl"
DEFAULT_WEB_CACHE_PATH = "data/web_cache.json"


@dataclass
class SearchResult:
    rank: int
    score: float
    url: str
    title: str
    media_type: str
    plot: str
    neural_score: float = 0.0
    lexical_score: float = 0.0
    rerank_score: float = 0.0


class NeuralRetriever:
    """Neural retriever for semantic search over CultureSearch data."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        reranker_model_name: str = DEFAULT_RERANKER_MODEL_NAME,
        dataset_path: str = DEFAULT_DATASET_PATH,
        vector_db_path: str = DEFAULT_VECTOR_DB_PATH,
        vector_index_path: str = DEFAULT_VECTOR_INDEX_PATH,
        vector_metadata_path: str = DEFAULT_VECTOR_METADATA_PATH,
        web_cache_path: str = DEFAULT_WEB_CACHE_PATH,
    ) -> None:
        """Initializes model, storage paths and in-memory search structures."""
        self.model_name = model_name
        self.reranker_model_name = reranker_model_name
        self.dataset_path = dataset_path
        # Se mantiene por compatibilidad de firma, pero el backend principal ahora es .index
        self.vector_db_path = vector_db_path
        self.vector_index_path = vector_index_path
        self.vector_metadata_path = vector_metadata_path
        self.web_cache_path = web_cache_path
        self.model = SentenceTransformer(self.model_name)
        self.reranker: CrossEncoder | None = None
        self.web_expander: WebExpander | None = None
        self.faiss_index: faiss.Index | None = None

        self.documents: List[Dict] = []
        self.urls: List[str] = []
        self.url_to_doc_idx: Dict[str, int] = {}
        self.embeddings: np.ndarray | None = None
        self.lexical_index: Dict[str, Dict[int, int]] = {}
        self.doc_count: int = 0

    def _encode_query_normalized(self, query: str) -> np.ndarray:
        """Encodes a query into a normalized float32 embedding row vector."""
        q = self.model.encode([query]).astype(np.float32)
        q = np.ascontiguousarray(q)
        faiss.normalize_L2(q)
        return q

    def _rebuild_embeddings_from_index(self) -> None:
        """Reconstructs all vectors from FAISS into a dense numpy matrix."""
        if self.faiss_index is None:
            raise ValueError("El indice FAISS no esta cargado.")
        emb_rows: List[np.ndarray] = []
        for i in range(self.faiss_index.ntotal):
            emb_rows.append(self.faiss_index.reconstruct(i))
        self.embeddings = np.array(emb_rows, dtype=np.float32)

    def _load_metadata(self) -> Dict:
        """Loads metadata payload and adapts legacy formats for compatibility."""
        if not os.path.exists(self.vector_metadata_path):
            raise FileNotFoundError(f"No se encontro metadata en: {self.vector_metadata_path}")
        with open(self.vector_metadata_path, "rb") as f:
            payload = pickle.load(f)

        # Compatibilidad con metadata vieja (lista de URLs)
        if isinstance(payload, list):
            return {
                "urls": payload,
                "model_name": self.model_name,
                "normalized": True,
            }
        if not isinstance(payload, dict):
            raise ValueError("metadata.pkl tiene un formato invalido.")
        return payload

    def _load_reranker(self, model_name: str | None = None) -> CrossEncoder:
        """Lazily loads and caches the cross-encoder reranker model."""
        if self.reranker is not None and (model_name is None or model_name == self.reranker_model_name):
            return self.reranker

        selected_model = model_name or self.reranker_model_name
        self.reranker = CrossEncoder(selected_model)
        self.reranker_model_name = selected_model
        return self.reranker

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        """Applies a sigmoid transform to map raw scores into [0, 1]."""
        return sigmoid(values)

    def _build_text(self, movie: Dict) -> str:
        """Builds the canonical searchable text representation for one document."""
        title = str(movie.get("title", ""))
        plot = str(movie.get("plot", ""))
        year = str(movie.get("year", ""))
        rating = str(movie.get("rating", ""))
        media_type = str(movie.get("type", ""))
        country = str(movie.get("country", ""))

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

        directors = movie.get("director", [])
        if isinstance(directors, list):
            directors_text = " ".join(str(d) for d in directors)
        else:
            directors_text = str(directors)

        return (
            f"{title} {year} {rating} {media_type} {country} "
            f"{genres_text} {directors_text} {plot} {actors_text}"
        ).strip()

    def _load_dataset(self) -> List[Dict]:
        """Loads local dataset and merges web cache documents without URL duplicates."""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                f"No se encontro el dataset en: {self.dataset_path}. "
                "Primero ejecuta tu scraper para generar data/dataset.json."
            )

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        if not isinstance(dataset, list) or not dataset:
            raise ValueError("El archivo data/dataset.json esta vacio o no tiene el formato esperado.")

        merged_documents = list(dataset)
        seen_urls = {str(doc.get("url", "")) for doc in merged_documents if doc.get("url")}

        if os.path.exists(self.web_cache_path):
            with open(self.web_cache_path, "r", encoding="utf-8") as f:
                cached_docs = json.load(f)

            if isinstance(cached_docs, list):
                for doc in cached_docs:
                    if not isinstance(doc, dict):
                        continue
                    url = str(doc.get("url", ""))
                    if url and url not in seen_urls:
                        merged_documents.append(doc)
                        seen_urls.add(url)

        return merged_documents

    def _build_lexical_index(self) -> None:
        """Builds an in-memory term -> {doc_idx: tf} lexical inverted index."""
        self.lexical_index = {}
        for doc_idx, movie in enumerate(self.documents):
            text = self._build_text(movie)
            tokens = indexer.clean_text(text)
            tf_counter = Counter(tokens)
            for token, tf in tf_counter.items():
                posting = self.lexical_index.setdefault(token, {})
                posting[doc_idx] = tf
        self.doc_count = len(self.documents)

    def _resolve_result_document(self, result: SearchResult) -> Dict:
        """Resolves full document metadata for a ranked result."""
        doc_idx = self.url_to_doc_idx.get(result.url)
        if doc_idx is not None and 0 <= doc_idx < len(self.documents):
            return self.documents[doc_idx]
        return {
            "url": result.url,
            "title": result.title,
            "plot": result.plot,
            "type": result.media_type,
        }

    def _compute_lexical_scores(self, query: str) -> np.ndarray:
        """Delegate lexical scoring to positioning module using cleaned doc texts."""
        if not self.documents:
            return np.array([])

        query_tokens = indexer.clean_text(query)
        if not query_tokens:
            return np.zeros(len(self.documents), dtype=float)

        docs_for_scoring = []
        for doc in self.documents:
            text = self._build_text(doc)
            clean = " ".join(indexer.clean_text(text))
            docs_for_scoring.append({"clean_text": clean, "metadata": doc})

        scores = compute_lexical_scores(query_tokens, docs_for_scoring)
        return np.array(scores, dtype=float)

    def _query_lexical_coverage(self, query: str) -> float:
        """Returns fraction of unique query tokens present in lexical index."""
        tokens = set(indexer.clean_text(query))
        if not tokens:
            return 0.0

        covered = sum(1 for token in tokens if token in self.lexical_index)
        return float(covered) / float(len(tokens))

    def _load_web_expander(self) -> WebExpander:
        """Creates and caches the web expansion helper with project defaults."""
        if self.web_expander is None:
            self.web_expander = WebExpander(
                cache_path=self.web_cache_path,
                seed_file_paths=["seeds/seed_sensacine.json"],
            )
        return self.web_expander

    def build_vector_db(self, force_rebuild: bool = False) -> None:
        """Builds or reuses vector index artifacts and refreshes lexical index state."""
        expected_documents = self._load_dataset()
        expected_count = len(expected_documents)

        has_index_backend = os.path.exists(self.vector_index_path) and os.path.exists(self.vector_metadata_path)

        if has_index_backend and not force_rebuild:
            self.load_vector_db()
            embeddings_count = int(self.embeddings.shape[0]) if self.embeddings is not None and self.embeddings.ndim > 0 else 0
            documents_count = len(self.documents)
            urls_count = len(self.urls)

            # Reuse existing vector DB only when all index components are aligned.
            if (
                documents_count == expected_count
                and embeddings_count == expected_count
                and urls_count == expected_count
            ):
                self._build_lexical_index()
                return

        self.documents = expected_documents
        bd_vectorizer.build_embeddings(
            documents=self.documents,
            model_name=self.model_name,
            index_path=self.vector_index_path,
            metadata_path=self.vector_metadata_path,
            include_documents=True,
            text_builder=self._build_text,
            model=self.model,
        )
        self.load_vector_db()

    def load_vector_db(self) -> None:
        """Loads FAISS index plus metadata and reconstructs aligned document state."""
        if not (os.path.exists(self.vector_index_path) and os.path.exists(self.vector_metadata_path)):
            raise FileNotFoundError(
                f"No se encontro la BD vectorial en formato index+metadata: {self.vector_index_path} / {self.vector_metadata_path}. "
                "Ejecuta build_vector_db() primero."
            )

        self.faiss_index = faiss.read_index(self.vector_index_path)
        metadata = self._load_metadata()

        self.urls = [str(url) for url in metadata.get("urls", [])]
        self.url_to_doc_idx = {str(url): idx for idx, url in enumerate(self.urls)}

        stored_docs = metadata.get("documents")
        if isinstance(stored_docs, list) and stored_docs:
            self.documents = stored_docs
        else:
            dataset_docs = self._load_dataset()
            doc_by_url = {
                str(doc.get("url", "")): doc
                for doc in dataset_docs
                if isinstance(doc, dict)
            }
            self.documents = [
                doc_by_url.get(
                    url,
                    {
                        "url": url,
                        "title": "Sin titulo",
                        "plot": "",
                        "type": "desconocido",
                    },
                )
                for url in self.urls
            ]

        if self.faiss_index.ntotal == 0:
            raise ValueError("El indice FAISS esta vacio.")

        self._rebuild_embeddings_from_index()

        if self.embeddings.size == 0:
            raise ValueError("La matriz de embeddings esta vacia.")

        self._build_lexical_index()

    def ensure_ready(self, force_rebuild: bool = False) -> None:
        """Guarantees vector and lexical structures are ready for search."""
        self.build_vector_db(force_rebuild=force_rebuild)

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Runs pure neural similarity search over FAISS and returns ranked results."""
        if not query or not query.strip():
            raise ValueError("La consulta no puede estar vacia.")

        if self.embeddings is None or self.faiss_index is None:
            self.ensure_ready()

        raw_results = bd_vectorizer.search_by_similarity(
            query=query,
            top_k=top_k,
            index=self.faiss_index,
            model=self.model,
            urls=self.urls,
            return_indices=True,
        )

        results: List[SearchResult] = []
        for item in raw_results:
            idx = int(item.get("index", -1))
            score = float(item.get("score", 0.0))
            rank = int(item.get("rank", len(results) + 1))
            if idx < 0 or idx >= len(self.documents):
                continue
            doc = self.documents[idx]
            results.append(
                SearchResult(
                    rank=rank,
                    score=float(score),
                    url=str(doc.get("url", "")),
                    title=str(doc.get("title", "Sin titulo")),
                    media_type=str(doc.get("type", "desconocido")),
                    plot=str(doc.get("plot", "")),
                    neural_score=float(score),
                    lexical_score=0.0,
                    rerank_score=0.0,
                )
            )

        return results

    def search_hybrid(self, query: str, top_k: int = 5, alpha: float = 0.7) -> List[SearchResult]:
        """Combines neural and lexical signals into a single fused ranking score."""
        if not query or not query.strip():
            raise ValueError("La consulta no puede estar vacia.")

        if self.embeddings is None:
            self.ensure_ready()

        alpha = float(np.clip(alpha, 0.0, 1.0))

        query_vector = self._encode_query_normalized(query)
        # Con embeddings normalizados, producto punto == coseno.
        neural_scores = (self.embeddings @ query_vector[0]).astype(float)
        lexical_scores = self._compute_lexical_scores(query)
        hybrid_data = compute_hybrid_fusion(neural_scores=neural_scores, lexical_scores=lexical_scores, alpha=alpha)
        fused_scores = hybrid_data["fused_scores"]

        top_k = max(1, min(top_k, len(fused_scores)))
        sorted_indices = rank_by_positioning(
            primary_scores=fused_scores,
            query=query,
            docs=self.documents,
            original_positions=np.arange(len(self.documents)),
            top_k=top_k,
        )

        results: List[SearchResult] = []
        for rank, idx in enumerate(sorted_indices, start=1):
            doc = self.documents[idx]
            results.append(
                SearchResult(
                    rank=rank,
                    score=float(fused_scores[idx]),
                    url=str(doc.get("url", "")),
                    title=str(doc.get("title", "Sin titulo")),
                    media_type=str(doc.get("type", "desconocido")),
                    plot=str(doc.get("plot", "")),
                    neural_score=float(hybrid_data["neural_scores"][idx]),
                    lexical_score=float(hybrid_data["lexical_scores"][idx]),
                    rerank_score=0.0,
                )
            )

        return results

    def retrieve_candidates(self, query: str, candidate_k: int = 50, alpha: float = 0.9) -> List[SearchResult]:
        """Retrieves a broad candidate set using hybrid search for later reranking."""
        return self.search_hybrid(query=query, top_k=candidate_k, alpha=alpha)

    def rerank_candidates(
        self,
        query: str,
        candidates: List[SearchResult],
        top_k: int = 5,
        rerank_weight: float = 0.75,
        reranker_model_name: str | None = None,
    ) -> List[SearchResult]:
        """Reranks candidates with a cross-encoder and blends base and rerank scores."""
        if not candidates:
            return []

        rerank_weight = float(np.clip(rerank_weight, 0.0, 1.0))
        base_scores = np.array([float(c.score) for c in candidates], dtype=float)

        try:
            reranker = self._load_reranker(reranker_model_name)
            pair_texts = []
            for c in candidates:
                doc_idx = self.url_to_doc_idx.get(c.url)
                if doc_idx is not None:
                    doc_text = self._build_text(self.documents[doc_idx])
                else:
                    doc_text = self._build_text(
                        {
                            "title": c.title,
                            "plot": c.plot,
                            "type": c.media_type,
                        }
                    )
                pair_texts.append((query, doc_text))
            rerank_raw = np.array(reranker.predict(pair_texts), dtype=float)
            rerank_data = compute_rerank_fusion(base_scores=base_scores, rerank_raw=rerank_raw, rerank_weight=rerank_weight)
            final_scores = rerank_data["final_scores"]
        except Exception:
            rerank_raw = np.zeros(len(candidates), dtype=float)
            rerank_data = compute_rerank_fusion(base_scores=base_scores, rerank_raw=rerank_raw, rerank_weight=rerank_weight)
            final_scores = rerank_data["final_scores"]
        top_k = max(1, min(top_k, len(candidates)))
        candidate_docs = [self._resolve_result_document(candidate) for candidate in candidates]
        sorted_idx = rank_by_positioning(
            primary_scores=final_scores,
            query=query,
            docs=candidate_docs,
            original_positions=np.arange(len(candidates)),
            top_k=top_k,
        )

        reranked: List[SearchResult] = []
        for rank, idx in enumerate(sorted_idx, start=1):
            item = candidates[idx]
            reranked.append(
                SearchResult(
                    rank=rank,
                    score=float(final_scores[idx]),
                    url=item.url,
                    title=item.title,
                    media_type=item.media_type,
                    plot=item.plot,
                    neural_score=item.neural_score,
                    lexical_score=item.lexical_score,
                    rerank_score=float(rerank_raw[idx]),
                )
            )

        return reranked

    def search_advanced(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 50,
        alpha: float = 0.9,
        rerank_weight: float = 0.75,
        reranker_model_name: str | None = None,
    ) -> List[SearchResult]:
        """Executes the full retrieval pipeline: candidate retrieval + reranking."""
        candidates = self.retrieve_candidates(query=query, candidate_k=max(candidate_k, top_k), alpha=alpha)
        return self.rerank_candidates(
            query=query,
            candidates=candidates,
            top_k=top_k,
            rerank_weight=rerank_weight,
            reranker_model_name=reranker_model_name,
        )

    def search_with_web_expansion(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 50,
        alpha: float = 0.9,
        rerank_weight: float = 0.75,
        min_local_score: float = 0.72,
        min_lexical_coverage: float = 0.75,
        hard_min_local_score: float = 0.65,
        web_max_results: int = 10,
    ) -> List[SearchResult]:
        """Runs local retrieval and triggers web expansion when confidence is low."""
        local_results = self.search_advanced(
            query=query,
            top_k=top_k,
            candidate_k=candidate_k,
            alpha=alpha,
            rerank_weight=rerank_weight,
        )

        top_score = float(local_results[0].score) if local_results else 0.0
        lexical_coverage = self._query_lexical_coverage(query)

        # If query terms are well covered in the local index, trust local results even
        # when rerank scores are low (reranker may underestimate title-only queries).
        high_lexical_coverage = lexical_coverage >= min_lexical_coverage
        needs_web_expansion = (
            (top_score < hard_min_local_score and not high_lexical_coverage)
            or (top_score < min_local_score and not high_lexical_coverage)
        )

        if not needs_web_expansion:
            return local_results

        web_expander = self._load_web_expander()
        web_documents = web_expander.expand(query=query, max_results=web_max_results)

        if not web_documents:
            return local_results

        self.build_vector_db(force_rebuild=True)

        return self.search_advanced(
            query=query,
            top_k=top_k,
            candidate_k=candidate_k,
            alpha=alpha,
            rerank_weight=rerank_weight,
        )
