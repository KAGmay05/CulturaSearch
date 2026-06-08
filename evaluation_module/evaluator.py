"""Retrieval benchmark runner for CulturaSearch."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from neural_based_model.neural_retriever import NeuralRetriever, SearchResult

from evaluation_module.metrics import f1_at_k, mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k


@dataclass(frozen=True)
class RelevantDocument:
    url: str
    relevance: int = 1


@dataclass(frozen=True)
class BenchmarkQuery:
    query_id: str
    query_type: str
    query: str
    relevant_documents: Sequence[RelevantDocument]


@dataclass
class QueryEvaluation:
    query_id: str
    query_type: str
    query: str
    top_k: int
    precision: float
    recall: float
    f1: float
    ndcg: float
    mrr: float
    relevant_retrieved: int
    total_relevant: int
    ranked_results: List[SearchResult]


@dataclass
class RetrievalEvaluation:
    top_k: int
    query_results: List[QueryEvaluation]

    def average(self, attribute: str) -> float:
        if not self.query_results:
            return 0.0
        return sum(getattr(item, attribute) for item in self.query_results) / float(len(self.query_results))

    def summary(self) -> Dict[str, float]:
        return {
            "queries": float(len(self.query_results)),
            f"precision@{self.top_k}": self.average("precision"),
            f"recall@{self.top_k}": self.average("recall"),
            f"f1@{self.top_k}": self.average("f1"),
            f"ndcg@{self.top_k}": self.average("ndcg"),
            f"mrr@{self.top_k}": self.average("mrr"),
        }

    def to_json_dict(self) -> Dict[str, object]:
        return {
            "top_k": self.top_k,
            "summary": self.summary(),
            "queries": [
                {
                    "query_id": result.query_id,
                    "query_type": result.query_type,
                    "query": result.query,
                    "precision": result.precision,
                    "recall": result.recall,
                    "f1": result.f1,
                    "ndcg": result.ndcg,
                    "mrr": result.mrr,
                    "relevant_retrieved": result.relevant_retrieved,
                    "total_relevant": result.total_relevant,
                    "ranked_results": [
                        {
                            "rank": item.rank,
                            "score": item.score,
                            "url": item.url,
                            "title": item.title,
                            "media_type": item.media_type,
                            "plot": item.plot,
                            "neural_score": item.neural_score,
                            "lexical_score": item.lexical_score,
                            "rerank_score": item.rerank_score,
                        }
                        for item in result.ranked_results
                    ],
                }
                for result in self.query_results
            ],
        }

    def export_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_json_dict(), handle, ensure_ascii=False, indent=2)

    def export_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "query_id",
                    "query_type",
                    "query",
                    "top_k",
                    "precision",
                    "recall",
                    "f1",
                    "ndcg",
                    "mrr",
                    "relevant_retrieved",
                    "total_relevant",
                ],
            )
            writer.writeheader()
            for result in self.query_results:
                writer.writerow(
                    {
                        "query_id": result.query_id,
                        "query_type": result.query_type,
                        "query": result.query,
                        "top_k": result.top_k,
                        "precision": f"{result.precision:.6f}",
                        "recall": f"{result.recall:.6f}",
                        "f1": f"{result.f1:.6f}",
                        "ndcg": f"{result.ndcg:.6f}",
                        "mrr": f"{result.mrr:.6f}",
                        "relevant_retrieved": result.relevant_retrieved,
                        "total_relevant": result.total_relevant,
                    }
                )


class RetrievalEvaluator:
    def __init__(self, queries_path: Path | None = None) -> None:
        if queries_path is not None:
            self.queries_path = queries_path
        else:
            default_path = Path(__file__).with_name("test_queries.json")
            fallback_path = Path(__file__).with_name("test_queries_actualizado.json")
            self.queries_path = default_path if default_path.exists() else fallback_path

    def load_queries(self) -> List[BenchmarkQuery]:
        with open(self.queries_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        queries: List[BenchmarkQuery] = []
        for item in payload:
            relevant_documents = [
                RelevantDocument(url=document["url"], relevance=int(document.get("relevance", 1)))
                for document in item.get("relevant_documents", [])
            ]
            queries.append(
                BenchmarkQuery(
                    query_id=str(item["query_id"]),
                    query_type=str(item.get("query_type", "tema")),
                    query=str(item["query"]),
                    relevant_documents=relevant_documents,
                )
            )
        return queries

    @staticmethod
    def _relevance_map(query: BenchmarkQuery) -> Dict[str, int]:
        return {document.url: int(document.relevance) for document in query.relevant_documents}

    @staticmethod
    def _ranked_relevances(ranked_results: Sequence[SearchResult], relevance_map: Dict[str, int]) -> List[int]:
        return [int(relevance_map.get(result.url, 0)) for result in ranked_results]

    def evaluate_query(
        self,
        retriever: NeuralRetriever,
        query: BenchmarkQuery,
        top_k: int = 10,
        candidate_k: int = 50,
        alpha: float = 0.9,
        rerank_weight: float = 0.75,
        use_web_expansion: bool = False,
    ) -> QueryEvaluation:
        if use_web_expansion:
            ranked_results = retriever.search_with_web_expansion(
                query=query.query,
                top_k=top_k,
                candidate_k=candidate_k,
                alpha=alpha,
                rerank_weight=rerank_weight,
            )
        else:
            ranked_results = retriever.search_advanced(
                query=query.query,
                top_k=top_k,
                candidate_k=candidate_k,
                alpha=alpha,
                rerank_weight=rerank_weight,
            )

        relevance_map = self._relevance_map(query)
        ranked_relevances = self._ranked_relevances(ranked_results, relevance_map)
        relevant_retrieved = sum(1 for relevance in ranked_relevances[:top_k] if relevance > 0)
        total_relevant = len(relevance_map)
        precision = precision_at_k(relevant_retrieved, top_k)
        recall = recall_at_k(relevant_retrieved, total_relevant)
        f1 = f1_at_k(precision, recall)
        ndcg = ndcg_at_k(ranked_relevances, top_k)
        mrr = mrr_at_k(ranked_relevances, top_k)

        return QueryEvaluation(
            query_id=query.query_id,
            query_type=query.query_type,
            query=query.query,
            top_k=top_k,
            precision=precision,
            recall=recall,
            f1=f1,
            ndcg=ndcg,
            mrr=mrr,
            relevant_retrieved=relevant_retrieved,
            total_relevant=total_relevant,
            ranked_results=list(ranked_results),
        )

    def evaluate(
        self,
        retriever: NeuralRetriever,
        top_k: int = 10,
        candidate_k: int = 50,
        alpha: float = 0.9,
        rerank_weight: float = 0.75,
        use_web_expansion: bool = False,
    ) -> RetrievalEvaluation:
        query_results = [
            self.evaluate_query(
                retriever=retriever,
                query=query,
                top_k=top_k,
                candidate_k=candidate_k,
                alpha=alpha,
                rerank_weight=rerank_weight,
                use_web_expansion=use_web_expansion,
            )
            for query in self.load_queries()
        ]
        return RetrievalEvaluation(top_k=top_k, query_results=query_results)


def format_query_result(result: QueryEvaluation) -> str:
    return (
        f"{result.query_id:>2} [{result.query_type}] | P@{result.top_k}={result.precision:.3f} "
        f"R@{result.top_k}={result.recall:.3f} F1={result.f1:.3f} "
        f"NDCG@{result.top_k}={result.ndcg:.3f} MRR@{result.top_k}={result.mrr:.3f} "
        f"| relevantes recuperados: {result.relevant_retrieved}/{result.total_relevant} "
        f"| {result.query}"
    )


def format_summary(evaluation: RetrievalEvaluation) -> str:
    summary = evaluation.summary()
    return (
        f"Consultas evaluadas: {int(summary['queries'])}\n"
        f"Precision@{evaluation.top_k}: {summary[f'precision@{evaluation.top_k}']:.3f}\n"
        f"Recall@{evaluation.top_k}: {summary[f'recall@{evaluation.top_k}']:.3f}\n"
        f"F1@{evaluation.top_k}: {summary[f'f1@{evaluation.top_k}']:.3f}\n"
        f"NDCG@{evaluation.top_k}: {summary[f'ndcg@{evaluation.top_k}']:.3f}\n"
        f"MRR@{evaluation.top_k}: {summary[f'mrr@{evaluation.top_k}']:.3f}"
    )