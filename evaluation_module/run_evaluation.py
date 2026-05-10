"""Command line entry point for retrieval evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation_module.evaluator import RetrievalEvaluator, format_query_result, format_summary
from neural_based_model.neural_retriever import NeuralRetriever


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CulturaSearch - evaluacion de recuperacion")
    parser.add_argument("--top-k", type=int, default=10, help="Numero de resultados evaluados por consulta")
    parser.add_argument("--candidate-k", type=int, default=50, help="Numero de candidatos antes del re-ranking")
    parser.add_argument("--alpha", type=float, default=0.9, help="Peso de combinacion semantica/lexica")
    parser.add_argument("--rerank-weight", type=float, default=0.75, help="Peso del re-ranking en la fusion final")
    parser.add_argument("--queries-path", type=Path, default=None, help="Ruta del archivo JSON de consultas")
    parser.add_argument("--with-web-expansion", action="store_true", help="Evalua con expansion web activada")
    parser.add_argument("--export-json", type=Path, default=None, help="Exporta el resultado completo a JSON")
    parser.add_argument("--export-csv", type=Path, default=None, help="Exporta el resumen por consulta a CSV")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    retriever = NeuralRetriever()
    retriever.ensure_ready(force_rebuild=False)

    evaluator = RetrievalEvaluator(queries_path=args.queries_path)
    evaluation = evaluator.evaluate(
        retriever=retriever,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        alpha=args.alpha,
        rerank_weight=args.rerank_weight,
        use_web_expansion=args.with_web_expansion,
    )

    print("=== Evaluacion de Recuperacion ===")
    for result in evaluation.query_results:
        print(format_query_result(result))
    print("----------------------------------")
    print(format_summary(evaluation))

    if args.export_json is not None:
        evaluation.export_json(args.export_json)
        print(f"JSON exportado en: {args.export_json}")

    if args.export_csv is not None:
        evaluation.export_csv(args.export_csv)
        print(f"CSV exportado en: {args.export_csv}")


if __name__ == "__main__":
    main()