from neural_based_model.neural_retriever import NeuralRetriever
from index import indexer


def print_indexing_stats(retriever: NeuralRetriever) -> None:
    docs = len(retriever.documents)
    emb_shape = retriever.embeddings.shape if retriever.embeddings is not None else (0, 0)
    vocab_size = len(retriever.lexical_index)
    postings = sum(len(posting) for posting in retriever.lexical_index.values())
    avg_docs_per_term = (postings / vocab_size) if vocab_size else 0.0

    print("\n=== Estadisticas de Indexacion ===")
    print(f"Documentos indexados: {docs}")
    print(f"Matriz de embeddings: {emb_shape}")
    print(f"Vocabulario lexico: {vocab_size} terminos")
    print(f"Postings totales: {postings}")
    print(f"Promedio docs por termino: {avg_docs_per_term:.2f}")
    print("==================================")


def print_query_lexical_stats(query: str, retriever: NeuralRetriever) -> None:
    tokens = indexer.clean_text(query)
    unique_tokens = sorted(set(tokens))
    covered_tokens = [tok for tok in unique_tokens if tok in retriever.lexical_index]
    lexical_scores = retriever._compute_lexical_scores(query)
    docs_with_lex_match = int((lexical_scores > 0).sum()) if lexical_scores.size else 0

    print("\n--- Estadisticas Lexicas de Consulta ---")
    print(f"Tokens normalizados: {tokens}")
    print(f"Tokens unicos: {len(unique_tokens)}")
    print(f"Tokens con match en indice: {len(covered_tokens)}")
    print(f"Cobertura lexica: {(len(covered_tokens) / len(unique_tokens) * 100.0):.2f}%" if unique_tokens else "Cobertura lexica: 0.00%")
    print(f"Documentos con score lexico > 0: {docs_with_lex_match}")
    print("---------------------------------------")


def print_results(query: str, results) -> None:
    print(f"\nConsulta: {query}")
    print("-" * 80)
    for item in results:
        snippet = item.plot[:140].strip()
        if len(item.plot) > 140:
            snippet += "..."
        print(
            f"#{item.rank} | fused={item.score:.4f} | neural={item.neural_score:.4f} "
            f"| lexical={item.lexical_score:.2f} | rerank={item.rerank_score:.4f} "
            f"| {item.title} ({item.media_type})"
            f"\nURL: {item.url}\nPlot: {snippet}\n"
        )


def main() -> None:
    retriever = NeuralRetriever()
    retriever.ensure_ready(force_rebuild=False)
    print_indexing_stats(retriever)

    test_queries = [
        "pelicula de tom y jerry",
    ]

    for query in test_queries:
        print_query_lexical_stats(query, retriever)
        results = retriever.search_advanced(
            query=query,
            top_k=5,
            candidate_k=50,
            alpha=0.9,
            rerank_weight=0.75,
        )
        print_results(query, results)
        new_results = retriever.search_with_web_expansion(query)
        print_results(query + " (con expansion web)", new_results)


if __name__ == "__main__":
    main()
