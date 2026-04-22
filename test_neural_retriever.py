from neural_based_model.neural_retriever import NeuralRetriever


def print_results(query: str, results) -> None:
    print(f"\nConsulta: {query}")
    print("-" * 80)
    for item in results:
        snippet = item.plot[:140].strip()
        if len(item.plot) > 140:
            snippet += "..."
        print(
            f"#{item.rank} | score={item.score:.4f} | {item.title} ({item.media_type})"
            f"\nURL: {item.url}\nPlot: {snippet}\n"
        )


def main() -> None:
    retriever = NeuralRetriever()
    retriever.ensure_ready(force_rebuild=False)

    queries = [
        "pelicula de accion",
        "comedia romantica",
    ]

    for query in queries:
        results = retriever.search(query=query, top_k=5)
        print_results(query, results)


if __name__ == "__main__":
    main()
