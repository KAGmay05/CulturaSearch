from neural_retriever import NeuralRetriever


def print_results(query: str, results) -> None:
    print(f"\nConsulta: {query}")
    print("-" * 80)
    for item in results:
        snippet = item.plot[:140].strip()
        if len(item.plot) > 140:
            snippet += "..."
        print(
            f"#{item.rank} | score={item.score:.4f} | {item.title} "
            f"({item.media_type})\nURL: {item.url}\nPlot: {snippet}\n"
        )


def main() -> None:
    retriever = NeuralRetriever()
    retriever.ensure_ready(force_rebuild=False)

    test_queries = [
        "pelicula de ciencia ficcion sobre viajes en el tiempo",
        "serie policial con detectives y asesino en serie",
        "drama romantico ambientado despues de la guerra",
    ]

    for query in test_queries:
        results = retriever.search(query, top_k=3)
        print_results(query, results)


if __name__ == "__main__":
    main()
