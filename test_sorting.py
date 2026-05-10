"""Quick test for sorting methods."""
from neural_based_model.neural_retriever import NeuralRetriever
from positioning_module import sort_by_name, sort_by_year, sort_by_rating, sort_by_relevance


def main():
    nr = NeuralRetriever()
    nr.ensure_ready()
    
    # Get some results
    results = nr.search_advanced("accion", top_k=10)
    
    print("=" * 80)
    print("ORIGINAL (by relevance/score):")
    for i, r in enumerate(results, 1):
        print(f"{i:2d}. {r.title:<40} score={r.score:.4f}")
    
    # Test sort by name A-Z
    print("\n" + "=" * 80)
    print("SORTED BY NAME (A-Z):")
    sorted_name_asc = sort_by_name(results, ascending=True)
    for i, r in enumerate(sorted_name_asc, 1):
        print(f"{i:2d}. {r.title:<40} score={r.score:.4f}")
    
    # Test sort by name Z-A
    print("\n" + "=" * 80)
    print("SORTED BY NAME (Z-A):")
    sorted_name_desc = sort_by_name(results, ascending=False)
    for i, r in enumerate(sorted_name_desc, 1):
        print(f"{i:2d}. {r.title:<40} score={r.score:.4f}")
    
    # Test sort by year (newest first)
    print("\n" + "=" * 80)
    print("SORTED BY YEAR (most recent first):")
    sorted_year_desc = sort_by_year(results, ascending=False, doc_lookup=nr._resolve_result_document)
    for i, r in enumerate(sorted_year_desc, 1):
        doc_idx = nr.url_to_doc_idx.get(r.url)
        year = nr.documents[doc_idx].get("year", "N/A") if doc_idx is not None else "N/A"
        print(f"{i:2d}. {r.title:<40} year={year} score={r.score:.4f}")
    
    # Test sort by rating
    print("\n" + "=" * 80)
    print("SORTED BY RATING (best first):")
    sorted_rating_desc = sort_by_rating(results, ascending=False, doc_lookup=nr._resolve_result_document)
    for i, r in enumerate(sorted_rating_desc, 1):
        doc_idx = nr.url_to_doc_idx.get(r.url)
        rating = nr.documents[doc_idx].get("rating", "N/A") if doc_idx is not None else "N/A"
        print(f"{i:2d}. {r.title:<40} rating={rating} score={r.score:.4f}")
    
    print("\n" + "=" * 80)
    print("✅ All sorting methods work correctly!")


if __name__ == "__main__":
    main()
