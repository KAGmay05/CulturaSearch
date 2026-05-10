"""Demo to exercise positioning module with NeuralRetriever."""
from neural_based_model.neural_retriever import NeuralRetriever
from positioning_module.ranking import compute_positioning_score


def main():
    nr = NeuralRetriever()
    try:
        nr.ensure_ready()
    except Exception as e:
        print("Index/data not ready:", e)
        return

    q = "accion artes marciales"
    cand = nr.retrieve_candidates(query=q, candidate_k=20)
    neural = [c.neural_score for c in cand]
    lexical = [c.lexical_score for c in cand]
    docs = [nr._resolve_result_document(c) for c in cand]

    final = compute_positioning_score(neural, lexical, docs)
    ranked = sorted(zip(cand, final), key=lambda x: x[1], reverse=True)
    for i, (c, s) in enumerate(ranked, start=1):
        print(f"{i:02d}. {c.title} - {c.url} score={s:.4f} neural={c.neural_score:.4f} lex={c.lexical_score:.2f}")


if __name__ == "__main__":
    main()
