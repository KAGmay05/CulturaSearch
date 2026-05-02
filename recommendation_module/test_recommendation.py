import unittest

from neural_based_model.neural_retriever import SearchResult
from recommendation_module.recommendation import RecommendationEngine, recommend_for_user
from recommendation_module.user_profile import User


def print_ranked_results(label, results):
    print(f"\n{label}")
    print("-" * 88)
    print(f"{'#':<3} {'Title':<28} {'Type':<10} {'Score':<8} URL")
    print("-" * 88)
    for idx, item in enumerate(results, start=1):
        title = (item.title[:25] + "...") if len(item.title) > 28 else item.title
        print(f"{idx:<3} {title:<28} {item.media_type:<10} {item.score:<8.3f} {item.url}")


def print_rank_changes(baseline, personalized):
    base_pos = {item.title: idx + 1 for idx, item in enumerate(baseline)}
    print("\nCambios de posicion")
    print("-" * 48)
    print(f"{'Title':<30} {'Base':<6} {'Personalizado':<13} {'Delta':<6}")
    print("-" * 48)
    for idx, item in enumerate(personalized, start=1):
        base = base_pos.get(item.title, "-")
        if isinstance(base, int):
            delta = base - idx
            delta_txt = f"{delta:+d}"
        else:
            delta_txt = "n/a"
        print(f"{item.title[:30]:<30} {str(base):<6} {idx:<13} {delta_txt:<6}")


def make_result(rank, score, url, title, media_type, plot):
    return SearchResult(
        rank=rank,
        score=score,
        url=url,
        title=title,
        media_type=media_type,
        plot=plot,
    )


def make_doc(url, title, media_type, plot, genres=None):
    return {
        "url": url,
        "title": title,
        "type": media_type,
        "plot": plot,
        "genres": genres or [],
        "actors": [],
        "director": [],
    }


class FakeRetriever:
    def __init__(self, results):
        self.results = list(results)
        self.queries = []

    def ensure_ready(self, force_rebuild=False):
        return None

    def search_advanced(self, query, top_k=5, candidate_k=50, alpha=0.9, rerank_weight=0.75):
        self.queries.append(query)
        return list(self.results[:top_k])


class RecommendationModuleTestCase(unittest.TestCase):
    def test_normalize_user_profile_combines_history_and_terms(self):
        user = User("norm")
        user.search_history = ["comedia romantica", "series detectives"]
        user.term_preferences = {"comedia": 3, "drama": 1}

        engine = RecommendationEngine()
        normalized = engine.normalize_user_profile(user)

        self.assertEqual(normalized["user_id"], "norm")
        self.assertIn("comedia romantica", normalized["combined_terms"])
        self.assertIn("series detectives", normalized["combined_terms"])
        self.assertIn("comedia", normalized["combined_terms"])
        self.assertIn("drama", normalized["combined_terms"])

    def test_genre_similarity_edge_cases_and_matches(self):
        engine = RecommendationEngine()

        self.assertEqual(engine.calculate_genre_similarity({}, ["comedia"]), 0.5)
        self.assertEqual(engine.calculate_genre_similarity({"comedia": 1}, []), 0.0)
        self.assertAlmostEqual(
            engine.calculate_genre_similarity({"comedia": 2, "drama": 1}, ["comedia", "accion"]),
            0.5,
        )

    def test_recency_boost_prefers_recent_matches(self):
        engine = RecommendationEngine()
        history = ["accion", "thriller", "comedia romantica"]

        score_recent = engine.calculate_recency_boost(history, "Gran comedia romantica")
        score_old = engine.calculate_recency_boost(history, "Aventura de accion")
        score_none = engine.calculate_recency_boost(history, "documental natural")

        self.assertGreater(score_recent, score_old)
        self.assertGreater(score_old, 0.0)
        self.assertEqual(score_none, 0.0)

    def test_viewed_penalty_reduces_final_score(self):
        user = User("alice")
        user.search_history = ["comedia romantica"]
        user.term_preferences = {"comedia": 2, "romantica": 2}
        user.viewed_urls = ["https://example.com/viewed"]

        docs = [
            make_doc(
                "https://example.com/viewed",
                "Comedia vista",
                "pelicula",
                "comedia romantica divertida",
                genres=["comedia"],
            ),
            make_doc(
                "https://example.com/new",
                "Comedia nueva",
                "pelicula",
                "comedia romantica divertida",
                genres=["comedia"],
            ),
        ]

        engine = RecommendationEngine(viewed_penalty=0.3)
        scored = engine.content_based_score(user, docs)
        by_url = {item.document["url"]: item for item in scored}

        self.assertTrue(by_url["https://example.com/viewed"].already_viewed)
        self.assertFalse(by_url["https://example.com/new"].already_viewed)
        self.assertLess(
            by_url["https://example.com/viewed"].final_score,
            by_url["https://example.com/new"].final_score,
        )

    def test_min_score_threshold_filters_results(self):
        user = User("threshold")
        user.search_history = ["comedia"]
        user.term_preferences = {"comedia": 1}

        docs = [
            make_doc("u1", "Comedia 1", "pelicula", "comedia divertida", ["comedia"]),
            make_doc("u2", "Drama 1", "serie", "drama oscuro", ["drama"]),
        ]

        engine = RecommendationEngine(min_score_threshold=0.4)
        filtered = engine.personalize_results(user, docs, top_k=10)

        self.assertGreaterEqual(len(filtered), 1)
        # Si hay filtrado correcto, no deben regresar resultados por debajo del umbral
        rescored = engine.content_based_score(user, docs)
        allowed_urls = {d.document["url"] for d in rescored if d.final_score >= 0.4}
        returned_urls = {d["url"] for d in filtered}
        self.assertTrue(returned_urls.issubset(allowed_urls))

    def test_personalization_changes_query_ranking(self):
        user = User("alice")
        user.search_history = ["comedia romantica"]
        user.term_preferences = {"comedia": 3, "romantica": 2}
        user.genre_preferences = {"comedia": 2}
        user.viewed_urls = ["https://example.com/action"]

        raw_results = [
            make_result(
                1,
                0.95,
                "https://example.com/action",
                "Accion explosiva",
                "pelicula",
                "Explosiones, persecuciones y un agente secreto.",
            ),
            make_result(
                2,
                0.72,
                "https://example.com/comedy",
                "Comedia romantica",
                "pelicula",
                "Una historia divertida con romance y muchas risas.",
            ),
            make_result(
                3,
                0.50,
                "https://example.com/documentary",
                "Documental musical",
                "serie",
                "Un recorrido por la historia de la musica.",
            ),
        ]

        engine = RecommendationEngine()
        personalized = engine.personalize_results(user, raw_results, top_k=3)

        raw_titles = [item.title for item in raw_results]
        personalized_titles = [item.title for item in personalized]

        self.assertEqual(raw_titles, ["Accion explosiva", "Comedia romantica", "Documental musical"])
        self.assertNotEqual(raw_titles, personalized_titles)
        self.assertEqual(personalized_titles[0], "Comedia romantica")

    def test_recommend_for_user_without_query_uses_term_preferences_first(self):
        user = User("bob")
        user.search_history = ["comedia romantica"]
        user.term_preferences = {"romantica": 4, "comedia": 2}
        user.viewed_urls = ["https://example.com/action"]

        raw_results = [
            make_result(
                1,
                0.88,
                "https://example.com/action",
                "Accion explosiva",
                "pelicula",
                "Explosiones, persecuciones y un agente secreto.",
            ),
            make_result(
                2,
                0.76,
                "https://example.com/comedy",
                "Comedia ligera",
                "pelicula",
                "Una comedia ligera con situaciones divertidas y calidez.",
            ),
            make_result(
                3,
                0.55,
                "https://example.com/drama",
                "Drama intenso",
                "serie",
                "Una historia emotiva y profunda.",
            ),
        ]

        fake_retriever = FakeRetriever(raw_results)
        recommendations = recommend_for_user(user, retriever=fake_retriever, top_k=2)

        generated_query = fake_retriever.queries[0]
        recommended_titles = [item.title for item in recommendations]
        baseline_titles = [item.title for item in raw_results[:2]]

        self.assertEqual(generated_query, "romantica comedia")
        self.assertEqual(len(recommendations), 2)
        self.assertNotEqual(baseline_titles, recommended_titles)
        self.assertEqual(recommended_titles[0], "Comedia ligera")

    def test_recommend_for_user_without_query_uses_genres_if_no_terms(self):
        user = User("genre-user")
        user.genre_preferences = {"accion": 3, "drama": 2, "comedia": 1}

        retriever = FakeRetriever([])
        recommend_for_user(user, retriever=retriever, top_k=2)

        self.assertEqual(retriever.queries[0], "accion drama comedia")

    def test_recommend_for_user_without_profile_falls_back_to_default_query(self):
        user = User("new-user")
        retriever = FakeRetriever([])

        recommend_for_user(user, retriever=retriever, top_k=2)

        self.assertEqual(retriever.queries[0], "peliculas series")

    def test_recommend_for_user_returns_empty_when_user_is_none(self):
        retriever = FakeRetriever([])
        result = recommend_for_user(None, retriever=retriever, top_k=3)
        self.assertEqual(result, [])
        self.assertEqual(retriever.queries, [])

    def test_recommend_for_user_keeps_order_if_personalization_fails(self):
        class FailingRetriever(FakeRetriever):
            def search_advanced(self, query, top_k=5, candidate_k=50, alpha=0.9, rerank_weight=0.75):
                raise RuntimeError("boom")

        user = User("fallback")
        user.term_preferences = {"accion": 1}
        retriever = FailingRetriever([])

        result = recommend_for_user(user, retriever=retriever, top_k=5)
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)