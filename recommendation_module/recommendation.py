from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from recommendation_module.content_extractor import extract_doc_feature
from neural_based_model.neural_retriever import NeuralRetriever, SearchResult
from recommendation_module.user_profile import User

class ScoredDocument:
    def __init__(self, document):
        self.document = document
        self.content_score = 0.0         #similitud con busqueda del usuario
        self.genre_score = 0.0           #match con generos preferido
        self.recency_score = 0.0         #busqueda recinte similar
        self.already_viewed = False      #si el usurio ya la vio
        self.final_score = 0.0           #puntaje final

class RecommendationEngine:
    def __init__(self, content_weight = 0.5, genre_weight = 0.3, recency_weight = 0.2, viewed_penalty = 0.1, min_score_threshold = 0.0):
        self.content_weight = content_weight
        self.genre_weight = genre_weight
        self.recency_weight = recency_weight
        self.viewed_penalty = viewed_penalty
        self.min_score_threshold = min_score_threshold
        self.tfidf_vectorizer = None

    def doc_get(self, doc, field, default=None):
        if isinstance(doc, dict):
            return doc.get(field, default)
        return getattr(doc, field, default)


    def normalize_user_profile(self, user_profile):
        if user_profile is None:
            return {
                "user_id": "anonymous",
                "search_history": [],
                "clicked_urls": [],
                "viewed_urls": [],
                "genre_preferences": {},
                "type_preferences": {},
                "term_preferences": {},
                "combined_terms": "",
            }
        data = user_profile.to_dict()
        combined_terms = []

        for query in data.get("search_history", []):
            if query:
                combined_terms.append(str(query))

        for term in data.get("term_preferences", {}).keys():
            if term:
                combined_terms.append(str(term))

        return {
            "user_id": data.get("user_id", "anonymous"),
            "search_history": data.get("search_history", []),
            "clicked_urls": data.get("clicked_urls", []),
            "viewed_urls": data.get("viewed_urls", []),
            "genre_preferences": data.get("genre_preferences", {}),
            "type_preferences": data.get("type_preferences", {}),
            "term_preferences": data.get("term_preferences", {}),
            "combined_terms": " ".join(combined_terms).strip(),
        }
    
    def calculate_content_similarity(self, user_profile, documents):
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer()

        normalized_profile = self.normalize_user_profile(user_profile)
        user_text = normalized_profile["combined_terms"].strip()

        if not documents:
            return
        
        doc_texts = []
        for doc in documents:
            if isinstance(doc, dict):
                doc_data = doc
            else:
                doc_data = {
                    "url": self.doc_get(doc, "url", ""),
                    "title": self.doc_get(doc, "title", ""),
                    "type": self.doc_get(doc, "media_type", self.doc_get(doc, "type", "")),
                    "plot": self.doc_get(doc, "plot", ""),
                    "genres": self.doc_get(doc, "genres", []),
                    "actors": self.doc_get(doc, "actors", []),
                    "director": self.doc_get(doc, "director", []),
                }
            feature = extract_doc_feature(doc_data)
            tokens = feature.get("tokens", [])
            doc_text = " ".join(tokens).strip()
            doc_texts.append(doc_text)

        texts = [user_text] + doc_texts
        if not user_text or not any(texts[1:]):
            return [0.0] * len(documents)

        self.tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        return similarities.tolist()

    def calculate_final_score(self, scored_doc):
        weighted_score = (
            self.content_weight * float(scored_doc.content_score)
            + self.genre_weight * float(scored_doc.genre_score)
            + self.recency_weight * float(scored_doc.recency_score)
        )
        if scored_doc.already_viewed:
            weighted_score -= self.viewed_penalty
        scored_doc.final_score = max(0.0, min(1.0, weighted_score))
        return scored_doc.final_score
    
    def calculate_genre_similarity(self, user_genres, document_genres):
        if not user_genres:
            return 0.5
        
        if not document_genres:
            return 0.0
        
        user_genres_lower = {g.lower() for g in user_genres.keys()}
        doc_genres_lower = {str(g).lower() for g in document_genres}
        matches = len(user_genres_lower.intersection(doc_genres_lower))
        
        if len(doc_genres_lower) == 0:
            return 0.0
        
        score = matches / len(doc_genres_lower)
        return min(1.0, score)
    
    def calculate_recency_boost(self, search_history, document_title):
        if not search_history or not document_title:
            return 0.0
        
        doc_title_lower = str(document_title).lower()
        doc_words = set(doc_title_lower.split())
        
        for idx, search_query in enumerate(reversed(search_history)):
            search_lower = str(search_query).lower()
            search_words = set(search_lower.split())
            if doc_words.intersection(search_words):
                score = 0.9 ** idx
                return min(1.0, score)
        return 0.0
    
    def content_based_score(self, user_profile, documents):
        if not documents:
            return []
        
        normalized_profile = self.normalize_user_profile(user_profile)
        content_similarities = self.calculate_content_similarity(user_profile, documents)
        scored_documents = []
        
        for idx, doc in enumerate(documents):
            scored_doc = ScoredDocument(doc)
            scored_doc.content_score = content_similarities[idx] if idx < len(content_similarities) else 0.0
            
            doc_genres = self.doc_get(doc, "genres", [])
            user_genres = normalized_profile.get("genre_preferences", {})
            scored_doc.genre_score = self.calculate_genre_similarity(user_genres, doc_genres)
            
            doc_title = self.doc_get(doc, "title", "")
            search_history = normalized_profile.get("search_history", [])
            scored_doc.recency_score = self.calculate_recency_boost(search_history, doc_title)
            
            doc_url = str(self.doc_get(doc, "url", "")).lower()
            viewed_urls = [url.lower() for url in normalized_profile.get("viewed_urls", [])]
            scored_doc.already_viewed = doc_url in viewed_urls
            
            self.calculate_final_score(scored_doc)
            scored_documents.append(scored_doc)

        return scored_documents    
    
    def personalize_results(self, user, retriever_results, top_k=10):
        if not retriever_results:
            return []

        scored_docs = self.content_based_score(user, retriever_results)
        filtered_docs = [
            doc for doc in scored_docs 
            if doc.final_score >= self.min_score_threshold
        ]
        
        sorted_docs = sorted(filtered_docs, key=lambda doc: doc.final_score, reverse=True)
        result = [doc.document for doc in sorted_docs[:top_k]]
        
        return result


def recommend_for_user(user_profile: User, retriever: NeuralRetriever | None = None, top_k: int = 5) -> list[SearchResult]:
    if user_profile is None:
        return []

    # construir query sintética a partir de term_preferences o genre_preferences
    terms = sorted(user_profile.term_preferences.items(), key=lambda x: -x[1])[:3]
    if terms:
        query = " ".join([t for t, _ in terms])
    else:
        genres = sorted(user_profile.genre_preferences.items(), key=lambda x: -x[1])[:3]
        if genres:
            query = " ".join([g for g, _ in genres])
        else:
            query = "peliculas series"

    if retriever is None:
        retriever = NeuralRetriever()
        retriever.ensure_ready(force_rebuild=False)

    try:
        results = retriever.search_advanced(query=query, top_k=top_k, candidate_k=max(50, top_k), alpha=0.9, rerank_weight=0.75)
    except Exception:
        results = []

    try:
        recommender = RecommendationEngine()
        personalized = recommender.personalize_results(user_profile, results, top_k=top_k)
        return personalized if personalized else results
    except Exception:
        return results