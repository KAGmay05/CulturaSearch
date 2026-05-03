from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from recommendation_module.content_extractor import extract_doc_feature
from neural_based_model.neural_retriever import NeuralRetriever, SearchResult
from recommendation_module.user_profile import User
import random

class ScoredDocument:
    """Documento con múltiples dimensiones de puntuación para personalización."""
    def __init__(self, document):
        self.document = document
        self.content_score = 0.0         # Similitud TF-IDF con perfil del usuario
        self.genre_score = 0.0           # Coincidencia con géneros preferidos
        self.recency_score = 0.0         # Similaridad con búsquedas recientes
        self.type_score = 0.0            # Coincidencia con tipo preferido (serie/película)
        self.viewed_penalty = 0.0        # Penalización si ya fue visto
        self.final_score = 0.0           # Puntaje final ponderado

class RecommendationEngine:
    """Motor de recomendación personalizado basado en perfil de usuario y contenido."""
    
    def __init__(self, content_weight=0.45, genre_weight=0.25, recency_weight=0.15, 
                 type_weight=0.1, viewed_penalty=0.05, min_score_threshold=0.0):
        self.content_weight = content_weight
        self.genre_weight = genre_weight
        self.recency_weight = recency_weight
        self.type_weight = type_weight
        self.viewed_penalty = viewed_penalty
        self.min_score_threshold = min_score_threshold
        self.tfidf_vectorizer = None

    def doc_get(self, doc, field, default=None):
        """Accede a campos de documento (dict o objeto)."""
        if isinstance(doc, dict):
            return doc.get(field, default)
        return getattr(doc, field, default)


    def normalize_user_profile(self, user_profile):
        """Normaliza el perfil del usuario a un diccionario estándar."""
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
                "top_genres": [],
                "top_type": None,
            }
        data = user_profile.to_dict()
        
        # Extraer top-3 géneros y top-1 tipo
        top_genres = [g for g, _ in sorted(data.get("genre_preferences", {}).items(), key=lambda x: -x[1])[:3]]
        top_type_list = [t for t, _ in sorted(data.get("type_preferences", {}).items(), key=lambda x: -x[1])[:1]]
        top_type = top_type_list[0] if top_type_list else None

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
            "top_genres": top_genres,
            "top_type": top_type,
        }
    
    def calculate_content_similarity(self, user_profile, documents):
        """Calcula similitud TF-IDF entre perfil del usuario y documentos."""
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer()

        normalized_profile = self.normalize_user_profile(user_profile)
        user_text = normalized_profile["combined_terms"].strip()

        if not documents:
            return []
        
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
        """Calcula puntuación final ponderada."""
        weighted_score = (
            self.content_weight * float(scored_doc.content_score)
            + self.genre_weight * float(scored_doc.genre_score)
            + self.recency_weight * float(scored_doc.recency_score)
            + self.type_weight * float(scored_doc.type_score)
        )
        # Aplicar penalización por visualización previa
        weighted_score -= scored_doc.viewed_penalty
        scored_doc.final_score = max(0.0, min(1.0, weighted_score))
        return scored_doc.final_score
    
    def calculate_genre_similarity(self, user_genres, document_genres):
        """Calcula similitud entre géneros preferidos del usuario y del documento."""
        if not user_genres:
            return 0.5  # Neutral si el usuario no tiene preferencias
        
        if not document_genres:
            return 0.0
        
        user_genres_lower = {g.lower() for g in user_genres.keys()}
        doc_genres_lower = {str(g).lower() for g in document_genres}
        matches = len(user_genres_lower.intersection(doc_genres_lower))
        
        if len(doc_genres_lower) == 0:
            return 0.0
        
        score = matches / len(doc_genres_lower)
        return min(1.0, score)
    
    def calculate_type_match(self, user_type, document_type):
        """Calcula si el tipo preferido del usuario coincide con el documento."""
        if not user_type or not document_type:
            return 0.5  # Neutral
        
        user_type_lower = str(user_type).lower().strip()
        doc_type_lower = str(document_type).lower().strip()
        
        if user_type_lower == doc_type_lower:
            return 1.0
        return 0.0
    
    def calculate_recency_boost(self, search_history, document_title):
        """Calcula boost de recencia si el documento es similar a búsquedas recientes."""
        if not search_history or not document_title:
            return 0.0
        
        doc_title_lower = str(document_title).lower()
        doc_words = set(doc_title_lower.split())
        
        for idx, search_query in enumerate(reversed(search_history)):
            search_lower = str(search_query).lower()
            search_words = set(search_lower.split())
            if doc_words.intersection(search_words):
                score = 0.9 ** idx  # Decae exponencialmente con antigüedad
                return min(1.0, score)
        return 0.0
    
    def content_based_score(self, user_profile, documents):
        """Calcula puntuaciones de contenido para múltiples dimensiones."""
        if not documents:
            return []
        
        normalized_profile = self.normalize_user_profile(user_profile)
        content_similarities = self.calculate_content_similarity(user_profile, documents)
        scored_documents = []
        
        # Preparar URLs vistas para comparación rápida
        viewed_urls_lower = {url.lower().strip() for url in normalized_profile.get("viewed_urls", [])}
        
        for idx, doc in enumerate(documents):
            scored_doc = ScoredDocument(doc)
            scored_doc.content_score = content_similarities[idx] if idx < len(content_similarities) else 0.0
            
            # Género
            doc_genres = self.doc_get(doc, "genres", [])
            user_genres = normalized_profile.get("genre_preferences", {})
            scored_doc.genre_score = self.calculate_genre_similarity(user_genres, doc_genres)
            
            # Tipo (serie/película)
            doc_type = self.doc_get(doc, "media_type", self.doc_get(doc, "type", ""))
            user_type = normalized_profile.get("top_type")
            scored_doc.type_score = self.calculate_type_match(user_type, doc_type)
            
            # Recencia
            doc_title = self.doc_get(doc, "title", "")
            search_history = normalized_profile.get("search_history", [])
            scored_doc.recency_score = self.calculate_recency_boost(search_history, doc_title)
            
            # Penalización por visualización
            doc_url = str(self.doc_get(doc, "url", "")).lower().strip()
            if doc_url in viewed_urls_lower:
                scored_doc.viewed_penalty = self.viewed_penalty
            
            self.calculate_final_score(scored_doc)
            scored_documents.append(scored_doc)

        return scored_documents    
    
    def personalize_results(self, user, retriever_results, top_k=10, diversity_ratio=0.3):
        """Personaliza resultados agregando diversidad (exploración + explotación).
        
        diversity_ratio: proporción de resultados exploratoria (baja similitud) vs explotación (alta similitud).
                        0.3 = 30% exploración, 70% explotación.
        """
        if not retriever_results:
            return []

        scored_docs = self.content_based_score(user, retriever_results)
        
        # Filtrar por threshold mínimo
        filtered_docs = [
            doc for doc in scored_docs 
            if doc.final_score >= self.min_score_threshold
        ]
        
        # Ordenar por puntuación final (explotación)
        sorted_docs = sorted(filtered_docs, key=lambda doc: doc.final_score, reverse=True)
        
        # Aplicar diversidad: mezclar exploración (scores bajos pero válidos) con explotación (scores altos)
        if len(sorted_docs) > top_k:
            num_exploitation = max(1, int(top_k * (1.0 - diversity_ratio)))
            num_exploration = top_k - num_exploitation
            
            exploitation_results = sorted_docs[:num_exploitation]
            exploration_results = sorted_docs[num_exploitation:]
            
            # Muestrear exploración aleatoriamente para diversidad
            if exploration_results and num_exploration > 0:
                exploration_sample = random.sample(exploration_results, min(num_exploration, len(exploration_results)))
                result = exploitation_results + exploration_sample
            else:
                result = exploitation_results
        else:
            result = sorted_docs[:top_k]
        
        return [doc.document for doc in result]
def _build_personalized_query(user_profile: User) -> str:
    """Construye una query sintética mejorada desde el perfil del usuario.
    
    Prioridades:
    1. Top-3 géneros (contexto temático principal)
    2. Top-5 términos (intereses específicos)
    3. Top-1 tipo (serie o película)
    
    Los términos se ponderan implícitamente al incluirlos primero.
    """
    if user_profile is None:
        return "películas series"
    
    query_parts = []
    
    # Top-3 géneros - contexto temático principal
    top_genres = [g for g, _ in sorted(user_profile.genre_preferences.items(), key=lambda x: -x[1])[:3]]
    if top_genres:
        query_parts.extend(top_genres)
    
    # Top-5 términos - intereses específicos más relevantes
    top_terms = [t for t, _ in sorted(user_profile.term_preferences.items(), key=lambda x: -x[1])[:5]]
    if top_terms:
        query_parts.extend(top_terms)
    
    # Top-1 tipo - filtro por tipo de contenido
    top_types = [t for t, _ in sorted(user_profile.type_preferences.items(), key=lambda x: -x[1])[:1]]
    if top_types:
        query_parts.extend(top_types)
    
    query = " ".join(query_parts) if query_parts else "películas series"
    return query.strip()


def recommend_for_user(user_profile: User, top_k: int = 5, diversity_ratio: float = 0.3) -> list[SearchResult]:
    """Recomendaciones personalizadas basadas en perfil del usuario.
    
    Args:
        user_profile: Perfil del usuario (con historial, preferencias, URLs vistas/clickeadas)
        top_k: Número de recomendaciones a retornar
        diversity_ratio: Proporción de exploración vs explotación (0.3 = 30% nuevos, 70% similares)
    
    Returns:
        Lista de documentos recomendados, personalizada y filtrada.
    """
    if user_profile is None:
        return []

    # 1. Construir query sintética mejorada
    query = _build_personalized_query(user_profile)
    
    try:
        retriever = NeuralRetriever()
        retriever.ensure_ready()
        
        # 2. Obtener más candidatos para poder filtrar y personalizar
        fetch_k = max(100, top_k * 10)  # Aumentado para mejor filtrado
        raw_results = retriever.search_advanced(
            query=query,
            top_k=fetch_k,
            candidate_k=fetch_k,
            alpha=0.7,
            rerank_weight=0.5
        )
        
        # 3. Preparar URLs filtradas (clicked + viewed)
        clicked_urls = {u.lower().strip() for u in getattr(user_profile, 'clicked_urls', []) if u}
        viewed_urls = {u.lower().strip() for u in getattr(user_profile, 'viewed_urls', []) if u}
        excluded_urls = clicked_urls | viewed_urls  # Unión de ambas
        
        # 4. Filtrar resultados que el usuario ya ha visto o clickeado
        filtered_results = [
            r for r in raw_results 
            if (getattr(r, 'url', '') or '').strip().lower() not in excluded_urls
        ]
        
        # 5. Personalizar con RecommendationEngine (re-puntuación + diversidad)
        engine = RecommendationEngine()
        personalized = engine.personalize_results(
            user_profile,
            filtered_results,
            top_k=top_k,
            diversity_ratio=diversity_ratio
        )
        
        return personalized
        
    except Exception as e:
        print(f"Error en recommend_for_user: {e}")
        return []
