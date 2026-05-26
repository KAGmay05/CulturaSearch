from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from recommendation_module.content_extractor import extract_doc_feature, clean_text
from neural_based_model.neural_retriever import NeuralRetriever, SearchResult
from recommendation_module.user_profile import User
import random
import numpy as np
import re
import unicodedata

class ScoredDocument:
    """Documento con múltiples dimensiones de puntuación para personalización."""
    def __init__(self, document):
        self.document = document
        self.content_score = 0.0         # Similitud TF-IDF con perfil del usuario
        self.genre_score = 0.0           # Coincidencia con géneros preferidos
        self.recency_score = 0.0         # Similaridad con búsquedas recientes
        self.type_score = 0.0            # Coincidencia con tipo preferido (serie/película)
        self.viewed_penalty = 0.0        # Penalización si ya fue visto
        self.already_viewed = False      # Marca explícita de visualización previa
        self.metadata_score = 0.0        # Metadata signals del retriever (país, actores, temporadas, etc.)
        self.exact_title_match = False    # Coincidencia literal con el título consultado
        self.final_score = 0.0           # Puntaje final ponderado


def _normalize_plain_text(value):
    """Normaliza texto para comparaciones exactas de entidades y títulos."""
    if not value:
        return ""
    text = str(value).lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8", "ignore")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

class RecommendationEngine:
    """Motor de recomendación personalizado basado en perfil de usuario y contenido."""
    
    def __init__(self, content_weight=0.15, genre_weight=0.15, recency_weight=0.05, 
                 type_weight=0.10, viewed_penalty=0.05, min_score_threshold=0.0, metadata_weight=0.55):
        """Inicializa el motor de recomendación con pesos configurables.
        
        PESOS UTILIZADOS (deben sumar ~1.0):
        - metadata_weight (0.55): Preserva relevancia del retriever como signal principal
        - content_weight (0.15): Similitud TF-IDF con historial del usuario
        - genre_weight (0.15): Coincidencia con géneros preferidos del usuario
        - type_weight (0.10): Coincidencia con tipo preferido (serie/película)
        - recency_weight (0.05): Boost de búsquedas recientes
        - viewed_penalty (0.05): Penalización si el contenido ya fue visto
        
        NOTA: metadata_weight se incrementa automáticamente a 0.85 para búsquedas 
        de metadata específica (director, actor, creador) para preservar ranking.
        """
        self.content_weight = content_weight
        self.genre_weight = genre_weight
        self.recency_weight = recency_weight
        self.type_weight = type_weight
        self.viewed_penalty = viewed_penalty
        self.min_score_threshold = min_score_threshold
        self.metadata_weight = metadata_weight
        self._tfidf_vectorizer = None
        self._tfidf_vocab = None

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
        """Calcula similitud TF-IDF entre perfil del usuario y documentos. Cachea vectorizer."""
        normalized_profile = self.normalize_user_profile(user_profile)
        user_text = normalized_profile["combined_terms"].strip()
        user_text = " ".join(clean_text(user_text))

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

        # Cachear vectorizer para evitar refitting innecesario
        if self._tfidf_vectorizer is None:
            self._tfidf_vectorizer = TfidfVectorizer()
            tfidf_matrix = self._tfidf_vectorizer.fit_transform(texts)
        else:
            try:
                # Usar vocabulario cacheado si es posible
                tfidf_matrix = self._tfidf_vectorizer.transform(texts)
            except ValueError:
                # Si hay términos nuevos, refitear
                self._tfidf_vectorizer = TfidfVectorizer()
                tfidf_matrix = self._tfidf_vectorizer.fit_transform(texts)
        
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        return similarities.tolist()

    def calculate_final_score(self, scored_doc):
        """Calcula puntuación final ponderada, preservando ranking del retriever como signal principal.
        
        El score final se mantiene en [0, 1] para ser compatible con la UI y el ranking.
        """
        if scored_doc.exact_title_match:
            scored_doc.final_score = 1.0
            return scored_doc.final_score

        weighted_score = (
            self.metadata_weight * float(scored_doc.metadata_score)  # PRIORITARIO
            + self.content_weight * float(scored_doc.content_score)
            + self.genre_weight * float(scored_doc.genre_score)
            + self.recency_weight * float(scored_doc.recency_score)
            + self.type_weight * float(scored_doc.type_score)
        )
        
        # Aplicar penalización por visualización previa
        weighted_score -= float(scored_doc.viewed_penalty)
        scored_doc.final_score = float(np.clip(weighted_score, 0.0, 1.0))
        return scored_doc.final_score

    def _materialize_document_with_score(self, doc, final_score):
        """Adjunta el final_score al documento devuelto sin perder sus campos originales."""
        if isinstance(doc, dict):
            materialized = dict(doc)
            materialized["final_score"] = float(final_score)
            # Mantener score por compatibilidad, pero no sobreescribir el score base si existe.
            materialized.setdefault("score", materialized.get("score", float(final_score)))
            return materialized

        try:
            setattr(doc, "final_score", float(final_score))
        except Exception:
            pass
        return doc
    
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
    
    def calculate_weighted_genre_score(self, user_genres, document_genres):
        """Calcula similitud ponderada entre géneros (considera preferencia de usuario).
        
        A diferencia de calculate_genre_similarity, pesa los géneros según la preferencia del usuario.
        Géneros más preferidos tienen mayor impacto en la puntuación final.
        """
        if not user_genres:
            return 0.5  # Neutral si el usuario no tiene preferencias
        
        if not document_genres:
            return 0.0
        
        # Suma de pesos de géneros coincidentes
        matching_weight = 0.0
        doc_genres_lower = {str(g).lower() for g in document_genres}
        
        for genre, weight in user_genres.items():
            if genre.lower() in doc_genres_lower:
                matching_weight += weight
        
        # Suma total de preferencias del usuario
        total_weight = sum(user_genres.values())
        
        if total_weight == 0:
            return 0.0
        
        score = matching_weight / total_weight
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
    
    def content_based_score(self, user_profile, documents, query: str = None, apply_viewed_penalty: bool = False):
        """Calcula puntuaciones de contenido para múltiples dimensiones, preservando metadata signals."""
        if not documents:
            return []
        
        normalized_profile = self.normalize_user_profile(user_profile)
        content_similarities = self.calculate_content_similarity(user_profile, documents)
        scored_documents = []
        
        # Preparar URLs vistas para comparación rápida (solo se usarán si apply_viewed_penalty)
        viewed_urls_lower = {url.lower().strip() for url in normalized_profile.get("viewed_urls", [])} if apply_viewed_penalty else set()
        
        for idx, doc in enumerate(documents):
            scored_doc = ScoredDocument(doc)
            scored_doc.content_score = content_similarities[idx] if idx < len(content_similarities) else 0.0
            
            # Género
            doc_genres = self.doc_get(doc, "genres", [])
            user_genres = normalized_profile.get("genre_preferences", {})
            scored_doc.genre_score = self.calculate_weighted_genre_score(user_genres, doc_genres)
            
            # Tipo (serie/película)
            doc_type = self.doc_get(doc, "media_type", self.doc_get(doc, "type", ""))
            user_type = normalized_profile.get("top_type")
            scored_doc.type_score = self.calculate_type_match(user_type, doc_type)
            
            # Recencia
            doc_title = self.doc_get(doc, "title", "")
            search_history = normalized_profile.get("search_history", [])
            scored_doc.recency_score = self.calculate_recency_boost(search_history, doc_title)

            # Coincidencia literal de título: si la query contiene el título exacto,
            # el resultado debe mantenerse en la primera posición.
            if query:
                query_plain = _normalize_plain_text(query)
                title_plain = _normalize_plain_text(doc_title)
                if title_plain and title_plain in query_plain:
                    scored_doc.exact_title_match = True
            
            # Preservar metadata signals del retriever (si están disponibles en SearchResult)
            # Intentar obtener score del retriever (puede ser .score, .final_score, o estar normalizado 0-1)
            # IMPORTANTE: El retriever asegura que .final_score está en rango [0, 1]
            retriever_score = getattr(doc, 'final_score', getattr(doc, 'score', 0.0))
            if retriever_score > 0:
                # final_score está ya normalizado a [0, 1] por el retriever
                scored_doc.metadata_score = min(1.0, float(retriever_score))
            else:
                scored_doc.metadata_score = 0.0
            
            # Penalización por visualización (aplica solo si apply_viewed_penalty=True)
            if apply_viewed_penalty:
                doc_url = str(self.doc_get(doc, "url", "")).lower().strip()
                if doc_url in viewed_urls_lower:
                    scored_doc.viewed_penalty = self.viewed_penalty
                    scored_doc.already_viewed = True
            
            self.calculate_final_score(scored_doc)
            scored_documents.append(scored_doc)

        return scored_documents    
    
    def _is_metadata_specific_query(self, query: str) -> bool:
        """Detecta si la query busca metadata específica (director, actor, creador, etc.).
        
        Ejemplos:
        - 'películas de joe johnston' → True (director específico)
        - 'series de alex pina' → True (creador específico)
        - 'películas protagonizadas por elle fanning' → True (actor específico)
        - 'películas de romance' → False (género general)
        - 'series de españa' → False (país general)
        """
        q = query.lower().strip()
        
        # Palabras clave que indican búsqueda de metadata específica
        metadata_keywords = [
            'de ', 'director', 'creador', 'protagonizada', 'actor', 'actriz',
            'dirigida', 'creada', 'trabajo', 'trabajos'
        ]
        
        # Si contiene "de " + nombre propio (típicamente capitalizado)
        if ' de ' in q:
            parts = q.split(' de ')
            if len(parts) > 1:
                # Siguiente parte después de "de" probablemente sea nombre propio
                name_part = parts[-1].strip()
                # Si empieza con mayúscula o es un nombre, es metadata
                if name_part and (name_part[0].isupper() or any(c.isalpha() for c in name_part)):
                    return True
        
        # Si contiene palabras clave de actor/director/creador
        for keyword in metadata_keywords:
            if keyword in q:
                return True
        
        return False
    
    def personalize_results(self, user, retriever_results, top_k=10, diversity_ratio=0.3, query: str = None, apply_viewed_penalty: bool = False):
        """Personaliza resultados agregando diversidad (exploración + explotación).
        
        diversity_ratio: proporción de resultados exploratoria (baja similitud) vs explotación (alta similitud).
                        0.3 = 30% exploración, 70% explotación.
        
        query: Query actual (opcional). Si es metadata-específica, preserva ranking del retriever.
        """
        # Detectar si es búsqueda de metadata específica
        is_metadata_query = query is not None and self._is_metadata_specific_query(query)

        # Preparar mapeo de ranks originales (url -> rank) para preservación de top-N
        original_rank_map = {}
        try:
            for r in retriever_results:
                if isinstance(r, dict):
                    url = (r.get('url', '') or '').lower().strip()
                    rank = r.get('original_rank', r.get('rank'))
                else:
                    url = (getattr(r, 'url', '') or '').lower().strip()
                    rank = getattr(r, 'rank', None)
                if url:
                    try:
                        original_rank_map[url] = int(rank) if rank is not None else None
                    except Exception:
                        original_rank_map[url] = None
        except Exception:
            original_rank_map = {}

        # Si es búsqueda de metadata, aumentar metadata_weight para preservar ranking
        if is_metadata_query:
            original_metadata_weight = self.metadata_weight
            self.metadata_weight = 0.85  # Casi todo peso en metadata (preserva ranking retriever)
        else:
            original_metadata_weight = None
        if not retriever_results:
            return []

        scored_docs = self.content_based_score(user, retriever_results, query=query, apply_viewed_penalty=apply_viewed_penalty)
        
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
        
        # Reordenar por score final para que la salida sea coherente y estable.
        result = sorted(result, key=lambda doc: doc.final_score, reverse=True)
        
        # Si es query de metadata, forzar preservación de los top-2 del retriever
        if is_metadata_query and original_rank_map:
            # Determinar URLs a preservar (rank 1 y 2) en orden
            preserve_urls = [u for u, r in sorted(original_rank_map.items(), key=lambda x: (x[1] if x[1] is not None else 9999)) if r in (1, 2)]
            preserve_urls = [u for u in preserve_urls if u]

            if preserve_urls:
                ordered = []
                seen = set()
                # Añadir primero los preservados en el orden original
                for pu in preserve_urls:
                    for doc in result:
                        doc_url = (str(self.doc_get(doc.document, 'url', '')).lower().strip())
                        if doc_url == pu and doc_url not in seen:
                            ordered.append(doc)
                            seen.add(doc_url)
                            break

                # Añadir el resto manteniendo el orden por score
                for doc in result:
                    doc_url = (str(self.doc_get(doc.document, 'url', '')).lower().strip())
                    if doc_url not in seen:
                        ordered.append(doc)

                # Recortar al top_k pedido
                result = ordered[:top_k]

        # Restaurar weights originales
        if original_metadata_weight is not None:
            self.metadata_weight = original_metadata_weight

        return [self._materialize_document_with_score(doc.document, doc.final_score) for doc in result]
def _build_personalized_query(user_profile: User) -> str:
    """Construye una query sintética mejorada desde el perfil del usuario.
    
    Estructura mejorada:
    - Top-3 géneros (contexto temático principal)
    - Top-5 términos (intereses específicos)
    - Top-1 tipo (serie o película)
    
    Resultado: consulta más estructurada para mejorar relevancia del retriever.
    """
    if user_profile is None:
        return "peliculas series"
    
    # Top-3 géneros - contexto temático principal
    top_genres = [g for g, _ in sorted(user_profile.genre_preferences.items(), key=lambda x: -x[1])[:3]]
    
    # Top-5 términos - intereses específicos más relevantes
    top_terms = [t for t, _ in sorted(user_profile.term_preferences.items(), key=lambda x: -x[1])[:5]]
    
    # Top-1 tipo - filtro por tipo de contenido
    top_types = [t for t, _ in sorted(user_profile.type_preferences.items(), key=lambda x: -x[1])[:1]]
    
    # Construir query con estructura explícita
    query_parts = []
    
    # Géneros primero (mayor prioridad)
    if top_genres:
        query_parts.append(" ".join(top_genres))
    
    # Términos segundo (media prioridad)
    if top_terms:
        query_parts.append(" ".join(top_terms))
    
    # Tipo último (menor prioridad pero útil como filtro)
    if top_types:
        query_parts.extend(top_types)
    
    # Unir con espacios: cada grupo está separado naturalmente
    query = " ".join(query_parts) if query_parts else "peliculas series"
    return query.strip()


def _result_to_document(result, retriever: NeuralRetriever | None = None):
    """Convierte un SearchResult o dict en un documento dict completo cuando sea posible."""
    if isinstance(result, dict):
        # Preserve any existing rank info as original_rank for later use
        doc = dict(result)
        if 'original_rank' not in doc:
            doc['original_rank'] = doc.get('rank', None)
        return doc

    if retriever is not None:
        try:
            doc_idx = retriever.url_to_doc_idx.get(getattr(result, "url", ""))
            if doc_idx is not None and 0 <= doc_idx < len(retriever.documents):
                doc = retriever.documents[doc_idx]
                if isinstance(doc, dict):
                    return dict(doc)
        except Exception:
            pass

    return {
        "url": getattr(result, "url", ""),
        "title": getattr(result, "title", "Sin titulo"),
        "type": getattr(result, "media_type", getattr(result, "type", "desconocido")),
        "plot": getattr(result, "plot", ""),
        "genres": getattr(result, "genres", []),
        "actors": getattr(result, "actors", []),
        "director": getattr(result, "director", []),
        "final_score": getattr(result, "final_score", getattr(result, "score", 0.0)),
        "score": getattr(result, "score", 0.0),
        "original_rank": getattr(result, "rank", None),
    }


def recommend_for_user(
    user_profile: User,
    top_k: int = 5,
    diversity_ratio: float = 0.3,
    retriever: NeuralRetriever | None = None,
) -> list[SearchResult]:
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
        retriever = retriever or NeuralRetriever()
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
        filtered_results = []
        for r in raw_results:
            result_url = (getattr(r, 'url', '') or '').strip().lower()
            if result_url in excluded_urls:
                continue
            filtered_results.append(_result_to_document(r, retriever=retriever))
        
        # 5. Personalizar con RecommendationEngine (re-puntuación + diversidad)
        engine = RecommendationEngine()
        personalized = engine.personalize_results(
            user_profile,
            filtered_results,
            top_k=top_k,
            diversity_ratio=diversity_ratio,
            apply_viewed_penalty=True,
        )

        # Devolver SearchResult-like objects para compatibilidad con el resto del código y tests
        final_results: list[SearchResult] = []
        for rank, item in enumerate(personalized, start=1):
            if isinstance(item, dict):
                final_results.append(
                    SearchResult(
                        rank=rank,
                        score=float(item.get("final_score", item.get("score", 0.0))),
                        url=str(item.get("url", "")),
                        title=str(item.get("title", "Sin titulo")),
                        media_type=str(item.get("type", "desconocido")),
                        plot=str(item.get("plot", "")),
                        neural_score=float(item.get("score", item.get("final_score", 0.0))),
                        lexical_score=0.0,
                        rerank_score=0.0,
                        final_score=float(item.get("final_score", item.get("score", 0.0))),
                    )
                )
            else:
                final_results.append(item)

        return final_results
        
    except Exception as e:
        print(f"Error en recommend_for_user: {e}")
        return []
