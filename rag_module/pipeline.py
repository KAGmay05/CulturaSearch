import re

from rag_module.retriever_wrapper import RetrieverWrapped
from rag_module.generator import OllamaGenerator
from rag_module.config import RAGConfig
from recommendation_module.recommendation import RecommendationEngine
from recommendation_module.profile_store import load_user


class RAGPipeline:

    def __init__(self, config=RAGConfig()):
        self.retriever = RetrieverWrapped()
        self.generator = OllamaGenerator(config.model_name)
        self.temperature = config.temperature
        self.top_k = config.top_k
        self.max_tokens = config.max_tokens
        self.rewrite_query = config.rewrite_query
        self.rewrite_max_tokens = config.rewrite_max_tokens
        self.recommender = RecommendationEngine()
        self.default_user_id = None
        # Caché en memoria para descripciones generadas (evita regenerar en clics "ver más")
        self._description_cache: dict = {}

    def build_context(self, docs, full: bool = False):
        """Construir contexto desde documentos recuperados."""
        context_parts = []

        for i, doc in enumerate(docs, 1):
            title = getattr(doc, 'title', 'Sin título')
            plot = getattr(doc, 'plot', None) or getattr(doc, 'description', None)
            if plot:
                # Si se solicita contexto completo (p. ej. "ver más"), no truncar demasiado
                if full:
                    plot_clean = plot.strip()[:2000]
                else:
                    plot_clean = plot.strip()[:400]
                context_parts.append(
                    f"[{i}] {title}\n"
                    f"Descripción: {plot_clean}")
            else:
                context_parts.append(f"[{i}] {title}")

        if not context_parts:
            return "(No hay información disponible)"

        return "\n\n".join(context_parts)

    def build_prompt(self, query, context):
        """Construir prompt para el modelo."""
        # Prompt estricto: obligar al modelo a usar SOLO la información proporcionada
        return (
            f"Eres un experto en películas y series. Responde siempre en español.\n\n"
            f"INFORMACIÓN DISPONIBLE:\n{context}\n\n"
            f"PREGUNTA: {query}\n\n"
            "INSTRUCCIONES (cumplir exactamente):\n"
            "- Usa SÓLO la información listada en 'INFORMACIÓN DISPONIBLE'. No añadas títulos ni datos externos.\n"
            "- Si NINGÚN documento coincide con la consulta, responde ÚNICAMENTE con: 'No hay suficiente información en los documentos proporcionados.' No mezcles esto con recomendaciones.\n"
            "- Para cada título relevante, escribe UN párrafo numerado (2-3 oraciones) explicando por qué encaja con la pregunta.\n"
            "- No inventes títulos, ni uses conocimiento externo, ni agregues aclaraciones o texto adicional fuera de los párrafos numerados o advertencias al final de la respuesta.\n"
            "- Responde sólo en español y mantén el formato estricto solicitado.\n"
        )

    def rewrite_search_query(self, question: str) -> str:
        """Reescribe la consulta para mejorar el retrieval sin cambiar la intención."""
        original = (question or "").strip()
        if not original:
            return original

        if not self.rewrite_query or not self.generator.is_available:
            return original

        # Detectar y preservar el tipo de contenido buscado (película/serie)
        content_type_markers = ["pelicula", "peliculas", "serie", "series", "film", "films", "movie", "movies", "show", "shows"]
        content_type_keywords = []
        q_lower = original.lower()
        for marker in content_type_markers:
            if marker in q_lower:
                # Normalizar a término común
                if marker in ["pelicula", "peliculas", "film", "films", "movie", "movies"]:
                    content_type_keywords.append("pelicula")
                elif marker in ["serie", "series", "show", "shows"]:
                    content_type_keywords.append("serie")
        content_type = " ".join(set(content_type_keywords))  # Evitar duplicados

        # PASO 1: Limpieza local de la consulta original (sin LLM)
        stopwords = {
            "recomiendame", "recomienda", "recomendar", "recoemidame", "recoemindame",
            "quiero", "busco", "dame", "deme", "muestrame", "muestra",
            "parecida", "parecidas", "parecido", "parecidos",
            "similar", "similares", "igualmente", "igual", "iguales", "tipo", "tipos",
            "pelicula", "peliculas", "serie", "series", "film", "films", "movie", "movies", "show", "shows",
            "como", "cual", "cuales", "cualesquiera",
            "de", "del", "la", "las", "el", "los", "un", "una", "unos", "unas",
            "por", "para", "que", "me", "y", "o", "a", "en", "con", "sin", "entre", "sobre",
            "es", "son", "sea", "seas", "seamos", "siendo", "eso", "esto",
            "me", "te", "se", "le", "nos", "os", "les", "sigue", "funcionar", "funcione", "funcionando", "funciona",
            "q", "salga", "sale", "salir", "salen", "aparezca", "aparece", "aparecen",
            "actor", "actriz", "actores", "actrices", "reparto", "elenco",
            "actua", "actuan", "protagoniza", "protagonizan", "protagonizada", "protagonizado",
        }
        
        tokens = re.findall(r"[A-Za-z0-9ÁÉÍÓÚÜÑáéíóúüñ']+", original)
        local_cleaned = [t for t in tokens if t.lower() not in stopwords]

        # PASO 1B: Consultas por actor/reparto.
        
        if self._is_cast_query(original):
            aliases = self._expand_cast_aliases(original)
            combined = local_cleaned + aliases
            if combined:
                # Mantener orden y evitar duplicados conservando la primera aparición.
                deduped = list(dict.fromkeys(t.lower() for t in combined if t.strip()))
                result = " ".join(deduped)[:300]
                # Añadir tipo de contenido al final
                if content_type:
                    result = f"{result} {content_type}"
                return result[:300]
            return original

        # PASO 2: Si es query de similitud, pide al LLM que infiera géneros del título
        is_similarity = self._is_similarity_query(original)
        if is_similarity and len(local_cleaned) >= 1 and self.generator.is_available:
            # Construye una lista de posibles títulos mencionados
            title_candidates = " ".join(local_cleaned)
            
            prompt = (
                "Eres un experto en cine y series. "
                "Dada una película o serie, infiere sus 3-4 géneros/temáticas principales en una palabra cada uno. "
                "SIN explicaciones, SIN preposiciones. "
                "Devuelve solo palabras clave separadas por espacios.\n\n"
                f"Película/Serie: {title_candidates}\n"
                "Géneros/Temáticas:"
            )
            
            genres = self.generator.generate(
                prompt,
                temperature=0.1,
                max_tokens=32,
            )
            
            genres = (genres or "").strip().strip('"').strip("'").strip(".")
            genre_tokens = re.findall(r"[A-Za-z0-9ÁÉÍÓÚÜÑáéíóúüñ']+", genres)
            genre_cleaned = [t for t in genre_tokens if t.lower() not in stopwords]
            
            # Combina: local_cleaned + géneros inferidos
            combined = local_cleaned + genre_cleaned
            if combined:
                result = " ".join(combined)[:300]
                # Añadir tipo de contenido al final
                if content_type:
                    result = f"{result} {content_type}"
                return result[:300]
        
        # PASO 3: Si no hay un título específico, pide al LLM expansión semántica.
        # Esto cubre consultas temáticas como "películas sobre viajes en el tiempo".
        if self.generator.is_available and (1 <= len(local_cleaned) <= 3 or not self._looks_like_specific_title(original)):
            keywords = " ".join(local_cleaned)
            
            prompt = (
                "Eres experto en búsqueda. Dada una temática/concepto de película o serie, "
                "genera 2-4 palabras clave o frases cortas relacionadas que amplíen la búsqueda. "
                "Si la consulta no menciona una película concreta, transforma la idea en términos de búsqueda útiles "
                "y relacionados. Prioriza conceptos, géneros, sinónimos y términos cercanos. "
                "Mantén SIEMPRE el dominio cine/series y no desvíes conceptos a otros contextos. "
                "Si un término puede ser nombre propio, consérvalo como nombre propio. "
                "NO devuelvas títulos de películas ni explicaciones. "
                "Devuelve solo palabras clave separadas por espacios.\n\n"
                f"Temática/Concepto: {keywords}\n"
                "Palabras clave relacionadas:"
            )
            
            expansions = self.generator.generate(
                prompt,
                temperature=0.2,
                max_tokens=32,
            )
            
            expansions = (expansions or "").strip().strip('"').strip("'").strip(".")
            expansion_tokens = re.findall(r"[A-Za-z0-9ÁÉÍÓÚÜÑáéíóúüñ']+", expansions)
            expansion_cleaned = [t for t in expansion_tokens if t.lower() not in stopwords]
            
            # Combina: original + expansiones
            combined = local_cleaned + expansion_cleaned
            if combined:
                result = " ".join(combined)[:300]
                # Añadir tipo de contenido al final
                if content_type:
                    result = f"{result} {content_type}"
                return result[:300]
        
        # Si la limpieza local produce buenos resultados (2+ palabras), usarlos
        if len(local_cleaned) >= 2:
            result = " ".join(local_cleaned)[:300]
            if content_type:
                result = f"{result} {content_type}"
            return result[:300]
        
        # FALLBACK: devolver lo que quedó de la limpieza local
        if local_cleaned:
            result = " ".join(local_cleaned)[:300]
            if content_type:
                result = f"{result} {content_type}"
            return result[:300]
        
        return original

    @staticmethod
    def _is_cast_query(question: str) -> bool:
        """Detecta si la consulta pide resultados por actor/actriz/reparto."""
        q = (question or "").lower().strip()
        if not q:
            return False

        cast_markers = (
            "actor", "actriz", "actores", "actrices", "reparto", "elenco",
            "salga", "sale", "salen", "aparezca", "aparece", "aparecen",
            "actua", "actuan", "protagoniza", "protagonizan",
        )
        return any(marker in q for marker in cast_markers)

    @staticmethod
    def _expand_cast_aliases(question: str) -> list[str]:
        """Expande alias y corrige errores comunes en nombres de actores."""
        q = (question or "").lower()
        aliases: list[str] = []

        if "la roca" in q or "the rock" in q:
            aliases.extend(["dwayne", "johnson"])

        return aliases

    @staticmethod
    def _looks_like_specific_title(question: str) -> bool:
        """Detecta si la consulta parece mencionar un título concreto de película o serie."""
        q = (question or "").strip()
        if not q:
            return False

        title_markers = (
            '"', "'", " llamada ", " titulado ", " titulada ",
        )
        if any(marker in f" {q.lower()} " for marker in title_markers):
            return True

        tokens = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ][A-Za-z0-9ÁÉÍÓÚÜÑáéíóúüñ']*", q)
        capitalized_tokens = sum(1 for token in tokens if token[:1].isupper())
        return capitalized_tokens >= 2

    @staticmethod
    def _is_similarity_query(question: str) -> bool:
        """Detecta si la query pide contenido similar a un título concreto."""
        import re as _re
        patterns = [
            r'parecid[ao]s?\s+a', r'similar(?:es)?\s+a', r'como\s+[\"\']?[A-Z]',
            r'tipo\s+[\"\']?[A-Z]', r'al\s+estilo\s+de', r'del\s+estilo\s+de',
            r'recomienda.*(?:similar|parecid)', r'(?:busco|quiero).*como',
        ]
        q = question.lower().strip()
        return any(_re.search(p, q, _re.IGNORECASE) for p in patterns)

    def generate_descriptions(self, question: str, results: list) -> dict:
        """Generate a per-result RAG description, keyed by result URL.

        Takes the already-retrieved SearchResult objects so that each
        description is strictly anchored to the title it belongs to.
        Returns a dict {url: description_text}.
        """
        if not results:
            return {}

        if not self.generator.is_available:
            return {}

        # Construir contexto completo (para descripciones detalladas) y usar caché.
        urls = tuple(getattr(r, 'url', '') or '' for r in results)
        cache_key = (question or '', urls)
        if cache_key in self._description_cache:
            return self._description_cache[cache_key]

        # Construir contexto con texto más largo para descripciones
        context = self.build_context(results, full=True)
        n = len(results)

        # Scale token budget: ~120 tokens per result, cap at 400 total
        dynamic_max_tokens = min(400, max(300, n * 120))

        is_similarity = self._is_similarity_query(question)

        if is_similarity:
            task_instruction = (
                f"La consulta pide contenido SIMILAR o PARECIDO a un título de referencia. "
                f"Para cada título numerado explica: de qué trata, qué elementos concretos comparte "
                f"con el referente mencionado en la consulta (estilo, humor, personajes, ambientación, temas), "
                f"y por qué lo disfrutaría alguien que le gustó ese referente. "
                f"Sé específico sobre las similitudes, no digas solo 'es parecido'."
            )
        else:
            task_instruction = (
                f"Para cada título numerado explica de qué trata, qué lo hace especial "
                f"y por qué encaja con la consulta del usuario. Sé específico y narrativo."
            )

        prompt = (
            f"Eres un experto en películas y series. Responde siempre en español.\n\n"
            f"CONSULTA DEL USUARIO: {question}\n\n"
            f"INFORMACIÓN DISPONIBLE:\n{context}\n\n"
            f"{task_instruction}\n\n"
            f"Escribe EXACTAMENTE {n} párrafos numerados en el MISMO orden que aparecen arriba. "
            f"Cada párrafo: 3-5 oraciones, narrativo y basado SOLAMENTE "
            f"en la información proporcionada. "
            f"No inventes detalles ni uses conocimiento externo. "
            f"Si falta información, dilo explícitamente. "
            f"Formato estricto:\n"
            f"1. [párrafo del título 1]\n"
            f"2. [párrafo del título 2]\n"
            f"(y así hasta el {n})\n"
            f"No escribas el nombre del título al inicio del párrafo."
        )

        response = self.generator.generate(
            prompt,
            temperature=self.temperature,
            max_tokens=dynamic_max_tokens,
        )

        # Parse numbered response → dict URL → description
        # Capture everything from one numbered item to the next (multi-line)
        descriptions: dict = {}
        pattern = re.compile(
            r'^\s*(\d{1,2})[.):\-]\s+(.*?)(?=^\s*\d{1,2}[.):\-]\s+|\Z)',
            re.MULTILINE | re.DOTALL,
        )
        for match in pattern.finditer(response):
            idx = int(match.group(1)) - 1
            desc = match.group(2).strip()
            if 0 <= idx < len(results):
                url = getattr(results[idx], 'url', '') or ''
                if url:
                    descriptions[url] = desc

        # Guardar en caché y devolver
        self._description_cache[cache_key] = descriptions
        return descriptions

    def query_with_docs(self, question: str, docs: list) -> str:
        """Genera respuesta LLM usando docs ya recuperados (evita doble búsqueda).

        Úsalo desde la UI pasando los resultados de run_query() para que
        la narrativa y las descripciones por tarjeta compartan el mismo contexto.
        """
        if not docs:
            return "No encontré resultados."

        if not self.generator.is_available:
            return "⚠️ Ollama no está disponible. Ejecuta: ollama serve"

        # Por defecto usamos contexto corto para respuestas rápidas en la UI.
        context = self.build_context(docs, full=False)
        prompt = self.build_prompt(question, context)
        return self.generator.generate(
            prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def query(self, question, user_id: str | None = None):
        search_query = self.rewrite_search_query(question)
        if search_query.strip() and search_query.strip() != (question or "").strip():
            print(f"[RAG] Consulta optimizada: {search_query}")
        docs = self.retriever.search_with_web_expansion(search_query, top_k=self.top_k)
        print("\n===== DOCUMENTOS RECUPERADOS =====")

        for i, doc in enumerate(docs, 1):
          title = getattr(doc, "title", "Sin título")
          plot = getattr(doc, "plot", "") or getattr(doc, "description", "")

          print(f"\n[{i}] {title}")

          if plot:
            print(plot[:400])

        print("\n==================================\n")
        if not docs:
            return "No encontré resultados."

        uid = user_id if user_id is not None else self.default_user_id
        user = load_user(uid) if uid else None

        try:
            personalized = self.recommender.personalize_results(user, docs, top_k=self.top_k)
            if personalized:
                docs = personalized
        except Exception:
            pass

        return self.query_with_docs(question, docs)
