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
        self.recommender = RecommendationEngine()
        self.default_user_id = None

    def build_context(self, docs):
        """Construir contexto desde documentos recuperados."""
        context_parts = []

        for i, doc in enumerate(docs, 1):
            title = getattr(doc, 'title', 'Sin título')
            plot = getattr(doc, 'plot', None) or getattr(doc, 'description', None)

            if plot:
                plot_clean = plot.strip()[:300]
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
        return f"""Eres un experto en películas y series. Responde siempre en español.

INFORMACION DISPONIBLE:
{context}

PREGUNTA: {query}

Recomienda las opciones más relevantes, explica brevemente por qué cada una encaja con la pregunta y sé específico."""

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

        # Build numbered list of titles + plots so the LLM has full context
        items_lines = []
        for i, result in enumerate(results, 1):
            title = getattr(result, 'title', '') or 'Sin título'
            plot = (getattr(result, 'plot', '') or '').strip()[:300]
            items_lines.append(f"{i}. {title}" + (f": {plot}" if plot else ""))

        items_text = "\n".join(items_lines)
        n = len(results)

        # Scale token budget: ~400 tokens per result, minimum 512
        dynamic_max_tokens = max(self.max_tokens, n * 400)

        prompt = (
            f"Eres un experto en películas y series. Responde siempre en español.\n\n"
            f"CONSULTA DEL USUARIO: {question}\n\n"
            f"TÍTULOS RECUPERADOS:\n{items_text}\n\n"
            f"Escribe EXACTAMENTE {n} descripciones numeradas, una por cada título en el MISMO orden que aparecen arriba. "
            f"Cada descripción (3-5 oraciones) debe: (a) resumir de qué trata ese título concreto y (b) explicar por qué encaja con la consulta. "
            f"Sé detallado y específico. No cambies el orden. Usa este formato estricto:\n"
            f"1. [descripción del título 1]\n"
            f"2. [descripción del título 2]\n"
            f"(y así hasta el {n})\n"
            f"No pongas el nombre del título en la descripción, empieza directamente con el contenido."
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

        return descriptions

    def query(self, question, user_id: str | None = None):
        docs = self.retriever.search_with_web_expansion(question, top_k=self.top_k)
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

        context = self.build_context(docs)
        prompt = self.build_prompt(question, context)
        response = self.generator.generate(
            prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response
