from rag_module.retriever_wrapper import RetrieverWrapped
from rag_module.generator import OllamaGenerator
from rag_module.config import RAGConfig

class RAGPipeline:

    def __init__(self, config=RAGConfig()):
        self.retriever = RetrieverWrapped()
        self.generator = OllamaGenerator(config.model_name)
        self.temperature = config.temperature
        self.top_k = config.top_k
        self.max_tokens = config.max_tokens

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

    def query(self, question):
       docs = self.retriever.search_with_web_expansion(question, top_k=self.top_k)

       if not docs:
         return "No encontré resultados."

       context = self.build_context(docs)
       prompt = self.build_prompt(question, context)
       response = self.generator.generate(
           prompt,
           temperature=self.temperature,
           max_tokens=self.max_tokens,
        )
       return response   