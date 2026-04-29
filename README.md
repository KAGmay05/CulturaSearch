# CulturaSearch

Proyecto de recuperación de información en el dominio de cultura y entretenimiento.

## Instalación

Instala las dependencias necesarias:

```bash
pip install -r requirements.txt
```

## Estructura del Proyecto

El proyecto está dividido en módulos independientes pero integrados:

- **`crawler/`** - Descubridor de URLs desde SensaCine
- **`scraper/`** - Extractor de datos de URLs
- **`neural_based_model/`** - Recuperador neuronal con búsqueda semántica
- **`rag_module/`** - Pipeline RAG con generación LLM
- **`web_search/`** - Expansión web bajo demanda

## Integración: RAG + Módulo Web

El módulo RAG y el módulo web están **completamente integrados** en un único flujo:

1. Cuando haces una consulta en RAG, el sistema busca localmente.
2. Si la confianza local es baja, **automáticamente** activa la expansión web (`WebExpander`).
3. Los documentos web se scrapean, se re-indexan y se rankean junto con los locales.
4. La respuesta final usa los mejores documentos, sin importar si vinieron del corpus local o de la web.

 El RAG siempre usa `NeuralRetriever.search_with_web_expansion()`, que decide internamente cuándo buscar en la web.

### Configuración de Web Expansion

Edita `neural_based_model/neural_retriever.py` si quieres ajustar los umbrales:

```python
def search_with_web_expansion(
    self,
    query: str,
    top_k: int = 5,
    min_local_score: float = 0.72,          # Umbral de confianza
    min_lexical_coverage: float = 0.75,     # Cobertura de términos
    hard_min_local_score: float = 0.65,     # Mínimo absoluto
    web_max_results: int = 10,              # Máx docs web a scrapear
):
```

## Ejecución del Proyecto

### Flujo Completo (Recomendado)

Para ejecutar todo el pipeline (crawler → scraper → RAG interactivo):

```bash
python main.py
```

O explícitamente:

```bash
python main.py full
```

### Ejecutar Componentes Individuales

**Crawler** (descubre URLs de películas y series):
```bash
python -m crawler.run_crawler
```

**Scraper** (descarga datos de las URLs descubiertas):
```bash
python -m scraper.run_scraper
```

**Recuperador Neuronal Demo** (diagnóstico de búsqueda semántica):
```bash
python run_retrieval_demo.py
```

**RAG Interactivo** (búsqueda + generación con LLM):
```bash
python -m rag_module.run_rag
```

**Interfaz Visual** (búsqueda en lenguaje natural con ranking visual):
```bash
streamlit run app.py
```

También puedes abrirla desde el lanzador general:
```bash
python main.py ui
```

La interfaz está pensada para mostrar los resultados en tarjetas rankeadas, con métricas de relevancia, tipo de contenido y una presentación más limpia para navegar mejor los hallazgos.

### Desde el Lanzador General

Si prefieres usar `main.py` para cada subsistema:

```bash
python main.py crawl    # Solo crawler
python main.py scrape   # Solo scraper
python main.py rag      # Solo RAG
python main.py full     # Todo (default)
```

## Módulo Neural (Red Neuronal)

El recuperador neuronal usa embeddings densos con `sentence-transformers`.

### Uso Programático

```python
from neural_based_model.neural_retriever import NeuralRetriever

retriever = NeuralRetriever()
retriever.ensure_ready(force_rebuild=False)
results = retriever.search("serie policial con detectives", top_k=5)

for item in results:
    print(item.rank, item.score, item.title, item.url)
```

### Búsqueda con Expansión Web

Para activar expansión web automática si la confianza local es baja:

```python
results = retriever.search_with_web_expansion("tu consulta", top_k=5)
```

## Módulo RAG (Retrieval-Augmented Generation)

El módulo RAG integra **búsqueda semántica + expansión web + generación LLM** en un único pipeline.

### Características

- ✅ **Recuperación unificada**: búsqueda local + expansión web automática
- ✅ **Generación con Ollama**: respuestas en lenguaje natural
- ✅ **Re-ranking de resultados**: mejor precisión
- ✅ **Loop interactivo**: preguntas y respuestas por consola
- ✅ **Prompts en español**: respuestas contextualizadas

### Configuración

Edita `rag_module/config.py`:

```python
@dataclass
class RAGConfig:
    model_name: str = "neural-chat"    # Modelo Ollama local
    top_k: int = 4                     # Documentos a recuperar
    max_tokens: int = 512              # Máximo de tokens en respuesta
    temperature: float = 0.7           # Creatividad (0=determinístico, 1=creativo)
```

### Requisitos Previos

1. **Ollama** descargado e instalado, corriendo con:
```bash
ollama serve
```

2. **Dataset local** generado primero:
```bash
python main.py crawl
python main.py scrape
```

3. **Índice neural** construido:
```bash
python run_retrieval_demo.py
```

### Ejecutar RAG Interactivo

```bash
python -m rag_module.run_rag
```

Luego escribe preguntas sobre películas y series:
```
🔍 Pregunta: ¿Qué películas de acción hay?
💡 RESPUESTA: [respuesta generada con RAG + web]
```

El sistema automáticamente decidirá si buscar solo localmente o expandir con documentos web.