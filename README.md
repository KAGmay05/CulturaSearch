# CulturaSearch

Proyecto de recuperacion de informacion en el dominio de cultura y entretenimiento.

## Modelo no basico (Red neuronal)

Para el primer corte se incluyo un recuperador neuronal en `non_basic_model/` basado en embeddings densos con `sentence-transformers`.

### Dependencias

Instala estas librerias en tu entorno de Python:

```bash
pip install sentence-transformers scikit-learn numpy
```

O instala todo desde el archivo del proyecto:

```bash
pip install -r requirements.txt
```

### Ejecutar demo de busqueda semantica

```bash
python -m non_basic_model.demo
```

Este flujo realiza:

1. Carga del modelo `paraphrase-multilingual-MiniLM-L12-v2`.
2. Construccion/carga de la base vectorial inicial en `bd/movies_vectors.pkl`.
3. Ranking de resultados por similitud coseno para consultas en lenguaje natural.

### Uso programatico

```python
from non_basic_model.neural_retriever import NeuralRetriever

retriever = NeuralRetriever()
retriever.ensure_ready(force_rebuild=False)
results = retriever.search("serie policial con detectives", top_k=5)

for item in results:
	print(item.rank, item.score, item.title, item.url)
```

## Módulo RAG (Retrieval-Augmented Generation)

El módulo RAG integra búsqueda semántica con generación de lenguaje para proporcionar respuestas enriquecidas y contextualmente relevantes.

### ¿Qué es RAG?

RAG combina:
1. **Retrieval**: Recuperar documentos relevantes del corpus
2. **Generation**: Generar respuestas basadas en el contexto recuperado

### Características

- ✅ Integración transparente con NeuralRetriever existente
- ✅ Soporte para múltiples backends LLM (Ollama, OpenAI, HuggingFace)
- ✅ Re-ranking de resultados para mayor precisión
- ✅ Prompts personalizables
- ✅ Batch processing de consultas
- ✅ Exportación de resultados a JSON

### Instalación Rápida

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Configurar backend LLM (Ollama recomendado)
# Opción A: Ollama (local)
ollama pull mistral
ollama serve

# Opción B: OpenAI
export OPENAI_API_KEY=sk-...

# 3. Usar el pipeline RAG
python rag_demo.py --interactive
```

### Uso Básico

```python
from rag_module import RAGConfig, RAGPipeline, LLMBackend

# Crear configuración
config = RAGConfig(
    llm_backend=LLMBackend.OLLAMA,
    model_name="mistral",
    retriever_top_k=5
)

# Inicializar pipeline
rag = RAGPipeline(config)
rag.initialize()

# Realizar consulta
result = rag.query("¿Qué películas de acción puedo ver?")

# Mostrar resultado
rag.print_result(result)
```

### Ejemplos

```bash
# Modo interactivo
python rag_demo.py --interactive

# Ejecutar ejemplos específicos
python rag_demo.py --example 1  # RAG básico
python rag_demo.py --example 2  # Con re-ranking
python rag_demo.py --example 3  # Batch processing
python rag_demo.py --example 4  # OpenAI API
python rag_demo.py --example 5  # Prompts personalizados
python rag_demo.py --example 6  # Verificar backends
```

### Uso Programático Avanzado

```python
from rag_module import RAGConfig, RAGPipeline, LLMBackend

# Consulta con re-ranking
result = rag.query_with_reranking(
    "Tu pregunta aquí",
    initial_top_k=20,
    final_top_k=5
)

# Batch de consultas
questions = [
    "¿Qué películas de acción hay?",
    "¿Mejores series de drama?"
]
results = rag.batch_query(questions)
rag.export_results(results, "resultados.json")
```

### Arquitectura del Módulo RAG

```
rag_module/
├── config.py       # Configuración centralizada
├── generator.py    # Generadores LLM (Ollama, OpenAI, HF)
├── pipeline.py     # Pipeline RAG completo
└── utils.py        # Funciones de utilidad
└── demo.py         # Ejemplos de uso


### Parámetros Configurables

| Parámetro | Rango | Descripción |
|-----------|-------|-------------|
| `retriever_top_k` | 1-20 | Documentos a recuperar |
| `temperature` | 0.0-1.0 | Creatividad de respuesta (0=determinístico, 1=creativo) |
| `max_tokens` | 100-2000 | Longitud máxima de respuesta |
| `reranker_enabled` | True/False | Usar re-ranking CrossEncoder |


### Documentación Completa

- Consulta [RAG_SETUP.py](RAG_SETUP.py) para guía de instalación detallada
- Consulta [RAG_REFERENCE.py](RAG_REFERENCE.py) para referencia rápida
- Ver ejemplos en [rag_demo.py](rag_demo.py)