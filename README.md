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