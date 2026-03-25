# Documentacion - CulturaSearch

## 1. Definicion del dominio seleccionado

El dominio seleccionado es cultura y entretenimiento, con enfasis en peliculas y series.

El sistema recupera documentos que describen obras audiovisuales (titulo, sinopsis, genero, reparto, direccion, país y año), extraidos desde fuentes web especializadas mediante el crawler/scraper del proyecto.

Objetivo de recuperacion:

- Permitir busquedas semanticas en lenguaje natural sobre peliculas y series.
- Recuperar resultados relevantes aunque la consulta no use exactamente las mismas palabras del documento.

Alcance del corte 1:

- Construccion de un corpus inicial del dominio.
- Generacion de representaciones vectoriales densas (embeddings).
- Ranking basico por similitud coseno.

## 2. Documentacion del modelo de recuperacion implementado

### 2.1 Tipo de modelo

Se implemento un modelo de recuperacion neuronal no basico basado en embeddings con Sentence Transformers.

Modelo usado:

- sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

Este modelo produce un vector denso para cada documento y para cada consulta. La recuperacion se hace comparando vectores en el espacio latente.

En terminos de IR neuronal, este enfoque corresponde a un esquema bi-encoder (dual encoder):

- Un encoder transforma documentos a vectores.
- El mismo encoder transforma la consulta a un vector.
- La relevancia se estima con una funcion de similitud vectorial.

Este diseno permite indexar el corpus una vez y reutilizar los embeddings para multiples consultas.

### 2.2 Modelo hibrido: motivacion y arquitectura

Si bien el enfoque puramente neuronal es efectivo, se ha evolucionado hacia un **modelo hibrido** que combina embeddings densos con indexacion lexico-estadistica (TF-IDF) para obtener mayor precision y cobertura.

#### Por que hibrido?

Un modelo hibrido combina dos paradigmas complementarios:

1. **Recuperacion densa (embeddings neurales)**:
   - Captura relaciones semanticas profundas entre consulta y documentos.
   - Funciona bien con parafraseos y sinonimos.
   - Entiende intencionalidad semantica detras de la consulta.

2. **Recuperacion lexico-estadistica (TF-IDF)**:
   - Excelente para coincidencias de terminos exactos y frecuencia.
   - El efecto IDF captura la importancia de palabras poco comunes.
   - Soporta busquedas por metadatos especificos (directores, paises, actores).

#### Ventajas de la combinacion

- **Mayor cobertura**: Algunos documentos relevantes se recuperan por semantica, otros por exactitud lexical.
- **Mayor precision**: En consultas que mencionan tipos explicitamente ("dame una serie policial"), se aplican restricciones de metadatos.
- **Robustez**: Si la semantica falla en un ambito, la lexical compensa (y viceversa).
- **Escalabilidad**: El indice lexico es ligero y rapido, permitiendo filtrado eficiente antes del ranking neural.

#### Arquitectura de dos etapas

**Etapa 1: Recuperacion de candidatos (hibrida)**
- Busqueda densa: top-50 resultados por similitud coseno de embeddings.
- Busqueda lexico-estadistica: puntuacion TF-IDF para terminos de consulta.
- Fusion adaptativa: combinacion ponderada neural + lexico con peso ajustable ($\alpha$).
- Restricciones de tipo: filtro o penalizacion para consultas que especifican "serie" o "pelicula".

**Etapa 2: Re-ranking neuronal (CrossEncoder)**
- Los candidatos se re-evaluan con un modelo CrossEncoder especializado.
- El CrossEncoder analiza directamente cada par (consulta, documento) en lugar de embeddings independientes.
- Fusion final: score base combinado con score de re-ranking mediante fusion ponderada para mayor precision.

### 2.3 Flujo implementado en el proyecto

1. Carga de documentos desde [data/movies.json](data/movies.json).
2. Construccion de texto por documento usando campos: titulo + generos + plot + actores.
3. Codificacion neuronal del corpus con SentenceTransformer.
4. Almacenamiento de la base vectorial inicial en [bd/movies_vectors.pkl](bd/movies_vectors.pkl).
5. En consulta:
- Se codifica la consulta con el mismo modelo.
- Se calcula similitud coseno contra todos los documentos.
- Se ordenan resultados de mayor a menor similitud (top-k).

Implementacion principal:

- [neural_based_model/neural_retriever.py](neural_based_model/neural_retriever.py)
- [neural_based_model/demo.py](neural_based_model/demo.py)

Detalle del pipeline interno:

1. Preprocesamiento estructural:
- Se consolidan campos relevantes en una sola cadena semantica por documento.
- Se conserva informacion de contenido (plot) y contexto (generos, actores).

2. Indexacion densa:
- Cada cadena se codifica con el modelo multilingual MiniLM.
- El resultado es una matriz de embeddings de dimension fija.

3. Persistencia de indice:
- Se serializa en [bd/movies_vectors.pkl](bd/movies_vectors.pkl) el conjunto:
  - urls
  - embeddings
  - metadatos documentales
  - nombre del modelo

4. Recuperacion en tiempo de consulta:
- Se codifica la consulta con el mismo encoder.
- Se calcula similitud coseno contra toda la matriz.
- Se retorna top-k ordenado de mayor a menor score.

### 2.4 Funcion de ranking

La similitud utilizada es coseno:

$$
\text{sim}(q, d)=\frac{\vec{q}\cdot\vec{d}}{||\vec{q}||\,||\vec{d}||}
$$

Donde:

- $\vec{q}$ es el embedding de la consulta.
- $\vec{d}$ es el embedding del documento.

Los documentos se ordenan por valor de similitud descendente.

Interpretacion de score:

- Score cercano a 1: alta cercania semantica entre consulta y documento.
- Score cercano a 0: baja relacion semantica.
- Score negativo: orientacion semantica opuesta (poco frecuente en este escenario).

Procedimiento de ranking implementado:

1. Se obtiene el vector de similitudes para todos los documentos.
2. Se aplica ordenamiento descendente de indices por score.
3. Se recorta al top-k solicitado.
4. Se construye la salida final con rank, score y metadatos del documento.

Para el ordenamiento se usa una estrategia exacta sobre todo el corpus (no aproximada), adecuada para el volumen actual del corte 1.

### 2.5 Analisis de complejidad y comportamiento

Si $N$ es el numero de documentos y $D$ la dimension del embedding:

- Costo de indexacion inicial: $O(N \cdot C_{enc})$, donde $C_{enc}$ es el costo del encoder por documento.
- Costo por consulta (similitud coseno vectorizada): aproximadamente $O(N \cdot D)$.
- Costo de ordenar resultados completos: $O(N \log N)$.

Para el corpus actual (1370 documentos), este costo es aceptable en CPU para una primera entrega.

### 2.6 Criterios de desempate y estabilidad del ranking

El criterio primario de orden es el score de similitud coseno.

Cuando dos documentos tienen scores muy cercanos, el orden relativo depende del resultado del ordenamiento numerico sobre el arreglo de similitudes. Para una etapa posterior se puede agregar desempate explicito por campos como:

- mayor coincidencia de genero con la consulta,
- mayor completitud de metadatos,
- popularidad o rating.

### 2.7 Fortalezas y limitaciones del enfoque actual

Fortalezas:

- Recupera por significado y no solo por palabras exactas.
- Funciona bien en consultas parafraseadas.
- Es simple de mantener para el corte 1.

Limitaciones actuales:

- No incorpora retroalimentacion de usuario ni aprendizaje de ranking supervisado.
- No usa re-ranking cruzado (cross-encoder), por lo que el score final depende solo del bi-encoder.
- Al ser busqueda densa exacta, no escala tan bien como ANN para corpus muy grandes.

Mejoras naturales para siguiente corte:

- ANN con FAISS/Chroma para escalar.
- Re-ranking con cross-encoder sobre top-k inicial.
- Combinacion hibrida: BM25 + denso (fusion de rankings).

### 2.8 Justificacion tecnica

El enfoque neuronal mejora frente a modelos puramente lexicales porque:

- Captura relaciones semanticas entre terminos.
- Reduce dependencia de coincidencia exacta de palabras.
- Soporta consultas parafraseadas o con vocabulario distinto.

## 3. Fuentes bibliograficas utilizadas

1. Mitra, B., y Craswell, N. (2018). An Introduction to Neural Information Retrieval. Foundations and Trends in Information Retrieval

## 4. Estadisticas basicas del corpus recopilado

Fuente analizada: [data/movies.json](data/movies.json)

### 4.1 Tamano general

- Cantidad total de documentos: 1370
- Documentos tipo pelicula: 1040
- Documentos tipo serie: 330

### 4.2 Cobertura de campos

- Documentos con titulo no vacio: 1370 (100%)
- Documentos con plot no vacio: 1370 (100%)

### 4.3 Longitud textual (titulo + plot)

- Promedio de palabras por documento: 70.15
- Promedio de caracteres por documento: 419.80
- Minimo de palabras en un documento: 11
- Maximo de palabras en un documento: 228

### 4.4 Diversidad del corpus

- Generos unicos detectados: 18
- Top 5 generos mas frecuentes:
  - Drama: 658
  - Thriller: 398
  - Comedia: 340
  - Serie de TV: 330
  - Accion: 207

- Paises unicos detectados: 52
- Top 5 paises mas frecuentes:
  - Estados Unidos: 671
  - Espana: 232
  - Reino Unido: 135
  - Francia: 69
  - Japon: 32

### 4.5 Rango temporal

- Año mínimo detectado: 2016
- Año máximo detectado: 2026

## 5. Trazabilidad de implementacion

- Modelo neuronal y recuperador: [neural_based_model/neural_retriever.py](neural_based_model/neural_retriever.py)
- Script de demostracion: [neural_based_model/demo.py](neural_based_model/demo.py)
- Dataset base: [data/movies.json](data/movies.json)
- Base vectorial serializada: [bd/movies_vectors.pkl](bd/movies_vectors.pkl)
