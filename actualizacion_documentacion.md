# Actualizaciones a la Documentación (Corte 2)

Este documento estructura las adiciones y modificaciones que deben integrarse a la documentación actual del proyecto, manteniendo el tono académico original pero integrando los nuevos componentes.

### 1.3 Alcance del Corte 2 *(Añadir a la Sección 1)*
• Refinamiento y depuración del corpus documental (1,029 obras audiovisuales con metadatos expandidos).
• Implementación de un modelo de Generación Aumentada por Recuperación (RAG) para ofrecer respuestas sintetizadas en lenguaje natural.
• Incorporación de un módulo de expansión web (web_search) para enriquecer las respuestas con información externa actualizada cuando la consulta lo amerita.
• Ajuste de la infraestructura de almacenamiento mutando a un formato JSON estandarizado para los metadatos y un índice vectorial renovado.

---

### *Nota sobre la Sección 2.4 y 2.6:*
Las secciones **2.4 (Función de Ranking)** y **2.6 (Criterios de Desempate)** no necesitan ser reescritas, ya que las bases matemáticas de exactitud (Similitud Coseno) de la etapa base de recuperación densa siguen siendo ciertas. El gran cambio es que el resultado de este ranking ahora se usa como "Contexto" en vez de ser la salida final. Tu intuición es correcta.

---

## 2.3 Flujo Implementado en el Proyecto *(Reescribir la sección 2.3 original)*
1. **Recolección y Limpieza:** Se procesan los documentos a través del crawler/scraper modificado, generando el archivo unificado en data/dataset.json.
2. **Preprocesamiento:** Construcción de una cadena semántica por documento unificando: título + géneros + directores + actores + plot.
3. **Indexación Dual:**
   • *Densa:* Codificación neuronal del corpus con SentenceTransformers y almacenamiento de vectores en d/movies_vectors.index.
   • *Léxica:* Procesamiento tradicional (limpieza, tokenización, stemming) y persistencia en index/index.json.
4. **Flujo de Ejecución RAG en tiempo de consulta:**
   • El pipeline es orquestado por `rag_module/pipeline.py`.
   • La consulta del usuario entra al `retriever_wrapper.py`, el cual consulta los índices y extrae el top-K de documentos candidatos empleando internamente `NeuralRetriever.search_with_web_expansion()`.
   • Si el umbral de confianza local es menor a un factor predefinido (ej. 0.72) o la cobertura léxica es baja, se dispara automáticamente el `web_expander.py` para recabar información en la red, integrando lo scrapeado temporalmente.
   • Se transmite la información recuperada y la consulta a `generator.py`, quien ensambla un *prompt* y lo inyecta a un Modelo de Lenguaje Local (Ollama - Neural-Chat).
   • El modelo sintetiza los datos, formulando la respuesta conversacional definitiva que lee el usuario final.

**Implementación Principal y Ejecución:**
La orquestación actual centralizó las operaciones en `main.py`, permitiendo ejecutar el recabado y la interfáz final a través de comandos escalonados:
- **Crawler y Scraper:** Se pueden invocar con `python main.py crawl` y `python main.py scrape` (que impactan sobre `crawler/crawler.py` y `scraper/scraper.py`).
- **Recuperador Híbrido (Base):** Configurado principalmente en `neural_based_model/neural_retriever.py`, es el motor base. Para validaciones, puede verificarse aislando el backend con `python run_retrieval_demo.py`.
- **Evaluación RAG Asistida:** Lanzada mediante `python -m rag_module.run_rag`, abre consola para testeos con Ollama de manera interactiva.

- **Ejecución completa:** Invocando `python main.py `.
## 2.5 Análisis de Complejidad y Comportamiento *(Pequeña actualización)*
*(Mantener todo el texto y las fórmulas igual, solo reemplazar el párrafo final por lo siguiente:)*
Para el corpus actual ajustado del corte 2 (1,029 documentos), este costo de clasificación de vectores es de ejecución casi inmediata en CPU. El costo temporal principal en esta nueva iteración ha pasado a concentrarse en la latencia de inferencia propia de la etapa de generación de texto del modelo RAG y el retraso de red producto de posibles consultas web externas.

## 2.6 Criterios de Desempate y Estabilidad del Ranking

El criterio primario de ordenación es el score de similitud coseno (producto punto sobre embeddings normalizados). En la implementación actual la ordenación se realiza numéricamente por ese score y no existe un desempate explícito codificado; en igualdad exacta el orden relativo queda determinado por el índice interno y el comportamiento del algoritmo de ordenamiento.

Para evitar reordenamientos indeseados por diferencias numéricas insignificantes se recomienda definir una tolerancia (epsilon) para considerar dos scores como empate; por ejemplo, epsilon = 1e-3. Solo cuando |score_a - score_b| < epsilon se deben aplicar criterios secundarios de desempate.

Política de desempate recomendada (aplicada en orden, de mayor a menor prioridad):

1. Rerank score (cross-encoder), si está disponible.
2. Score neural bruto (producto punto / similitud coseno antes de cualquier normalización adicional).
3. Score léxico normalizado (TF‑IDF o similar) — favorece coincidencia léxica exacta.
4. Mayor coincidencia de géneros con la consulta (conteo de géneros en común).
5. Mayor completitud de metadatos (presencia/ausencia de campos clave: plot, director, elenco, año, rating), como un índice compuesto normalizado a [0,1].
6. Popularidad / rating (valor numérico normalizado; documentar parsing de strings con comas o nulls).
7. Año de publicación (preferir más reciente si procede al caso de uso).
8. Identificador estable (url o índice) como desempate final para garantizar determinismo entre ejecuciones.

Nota: los campos numéricos deben normalizarse previamente y definirse de forma consistente. Dado que el ranking final se utiliza como "contexto" para el módulo RAG (y no como salida final en bruto), pequeñas variaciones mantienen un impacto limitado sobre la generación; no obstante, la estabilidad es importante para la experiencia de usuario en listados, snippets y resaltados.

## 2.7 Fortalezas y Limitaciones del Enfoque Actual *(Reescribir)*
**Fortalezas del Corte 2:**
• **Comprensión y Síntesis:** Gracias a RAG, el modelo ya no solo recupera datos, sino que redacta respuestas hiladas, explica relaciones complejas y elabora justificaciones en lenguaje humano.
• **Cobertura Viva:** La integración de un motor de búsqueda web en tiempo real elimina la frontera del *dataset* estático, permitiendo cubrir estrenos o temas altamente específicos.
• **Recuperación Resiliente:** La base híbrida neuronal-léxica garantiza un soporte sólido de contexto para el LLM, compensando léxico exacto frente a aproximaciones de parafraseo.

**Limitaciones actuales:**
• **Latencia Global:** El tiempo de respuesta se incrementó notablemente (de milisegundos a varios segundos) debido al tiempo requerido por la inferencia del generador (LLM) y las lecturas de los crawlers externos en tiempo real.
• **Riesgo de Alucinación Menor:** Pese al anclaje en documentos verificados (RAG), existe la posibilidad inherente de que el LLM sufra desviaciones generativas si el prompt recuperado es difuso.

## 2.8 Justificación Técnica *(Reescribir)*
La migración de un sistema exclusivo de recuperación (Corte 1) a una arquitectura conversacional aumentada (Corte 2) se justifica por la evolución en la interacción hombre-máquina moderna. Un sistema puramente extractivo obliga al usuario a evaluar individualmente múltiples resultados. Al implementar un *Pipeline* RAG con *fall-back* a web automática, la plataforma asume el costo cognitivo de analizar los resultados, sintetizando recomendaciones directas, escalando enormemente la calidad percibida de las respuestas y evitando devolver "vacíos" cuando las bases locales no son suficientes.

---

## 3. Estadísticas Básicas del Corpus Recopilado (Reescribir Cap. 3 Completo)
Tras las recientes iteraciones, el corpus ha sufrido una reestructuración, mejorando en riqueza textual. 

### 3.1 Tamaño General del Corpus
- **Volumen Total:** 1,029 documentos.
- **Formato:** 646 Series frente a 383 Películas. *(Se invirtió la proporción respecto al corte 1)*

### 3.2 Análisis de Longitud Textual
- **Promedios:** 139.76 palabras y 831.64 caracteres por documento.
- **Rango:** Desde 2 hasta 429 palabras. (Garantiza una densidad semántica alta para los embeddings).

### 3.3 Diversidad y Riqueza del Corpus
**Distribución Temática y Geográfica**
- **Géneros únicos:** 31 categorías. El Top 5 lo lideran: Drama (453), Comedia (239), Animación (223), Aventura (182) y Suspense (170).
- **Países detectados:** 109 países. Liderados ampliamente por EE.UU. (518) y España (135).

**Entidades y Personalidades (Elenco)**
- **Directores:** Se registran 382 realizadores únicos.
- **Elenco principal (Actores):** La base cuenta con 2,352 actores únicos.

**Valoraciones (Ratings)**
- **Rating medio:** 3.69 sobre un rango existente entre 1.2 y 4.7.

### 3.4 Análisis del Rango Temporal
- **Periodo cubierto:** Desde el año 1940 hasta proyecciones del 2028.
- **Total:** 88 años de cobertura histórica disponible para búsqueda.

### 3.5 Estructura del Índice Invertido
Al estabilizarse el corpus, el índice reflejó las mejoras cualitativas:
- **Vocabulario:** 14,507 términos únicos limpios y consolidados.
- **Densidad del Índice:** 5.34 documentos por término en promedio. Ayuda enormente a la etapa híbrida del *retriever*.
- **Términos principales:** Raíces puras de la naturaleza del sistema como 'seri' (683 docs), 'eeuu' (566 docs), 'dram' (463 docs) y 'pelicul' (410 docs).
