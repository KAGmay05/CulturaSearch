import json
import os
import faiss
import numpy as np
import pickle 
from sentence_transformers import SentenceTransformer

MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
INDEX_TYPE = "IndexFlatIP"
INDEX_PATH = 'bd/movies_vectors.index'
METADATA_PATH = 'bd/metadata.pkl'


def _safe_text(value):
    if value is None:
        return ""
    return str(value)


def _load_metadata(metadata_path=METADATA_PATH):
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"No se encontro metadata en: {metadata_path}")

    with open(metadata_path, 'rb') as f: #carga los metadatos 
        payload = pickle.load(f)

    if isinstance(payload, list): #verifica si es diccionario, si no lo es lo convierte
        return {"urls": payload, "model_name": MODEL_NAME, "normalized": True}
    if not isinstance(payload, dict):
        raise ValueError("metadata.pkl tiene un formato invalido.")
    return payload #devuelve el diccionario

def build_embeddings():
    model = SentenceTransformer(MODEL_NAME) #cargamos el modelo multilingue
    urls = []
    text = []

    with open('data/movies.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    for mov in data:
        if not isinstance(mov, dict):
            continue

        url = _safe_text(mov.get('url', ''))
        title = _safe_text(mov.get('title', ''))
        plot = _safe_text(mov.get('plot', ''))
        genres = _safe_text(mov.get('genres', ''))

        urls.append(url)  #listas con las urls
        text.append((title + " " + plot + " " + genres).strip()) #listas con la info, o sea, el titulo y el argumento

    if not text:
        raise ValueError("No hay documentos validos para vectorizar.")

    embeddings = model.encode(text).astype('float32') # creamos la matriz de embeddings con el metodo encode de model, donde cada vector representa una url
    embeddings = np.ascontiguousarray(embeddings)
    faiss.normalize_L2(embeddings) #normalizar embeddings
    d = embeddings.shape[1] #dimesion del vector 
    index = faiss.IndexFlatIP(d) #cargamos el indice para similitud coseno
    index.add(embeddings) #anadimos los embeddings al indice
    
    faiss.write_index(index, INDEX_PATH)  #guardamos el indice
    
    metadata = {
        "urls": urls,
        "model_name": MODEL_NAME,
        "index_type": INDEX_TYPE,
        "vector_count": int(embeddings.shape[0]),
        "vector_dim": int(embeddings.shape[1]),
        "dtype": str(embeddings.dtype),
        "normalized": True,
        "similarity": "cosine_via_ip",
    }

    with open(METADATA_PATH, 'wb') as f: #guardamos metadatos para mapear y control
        pickle.dump(metadata, f)


def search_by_similarity(query, top_k=5, index_path=INDEX_PATH, metadata_path=METADATA_PATH):
    if not query or not str(query).strip():
        raise ValueError("La consulta no puede estar vacia.")

    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"No se encontro el indice en: {index_path}. Ejecuta build_embeddings() primero."
        )

    metadata = _load_metadata(metadata_path)  #carga los metadatos
    urls = metadata.get("urls", [])  
    model_name = metadata.get("model_name", MODEL_NAME)

    index = faiss.read_index(index_path) #carga el inidce
    model = SentenceTransformer(model_name) #carga el modelo

    query_vec = model.encode([str(query)]).astype('float32')
    query_vec = np.ascontiguousarray(query_vec)
    faiss.normalize_L2(query_vec)  #convierte la query en embeddings

    top_k = max(1, int(top_k))
    distances, indices = index.search(query_vec, top_k) #busca similitud con query maximo 5

    results = []  #convierte la salida en formato legible
    for rank, (idx, score) in enumerate(zip(indices[0], distances[0]), start=1):
        if idx < 0:
            continue
        url = urls[idx] if idx < len(urls) else ""
        results.append(
            {
                "rank": rank,
                "url": url,
                "score": float(score),
            }
        )
    return results
        