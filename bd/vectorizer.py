import json
import os
import faiss
import numpy as np
import pickle 
from typing import Callable
from sentence_transformers import SentenceTransformer

MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
INDEX_TYPE = "IndexFlatIP"
INDEX_PATH = 'bd/movies_vectors.index'
METADATA_PATH = 'bd/metadata.pkl'


def _safe_text(value):
    if value is None:
        return ""
    return str(value)


def _default_text_builder(mov):
    title = _safe_text(mov.get('title', ''))
    plot = _safe_text(mov.get('plot', ''))
    genres = _safe_text(mov.get('genres', ''))
    return (title + " " + plot + " " + genres).strip()


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

def build_embeddings(
    documents=None,
    model_name=MODEL_NAME,
    index_path=INDEX_PATH,
    metadata_path=METADATA_PATH,
    dataset_path='data/movies.json',
    include_documents=False,
    text_builder: Callable | None = None,
    model: SentenceTransformer | None = None,
):
    model = model or SentenceTransformer(model_name) #cargamos el modelo multilingue
    urls = []
    text = []

    if documents is None:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = documents

    build_text = text_builder or _default_text_builder
    valid_documents = []

    for mov in data:
        if not isinstance(mov, dict):
            continue

        url = _safe_text(mov.get('url', ''))
        combined_text = build_text(mov)

        urls.append(url)  #listas con las urls
        text.append(_safe_text(combined_text)) #listas con la info, o sea, el titulo y el argumento
        valid_documents.append(mov)

    if not text:
        raise ValueError("No hay documentos validos para vectorizar.")

    embeddings = model.encode(text).astype('float32') # creamos la matriz de embeddings con el metodo encode de model, donde cada vector representa una url
    embeddings = np.ascontiguousarray(embeddings)
    faiss.normalize_L2(embeddings) #normalizar embeddings
    d = embeddings.shape[1] #dimesion del vector 
    index = faiss.IndexFlatIP(d) #cargamos el indice para similitud coseno
    index.add(embeddings) #anadimos los embeddings al indice
    
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)  #guardamos el indice
    
    metadata = {
        "urls": urls,
        "model_name": model_name,
        "index_type": INDEX_TYPE,
        "vector_count": int(embeddings.shape[0]),
        "vector_dim": int(embeddings.shape[1]),
        "dtype": str(embeddings.dtype),
        "normalized": True,
        "similarity": "cosine_via_ip",
    }

    if include_documents:
        metadata["documents"] = valid_documents

    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, 'wb') as f: #guardamos metadatos para mapear y control
        pickle.dump(metadata, f)


def search_by_similarity(
    query,
    top_k=5,
    index_path=INDEX_PATH,
    metadata_path=METADATA_PATH,
    index=None,
    model: SentenceTransformer | None = None,
    urls=None,
    model_name=None,
    return_indices=False,
):
    if not query or not str(query).strip():
        raise ValueError("La consulta no puede estar vacia.")

    if index is None:
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"No se encontro el indice en: {index_path}. Ejecuta build_embeddings() primero."
            )
        index = faiss.read_index(index_path) #carga el inidce

    if urls is None or (model is None and model_name is None):
        metadata = _load_metadata(metadata_path)  #carga los metadatos
        if urls is None:
            urls = metadata.get("urls", [])
        if model is None and model_name is None:
            model_name = metadata.get("model_name", MODEL_NAME)

    urls = urls or []
    model_name = model_name or MODEL_NAME
    model = model or SentenceTransformer(model_name) #carga el modelo

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
        item = {
            "rank": rank,
            "url": url,
            "score": float(score),
        }
        if return_indices:
            item["index"] = int(idx)
        results.append(item)
    return results
        