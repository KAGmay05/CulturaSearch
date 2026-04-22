import os
import pickle

from vectorizer import (
    INDEX_PATH,
    METADATA_PATH,
    build_embeddings,
    search_by_similarity,
)


def verificar_vectorizer() -> None:
    print("--- Verificando flujo completo de vectorizer.py ---")

    # 1) Construcción del índice + metadata
    build_embeddings()
    print("[OK] build_embeddings() ejecutado.")

    # 2) Archivos esperados
    if not os.path.exists(INDEX_PATH):
        print(f"[ERROR] No se encontró el índice FAISS: {INDEX_PATH}")
        return
    if not os.path.exists(METADATA_PATH):
        print(f"[ERROR] No se encontró metadata: {METADATA_PATH}")
        return
    print(f"[OK] Archivos generados: {INDEX_PATH} y {METADATA_PATH}")

    # 3) Validar metadata
    try:
        with open(METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)
    except Exception as e:
        print(f"[ERROR] No se pudo leer metadata.pkl: {e}")
        return

    if not isinstance(metadata, dict):
        print("[ERROR] metadata.pkl no tiene formato dict.")
        return

    required_keys = [
        "urls",
        "model_name",
        "index_type",
        "vector_count",
        "vector_dim",
        "dtype",
        "normalized",
        "similarity",
    ]
    missing = [k for k in required_keys if k not in metadata]
    if missing:
        print(f"[ERROR] Faltan claves en metadata: {missing}")
        return

    urls = metadata.get("urls", [])
    print("\n--- Estadísticas de metadata ---")
    print(f"Total URLs: {len(urls)}")
    print(f"Modelo: {metadata.get('model_name')}")
    print(f"Índice: {metadata.get('index_type')}")
    print(f"Vectores: {metadata.get('vector_count')}")
    print(f"Dimensión: {metadata.get('vector_dim')}")
    print(f"Tipo: {metadata.get('dtype')}")
    print(f"Normalizado: {metadata.get('normalized')}")
    print(f"Similitud: {metadata.get('similarity')}")

    # 4) Probar búsqueda semántica
    query = "pelicula de accion"
    top_k = 5
    try:
        results = search_by_similarity(query=query, top_k=top_k)
    except Exception as e:
        print(f"[ERROR] Falló search_by_similarity(): {e}")
        return

    print("\n--- Resultados de búsqueda ---")
    print(f"Consulta: {query}")
    print(f"Resultados devueltos: {len(results)} (top_k={top_k})")

    if len(results) == 0:
        print("[ERROR] search_by_similarity() devolvió 0 resultados.")
        return

    if len(results) > top_k:
        print("[ERROR] search_by_similarity() devolvió más resultados que top_k.")
        return

    for item in results[:3]:
        print(f"#{item['rank']} | score={item['score']:.4f} | url={item['url']}")

    print("\n[OK] vectorizer.py funciona correctamente (indexación + metadata + búsqueda).")


if __name__ == "__main__":
    verificar_vectorizer()