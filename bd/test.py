import pickle
import os
from vectorizer import build_embeddings

def verificar_bd_vectorial():
    build_embeddings()
    ruta_archivo = 'bd/movies_vectors.pkl'
    
    print("--- Verificando Base de Datos Vectorial ---")
    
    # 1. Verificar si el archivo existe
    if not os.path.exists(ruta_archivo):
        print(f"[ERROR] No se encontró el archivo en {ruta_archivo}")
        return

    # 2. Cargar el archivo pkl
    try:
        with open(ruta_archivo, 'rb') as f:
            datos = pickle.load(f)
        print("[OK] Archivo cargado correctamente.")
    except Exception as e:
        print(f"[ERROR] No se pudo leer el archivo: {e}")
        return

    # 3. Validar contenido
    urls = datos.get("urls", [])
    embeddings = datos.get("embeddings", [])

    print(f"\n--- Estadísticas de la BD Vectorial ---")
    print(f"Total de URLs indexadas: {len(urls)}")
    
    # 4. Validar Dimensiones (Lo más importante para tu informe)
    # Los embeddings son una matriz de (filas x columnas)
    # filas = número de películas
    # columnas = dimensiones del modelo (384 para MiniLM)
    if len(embeddings) > 0:
        print(f"Forma de la matriz (Shape): {embeddings.shape}")
        print(f"Dimensiones por vector: {embeddings.shape[1]}")
        
        if embeddings.shape[1] == 384:
            print("[OK] Las dimensiones coinciden con el modelo MiniLM (384).")
        
        # 5. Ver una muestra
        print(f"\nEjemplo de la primera película:")
        print(f"URL: {urls[0]}")
        # Mostramos solo los primeros 5 números del vector para no llenar la pantalla
        print(f"Primeros 5 valores del vector: {embeddings[0][:5]}")
    else:
        print("[ERROR] La matriz de embeddings está vacía.")

if __name__ == "__main__":
    verificar_bd_vectorial()