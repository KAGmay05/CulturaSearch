import json
import os
from indexer import build_index, clean_text 

def test_indexing():
    print("--- Iniciando Pruebas del Módulo de Indexación ---")
    dataset_path = 'data/dataset.json'
    
    if not os.path.exists(dataset_path):
        print(f"Error: No se encuentra el archivo {dataset_path}")
        return

    print("Construyendo índice...")
    index = build_index(dataset_path)
    
    if os.path.exists('index.json'):
        print("[OK] Archivo 'index.json' generado correctamente.")
    else:
        print("[ERROR] El archivo 'index.json' no fue creado.")

    sample_term = list(index.keys())[0]
    sample_value = index[sample_term]
    
    if isinstance(sample_value, dict):
        print(f"[OK] Estructura validada: El término '{sample_term}' mapea a un diccionario de URLs.")
        sample_url = list(sample_value.keys())[0]
        if isinstance(sample_value[sample_url], int):
            print(f"[OK] Frecuencia (TF) detectada: {sample_term} -> {sample_url}: {sample_value[sample_url]}")
    else:
        print("[ERROR] La estructura no es un diccionario de diccionarios.")

   
    print("\n--- Verificando Calidad de la Indexación ---")
    test_word = "Hole"
    stemmed_word = clean_text(test_word)[0]
    
    if stemmed_word in index:
        print(f"[OK] Búsqueda de prueba: La raíz '{stemmed_word}' (de '{test_word}') existe en el índice.")
        print(f"     Aparece en {len(index[stemmed_word])} documentos.")
    else:
        print(f"[AVISO] La raíz '{stemmed_word}' no está en el índice (quizás no está en tu JSON).")


    print("\n--- Estadísticas para tu Informe ---")
    print(f"Total de términos únicos (Vocabulario): {len(index)}")
    
  
    most_common = max(index, key=lambda k: len(index[k]))
    print(f"Término con más presencia en documentos: '{most_common}' ({len(index[most_common])} URLs)")

if __name__ == "__main__":
    test_indexing()