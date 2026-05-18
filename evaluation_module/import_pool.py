import csv
import json
from collections import defaultdict

def actualizar_test_queries(csv_anotado="pool_a_anotar.csv", output_json="evaluation_module/test_queries_actualizado.json"):
    queries_dict = {}

    with open(csv_anotado, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row["query_id"]
            
            if qid not in queries_dict:
                queries_dict[qid] = {
                    "query_id": qid,
                    "query_type": row["query_type"],
                    "query": row["query"],
                    "relevant_documents": []
                }
            
            relevancia_str = row["relevancia (0-3)"].strip()
            # Validamos que hayas rellenado el número y que sea mayor a 0 (el sistema asume 0 si no existe)
            if relevancia_str.isdigit() and int(relevancia_str) > 0:
                queries_dict[qid]["relevant_documents"].append({
                    "url": row["url"],
                    "relevance": int(relevancia_str)
                })

    # Guardar en el nuevo JSON
    datos_finales = list(queries_dict.values())
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(datos_finales, f, indent=2, ensure_ascii=False)
        
    print(f"¡Nuevo set de pruebas guardado en {output_json}!")

if __name__ == "__main__":
    actualizar_test_queries()