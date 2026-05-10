import csv
from neural_based_model.neural_retriever import NeuralRetriever
from evaluation_module.evaluator import RetrievalEvaluator

def generar_pool_csv(salida_csv="pool_a_anotar.csv", top_k=15):
    print("Inicializando Retriever (esto puede tardar unos segundos)...")
    # Ajusta los parámetros si tu NeuralRetriever requiere algo específico al inicializar
    retriever = NeuralRetriever() 
    evaluator = RetrievalEvaluator()
    queries = evaluator.load_queries()

    registros = set() # Para evitar anotar duplicados si ya los tenías

    with open(salida_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "query_type", "query", "url", "title", "plot", "relevancia (0-3)"])
        
        for q in queries:
            print(f"Ejecutando query: {q.query}")
            resultados_base = retriever.search_advanced(query=q.query, top_k=top_k)
            resultados_web = retriever.search_with_web_expansion(query=q.query, top_k=top_k)

            todos_los_resultados = resultados_base + resultados_web

            for res in todos_los_resultados:
                # Clave única para el set
                k = f"{q.query_id}_{res.url}"
                if k not in registros:
                    registros.add(k)
                    # Dejamos la columna 'relevancia' vacía para que la llenes a mano
                    writer.writerow([q.query_id, q.query_type, q.query, res.url, res.title, res.plot[:300] + "...", ""])
                    
    print(f"¡Pool generado exitosamente en {salida_csv}!")

if __name__ == "__main__":
    generar_pool_csv()