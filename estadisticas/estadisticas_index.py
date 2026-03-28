import json

# Cargar el índice
with open('index/index.json', 'r', encoding='utf-8') as f:
    index = json.load(f)

# Cálculos
total_terminos = len(index)
terminos = list(index.keys())

# Calcular frecuencia total por término y cantidad de documentos por término
stats = []
for term, docs in index.items():
    num_docs = len(docs)
    freq_total = sum(docs.values())
    stats.append({
        'term': term,
        'num_docs': num_docs,
        'freq_total': freq_total
    })

# Ordenar para ver los términos más pesados
stats_sorted = sorted(stats, key=lambda x: x['num_docs'], reverse=True)

print(f"--- ESTADÍSTICAS DEL ÍNDICE INVERTIDO ---")
print(f"Vocabulario total (Términos únicos): {total_terminos}")
print(f"\nTérminos con mayor alcance (Presencia en documentos):")
for s in stats_sorted[:5]:
    print(f" - '{s['term']}': presente en {s['num_docs']} documentos (Frecuencia total: {s['freq_total']})")

# Calcular densidad media
avg_docs_per_term = sum(s['num_docs'] for s in stats) / total_terminos
print(f"\nDensidad media: {avg_docs_per_term:.2f} documentos por término.")