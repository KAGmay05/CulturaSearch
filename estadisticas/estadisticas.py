import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Cargar el corpus actual

with open('data/movies.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
df = pd.DataFrame(data)

    # Conteo actualizado por tipo
counts = df['type'].value_counts()
total = len(df)

    # Generar gráfica de barras
plt.figure(figsize=(6, 4))
counts.plot(kind='bar', color=['#4D4D4D', '#A6A6A6'])
plt.title('Distribución por Tipo de Documento')
plt.ylabel('Cantidad')
plt.xticks(rotation=0)
plt.show()
    

    
print(f"Total de documentos: {total}")
print(counts)

# 2. Calcular longitud (Título + Plot)
df['text_completo'] = df['title'].fillna('') + " " + df['plot'].fillna('')
df['palabras'] = df['text_completo'].apply(lambda x: len(x.split()))
df['caracteres'] = df['text_completo'].apply(len)

# 3. Extraer métricas para el documento
stats = {
    "promedio_pal": df['palabras'].mean(),
    "promedio_char": df['caracteres'].mean(),
    "min_pal": df['palabras'].min(),
    "max_pal": df['palabras'].max(),
    "total": len(df)
}



# 4. Generar Histograma "Estilo Serio"
print(f"Estadísticas calculadas: {stats}")


# --- ESTADÍSTICAS DE GÉNEROS ---
# Como 'genres' es una lista, aplanamos todas las listas en una sola
todos_los_generos = [genero for lista in df['genres'].dropna() for genero in lista]
conteo_generos = Counter(todos_los_generos)
generos_unicos = len(conteo_generos)
top_5_generos = conteo_generos.most_common(5)

# --- ESTADÍSTICAS DE PAÍSES ---
conteo_paises = df['country'].value_counts()
paises_unicos = len(conteo_paises)
top_5_paises = conteo_paises.head(5)

# --- GENERAR GRÁFICA DE PAÍSES (ESTILO ACADÉMICO GRIS) ---
plt.figure(figsize=(8, 5))
# Colores en escala de grises: Gris oscuro para el líder, grises medios para el resto
colores_grises = ['#333333', '#666666', '#888888', '#AAAAAA', '#CCCCCC']

top_5_paises.plot(kind='bar', color=colores_grises, edgecolor='black', linewidth=0.7)

plt.title('Top 5 Países con mayor presencia en el Corpus', fontsize=12, fontweight='bold')
plt.ylabel('Cantidad de Documentos', fontsize=10)
plt.xlabel('País de Origen', fontsize=10)
plt.xticks(rotation=0) # Países en horizontal para mejor lectura
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Añadir los números encima de las barras para que sea fácil de copiar al Word
for i, v in enumerate(top_5_paises):
    plt.text(i, v + (max(top_5_paises)*0.01), str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('grafica_paises_seria.png', dpi=300)
plt.show()

# --- IMPRESIÓN DE RESULTADOS PARA TU WORD ---
print("-" * 30)
print("DATOS PARA LA SECCIÓN 4.4 - A")
print("-" * 30)
print(f"Géneros únicos detectados: {generos_unicos}")
print("Top 5 Géneros:")
for g, c in top_5_generos:
    print(f" - {g}: {c}")

print(f"\nPaíses únicos detectados: {paises_unicos}")
print("Top 5 Países:")
for p, c in top_5_paises.items():
    print(f" - {p}: {c}")
print("-" * 30)

# 2. Preparar datos: Explotar géneros (ya que son listas)
df_exploded = df.explode('genres')

# 3. Filtrar por Top 5 Países y Top 5 Géneros
top_paises = df['country'].value_counts().head(5).index
top_generos = df_exploded['genres'].value_counts().head(5).index

df_filtered = df_exploded[
    (df_exploded['country'].isin(top_paises)) & 
    (df_exploded['genres'].isin(top_generos))
]

# 4. Crear tabla cruzada para la gráfica
pivot_df = pd.crosstab(df_filtered['country'], df_filtered['genres'])
pivot_df = pivot_df.reindex(top_paises) # Ordenar por importancia de país

# 5. Gráfica de Barras Apiladas (Escala de Grises)
plt.style.use('seaborn-v0_8-white')
ax = pivot_df.plot(kind='bar', stacked=True, figsize=(10, 6), 
                   color=['#333333', '#666666', '#999999', '#CCCCCC', '#E6E6E6'],
                   edgecolor='black', linewidth=0.5)

plt.title('Distribución de Géneros en los Países con Mayor Presencia', fontsize=12, fontweight='bold')
plt.ylabel('Cantidad de Documentos', fontsize=10)
plt.xlabel('País de Origen', fontsize=10)
plt.legend(title='Géneros', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig('generos_por_pais.png', dpi=300)
plt.show()

# 2. Convertir ratings a números (manejo de "4,8" -> 4.8)
df['rating_num'] = df['rating'].str.replace(',', '.').astype(float)

# 3. Calcular métricas
rating_promedio = df['rating_num'].mean()
rating_max = df['rating_num'].max()
rating_min = df['rating_num'].min()

# 4. Generar Histograma de Calidad (Estilo Académico Gris)
plt.figure(figsize=(8, 5))
plt.hist(df['rating_num'], bins=15, color='#777777', edgecolor='white')

plt.title('Distribución de Valoraciones (Ratings) en el Corpus', fontsize=12, fontweight='bold')
plt.xlabel('Puntuación (0 - 10)', fontsize=10)
plt.ylabel('Frecuencia de Documentos', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Guardar imagen para Word
plt.savefig('distribucion_ratings.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"--- RESULTADOS PARA EL WORD ---")
print(f"Rating Promedio: {rating_promedio:.2f}")
print(f"Rating Máximo: {rating_max}")
print(f"Rating Mínimo: {rating_min}")

df['year_clean'] = pd.to_numeric(df['year'], errors='coerce')

# 3. Obtener métricas
anio_min = int(df['year_clean'].min())
anio_max = int(df['year_clean'].max())
total_anios = anio_max - anio_min

print(f"Año mínimo: {anio_min}")
print(f"Año máximo: {anio_max}")
print(f"Amplitud temporal: {total_anios} años")