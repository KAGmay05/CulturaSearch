import json
import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Cargar el corpus actual

with open('data/dataset.json', 'r', encoding='utf-8') as f:
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

# 2. Preparar datos: Explotar géneros y RESETEAR ÍNDICE
df_exploded = df.explode('genres').reset_index(drop=True) # <-- Añade .reset_index(drop=True)

# 3. Filtrar por Top 5 Países y Top 5 Géneros
top_paises = df['country'].value_counts().head(5).index
top_generos = df_exploded['genres'].value_counts().head(5).index

df_filtered = df_exploded[
    (df_exploded['country'].isin(top_paises)) & 
    (df_exploded['genres'].isin(top_generos))
].copy() # Usar .copy() evita avisos de SettingWithCopyWarning

# 4. Crear tabla cruzada
# Ahora ya no habrá conflicto de etiquetas duplicadas
pivot_df = pd.crosstab(df_filtered['country'], df_filtered['genres'])

# Reindexar para mantener el orden de importancia de los países
pivot_df = pivot_df.reindex(top_paises)

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

# Normalize year: try 'year' numeric first, then extract from 'year_range' (ej. '2019-presente')
df['year_clean'] = pd.to_numeric(df.get('year'), errors='coerce')

def extract_year_from_range(range_val):
    if isinstance(range_val, str):
        m = re.search(r"(\d{4})", range_val)
        if m:
            return int(m.group(1))
    return None

# Fill missing year_clean from year_range
mask_missing = df['year_clean'].isna()
if mask_missing.any():
    df.loc[mask_missing, 'year_clean'] = df.loc[mask_missing, 'year_range'].apply(extract_year_from_range)

# Compute metrics safely
valid_years = df['year_clean'].dropna()
if valid_years.empty:
    print("No hay años válidos para calcular min/max.")
    anio_min = None
    anio_max = None
    total_anios = None
else:
    anio_min = int(valid_years.min())
    anio_max = int(valid_years.max())
    total_anios = anio_max - anio_min

print(f"Año mínimo: {anio_min}")
print(f"Año máximo: {anio_max}")
print(f"Amplitud temporal: {total_anios} años")

# --- CONTAR DIRECTORES Y ACTORES ---
# Normalizamos posibles strings o listas vacías y contamos apariciones
todos_directores = []
todos_actores = []
for _, row in df.iterrows():
    dirs = row.get('director')
    acts = row.get('actors')

    # Normalizar distintos tipos posibles en las columnas
    # director
    if isinstance(dirs, list):
        pass
    elif dirs is None:
        dirs = []
    elif isinstance(dirs, float) and pd.isna(dirs):
        dirs = []
    elif isinstance(dirs, str):
        dirs = [dirs]
    else:
        dirs = [dirs]

    # actors
    if isinstance(acts, list):
        pass
    elif acts is None:
        acts = []
    elif isinstance(acts, float) and pd.isna(acts):
        acts = []
    elif isinstance(acts, str):
        acts = [acts]
    else:
        acts = [acts]

    for d in dirs:
        name = str(d).strip()
        if name:
            todos_directores.append(name)

    for a in acts:
        name = str(a).strip()
        if name:
            todos_actores.append(name)

director_counts = Counter(todos_directores)
actor_counts = Counter(todos_actores)

print('\n--- CONTEO DE PERSONAS ---')
print(f'Directores únicos: {len(director_counts)}')
print(f'Actores únicos: {len(actor_counts)}')

print('\nTop 20 directores por número de apariciones:')
for name, cnt in director_counts.most_common(20):
    print(f"{cnt:4d}  {name}")

print('\nTop 20 actores por número de apariciones:')
for name, cnt in actor_counts.most_common(20):
    print(f"{cnt:4d}  {name}")