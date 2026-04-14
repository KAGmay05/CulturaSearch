import json
from index import indexer

# Cargar el dataset
with open("data/movies.json", "r", encoding="utf-8") as f:
    movies = json.load(f)

# Buscar "(500) días juntos"
target_movie = None
for movie in movies:
    if "(500) días juntos" in movie.get("title", ""):
        target_movie = movie
        break

if target_movie:
    print(f"\n========== ANÁLISIS DE '{target_movie['title']}' ==========")
    print(f"URL: {target_movie.get('url', 'N/A')}")
    
    # Construir el texto igual que hace el retriever
    title = str(target_movie.get("title", ""))
    plot = str(target_movie.get("plot", ""))
    year = str(target_movie.get("year", ""))
    rating = str(target_movie.get("rating", ""))
    media_type = str(target_movie.get("type", ""))
    country = str(target_movie.get("country", ""))
    
    genres = target_movie.get("genres", [])
    if isinstance(genres, list):
        genres_text = " ".join(str(g) for g in genres)
    else:
        genres_text = str(genres)
    
    actors = target_movie.get("actors", [])
    if isinstance(actors, list):
        actors_text = " ".join(str(a) for a in actors)
    else:
        actors_text = str(actors)
    
    directors = target_movie.get("director", [])
    if isinstance(directors, list):
        directors_text = " ".join(str(d) for d in directors)
    else:
        directors_text = str(directors)
    
    full_text = (
        f"{title} {year} {rating} {media_type} {country} "
        f"{genres_text} {directors_text} {plot} {actors_text}"
    ).strip()
    
    print(f"\nPrimero 500 caracteres del texto indexado:")
    print(f"{full_text[:500]}\n...")
    
    # Limpiar como hace el indexador
    cleaned_tokens = indexer.clean_text(full_text)
    
    print(f"\nTokens limpios totales: {len(cleaned_tokens)}")
    print(f"\nBuscando 'tom' y 'jerry' en los tokens:")
    
    tom_count = cleaned_tokens.count('tom')
    jerry_count = cleaned_tokens.count('jer')  # stemmed version
    
    print(f"  'tom': {tom_count} veces")
    print(f"  'jer' (stemmed de jerry): {jerry_count} veces")
    
    # Ahora calcular lo que hace _compute_lexical_scores
    print(f"\n========== CÁLCULO DE SCORE LÉXICO ==========")
    
    query = "tom y jerry"
    query_tokens = indexer.clean_text(query)
    print(f"Query: '{query}'")
    print(f"Query tokens después limpiar: {query_tokens}")
    
    # Contar ocurrencias en el documento
    for qtoken in query_tokens:
        count = cleaned_tokens.count(qtoken)
        print(f"  Token '{qtoken}': aparece {count} veces en el documento")
else:
    print("Película no encontrada")
