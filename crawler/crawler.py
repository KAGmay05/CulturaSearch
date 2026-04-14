from crawler.fetcher import fetch
from crawler.parser import extract_links


def crawl(seed_urls, max_pages=10):

    movie_urls = set()
    series_urls = set()

    for seed in seed_urls:

        print(f"\n[INFO] Seed: {seed}")

        for page in range(1, max_pages + 1):

            if page == 1:
                url = seed
            else:
                # Detecta si la URL ya tiene parámetros query
                separator = "&" if "?" in seed else "?"
                url = f"{seed}{separator}page={page}"

            print(f"[INFO] Visitando: {url}")

            html = fetch(url)

            if not html:
                print("[WARN] HTML vacío, fin de resultados para esta semilla")
                break

            movie_links, series_links = extract_links(html, url)

            before_movies = len(movie_urls)
            before_series = len(series_urls)
            movie_urls.update(movie_links)
            series_urls.update(series_links)

            print(
               f"[INFO] Página {page}: "
               f"{len(movie_urls) - before_movies} nuevas películas | "
               f"{len(series_urls) - before_series} nuevas series | "
               f"total_movies={len(movie_urls)} | "
               f"total_series={len(series_urls)}"
            )

    return movie_urls, series_urls