import json
from scraper import scrape_movie
from concurrent.futures import ThreadPoolExecutor, as_completed

MAX_WORKERS = 5 

with open("data/movie_urls.json", "r") as f:
    urls = json.load(f)

movies = []

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

    futures = {executor.submit(scrape_movie, url): url for url in urls}

    for i, future in enumerate(as_completed(futures)):

        url = futures[future]

        print(f"[INFO] Scraping {i+1}/{len(urls)}")

        try:
            movie = future.result()
            if movie:
                movies.append(movie)

        except Exception as e:
            print(f"[ERROR] {url} -> {e}")

with open("data/movies.json", "w", encoding="utf-8") as f:
    json.dump(movies, f, indent=2, ensure_ascii=False)