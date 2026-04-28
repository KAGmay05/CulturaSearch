import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from scraper.scraper import scrape_movie

MAX_WORKERS = 5


def main():
    with open("data/movie_urls.json", "r", encoding="utf-8") as f:
        movie_urls = json.load(f)

    with open("data/series_urls.json", "r", encoding="utf-8") as f:
        series_urls = json.load(f)

    urls = movie_urls + series_urls
    movies = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(scrape_movie, url): url for url in urls}

        for i, future in enumerate(as_completed(futures)):
            url = futures[future]
            print(f"[INFO] Scraping {i + 1}/{len(urls)}")

            try:
                movie = future.result()
                if movie:
                    movies.append(movie)
            except Exception as e:
                print(f"[ERROR] {url} -> {e}")

    with open("data/dataset.json", "w", encoding="utf-8") as f:
        json.dump(movies, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()