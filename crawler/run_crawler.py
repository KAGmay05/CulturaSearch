import json

from crawler.crawler import crawl


def load_seeds(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_movie_urls(urls):
    with open("data/movie_urls.json", "w", encoding="utf-8") as f:
        json.dump(list(urls), f, indent=4)


def save_series_urls(urls):
    with open("data/series_urls.json", "w", encoding="utf-8") as f:
        json.dump(list(urls), f, indent=4)


def main():
    print("[INFO] CulturaSearch - Web Crawler")
    print("[INFO] Usando SensaCine (sin bloqueos Cloudflare)\n")

    seeds = load_seeds("seeds/seed_sensacine.json")
    print(f"[INFO] Cargadas {len(seeds)} semillas")
    print("[INFO] Extrayendo películas y series...\n")

    movie_urls, series_urls = crawl(seeds, max_pages=30)

    save_movie_urls(movie_urls)
    save_series_urls(series_urls)

    print("\n" + "=" * 60)
    print(f"✓ PELÍCULAS: {len(movie_urls)}")
    print(f"✓ SERIES: {len(series_urls)}")
    print(f"✓ TOTAL: {len(movie_urls) + len(series_urls)} contenidos")
    print(f"✓ Guardado en: data/movie_urls.json y data/series_urls.json")
    print("=" * 60)


if __name__ == "__main__":
    main()