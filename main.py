import json
from crawler.crawler import crawl


def load_seeds(path):

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_movie_urls(urls):

    with open("data/movie_urls.json", "w", encoding="utf-8") as f:
        json.dump(list(urls), f, indent=4)


if __name__ == "__main__":

    seeds = [load_seeds("seeds/seed_film_affinity.json")[0]]

    movie_urls = crawl(seeds, max_pages=20)

    save_movie_urls(movie_urls)

    print("Películas encontradas:", len(movie_urls))