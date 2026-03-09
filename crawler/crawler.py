from crawler.fetcher import fetch
from crawler.parser import extract_links


def crawl(seed_urls, max_pages=20):

    movie_urls = set()

    for seed in seed_urls:

        print(f"\n[INFO] Seed: {seed}")

        for page in range(1, max_pages + 1):

            if page == 1:
                url = seed
            else:
                url = f"{seed}&page={page}"

            print(f"[INFO] Visitando: {url}")

            html = fetch(url)

            if not html:
                print("[WARN] HTML vacío")
                continue

            movie_links, _ = extract_links(html, url)

            before = len(movie_urls)
            movie_urls.update(movie_links)
            new_found = len(movie_urls) - before

            print(
                f"[INFO] Página {page}: "
                f"{len(movie_links)} encontradas | "
                f"{new_found} nuevas | "
                f"total={len(movie_urls)}"
            )

    return movie_urls