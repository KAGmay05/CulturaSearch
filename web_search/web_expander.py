import json
import os
from typing import Dict, List
from urllib.parse import parse_qs, urlencode, quote_plus, urlparse, urlunparse

from crawler.fetcher import fetch
from crawler.parser import extract_links
from scraper.scraper import scrape_movie


GLOBAL_SEARCH_URL = "https://www.filmaffinity.com/es/globalsearch.php"
GLOBAL_SEARCH_CX = "008177178803676006601:mrme5orikjy"
GLOBAL_SEARCH_COF = "FORID:9"


class WebExpander:
    def __init__(
        self,
        cache_path: str = "data/web_cache.json",
        seed_file_paths: List[str] | None = None,
        max_pages_per_seed: int = 2,
    ) -> None:
        self.cache_path = cache_path
        self.seed_file_paths = seed_file_paths or ["seeds/seed_film_affinity.json"]
        self.max_pages_per_seed = max_pages_per_seed

    def _load_seed_urls(self) -> List[str]:
        urls: List[str] = []
        for seed_file in self.seed_file_paths:
            if not os.path.exists(seed_file):
                continue

            try:
                with open(seed_file, "r", encoding="utf-8") as f:
                    raw_content = f.read().strip()
                    if not raw_content:
                        continue
                    loaded = json.loads(raw_content)
            except (OSError, json.JSONDecodeError):
                continue

            if isinstance(loaded, list):
                for url in loaded:
                    url_text = str(url).strip()
                    if url_text and url_text != "None":
                        urls.append(url_text)
        return urls

    @staticmethod
    def _build_query_url(seed_url: str, query: str, page: int) -> str:
        parsed = urlparse(seed_url)
        query_params = parse_qs(parsed.query, keep_blank_values=True)

        # Always inject/update the user query into the search text parameter.
        query_params["stext"] = [query.strip()]

        # Keep page canonical to avoid duplicated page parameters.
        if page > 1:
            query_params["page"] = [str(page)]
        else:
            query_params.pop("page", None)

        rebuilt_query = urlencode(query_params, doseq=True)
        rebuilt = parsed._replace(query=rebuilt_query)
        return urlunparse(rebuilt)

    @staticmethod
    def _build_global_search_url(query: str) -> str:
        encoded_query = quote_plus(query.strip())
        return (
            f"{GLOBAL_SEARCH_URL}?cx={GLOBAL_SEARCH_CX}"
            f"&cof={quote_plus(GLOBAL_SEARCH_COF)}"
            f"&q={encoded_query}&ie=latin1"
        )

    def _load_cache(self) -> List[Dict]:
        if not os.path.exists(self.cache_path):
            return []

        with open(self.cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)

        if isinstance(cached, list):
            return [doc for doc in cached if isinstance(doc, dict)]

        return []

    def _save_cache(self, new_documents: List[Dict]) -> None:
        existing_documents = self._load_cache()
        merged_by_url: Dict[str, Dict] = {}

        for document in existing_documents + new_documents:
            url = str(document.get("url", ""))
            if not url:
                continue
            merged_by_url[url] = document

        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(list(merged_by_url.values()), f, indent=2, ensure_ascii=False)

    def expand(self, query: str, max_results: int = 10) -> List[Dict]:
        if not query or not query.strip():
            return []

        print(f"\n========== WEB EXPANSION START ==========")
        print(f"Query: '{query}'")
        print(f"Max results: {max_results}")
        
        seed_urls = self._load_seed_urls()
        print(f"Seed URLs cargadas: {len(seed_urls)}")
        if not seed_urls:
            seed_urls = ["https://www.filmaffinity.com/es/advsearch.php?stext=&stype%5B%5D=title&genre%5B%5D=&country=&fromyear=&toyear=&ratingcount=1&runtimemin=0&runtimemax="]
            print(f"[WARNING] No seed URLs encontradas, usando seed por defecto")

        discovered_urls: List[str] = []
        seen_urls = set()
        fetch_ok = 0
        fetch_failed = 0
        pages_checked = 0

        for seed_url in seed_urls:
            for page in range(1, self.max_pages_per_seed + 1):
                search_url = self._build_query_url(seed_url, query, page)
                print(f"\n[SEED SEARCH] Intentando página {page}: {search_url}")
                pages_checked += 1
                html = fetch(search_url)
                print(f"[SEED SEARCH] HTML recibido: {len(html) if html else 0} bytes")
                if not html:
                    fetch_failed += 1
                    continue
                fetch_ok += 1

                movie_links, _ = extract_links(html, search_url)
                print(f"[SEED SEARCH] Links encontrados en página {page}: {len(movie_links)}")
                for movie_url in movie_links:
                    if movie_url not in seen_urls:
                        seen_urls.add(movie_url)
                        discovered_urls.append(movie_url)

                if len(discovered_urls) >= max_results:
                    break

            if len(discovered_urls) >= max_results:
                break

        global_search_url = self._build_global_search_url(query)
        print(f"\n[GLOBAL SEARCH] Intentando buscar en: {global_search_url}")
        pages_checked += 1
        html = fetch(global_search_url)
        print(f"[GLOBAL SEARCH] HTML recibido: {len(html) if html else 0} bytes")
        if html:
            print(f"[GLOBAL SEARCH] Primeros 300 caracteres: {html[:300]}")
            fetch_ok += 1
            movie_links, _ = extract_links(html, global_search_url)
            print(f"[GLOBAL SEARCH] Links encontrados: {len(movie_links)}")
            for movie_url in movie_links:
                if movie_url not in seen_urls:
                    seen_urls.add(movie_url)
                    discovered_urls.append(movie_url)
        else:
            print("[GLOBAL SEARCH] ERROR: No se pudo obtener HTML (posible bloqueo de bot)")
            fetch_failed += 1

        print(
            "Diagnostico web -> "
            f"paginas={pages_checked}, fetch_ok={fetch_ok}, fetch_failed={fetch_failed}, "
            f"links_descubiertos={len(discovered_urls)}"
        )

        documents: List[Dict] = []
        seen_doc_urls = set()
        scrape_ok = 0
        scrape_failed = 0
        print(f"\n[SCRAPING] Iniciando scraping de {min(len(discovered_urls), max_results)} URLs encontradas...")

        for movie_url in discovered_urls[:max_results]:
            movie = scrape_movie(movie_url)
            if not movie:
                print(f"[SCRAPING] Fallo al scrapear: {movie_url}")
                scrape_failed += 1
                continue
            scrape_ok += 1

            url = str(movie.get("url", movie_url))
            if not url or url in seen_doc_urls:
                continue

            print(f"[SCRAPING] OK: {movie.get('title', 'Sin título')}")
            movie["source"] = "web"
            movie["source_query"] = query
            movie["source_url"] = movie_url
            documents.append(movie)
            seen_doc_urls.add(url)

        if documents:
            self._save_cache(documents)
        print(f"Diagnostico scraping -> scrape_ok={scrape_ok}, scrape_failed={scrape_failed}")
        print(f"Documentos expandidos desde la web: {len(documents)}")
        print(f"========== WEB EXPANSION END ==========\n")
        return documents