import json
import os
import re
from typing import Dict, List, Tuple
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from bs4 import BeautifulSoup

from crawler.fetcher import PLAYWRIGHT_AVAILABLE, fetch
from crawler.parser import extract_links
from scraper.scraper import scrape_movie


DUCKDUCKGO_SEARCH_URL = "https://html.duckduckgo.com/html/"
SENSACINE_BASE_URL = "https://www.sensacine.com"
SENSACINE_SEARCH_URL = f"{SENSACINE_BASE_URL}/buscar/"
DEFAULT_SEED_FILE_PATHS = ["seeds/seed_sensacine.json"]
DEFAULT_WEB_CACHE_PATH = "data/web_cache_sensacine.json"


class WebExpander:
    def __init__(
        self,
        cache_path: str = DEFAULT_WEB_CACHE_PATH,
        seed_file_paths: List[str] | None = None,
        max_pages_per_seed: int = 2,
    ) -> None:
        self.cache_path = cache_path
        self.seed_file_paths = seed_file_paths or list(DEFAULT_SEED_FILE_PATHS)
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
    def _build_seed_page_url(seed_url: str, page: int) -> str:
        parsed = urlparse(seed_url)
        query_params = parse_qs(parsed.query, keep_blank_values=True)

        if page > 1:
            query_params["page"] = [str(page)]
        else:
            query_params.pop("page", None)

        rebuilt_query = urlencode(query_params, doseq=True)
        rebuilt = parsed._replace(query=rebuilt_query)
        return urlunparse(rebuilt)

    @staticmethod
    def _resolve_duckduckgo_result_url(href: str) -> str:
        if not href:
            return ""

        if href.startswith("//"):
            href = f"https:{href}"

        parsed = urlparse(href)
        if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
            redirect_url = parse_qs(parsed.query).get("uddg", [""])[0]
            return redirect_url

        return href

    @staticmethod
    def _normalize_sensacine_url(url: str) -> str:
        if not url:
            return ""

        parsed = urlparse(url)
        if "sensacine.com" not in parsed.netloc:
            return ""

        match = re.search(r"^(/peliculas/pelicula-\d+|/series/serie-\d+)", parsed.path)
        if not match:
            return ""

        normalized_path = match.group(1).rstrip("/") + "/"
        return urlunparse((parsed.scheme or "https", "www.sensacine.com", normalized_path, "", "", ""))

    def _discover_from_duckduckgo(self, query: str, max_results: int) -> Tuple[List[str], Dict[str, int]]:
        stats = {"pages_checked": 0, "fetch_ok": 0, "fetch_failed": 0}
        search_query = f"site:sensacine.com {query.strip()}"
        search_url = f"{DUCKDUCKGO_SEARCH_URL}?{urlencode({'q': search_query})}"

        print(f"\n[DDG] Intentando descubrir URLs con: {search_url}")
        stats["pages_checked"] += 1
        html = fetch(search_url)
        print(f"[DDG] HTML recibido: {len(html) if html else 0} bytes")
        if not html:
            stats["fetch_failed"] += 1
            return [], stats

        stats["fetch_ok"] += 1
        soup = BeautifulSoup(html, "html.parser")
        discovered_urls: List[str] = []
        seen_urls = set()

        for anchor in soup.select("a[data-testid='result-title-a'], a.result__a"):
            resolved_url = self._resolve_duckduckgo_result_url(anchor.get("href", ""))
            normalized_url = self._normalize_sensacine_url(resolved_url)
            if not normalized_url or normalized_url in seen_urls:
                continue

            discovered_urls.append(normalized_url)
            seen_urls.add(normalized_url)
            if len(discovered_urls) >= max_results:
                break

        print(f"[DDG] URLs relevantes encontradas: {len(discovered_urls)}")
        return discovered_urls, stats

    def _discover_from_sensacine_search(self, query: str, max_results: int) -> Tuple[List[str], Dict[str, int]]:
        stats = {"pages_checked": 0, "fetch_ok": 0, "fetch_failed": 0}
        search_url = f"{SENSACINE_SEARCH_URL}?{urlencode({'q': query.strip()})}"

        print(f"\n[SENSACINE SEARCH] Buscando en: {search_url}")
        stats["pages_checked"] += 1
        html = fetch(search_url)
        print(f"[SENSACINE SEARCH] HTML recibido: {len(html) if html else 0} bytes")
        if not html:
            stats["fetch_failed"] += 1
            return [], stats

        stats["fetch_ok"] += 1
        movie_links, series_links = extract_links(html, search_url)

        # Si requests devolvio HTML pero no enlaces utiles, forzar un reintento
        # con navegador real (Playwright) para sortear consent-wall/anti-bot.
        if not movie_links and not series_links and PLAYWRIGHT_AVAILABLE:
            print("[SENSACINE SEARCH] Sin enlaces utiles, reintentando con Playwright...")
            html = fetch(search_url, force_playwright=True)
            print(f"[SENSACINE SEARCH] HTML tras fallback browser: {len(html) if html else 0} bytes")
            if html:
                movie_links, series_links = extract_links(html, search_url)

        candidates = list(movie_links) + list(series_links)
        discovered_urls: List[str] = []
        seen_urls = set()

        for candidate_url in candidates:
            normalized_url = self._normalize_sensacine_url(candidate_url)
            if not normalized_url or normalized_url in seen_urls:
                continue

            seen_urls.add(normalized_url)
            discovered_urls.append(normalized_url)
            if len(discovered_urls) >= max_results:
                break

        print(f"[SENSACINE SEARCH] URLs relevantes encontradas: {len(discovered_urls)}")
        return discovered_urls, stats

    def _discover_from_seed_pages(self, max_results: int) -> Tuple[List[str], Dict[str, int]]:
        stats = {"pages_checked": 0, "fetch_ok": 0, "fetch_failed": 0}
        seed_urls = self._load_seed_urls()
        print(f"Seed URLs cargadas: {len(seed_urls)}")

        if not seed_urls:
            seed_urls = [
                f"{SENSACINE_BASE_URL}/peliculas/",
                f"{SENSACINE_BASE_URL}/peliculas/estrenos/es/",
                f"{SENSACINE_BASE_URL}/series/",
            ]
            print("[WARNING] No se encontraron seeds, usando seeds de SensaCine por defecto")

        discovered_urls: List[str] = []
        seen_urls = set()

        for seed_url in seed_urls:
            for page in range(1, self.max_pages_per_seed + 1):
                page_url = self._build_seed_page_url(seed_url, page)
                print(f"\n[SEED SEARCH] Intentando página {page}: {page_url}")
                stats["pages_checked"] += 1

                html = fetch(page_url)
                print(f"[SEED SEARCH] HTML recibido: {len(html) if html else 0} bytes")
                if not html:
                    stats["fetch_failed"] += 1
                    continue

                stats["fetch_ok"] += 1
                movie_links, series_links = extract_links(html, page_url)
                page_links = list(movie_links) + list(series_links)
                print(f"[SEED SEARCH] Links encontrados en página {page}: {len(page_links)}")

                for candidate_url in page_links:
                    normalized_url = self._normalize_sensacine_url(candidate_url)
                    if not normalized_url or normalized_url in seen_urls:
                        continue

                    seen_urls.add(normalized_url)
                    discovered_urls.append(normalized_url)

                    if len(discovered_urls) >= max_results:
                        break

                if len(discovered_urls) >= max_results:
                    break

            if len(discovered_urls) >= max_results:
                break

        print(
            "Diagnostico seeds -> "
            f"paginas={stats['pages_checked']}, fetch_ok={stats['fetch_ok']}, fetch_failed={stats['fetch_failed']}, "
            f"links_descubiertos={len(discovered_urls)}"
        )
        return discovered_urls, stats

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

        discovered_urls: List[str] = []
        seen_urls = set()
        pages_checked = 0
        fetch_ok = 0
        fetch_failed = 0

        search_urls, search_stats = self._discover_from_sensacine_search(query, max_results)
        pages_checked += search_stats["pages_checked"]
        fetch_ok += search_stats["fetch_ok"]
        fetch_failed += search_stats["fetch_failed"]

        for candidate_url in search_urls:
            if candidate_url not in seen_urls:
                seen_urls.add(candidate_url)
                discovered_urls.append(candidate_url)

        if len(discovered_urls) < max_results:
            ddg_urls, ddg_stats = self._discover_from_duckduckgo(query, max_results - len(discovered_urls))
            pages_checked += ddg_stats["pages_checked"]
            fetch_ok += ddg_stats["fetch_ok"]
            fetch_failed += ddg_stats["fetch_failed"]

            for candidate_url in ddg_urls:
                if candidate_url not in seen_urls:
                    seen_urls.add(candidate_url)
                    discovered_urls.append(candidate_url)

        # Seeds como ultimo recurso real: solo si no se encontro nada
        # con la busqueda directa de SensaCine ni con DDG.
        if not discovered_urls:
            seed_urls, seed_stats = self._discover_from_seed_pages(max_results)
            pages_checked += seed_stats["pages_checked"]
            fetch_ok += seed_stats["fetch_ok"]
            fetch_failed += seed_stats["fetch_failed"]

            for candidate_url in seed_urls:
                if candidate_url not in seen_urls:
                    seen_urls.add(candidate_url)
                    discovered_urls.append(candidate_url)
        elif len(discovered_urls) < max_results:
            print("[SEED SEARCH] Omitido: solo se usa como ultimo recurso")

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

            url = self._normalize_sensacine_url(str(movie.get("url", movie_url))) or movie_url
            if not url or url in seen_doc_urls:
                continue

            print(f"[SCRAPING] OK: {movie.get('title', 'Sin título')}")
            movie["url"] = url
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