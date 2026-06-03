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
FILMAFFINITY_BASE_URL = "https://www.filmaffinity.com"
FILMAFFINITY_SEARCH_URL = f"{FILMAFFINITY_BASE_URL}/es/search.php"
SENSACINE_SEARCH_URL = f"{SENSACINE_BASE_URL}/buscar/"
DEFAULT_SEED_FILE_PATHS = ["seeds/seed_sensacine.json"]
DEFAULT_FILMAFFINITY_SEED_FILE_PATHS = ["seeds/seed_film_affinity.json"]
DEFAULT_WEB_CACHE_PATH = "data/web_cache.json"
DEFAULT_QUERY_CACHE_PATH = "data/web_expansion_queries.json"


class WebExpander:
    """Discovers and scrapes external SensaCine documents for query expansion."""

    def __init__(
        self,
        cache_path: str = DEFAULT_WEB_CACHE_PATH,
        seed_file_paths: List[str] | None = None,
        max_pages_per_seed: int = 2,
        query_cache_path: str = DEFAULT_QUERY_CACHE_PATH,
    ) -> None:
        """Configures cache location, seeds and traversal limits for expansion."""
        self.cache_path = cache_path
        self.seed_file_paths = seed_file_paths or list(DEFAULT_SEED_FILE_PATHS)
        self.max_pages_per_seed = max_pages_per_seed
        # query_cache_path is derived from cache_path dir so both files stay together
        cache_dir = os.path.dirname(cache_path) or "data"
        self.query_cache_path = os.path.join(cache_dir, os.path.basename(query_cache_path))

    def _load_seed_urls(self) -> List[str]:
        """Loads valid seed URLs from configured JSON files."""
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
        """Builds paginated seed URL preserving existing query parameters."""
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
        """Resolves DuckDuckGo redirect links into final destination URLs."""
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
        """Normalizes SensaCine movie/series URLs into canonical form."""
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

    @staticmethod
    def _normalize_filmaffinity_url(url: str) -> str:
        """Normalizes FilmAffinity movie URLs into canonical form."""
        if not url:
            return ""

        parsed = urlparse(url)
        if "filmaffinity.com" not in parsed.netloc:
            return ""

        match = re.search(r"^(/es/film\d+\.html)", parsed.path)
        if not match:
            return ""

        normalized_path = match.group(1)
        return urlunparse((parsed.scheme or "https", "www.filmaffinity.com", normalized_path, "", "", ""))

    def _discover_from_duckduckgo(
        self,
        query: str,
        max_results: int,
        site_domain: str = "sensacine.com",
    ) -> Tuple[List[str], Dict[str, int]]:
        """Finds candidate URLs by querying DuckDuckGo HTML results for a site."""
        stats = {"pages_checked": 0, "fetch_ok": 0, "fetch_failed": 0}
        search_query = f"site:{site_domain} {query.strip()}"
        search_url = f"{DUCKDUCKGO_SEARCH_URL}?{urlencode({'q': search_query})}"

        print(f"\n[DDG:{site_domain.upper()}] Intentando descubrir URLs con: {search_url}")
        stats["pages_checked"] += 1
        html = fetch(search_url)
        print(f"[DDG:{site_domain.upper()}] HTML recibido: {len(html) if html else 0} bytes")
        if not html:
            stats["fetch_failed"] += 1
            return [], stats

        stats["fetch_ok"] += 1
        soup = BeautifulSoup(html, "html.parser")
        discovered_urls: List[str] = []
        seen_urls = set()

        for anchor in soup.select("a[data-testid='result-title-a'], a.result__a"):
            resolved_url = self._resolve_duckduckgo_result_url(anchor.get("href", ""))
            if site_domain == "filmaffinity.com":
                normalized_url = self._normalize_filmaffinity_url(resolved_url)
            else:
                normalized_url = self._normalize_sensacine_url(resolved_url)
            if not normalized_url or normalized_url in seen_urls:
                continue

            discovered_urls.append(normalized_url)
            seen_urls.add(normalized_url)
            if len(discovered_urls) >= max_results:
                break

        print(f"[DDG:{site_domain.upper()}] URLs relevantes encontradas: {len(discovered_urls)}")
        return discovered_urls, stats

    def _discover_from_sensacine_search(
        self,
        query: str,
        max_results: int,
        allow_playwright: bool = True,
    ) -> Tuple[List[str], Dict[str, int]]:
        """Finds candidate URLs from SensaCine internal search pages."""
        stats = {"pages_checked": 0, "fetch_ok": 0, "fetch_failed": 0}
        search_url = f"{SENSACINE_SEARCH_URL}?{urlencode({'q': query.strip()})}"

        print(f"\n[SENSACINE SEARCH] Buscando en: {search_url}")
        stats["pages_checked"] += 1
        html = fetch(search_url, allow_playwright=allow_playwright)
        print(f"[SENSACINE SEARCH] HTML recibido: {len(html) if html else 0} bytes")
        if not html:
            stats["fetch_failed"] += 1
            return [], stats

        stats["fetch_ok"] += 1
        movie_links, series_links = extract_links(html, search_url)

        # Si requests devolvio HTML pero no enlaces utiles, forzar un reintento
        # con navegador real (Playwright) para sortear consent-wall/anti-bot.
        if allow_playwright and not movie_links and not series_links and PLAYWRIGHT_AVAILABLE:
            print("[SENSACINE SEARCH] Sin enlaces utiles, reintentando con Playwright...")
            html = fetch(search_url, force_playwright=True, respect_robots=False)
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

    def _discover_from_filmaffinity_search(self, query: str, max_results: int) -> Tuple[List[str], Dict[str, int]]:
        """Finds candidate FilmAffinity URLs by querying FilmAffinity directly."""
        stats = {"pages_checked": 0, "fetch_ok": 0, "fetch_failed": 0}
        search_url = f"{FILMAFFINITY_SEARCH_URL}?{urlencode({'stext': query.strip()})}"

        print(f"\n[FILMAFFINITY SEARCH] Buscando en: {search_url}")
        stats["pages_checked"] += 1
        html = None
        if PLAYWRIGHT_AVAILABLE:
            html = fetch(search_url, force_playwright=True, respect_robots=False)
        if not html:
            html = fetch(search_url, allow_playwright=False)
        print(f"[FILMAFFINITY SEARCH] HTML recibido: {len(html) if html else 0} bytes")
        if not html:
            stats["fetch_failed"] += 1
            return [], stats

        stats["fetch_ok"] += 1
        movie_links, _ = extract_links(html, search_url)

        if not movie_links and PLAYWRIGHT_AVAILABLE:
            print("[FILMAFFINITY SEARCH] Sin enlaces útiles, reintentando con Playwright...")
            html = fetch(search_url, force_playwright=True, respect_robots=False)
            print(f"[FILMAFFINITY SEARCH] HTML tras fallback browser: {len(html) if html else 0} bytes")
            if html:
                movie_links, _ = extract_links(html, search_url)

        discovered_urls: List[str] = []
        seen_urls = set()

        for candidate_url in movie_links:
            normalized_url = self._normalize_filmaffinity_url(candidate_url)
            if not normalized_url or normalized_url in seen_urls:
                continue

            seen_urls.add(normalized_url)
            discovered_urls.append(normalized_url)
            if len(discovered_urls) >= max_results:
                break

        print(f"[FILMAFFINITY SEARCH] URLs relevantes encontradas: {len(discovered_urls)}")
        return discovered_urls, stats

    def _discover_from_seed_pages(
        self,
        max_results: int,
        seed_file_paths: List[str] | None = None,
        default_seed_urls: List[str] | None = None,
        label: str = "SEED SEARCH",
        normalize_to_filmaffinity: bool = False,
    ) -> Tuple[List[str], Dict[str, int]]:
        """Traverses seed listing pages when direct search discovery fails."""
        stats = {"pages_checked": 0, "fetch_ok": 0, "fetch_failed": 0}
        seed_urls: List[str] = []
        if seed_file_paths is None:
            seed_urls = self._load_seed_urls()
        else:
            for seed_file in seed_file_paths:
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
                            seed_urls.append(url_text)

        print(f"Seed URLs cargadas: {len(seed_urls)}")

        if not seed_urls:
            seed_urls = default_seed_urls or []
            if seed_urls:
                print("[WARNING] No se encontraron seeds, usando seeds por defecto")

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
                    if normalize_to_filmaffinity:
                        normalized_url = self._normalize_filmaffinity_url(candidate_url)
                    else:
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
            f"Diagnostico {label.lower()} -> "
            f"paginas={stats['pages_checked']}, fetch_ok={stats['fetch_ok']}, fetch_failed={stats['fetch_failed']}, "
            f"links_descubiertos={len(discovered_urls)}"
        )
        return discovered_urls, stats

    def _load_cache(self) -> List[Dict]:
        """Loads cached web documents from disk, returning only valid dict items."""
        if not os.path.exists(self.cache_path):
            return []

        with open(self.cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)

        if isinstance(cached, list):
            return [doc for doc in cached if isinstance(doc, dict)]

        return []

    def _save_cache(self, new_documents: List[Dict]) -> None:
        """Merges new web documents into cache keyed by normalized URL."""
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

    def _load_query_cache(self) -> set:
        """Returns the set of queries that have already been web-expanded."""
        if not os.path.exists(self.query_cache_path):
            return set()
        try:
            with open(self.query_cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return set(data)
        except Exception:
            pass
        return set()

    def _save_query_cache(self, query: str) -> None:
        """Persists a query to the expanded-queries cache."""
        existing = self._load_query_cache()
        existing.add(query.strip().lower())
        os.makedirs(os.path.dirname(self.query_cache_path) or ".", exist_ok=True)
        with open(self.query_cache_path, "w", encoding="utf-8") as f:
            json.dump(sorted(existing), f, ensure_ascii=False, indent=2)

    def expand(self, query: str, max_results: int = 10) -> List[Dict]:
        """Expands corpus with scraped web documents relevant to the query.

        Uses a discovery cascade (SensaCine search, FilmAffinity, DuckDuckGo, seeds), scrapes
        the resulting pages, persists cache and returns newly obtained docs.
        Skips re-expansion entirely if the query has already been expanded before.
        Skips individual URLs already present in the document cache.
        """
        if not query or not query.strip():
            return []

        # Skip re-expansion for queries already processed
        query_key = query.strip().lower()
        if query_key in self._load_query_cache():
            print(f"[WEB EXPANSION] Query ya expandida anteriormente, omitiendo: '{query}'")
            return []

        # Build set of URLs already in the doc cache so we don't re-scrape them
        cached_urls = {str(doc.get("url", "")) for doc in self._load_cache() if doc.get("url")}
        print(f"[WEB EXPANSION] URLs ya en cache: {len(cached_urls)}")

        print(f"\n========== WEB EXPANSION START ==========")
        print(f"Query: '{query}'")
        print(f"Max results: {max_results}")

        discovered_urls: List[str] = []
        seen_urls = set()
        pages_checked = 0
        fetch_ok = 0
        fetch_failed = 0

        search_urls, search_stats = self._discover_from_sensacine_search(query, max_results, allow_playwright=False)
        pages_checked += search_stats["pages_checked"]
        fetch_ok += search_stats["fetch_ok"]
        fetch_failed += search_stats["fetch_failed"]

        for candidate_url in search_urls:
            if candidate_url not in seen_urls:
                seen_urls.add(candidate_url)
                discovered_urls.append(candidate_url)

        if len(discovered_urls) < max_results:
            filmaffinity_urls, filmaffinity_stats = self._discover_from_filmaffinity_search(
                query,
                max_results - len(discovered_urls),
            )
            pages_checked += filmaffinity_stats["pages_checked"]
            fetch_ok += filmaffinity_stats["fetch_ok"]
            fetch_failed += filmaffinity_stats["fetch_failed"]

            for candidate_url in filmaffinity_urls:
                if candidate_url not in seen_urls:
                    seen_urls.add(candidate_url)
                    discovered_urls.append(candidate_url)

        if len(discovered_urls) < max_results and PLAYWRIGHT_AVAILABLE:
            playwright_urls, playwright_stats = self._discover_from_sensacine_search(
                query,
                max_results - len(discovered_urls),
                allow_playwright=True,
            )
            pages_checked += playwright_stats["pages_checked"]
            fetch_ok += playwright_stats["fetch_ok"]
            fetch_failed += playwright_stats["fetch_failed"]

            for candidate_url in playwright_urls:
                if candidate_url not in seen_urls:
                    seen_urls.add(candidate_url)
                    discovered_urls.append(candidate_url)

        # Seeds como ultimo recurso real: primero FilmAffinity y luego SensaCine.
        if len(discovered_urls) < max_results:
            seed_urls, seed_stats = self._discover_from_seed_pages(
                max_results - len(discovered_urls),
                seed_file_paths=DEFAULT_FILMAFFINITY_SEED_FILE_PATHS,
                normalize_to_filmaffinity=True,
                label="FILMAFFINITY SEEDS",
            )
            pages_checked += seed_stats["pages_checked"]
            fetch_ok += seed_stats["fetch_ok"]
            fetch_failed += seed_stats["fetch_failed"]

            for candidate_url in seed_urls:
                if candidate_url not in seen_urls:
                    seen_urls.add(candidate_url)
                    discovered_urls.append(candidate_url)

        if len(discovered_urls) < max_results:
            sensacine_seed_urls, sensacine_seed_stats = self._discover_from_seed_pages(
                max_results - len(discovered_urls),
                label="SENSACINE SEEDS",
            )
            pages_checked += sensacine_seed_stats["pages_checked"]
            fetch_ok += sensacine_seed_stats["fetch_ok"]
            fetch_failed += sensacine_seed_stats["fetch_failed"]

            for candidate_url in sensacine_seed_urls:
                if candidate_url not in seen_urls:
                    seen_urls.add(candidate_url)
                    discovered_urls.append(candidate_url)

        print(
            "Diagnostico web -> "
            f"paginas={pages_checked}, fetch_ok={fetch_ok}, fetch_failed={fetch_failed}, "
            f"links_descubiertos={len(discovered_urls)}"
        )

        documents: List[Dict] = []
        seen_doc_urls = set()
        scrape_ok = 0
        scrape_failed = 0
        skipped_cached = 0
        urls_to_scrape = [u for u in discovered_urls[:max_results] if u not in cached_urls]
        print(f"\n[SCRAPING] Iniciando scraping de {len(urls_to_scrape)} URLs nuevas (omitidas {len(discovered_urls[:max_results]) - len(urls_to_scrape)} ya en cache)...")

        for movie_url in urls_to_scrape:
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
            self._save_query_cache(query)
        elif not urls_to_scrape:
            # All discovered URLs were already cached — still mark query as done
            self._save_query_cache(query)
        print(f"Diagnostico scraping -> scrape_ok={scrape_ok}, scrape_failed={scrape_failed}")
        print(f"Documentos expandidos desde la web: {len(documents)}")
        print(f"========== WEB EXPANSION END ==========\n")
        return documents