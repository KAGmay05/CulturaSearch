from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlunparse
import re


SENSACINE_BASE_URL = "https://www.sensacine.com"


def _normalize_sensacine_url(href):
    if not href:
        return None

    href = href.strip()
    if not href or href.startswith(("javascript:", "mailto:", "#")):
        return None

    if href.startswith("//"):
        href = f"https:{href}"
    elif href.startswith("/"):
        href = urljoin(SENSACINE_BASE_URL, href)

    parsed = urlparse(href)
    if "sensacine.com" not in parsed.netloc:
        return None

    match = re.search(r"^(/peliculas/pelicula-\d+|/series/serie-\d+)", parsed.path)
    if not match:
        return None

    normalized_path = match.group(1).rstrip("/") + "/"
    return urlunparse((parsed.scheme or "https", "www.sensacine.com", normalized_path, "", "", ""))


def _extract_filmaffinity_links(html, current_url):
    """Extrae enlaces de películas de FilmAffinity."""
    soup = BeautifulSoup(html, "lxml")
    movie_links = set()
    
    for tag in soup.find_all("a", href=True):
        href = tag["href"]
        if re.search(r'/es/film\d+\.html', href):
            full_url = urljoin("https://www.filmaffinity.com", href)
            movie_links.add(full_url)
    
    return movie_links


def _extract_sensacine_links(html, current_url):
    """Extrae enlaces de películas Y series de SensaCine."""
    soup = BeautifulSoup(html, "lxml")
    movie_links = set()
    series_links = set()
    
    for tag in soup.find_all("a", href=True):
        href = tag["href"]

        normalized_url = _normalize_sensacine_url(href)
        if not normalized_url:
            continue

        if "/peliculas/pelicula-" in normalized_url:
            movie_links.add(normalized_url)
        elif "/series/serie-" in normalized_url:
            series_links.add(normalized_url)
    
    return movie_links, series_links


def extract_links(html, current_url):
    """
    Extrae enlaces de películas y series detectando automáticamente la fuente.
    
    Soporta:
    - FilmAffinity: /es/film{ID}.html
    - SensaCine: /peliculas/pelicula-{ID} y /series/serie-{ID}
    """
    
    movie_links = set()
    series_links = set()
    
    # Detecta la fuente por la URL actual
    if "filmaffinity.com" in current_url:
        movie_links = _extract_filmaffinity_links(html, current_url)
    elif "sensacine.com" in current_url:
        movie_links, series_links = _extract_sensacine_links(html, current_url)
    else:
        # Intenta ambas estrategias por si acaso
        movie_links.update(_extract_filmaffinity_links(html, current_url))
        sensacine_movies, sensacine_series = _extract_sensacine_links(html, current_url)
        movie_links.update(sensacine_movies)
        series_links.update(sensacine_series)

    return movie_links, series_links