from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re


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
        
        # SensaCine películas: /peliculas/pelicula-12345 o /peliculas/pelicula-12345/
        if re.search(r'/peliculas/pelicula-\d+/?', href):
            if not href.startswith(('//', 'http')):
                full_url = urljoin("https://www.sensacine.com", href)
            else:
                full_url = href
            movie_links.add(full_url)
        
        # SensaCine series: /series/serie-12345 o /series/serie-12345/
        elif re.search(r'/series/serie-\d+/?', href):
            if not href.startswith(('//', 'http')):
                full_url = urljoin("https://www.sensacine.com", href)
            else:
                full_url = href
            series_links.add(full_url)
    
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