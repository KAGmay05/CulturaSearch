import requests 
from bs4 import BeautifulSoup
import time
import random
import json

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0)",
    "Mozilla/5.0 (Macintosh)",
    "Mozilla/5.0 (X11; Linux)"
]

headers = {
    "User-Agent": random.choice(USER_AGENTS)
}

def fetch(url):

    for _ in range(3):   

        try:
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 429:
                time.sleep(random.uniform(2,5))
                continue

            return response.text

        except:
            time.sleep(1)

    return None

def parse_json_ld(soup):
    """Extract data from JSON-LD structured data (Movie or TVSeries)"""
    data = {}
    content_type = None
    
    try:
        for script in soup.find_all('script', type='application/ld+json'):
            json_data = json.loads(script.string)
            
            # Detectar tipo de contenido
            if json_data.get('@type') == 'Movie':
                content_type = 'película'
                
                # Extraer título
                if 'name' in json_data:
                    data['title'] = json_data['name']
                
                # Extraer géneros
                if 'genre' in json_data:
                    data['genres'] = json_data['genre'] if isinstance(json_data['genre'], list) else [json_data['genre']]
                
                # Extraer descripción (sinopsis)
                if 'description' in json_data:
                    data['plot'] = json_data['description']
                
                # Extraer director
                if 'director' in json_data:
                    director_data = json_data['director']
                    if isinstance(director_data, dict):
                        data['director'] = [director_data.get('name', '')]
                    elif isinstance(director_data, list):
                        data['director'] = [d.get('name', '') if isinstance(d, dict) else d for d in director_data]
                
                # Extraer actores de JSON-LD
                if 'actor' in json_data:
                    actors = []
                    actor_list = json_data['actor'] if isinstance(json_data['actor'], list) else [json_data['actor']]
                    for actor_item in actor_list:
                        if isinstance(actor_item, dict):
                            if 'actor' in actor_item and isinstance(actor_item['actor'], dict):
                                actors.append(actor_item['actor'].get('name', ''))
                            elif 'name' in actor_item:
                                actors.append(actor_item['name'])
                    if actors:  # Solo usar JSON-LD si tiene actores
                        data['actors'] = actors
                
                # Extraer rating
                if 'aggregateRating' in json_data:
                    rating_data = json_data['aggregateRating']
                    if 'ratingValue' in rating_data:
                        data['rating'] = rating_data['ratingValue']
                
                break
            
            elif json_data.get('@type') == 'TVSeries':
                content_type = 'serie'
                
                # Extraer título
                if 'name' in json_data:
                    data['title'] = json_data['name']
                
                # Extraer géneros
                if 'genre' in json_data:
                    data['genres'] = json_data['genre'] if isinstance(json_data['genre'], list) else [json_data['genre']]
                
                # Extraer descripción (sinopsis)
                if 'description' in json_data:
                    data['plot'] = json_data['description']
                
                # Extraer número de temporadas y episodios
                if 'numberOfSeasons' in json_data:
                    data['seasons'] = json_data['numberOfSeasons']
                
                if 'numberOfEpisodes' in json_data:
                    data['episodes'] = json_data['numberOfEpisodes']
                
                # Extraer director (si está disponible)
                if 'director' in json_data:
                    director_data = json_data['director']
                    if isinstance(director_data, dict):
                        data['director'] = [director_data.get('name', '')]
                    elif isinstance(director_data, list):
                        data['director'] = [d.get('name', '') if isinstance(d, dict) else d for d in director_data]
                
                # Extraer actores (si está disponible)
                if 'actor' in json_data:
                    actors = []
                    actor_list = json_data['actor'] if isinstance(json_data['actor'], list) else [json_data['actor']]
                    for actor_item in actor_list:
                        if isinstance(actor_item, dict):
                            if 'actor' in actor_item and isinstance(actor_item['actor'], dict):
                                actors.append(actor_item['actor'].get('name', ''))
                            elif 'name' in actor_item:
                                actors.append(actor_item['name'])
                    if actors:
                        data['actors'] = actors
                
                # Extraer rating
                if 'aggregateRating' in json_data:
                    rating_data = json_data['aggregateRating']
                    if 'ratingValue' in rating_data:
                        data['rating'] = rating_data['ratingValue']
                
                break
    except Exception as e:
        pass
    
    return data, content_type

def parse_html(soup):
    """Extract additional data from HTML"""
    data = {}
    
    # Extraer país, año, actores de acuerdo al tipo (película o serie)
    
    # 1. PELÍCULAS: Extraer de Especificaciones Técnicas
    tech_section = soup.find('h2', string='Especificaciones técnicas')
    if tech_section:
        current = tech_section.find_next()
        while current:
            if current.name == 'div' and 'item' in str(current.get('class', [])):
                what = current.find('span', class_='what')
                if what:
                    label = what.text.strip()
                    that = current.find('span', class_='that')
                    if that:
                        value = that.text.strip()
                        
                        if 'Nacionalidad' in label:
                            data['country'] = value
                        elif 'Año de producción' in label:
                            data['year'] = value
            elif current.name == 'h2':
                break
            current = current.find_next_sibling()
    
    # 2. PELÍCULAS: Extraer actores del HTML (sección Reparto)
    # Solo si no vinieron del JSON-LD
    reparto_h2 = soup.find('h2', string='Actores y actrices')
    if reparto_h2 and 'actors' not in data:
        container = reparto_h2.find_next('div', class_=True)
        if container:
            actors = []
            for link in container.find_all('a'):
                actor_name = link.text.strip()
                # Filtrar enlaces que son links a películas/series (contienen números de ID)
                href = link.get('href', '').lower()
                if actor_name and 'actor' in href:
                    actors.append(actor_name)
            if actors:
                data['actors'] = actors
    
    # 3. SERIES: Extraer de meta-body-item
    meta_items = soup.find_all('div', class_='meta-body-item')
    for item in meta_items:
        first_span = item.find('span', class_='light')
        if not first_span:
            continue
        
        label = first_span.text.strip()
        
        # Extraer país (Nacionalidad)
        if 'Nacionalidad' in label:
            # El país está en los spans siguientes
            all_spans = item.find_all('span')
            if len(all_spans) > 1:
                # El segundo span es el país
                value = all_spans[1].text.strip()
                data['country'] = value
        
        # Extraer creador (para series) - está en <a> tags
        elif 'Creada por' in label or 'Creador' in label:
            creators = []
            for link in item.find_all('a'):
                creator_name = link.text.strip()
                if creator_name:
                    creators.append(creator_name)
            if creators:
                data['creator'] = creators
        
        # Extraer actores (Reparto en series) - está en spans
        elif 'Reparto' in label:
            actors = []
            all_spans = item.find_all('span')[1:]  # Skip the label span
            for span in all_spans:
                actor_name = span.text.strip()
                if actor_name and ',' not in actor_name:
                    actors.append(actor_name)
            if actors:
                data['actors'] = actors
    
    return data

def parse_movie(html, url):
    soup = BeautifulSoup(html, "html.parser")

    # Filtrar críticas de SensaCine (URLs con /sensacine/ al final)
    if '/sensacine/' in url.lower():
        return None

    data = {}
    data["url"] = url

    # Extraer datos del JSON-LD (retorna tupla: data, content_type)
    json_ld_data, content_type = parse_json_ld(soup)
    data.update(json_ld_data)
    
    # Determinar tipo de contenido basado en JSON-LD
    if content_type == 'serie':
        data["type"] = "serie"
    else:
        data["type"] = "pelicula"
    
    # Extraer datos del HTML (tanto para películas como series)
    html_data = parse_html(soup)
    data.update(html_data)
    
    # Asignar valores por defecto si no se encontraron
    if 'title' not in data:
        title_elem = soup.find("h1")
        if title_elem:
            data['title'] = title_elem.text.strip().split("\n")[0]
    
    if 'genres' not in data:
        data['genres'] = []
    
    if 'plot' not in data:
        data['plot'] = None
    
    # Para películas: director, para series: creator
    if data["type"] == "pelicula":
        if 'director' not in data:
            data['director'] = []
    else:  # serie
        if 'creator' not in data:
            data['creator'] = []
        # Remover director si existía
        data.pop('director', None)
    
    if 'actors' not in data:
        data['actors'] = []
    
    if 'rating' not in data:
        data['rating'] = None
    
    if 'country' not in data:
        data['country'] = None

    return data
    
    if 'year' not in data:
        data['year'] = None

    return data

def scrape_movie(url):

    html= fetch(url)
    if not html:
        return None

    return parse_movie(html,url)            
