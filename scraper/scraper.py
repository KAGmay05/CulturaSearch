import requests 
from bs4 import BeautifulSoup
import time
import requests
import random

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

def parse_movie(html, url):
    soup = BeautifulSoup(html, "html.parser")

    data = {}
    data["url"] = url

    title = soup.find("h1")
    if title:
        clean_title = title.text.strip().split("\n")[0]
        data["title"] = clean_title

    year = soup.select_one("dd[itemprop='datePublished']")       
    if year:
        data["year"] = year.text.strip()

    rating = soup.find("div", id="movie-rat-avg")
    if rating:
        data["rating"] = rating.text.strip()

    genres = []    
    for g in soup.select("span[itemprop='genre']"):
        genres.append(g.text.strip())

    data["genres"] = genres

    if "Serie de TV" in genres:
        data["type"] = "serie"

    else:
        data["type"] = "pelicula"        

    plot = soup.find("dd", itemprop="description")
    if plot:
        data["plot"] = plot.text.strip()

    actors = []

    for dt in soup.select("dt"):
      if "Reparto" in dt.text:
        dd = dt.find_next("dd")

        if dd:
            for a in dd.select("a"):
                actors.append(a.text.strip())

        break

    data["actors"] = actors

    directors = []
    for d in soup.select('[itemprop="director"] span[itemprop="name"]'):
        directors.append(d.text.strip())

    data["director"] = directors

    country = None

    for dt in soup.select("dt"):
      if "País" in dt.text:
        dd = dt.find_next("dd")
        if dd:
            country = dd.text.strip()
            break

    data["country"] = country

    return data

def scrape_movie(url):

    html= fetch(url)
    if not html:
        return None

    return parse_movie(html,url)            
