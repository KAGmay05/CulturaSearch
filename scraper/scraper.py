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

    for _ in range(3):   # retry 3 veces

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
        data["title"] = title.text.strip()

    year = soup.find("span", class_="date")       
    if year:
        data["year"] = year.text.strip()

    rating = soup.find("div", id="movie-rat-avg")
    if rating:
        data["rating"] = rating.text.strip()

    genres = []    
    for g in soup.select("span[itemprop='genre']"):
        genres.append(g.text.strip())

    data["genres"] = genres

    plot = soup.find("dd", itemprop="description")
    if plot:
        data["plot"] = plot.text.strip()

    actors = []
    for a in soup.select("span[itemprop='actor']"):
        actors.append(a.text.strip())

    data["actors"] = actors

    return data

def scrape_movie(url):

    html= fetch(url)
    if not html:
        return None

    return parse_movie(html,url)            
