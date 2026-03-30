from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re

BASE_URL = "https://www.filmaffinity.com"


def extract_links(html, current_url):

    soup = BeautifulSoup(html, "lxml")

    movie_links = set()
    other_links = set()

    for tag in soup.find_all("a", href=True):

        href = tag["href"]

        if re.search(r'/es/film\d+\.html', href):

            full_url = urljoin(BASE_URL, href)
            movie_links.add(full_url)

    return movie_links, other_links