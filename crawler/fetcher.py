import requests
import time
import random

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
}


def fetch(url):

    try:

        time.sleep(random.uniform(1.5, 3))

        response = requests.get(url, headers=HEADERS, timeout=15)

        if response.status_code == 200:
            return response.text

        else:
            return None

    except Exception as e:

        print(f"Request failed: {url}")
        print(e)

        return None