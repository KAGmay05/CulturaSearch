from functools import lru_cache
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests


ROBOTS_USER_AGENT = "CulturaSearchBot"
ROBOTS_TIMEOUT_SECONDS = 15


def _robots_url_for(url):
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}/robots.txt"


@lru_cache(maxsize=128)
def _load_robot_parser(robots_url):
    parser = RobotFileParser()
    parser.set_url(robots_url)

    try:
        response = requests.get(
            robots_url,
            headers={"User-Agent": f"{ROBOTS_USER_AGENT}/1.0"},
            timeout=ROBOTS_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        print(f"[WARN] No se pudo leer robots.txt ({type(exc).__name__}) para: {robots_url}")
        return None

    if response.status_code == 404:
        return None

    if response.status_code != 200:
        print(f"[WARN] robots.txt no disponible (status={response.status_code}) para: {robots_url}")
        return None

    if not response.text.strip():
        return None

    parser.parse(response.text.splitlines())
    return parser


def can_fetch_url(url, user_agent=ROBOTS_USER_AGENT):
    robots_url = _robots_url_for(url)
    if not robots_url:
        return True

    parser = _load_robot_parser(robots_url)
    if parser is None:
        return True

    return parser.can_fetch(user_agent, url)