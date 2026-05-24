import random
import time
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from crawler.robots import ROBOTS_USER_AGENT, can_fetch_url

HEADERS = {
    "User-Agent": f"{ROBOTS_USER_AGENT}/1.0 (+https://github.com/kelen/CulturaSearch)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
    "Cache-Control": "max-age=0",
    "Connection": "keep-alive",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}

PLAYWRIGHT_AVAILABLE = False

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    pass


def _create_session_with_retries():
    """Crea sesión con retry strategy para transient errors."""
    session = requests.Session()
    
    retry_strategy = Retry(
        total=2,
        backoff_factor=1,
        status_forcelist=[429, 503, 504],
        allowed_methods=["GET"],
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session


SESSION = _create_session_with_retries()


def _looks_like_challenge(html):
    """Detecta si la respuesta es un challenge de Cloudflare."""
    if not html:
        return False

    markers = (
        "Just a moment...",
        "Attention Required! | Cloudflare",
        "cf-browser-verification",
        "/cdn-cgi/challenge-platform/",
        "cf_clearance",
        "Please wait...",
    )
    return any(marker in html for marker in markers)


def _fetch_with_requests(url, retries=3):
    """Intenta obtener HTML usando requests con estrategia de reintentos."""
    if not can_fetch_url(url):
        print(f"[WARN] Bloqueado por robots.txt: {url[:80]}...")
        return None

    for attempt in range(1, retries + 1):
        try:
            time.sleep(random.uniform(2.0, 4.0))
            
            response = SESSION.get(
                url,
                headers=HEADERS,
                timeout=25,
                allow_redirects=True,
            )

            if response.status_code == 200 and not _looks_like_challenge(response.text):
                return response.text

            if response.status_code in (403, 429, 503):
                print(
                    f"[WARN] Status {response.status_code} en intento {attempt}/{retries}"
                )
                if attempt < retries:
                    time.sleep(random.uniform(5.0, 10.0))
                    continue
                return None

            if _looks_like_challenge(response.text):
                print(
                    f"[WARN] Challenge detectado (status={response.status_code}) en intento {attempt}/{retries}"
                )
                if attempt < retries:
                    time.sleep(random.uniform(5.0, 10.0))
                    continue
                return None

        except requests.RequestException as e:
            print(f"[WARN] Error en request ({attempt}/{retries}): {type(e).__name__}")
            if attempt < retries:
                time.sleep(random.uniform(3.0, 6.0))
                continue

    return None


def _fetch_with_playwright(url):
    """Fallback: usa Playwright para ejecutar JS y burlar Cloudflare."""
    if not PLAYWRIGHT_AVAILABLE:
        print("[ERROR] Playwright no está disponible")
        return None

    if not can_fetch_url(url):
        print(f"[WARN] Bloqueado por robots.txt: {url[:80]}...")
        return None

    try:
        print(f"[INFO] Usando Playwright para: {url[:80]}...")
        with sync_playwright() as p:
            browser = None
            launch_errors = []

            for channel in ("msedge", "chrome"):
                try:
                    browser = p.chromium.launch(channel=channel, headless=True)
                    print(f"[INFO] Playwright lanzado con canal del sistema: {channel}")
                    break
                except Exception as e:
                    launch_errors.append(f"{channel}:{type(e).__name__}")

            if browser is None:
                try:
                    browser = p.chromium.launch(headless=True)
                    print("[INFO] Playwright lanzado con Chromium gestionado")
                except Exception as e:
                    launch_errors.append(f"managed:{type(e).__name__}")
                    print(f"[ERROR] No se pudo lanzar Playwright ({', '.join(launch_errors)})")
                    return None

            page = browser.new_page()
            page.set_extra_http_headers(HEADERS)
            
            try:
                page.goto(url, timeout=30000, wait_until="domcontentloaded")

                cookie_selectors = [
                    "#didomi-notice-agree-button",
                    "button:has-text('Aceptar y leer GRATIS')",
                    "button:has-text('Aceptar')",
                    "button:has-text('Accept')",
                ]
                for selector in cookie_selectors:
                    try:
                        button = page.locator(selector).first
                        if button.is_visible(timeout=1200):
                            button.click(timeout=1500)
                            break
                    except Exception:
                        continue

                time.sleep(2)
                html = page.content()
                
                if _looks_like_challenge(html):
                    print("[WARN] Challenge aún presente con Playwright")
                    browser.close()
                    return None
                
                browser.close()
                print("[INFO] Página obtenida con Playwright")
                return html
                
            except Exception as e:
                print(f"[ERROR] Playwright navegación falló: {type(e).__name__}")
                try:
                    browser.close()
                except:
                    pass
                return None

    except Exception as e:
        print(f"[ERROR] Playwright error: {type(e).__name__}")
        return None


def fetch(url, force_playwright=False):
    """
    Fetch inteligente con fallback requests→Playwright.
    
    1. Intenta requests (rápido, con reintentos y delays)
    2. Si falla/challenge, fallback a Playwright si está disponible
    3. Retorna HTML o None
    """
    
    if force_playwright:
        if PLAYWRIGHT_AVAILABLE:
            print("[INFO] Fetch forzado con Playwright...")
            return _fetch_with_playwright(url)
        print("[WARN] force_playwright=True pero Playwright no está disponible")
        return None

    html = _fetch_with_requests(url, retries=3)
    if html:
        parsed_url = urlparse(url)
        is_sensacine_search = (
            "sensacine.com" in parsed_url.netloc and parsed_url.path.startswith("/buscar/")
        )
        has_expected_results = (
            "/peliculas/pelicula-" in html or "/series/serie-" in html
        )
        if is_sensacine_search and not has_expected_results and PLAYWRIGHT_AVAILABLE:
            print("[INFO] Search HTML sin resultados detectados, fallback forzado a Playwright...")
            browser_html = _fetch_with_playwright(url)
            if browser_html:
                return browser_html

        return html

    # Fallback a Playwright si está disponible
    if PLAYWRIGHT_AVAILABLE:
        print("[INFO] Fallback a Playwright...")
        html = _fetch_with_playwright(url)
        if html:
            return html

    print(f"[WARN] No se obtuvo contenido para: {url[:80]}...")
    return None