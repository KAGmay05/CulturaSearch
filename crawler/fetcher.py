import random
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
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

    try:
        print(f"[INFO] Usando Playwright para: {url[:80]}...")
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_extra_http_headers(HEADERS)
            
            try:
                page.goto(url, timeout=30000, wait_until="domcontentloaded")
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


def fetch(url):
    """
    Fetch inteligente con fallback requests→Playwright.
    
    1. Intenta requests (rápido, con reintentos y delays)
    2. Si falla/challenge, fallback a Playwright si está disponible
    3. Retorna HTML o None
    """
    
    html = _fetch_with_requests(url, retries=3)
    if html:
        return html

    # Fallback a Playwright si está disponible
    if PLAYWRIGHT_AVAILABLE:
        print("[INFO] Fallback a Playwright...")
        html = _fetch_with_playwright(url)
        if html:
            return html

    print(f"[WARN] No se obtuvo contenido para: {url[:80]}...")
    return None