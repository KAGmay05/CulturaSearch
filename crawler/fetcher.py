import os
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
CURL_CFFI_AVAILABLE = False
PLAYWRIGHT_STEALTH_AVAILABLE = False

DEFAULT_HUMAN_DELAY_MIN_SECONDS = float(os.environ.get("FETCH_DELAY_MIN_SECONDS", "3"))
DEFAULT_HUMAN_DELAY_MAX_SECONDS = float(os.environ.get("FETCH_DELAY_MAX_SECONDS", "7"))
DEFAULT_PROXY_SERVER = os.environ.get("FETCH_PROXY_SERVER", "").strip()

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    pass

try:
    from curl_cffi import requests as curl_requests
    CURL_CFFI_AVAILABLE = True
except ImportError:
    curl_requests = None

try:
    from playwright_stealth import stealth_sync
    PLAYWRIGHT_STEALTH_AVAILABLE = True
except ImportError:
    stealth_sync = None


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


def _sleep_human(min_seconds=DEFAULT_HUMAN_DELAY_MIN_SECONDS, max_seconds=DEFAULT_HUMAN_DELAY_MAX_SECONDS):
    """Introduce una pausa aleatoria para simular navegación humana."""
    try:
        min_value = max(0.0, float(min_seconds))
        max_value = max(min_value, float(max_seconds))
        time.sleep(random.uniform(min_value, max_value))
    except Exception:
        time.sleep(3)


def _get_proxy_settings():
    """Devuelve la configuración de proxy si está activada por entorno."""
    proxy_server = DEFAULT_PROXY_SERVER
    if not proxy_server:
        return None

    return {
        "http": proxy_server,
        "https": proxy_server,
    }


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


def _create_curl_session():
    """Crea una sesión curl_cffi con impersonación de navegador si está disponible."""
    if not CURL_CFFI_AVAILABLE:
        return None

    try:
        session = curl_requests.Session(impersonate="chrome120")
        session.headers.update(HEADERS)
        return session
    except Exception:
        return None


CURL_SESSION = _create_curl_session()


def _fetch_with_requests(url, retries=3, respect_robots=True):
    """Intenta obtener HTML usando requests con estrategia de reintentos."""
    if respect_robots and not can_fetch_url(url):
        print(f"[WARN] Bloqueado por robots.txt: {url[:80]}...")
        return None

    for attempt in range(1, retries + 1):
        try:
            _sleep_human()
            
            session = CURL_SESSION or SESSION
            proxy_settings = _get_proxy_settings()
            response = session.get(
                url,
                headers=HEADERS if session is SESSION else None,
                timeout=25,
                allow_redirects=True,
                proxies=proxy_settings,
            )

            if response.status_code == 200 and not _looks_like_challenge(response.text):
                return response.text

            if response.status_code in (403, 429, 503):
                print(
                    f"[WARN] Status {response.status_code} en intento {attempt}/{retries}"
                )
                if attempt < retries:
                    _sleep_human(5.0, 10.0)
                    continue
                return None

            if _looks_like_challenge(response.text):
                print(
                    f"[WARN] Challenge detectado (status={response.status_code}) en intento {attempt}/{retries}"
                )
                if attempt < retries:
                    _sleep_human(5.0, 10.0)
                    continue
                return None

        except requests.RequestException as e:
            print(f"[WARN] Error en request ({attempt}/{retries}): {type(e).__name__}")
            if attempt < retries:
                _sleep_human(3.0, 6.0)
                continue

    return None


def _fetch_with_playwright(url, respect_robots=False, challenge_retries=3):
    """Fallback: usa Playwright para ejecutar JS y burlar Cloudflare."""
    if not PLAYWRIGHT_AVAILABLE:
        print("[ERROR] Playwright no está disponible")
        return None

    if respect_robots and not can_fetch_url(url):
        print(f"[WARN] Bloqueado por robots.txt: {url[:80]}...")
        return None

    try:
        print(f"[INFO] Usando Playwright para: {url[:80]}...")
        with sync_playwright() as p:
            browser = None
            launch_errors = []

            launch_attempts = [
                ("msedge", True, "canal del sistema"),
                ("chrome", True, "canal del sistema"),
                (None, True, "Chromium gestionado"),
            ]

            for channel, headless, label in launch_attempts:
                try:
                    launch_kwargs = {"headless": headless}
                    if channel is not None:
                        launch_kwargs["channel"] = channel
                    browser = p.chromium.launch(**launch_kwargs)
                    print(
                        f"[INFO] Playwright lanzado con {label}: {channel or 'managed'} (headless={headless})"
                    )
                    break
                except Exception as e:
                    launch_errors.append(f"{channel or 'managed'}:{headless}:{type(e).__name__}")

            if browser is None:
                print(f"[ERROR] No se pudo lanzar Playwright ({', '.join(launch_errors)})")
                return None

            context = browser.new_context()
            proxy_settings = _get_proxy_settings()
            if proxy_settings:
                context = browser.new_context(proxy={"server": DEFAULT_PROXY_SERVER})
            page = context.new_page()
            page.set_extra_http_headers(HEADERS)

            if PLAYWRIGHT_STEALTH_AVAILABLE and stealth_sync is not None:
                try:
                    stealth_sync(page)
                    print("[INFO] Playwright stealth aplicado")
                except Exception as e:
                    print(f"[WARN] No se pudo aplicar stealth: {type(e).__name__}")
            
            try:
                _sleep_human(2.0, 5.0)
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
                            _sleep_human(1.0, 2.0)
                            button.click(timeout=1500)
                            break
                    except Exception:
                        continue

                html = None
                for attempt in range(1, challenge_retries + 1):
                    _sleep_human(2.0, 4.0)
                    html = page.content()

                    if not _looks_like_challenge(html):
                        browser.close()
                        print("[INFO] Página obtenida con Playwright")
                        return html

                    print(
                        f"[WARN] Challenge aún presente con Playwright (intento {attempt}/{challenge_retries})"
                    )

                    if attempt < challenge_retries:
                        try:
                            _sleep_human(2.0, 4.0)
                            page.reload(timeout=30000, wait_until="domcontentloaded")
                        except Exception:
                            pass
                
                context.close()
                browser.close()
                return None
                
            except Exception as e:
                print(f"[ERROR] Playwright navegación falló: {type(e).__name__}")
                try:
                    context.close()
                except:
                    pass
                try:
                    browser.close()
                except:
                    pass
                return None

    except Exception as e:
        print(f"[ERROR] Playwright error: {type(e).__name__}")
        return None


def fetch(url, force_playwright=False, allow_playwright=True, respect_robots=True):
    """
    Fetch inteligente con fallback requests→Playwright.
    
    1. Intenta requests (rápido, con reintentos y delays)
    2. Si falla/challenge, fallback a Playwright si está disponible
    3. Retorna HTML o None
    """
    
    if force_playwright:
        if PLAYWRIGHT_AVAILABLE:
            print("[INFO] Fetch forzado con Playwright...")
            return _fetch_with_playwright(url, respect_robots=False)
        print("[WARN] force_playwright=True pero Playwright no está disponible")
        return None

    # Si curl_cffi está disponible, lo priorizamos porque suele sobrevivir mejor a TLS/WAF.
    if CURL_CFFI_AVAILABLE and CURL_SESSION is not None:
        html = _fetch_with_requests(url, retries=3, respect_robots=respect_robots)
    else:
        html = _fetch_with_requests(url, retries=3, respect_robots=respect_robots)
    if html:
        parsed_url = urlparse(url)
        is_sensacine_search = (
            "sensacine.com" in parsed_url.netloc and parsed_url.path.startswith("/buscar/")
        )
        has_expected_results = (
            "/peliculas/pelicula-" in html or "/series/serie-" in html
        )
        if (
            allow_playwright
            and is_sensacine_search
            and not has_expected_results
            and PLAYWRIGHT_AVAILABLE
        ):
            print("[INFO] Search HTML sin resultados detectados, fallback forzado a Playwright...")
            browser_html = _fetch_with_playwright(url)
            if browser_html:
                return browser_html

        return html

    # Fallback a Playwright si está disponible
    if allow_playwright and PLAYWRIGHT_AVAILABLE:
        print("[INFO] Fallback a Playwright...")
        html = _fetch_with_playwright(url, respect_robots=False)
        if html:
            return html

    print(f"[WARN] No se obtuvo contenido para: {url[:80]}...")
    return None