"""
Ranking and positioning utilities for CultureSearch.

Responsibilities:
- compute_lexical_scores: simple lexical overlap scoring (uses cleaned tokens)
- compute_positioning_score: blend neural+lexical with freshness
- stable_rank_indices: deterministic tie-breaking ordering
- sigmoid: numpy-aware sigmoid
- additional: simple UI placement bucketing helper
"""
from typing import List, Dict, Any, Sequence
import math
import time
import re
import unicodedata
import numpy as np

from index import indexer


COUNTRY_SYNONYMS = {
    "ee uu": {"eeuu", "estadounidense", "estadounidenses", "americana", "americano", "americanas", "americanos", "unidense", "unidenses"},
    "españa": {"espana", "española", "espanola", "español", "espanol", "españolas", "espanolas", "españoles", "espanoles"},
    "gran bretaña": {"gran bretaña", "gran bretana", "britanica", "britanico", "britanicas", "britanicos", "britanica", "britanico"},
    "turquía": {"turquia", "turca", "turco", "turcas", "turcos"},
    "japón": {"japon", "japonesa", "japones", "japonesa", "japoneses", "japonesas"},
    "francia": {"francesa", "frances", "franceses", "francesas"},
    "colombia": {"colombiana", "colombiano", "colombianas", "colombianos"},
    "corea del sur": {"corea del sur", "surcoreana", "surcoreano", "coreana", "coreano", "coreanas", "coreanos"},
    "méjico": {"mexico", "mejico", "mexicana", "mexicano", "mexicanas", "mexicanos"},
    "canadá": {"canada", "canadiense", "canadienses"},
    "alemania": {"alemana", "aleman", "alemanes", "alemanas"},
    "argentina": {"argentina", "argentino", "argentinas", "argentinos"},
    "israel": {"israeli", "israeli", "israelis"},
    "italia": {"italiana", "italiano", "italianas", "italianos"},
    "australia": {"australiana", "australiano", "australianas", "australianos"},
    "brasil": {"brasilena", "brasileña", "brasileno", "brasileño", "brasilenas", "brasileñas", "brasilenos", "brasileños"},
    "dinamarca": {"danesa", "danes", "danesas", "daneses"},
    "india": {"india", "indio", "india", "indias", "indios", "hindú", "hindu"},
    "bélgica": {"belgica", "belga", "belgas"},
    "suecia": {"sueca", "sueco", "suecas", "suecos"},
    "noruega": {"noruega", "noruego", "noruegas", "noruegos"},
    "finlandia": {"finlandesa", "finlandes", "finlandesas", "finlandeses"},
    "polonia": {"polaca", "polaco", "polacas", "polacos"},
    "egipto": {"egipcia", "egipcio", "egipcias", "egipcios"},
    "venezuela": {"venezolana", "venezolano", "venezolanas", "venezolanos"},
    "holanda": {"holandesa", "holandes", "holandesas", "holandeses", "neerlandesa", "neerlandes", "neerlandesas", "neerlandeses"},
    "indonesia": {"indonesia", "indonesio", "indonesia", "indonesias", "indonesios"},
    "portugal": {"portuguesa", "portugues", "portuguesas", "portugueses"},
}


def _find_country_canonicals(text: str) -> set[str]:
    """Return canonical country keys whose aliases appear in the provided text."""
    plain = _normalize_plain_text(text)
    if not plain:
        return set()

    found: set[str] = set()
    for canonical, aliases in COUNTRY_SYNONYMS.items():
        if canonical in plain or any(alias in plain for alias in aliases):
            found.add(canonical)
    return found


def _normalize_plain_text(value: str) -> str:
    """Normalize text without stemming for exact-name/entity comparisons."""
    if not value:
        return ""
    text = str(value).lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8", "ignore")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def sigmoid(x):
    """Numerically-stable sigmoid for scalars or numpy arrays."""
    try:
        return 1.0 / (1.0 + np.exp(-x))
    except Exception:
        return 1.0 / (1.0 + math.exp(-float(x)))


def compute_lexical_scores(query_tokens: List[str], docs: List[Dict[str, Any]]) -> List[float]:
    """Compute a lightweight lexical score based on token overlap and term frequency.

    Args:
        query_tokens: list of cleaned tokens from the query
        docs: list of dicts with a `clean_text` key containing space-separated tokens

    Returns:
        List[float] scores aligned with docs.
    """
    qset = set(query_tokens)
    scores = []
    for d in docs:
        text = d.get("clean_text", "")
        tokens = text.split()
        tf = 0
        for t in tokens:
            if t in qset:
                tf += 1
        scores.append(float(tf))
    return scores


def _normalize_text_tokens(value: str) -> List[str]:
    """Normalize text using the project's lexical pipeline."""
    if not value:
        return []
    return indexer.clean_text(str(value))


def _extract_numeric_values(value: str) -> set[int]:
    """Extract integer tokens from free text without lexical stemming."""
    if not value:
        return set()
    numbers: set[int] = set()
    for match in re.findall(r"\b\d{1,4}\b", str(value)):
        try:
            numbers.add(int(match))
        except Exception:
            continue
    return numbers


def _extract_requested_season_count(query: str) -> int | None:
    """Return the season count explicitly requested in the query, if any."""
    if not query:
        return None

    query_text = str(query).lower()
    if "temporada" not in query_text and "season" not in query_text:
        return None

    numbers = sorted(_extract_numeric_values(query))
    if not numbers:
        return None

    # Prefer the first explicit number mentioned near a seasons keyword.
    return int(numbers[0])


def _extract_requested_episode_count(query: str) -> int | None:
    """Return the episode/chapter count explicitly requested in the query, if any."""
    if not query:
        return None

    query_text = str(query).lower()
    if not any(keyword in query_text for keyword in ("episodio", "episodios", "capitulo", "capitulos", "chapter", "chapters")):
        return None

    numbers = sorted(_extract_numeric_values(query))
    if not numbers:
        return None

    return int(numbers[0])


def _extract_requested_year(query: str) -> int | None:
    """Return an explicit year mentioned in the query, if any."""
    if not query:
        return None

    numbers = sorted(_extract_numeric_values(query))
    if not numbers:
        return None

    query_text = str(query).lower()
    has_year_context = any(keyword in query_text for keyword in ("ano", "año", "year", "estreno", "estrenada", "estrenado", "lanzada", "lanzado"))
    for number in numbers:
        if 1900 <= number <= 2100 and (has_year_context or len(numbers) == 1):
            return int(number)
    return None


def _is_token_negated(query: str, token: str, window_chars: int = 40) -> bool:
    """Return True if `token` in `query` is preceded by a negation within a character window.

    This is a heuristic used for phrases like "no drama", "sin violencia", "excepto comedia".
    """
    if not query or not token:
        return False
    q = _normalize_plain_text(query)
    t = _normalize_plain_text(token)
    idx = q.find(t)
    if idx == -1:
        return False
    window_start = max(0, idx - window_chars)
    window = q[window_start:idx]
    negation_words = ("no", "sin", "excepto", "menos", "salvo", "except")
    for neg in negation_words:
        if re.search(r"\b" + re.escape(neg) + r"\b", window):
            return True
    # also check direct 'excepto {token}' anywhere
    for neg in ("excepto", "salvo", "except"):
        if re.search(r"\b" + re.escape(neg) + r"\s+" + re.escape(t) + r"\b", q):
            return True
    return False


def compute_genre_negation_scores(query: str, docs: Sequence[Dict[str, Any]]) -> List[float]:
    """Penalize documents whose genres are explicitly negated in the query.

    Returns a list of floats (negative for penalized docs, small positive otherwise).
    """
    if not query or not docs:
        return [0.0] * (len(docs) if docs is not None else 0)

    query_plain = _normalize_plain_text(query)
    # collect all genre tokens present in corpus
    genre_tokens: set[str] = set()
    for doc in docs:
        genres = doc.get("genres", [])
        if not isinstance(genres, list):
            genres = [genres]
        for g in genres:
            genre_tokens.update(_normalize_text_tokens(str(g)))

    negated_genres = {g for g in genre_tokens if _is_token_negated(query, g)}
    if not negated_genres:
        return [0.0 for _ in docs]

    scores: List[float] = []
    for doc in docs:
        doc_genre_tokens = set()
        genres = doc.get("genres", [])
        if not isinstance(genres, list):
            genres = [genres]
        for g in genres:
            doc_genre_tokens.update(_normalize_text_tokens(str(g)))

        if doc_genre_tokens & negated_genres:
            scores.append(-1.5)
        else:
            # small positive reward for not matching excluded genres
            scores.append(0.12)
    return scores


def compute_keywords_negation_scores(query: str, docs: Sequence[Dict[str, Any]]) -> List[float]:
    """Penalize documents that contain keywords explicitly excluded by the user ("sin X", "no X")."""
    if not query or not docs:
        return [0.0] * (len(docs) if docs is not None else 0)

    q_tokens = _normalize_text_tokens(query)
    negation_words = {"no", "sin", "excepto", "salvo", "menos", "except"}
    negated_keywords: set[str] = set()
    # simple heuristic: any token immediately following a negation word is treated as excluded
    toks = q_tokens
    for i, tok in enumerate(toks[:-1]):
        if tok in negation_words:
            negated_keywords.add(toks[i + 1])

    if not negated_keywords:
        return [0.0 for _ in docs]

    scores: List[float] = []
    for doc in docs:
        text = " ".join(_normalize_text_tokens(str(doc.get("plot", "")) + " " + str(doc.get("title", "")) + " " + str(doc.get("genres", ""))))
        text_tokens = set(text.split())
        if text_tokens & negated_keywords:
            scores.append(-1.0)
        else:
            scores.append(0.0)
    return scores


def compute_rating_negation_scores(query: str, docs: Sequence[Dict[str, Any]]) -> List[float]:
    """Handle simple rating negation patterns like 'sin puntuacion' or 'no puntuadas'.

    Returns negative values for docs to be deprioritized when user excludes rated/unrated content.
    """
    if not query or not docs:
        return [0.0] * (len(docs) if docs is not None else 0)

    q = _normalize_plain_text(query)
    patterns_no_rating = ("sin puntuacion", "sin calificacion", "no puntuad", "no calific", "sin rating", "sin valoracion")
    wants_unrated_exclusion = any(p in q for p in patterns_no_rating)
    if not wants_unrated_exclusion:
        return [0.0 for _ in docs]

    scores: List[float] = []
    for doc in docs:
        rating = doc.get("rating")
        if rating in (None, "", "null"):
            # keep unrated documents (user asked 'sin puntuacion' -> preferring unrated)
            scores.append(0.25)
        else:
            scores.append(-0.6)
    return scores


def compute_season_negation_scores(query: str, docs: Sequence[Dict[str, Any]]) -> List[float]:
    """Penalize documents matching an explicitly negated season count (e.g. 'sin 3 temporadas')."""
    if not query or not docs:
        return [0.0] * (len(docs) if docs is not None else 0)

    requested = _extract_requested_season_count(query)
    if requested is None:
        return [0.0 for _ in docs]

    if not _is_token_negated(query, str(requested)):
        return [0.0 for _ in docs]

    scores: List[float] = []
    for doc in docs:
        seasons_value = doc.get("seasons")
        if seasons_value in (None, "", "null"):
            scores.append(0.0)
            continue
        try:
            seasons_int = int(str(seasons_value).strip())
        except Exception:
            scores.append(0.0)
            continue
        if seasons_int == requested:
            scores.append(-1.2)
        else:
            scores.append(0.0)
    return scores


def _normalize_country_value(value: str) -> str:
    """Normalize a country string for comparison."""
    return _normalize_plain_text(value)


def _country_alias_matches(query_plain: str, country_plain: str) -> bool:
    if not query_plain or not country_plain:
        return False
    if country_plain in query_plain:
        return True
    aliases = COUNTRY_SYNONYMS.get(country_plain, set())
    return any(alias in query_plain for alias in aliases)


def _normalized_text(value: str) -> str:
    return " ".join(_normalize_text_tokens(value))


def extract_query_genre_tokens(query: str, docs: Sequence[Dict[str, Any]]) -> set[str]:
    """Extract query tokens that overlap with known genre labels in the corpus."""
    query_tokens = set(_normalize_text_tokens(query))
    if not query_tokens:
        return set()

    genre_tokens: set[str] = set()
    for doc in docs:
        genres = doc.get("genres", [])
        if not isinstance(genres, list):
            genres = [genres]
        for genre in genres:
            genre_tokens.update(_normalize_text_tokens(str(genre)))
    return query_tokens & genre_tokens


def genre_match_score(query_genre_tokens: set[str], doc: Dict[str, Any]) -> float:
    """Score how well document genres match the tokens present in the query."""
    if not query_genre_tokens:
        return 0.0

    genres = doc.get("genres", [])
    if not isinstance(genres, list):
        genres = [genres]

    doc_genre_tokens: set[str] = set()
    for genre in genres:
        doc_genre_tokens.update(_normalize_text_tokens(str(genre)))

    if not doc_genre_tokens:
        return 0.0
    return float(len(query_genre_tokens & doc_genre_tokens)) / float(len(query_genre_tokens))


def metadata_completeness_score(doc: Dict[str, Any]) -> float:
    fields = ["title", "genres", "type", "publish_date", "url"]
    present = sum(1 for f in fields if doc.get(f))
    return present / len(fields)


def metadata_match_score(query: str, doc: Dict[str, Any]) -> float:
    """Score how strongly a document matches explicit metadata cues in the query."""
    if not query or not doc:
        return 0.0

    query_text = str(query).lower()
    query_tokens = set(_normalize_text_tokens(query))
    query_numbers = _extract_numeric_values(query)

    score = 0.0

    title = str(doc.get("title", ""))
    if title:
        title_text = title.lower()
        title_tokens = set(_normalize_text_tokens(title))
        if title_text and title_text in query_text:
            score += 0.65
        elif title_tokens:
            overlap = len(query_tokens & title_tokens)
            if overlap:
                score += min(0.25, 0.08 * overlap)

    directors = doc.get("director", [])
    if not isinstance(directors, list):
        directors = [directors]
    for director in directors:
        director_text = str(director).strip().lower()
        if not director_text:
            continue
        director_tokens = set(_normalize_text_tokens(director_text))
        if director_text in query_text:
            score += 1.0
        elif director_tokens and director_tokens.issubset(query_tokens):
            score += 0.7
        elif director_tokens:
            overlap = len(query_tokens & director_tokens)
            if overlap >= 2:
                score += 0.5
            elif overlap == 1:
                score += 0.25

    creators = doc.get("creator", [])
    if not isinstance(creators, list):
        creators = [creators]
    for creator in creators:
        creator_text = str(creator).strip().lower()
        if not creator_text:
            continue
        creator_tokens = set(_normalize_text_tokens(creator_text))
        if creator_text in query_text:
            score += 0.95
        elif creator_tokens and creator_tokens.issubset(query_tokens):
            score += 0.65
        elif creator_tokens:
            overlap = len(query_tokens & creator_tokens)
            if overlap >= 2:
                score += 0.45
            elif overlap == 1:
                score += 0.2

    actor_query_hint = any(keyword in query_text for keyword in ("protagon", "reparto", "actor", "actriz", "interpreta", "protagonizada", "protagonizado", "starring", "cast"))
    actors = doc.get("actors", [])
    if not isinstance(actors, list):
        actors = [actors]
    actor_hits = 0
    for actor in actors:
        actor_text = str(actor).strip().lower()
        if not actor_text:
            continue
        actor_plain = _normalize_plain_text(actor_text)
        actor_tokens = set(_normalize_text_tokens(actor_text))
        if actor_plain and actor_plain in _normalize_plain_text(query):
            actor_hits += 3
        elif actor_tokens and actor_tokens.issubset(query_tokens):
            actor_hits += 2
        elif actor_query_hint and actor_tokens & query_tokens:
            actor_hits += 1
    if actor_hits:
        score += min(1.2, 0.35 * actor_hits)

    year_value = doc.get("year")
    if year_value not in (None, "", "null"):
        try:
            year_int = int(str(year_value).strip()[:4])
            if year_int in query_numbers:
                score += 0.4
        except Exception:
            pass

    seasons_value = doc.get("seasons")
    if seasons_value not in (None, "", "null"):
        try:
            seasons_int = int(str(seasons_value).strip())
            if seasons_int in query_numbers:
                score += 1.35
            elif query_numbers and ("temporada" in query_text or "temporadas" in query_text or "season" in query_text or "seasons" in query_text):
                # If the query explicitly asks for seasons, matching the count matters a lot.
                # Non-matching season counts should not outrank exact matches.
                score -= 0.18
        except Exception:
            pass

    media_type = str(doc.get("type", "")).lower()
    if media_type:
        type_tokens = set(_normalize_text_tokens(media_type))
        if type_tokens & query_tokens:
            score += 0.22
        elif media_type in ("serie", "series") and ({"seri", "seri"} & query_tokens or "serie" in query_text or "series" in query_text):
            score += 0.22
        elif media_type in ("pelicula", "movie", "film") and ("pelicul" in query_tokens or "movie" in query_text or "film" in query_text):
            score += 0.22

    return float(min(score, 1.0))


def actor_alignment_score(query: str, doc: Dict[str, Any]) -> float:
    """Strongly reward exact actor matches, especially when the query asks who stars in the title."""
    if not query or not doc:
        return 0.0

    query_plain = _normalize_plain_text(query)
    query_tokens = set(_normalize_text_tokens(query))
    actor_query_hint = any(keyword in query_plain for keyword in ("protagon", "reparto", "actor", "actriz", "interpreta", "starring", "cast"))

    actors = doc.get("actors", [])
    if not isinstance(actors, list):
        actors = [actors]

    score = 0.0
    for actor in actors:
        actor_text = str(actor).strip()
        if not actor_text:
            continue
        actor_plain = _normalize_plain_text(actor_text)
        actor_tokens = set(_normalize_text_tokens(actor_text))

        if actor_plain and actor_plain in query_plain:
            score += 3.0
        elif actor_tokens and actor_tokens.issubset(query_tokens):
            score += 2.0
        elif actor_query_hint and actor_tokens & query_tokens:
            score += 0.8

    if score == 0.0 and actor_query_hint:
        return -0.25
    return float(min(score, 3.5))


def season_alignment_score(query: str, doc: Dict[str, Any]) -> float:
    """Return a strong score for exact season matches and a penalty for mismatches."""
    requested = _extract_requested_season_count(query)
    if requested is None:
        return 0.0

    seasons_value = doc.get("seasons")
    if seasons_value in (None, "", "null"):
        return -0.15

    try:
        seasons_int = int(str(seasons_value).strip())
    except Exception:
        return -0.15

    if seasons_int == requested:
        return 2.5

    # Penalize by distance so 2 or 9 seasons cannot outrank exact matches.
    distance = abs(seasons_int - requested)
    return -min(1.5, 0.22 * distance)


def episode_alignment_score(query: str, doc: Dict[str, Any]) -> float:
    """Return a strong score for exact episode/chapter matches and a penalty for mismatches."""
    requested = _extract_requested_episode_count(query)
    if requested is None:
        return 0.0

    episodes_value = doc.get("episodes")
    if episodes_value in (None, "", "null"):
        return -0.1

    try:
        episodes_int = int(str(episodes_value).strip())
    except Exception:
        return -0.1

    if episodes_int == requested:
        return 1.8

    distance = abs(episodes_int - requested)
    return -min(1.2, 0.08 * distance)


def year_alignment_score(query: str, doc: Dict[str, Any]) -> float:
    """Return a strong score for exact year matches and a penalty for mismatches."""
    requested = _extract_requested_year(query)
    if requested is None:
        return 0.0

    candidates: List[int] = []
    year_value = doc.get("year")
    if year_value not in (None, "", "null"):
        try:
            candidates.append(int(str(year_value).strip()[:4]))
        except Exception:
            pass

    year_range = doc.get("year_range")
    if year_range:
        try:
            start_year = int(str(year_range).split("-")[0].strip()[:4])
            candidates.append(start_year)
        except Exception:
            pass

    if not candidates:
        return -0.1

    if requested in candidates:
        return 1.5

    closest = min(abs(requested - cand) for cand in candidates)
    return -min(1.0, 0.05 * closest)


def country_alignment_score(query: str, doc: Dict[str, Any]) -> float:
    """Return a score for explicit country queries. More conservative scoring."""
    if not query:
        return 0.0

    query_plain = _normalize_plain_text(query)
    query_canonicals = _find_country_canonicals(query_plain)
    if not query_canonicals:
        return 0.0

    doc_canonicals = _find_country_canonicals(str(doc.get("country", "")))
    if not doc_canonicals:
        return -0.15  # Reduced penalty: was -0.35

    if query_canonicals & doc_canonicals:
        return 1.8  # Reduced boost: was 3.25 (too aggressive)

    return -0.4  # Reduced penalty: was -0.85


def compute_metadata_alignment_scores(query: str, docs: Sequence[Dict[str, Any]]) -> List[float]:
    """Compute metadata alignment scores for a set of docs."""
    return [metadata_match_score(query, doc) for doc in docs]


def compute_actor_alignment_scores(query: str, docs: Sequence[Dict[str, Any]]) -> List[float]:
    """Compute actor alignment scores for a set of docs."""
    return [actor_alignment_score(query, doc) for doc in docs]


def compute_season_alignment_scores(query: str, docs: Sequence[Dict[str, Any]]) -> List[float]:
    """Compute season alignment scores for a set of docs."""
    return [season_alignment_score(query, doc) for doc in docs]


def compute_episode_alignment_scores(query: str, docs: Sequence[Dict[str, Any]]) -> List[float]:
    """Compute episode/chapter alignment scores for a set of docs."""
    return [episode_alignment_score(query, doc) for doc in docs]


def compute_year_alignment_scores(query: str, docs: Sequence[Dict[str, Any]]) -> List[float]:
    """Compute year alignment scores for a set of docs."""
    return [year_alignment_score(query, doc) for doc in docs]


def compute_country_alignment_scores(query: str, docs: Sequence[Dict[str, Any]]) -> List[float]:
    """Compute country alignment scores for a set of docs."""
    return [country_alignment_score(query, doc) for doc in docs]


def _freshness_score(metadata: Dict[str, Any], now_ts: float = None) -> float:
    now_ts = now_ts or time.time()
    # Prefer `year` because the dataset does not include publish_date.
    year_value = metadata.get("year") or metadata.get("publish_date")
    if not year_value:
        return 0.0
    try:
        if isinstance(year_value, (int, float)):
            year = int(year_value)
        else:
            text = str(year_value).strip()
            if len(text) >= 4 and text[:4].isdigit():
                year = int(text[:4])
            else:
                year = int(text)

        # Convert year into a freshness score against the current year.
        current_year = time.gmtime(now_ts).tm_year
        age = max(0, current_year - year)
        return math.exp(-age / 10.0)
    except Exception:
        return 0.0


def compute_positioning_score(
    neural_scores: Sequence[float],
    lexical_scores: Sequence[float],
    docs: List[Dict[str, Any]],
    weight_neural: float = 0.6,
    weight_lexical: float = 0.3,
    weight_fresh: float = 0.1,
) -> List[float]:
    """Blend signals into a final positioning score.

    Normalizes neural and lexical arrays to [0,1] then combines with freshness.
    """
    def _norm(arr):
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            return np.zeros(0)
        mn = float(np.min(arr))
        mx = float(np.max(arr))
        if abs(mx - mn) < 1e-9:
            return np.full_like(arr, 0.5)
        return (arr - mn) / (mx - mn)

    nn = _norm(neural_scores)
    lx = _norm(lexical_scores)

    final = []
    for i, d in enumerate(docs):
        meta = d.get("metadata", d)
        fresh = _freshness_score(meta)
        s = weight_neural * float(nn[i] if i < len(nn) else 0.0) + weight_lexical * float(lx[i] if i < len(lx) else 0.0) + weight_fresh * fresh
        final.append(float(s))
    return final


def _normalize_scores(values: Sequence[float], empty_value: float = 0.0) -> np.ndarray:
    """Normalize a score array to [0, 1] in a stable way."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.zeros(0, dtype=float)
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    if abs(mx - mn) < 1e-9:
        return np.full_like(arr, empty_value, dtype=float)
    return (arr - mn) / (mx - mn)


def compute_hybrid_fusion(
    neural_scores: Sequence[float],
    lexical_scores: Sequence[float],
    alpha: float = 0.7,
) -> Dict[str, Any]:
    """Compute the blended hybrid scores used before reranking.

    Returns a dict containing the normalized scores, automatic weight adjustment,
    and the final fused scores.
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))
    neural_scores_arr = np.asarray(neural_scores, dtype=float)
    lexical_scores_arr = np.asarray(lexical_scores, dtype=float)

    neural_norm = _normalize_scores((neural_scores_arr + 1.0) / 2.0, empty_value=0.5)
    max_lex = float(np.max(lexical_scores_arr)) if lexical_scores_arr.size else 0.0
    if max_lex > 0:
        lexical_norm = lexical_scores_arr / max_lex
    else:
        lexical_norm = np.zeros_like(neural_norm)

    lex_coverage = float((lexical_scores_arr > 0).sum()) / float(len(lexical_scores_arr)) if len(lexical_scores_arr) else 0.0
    lexical_weight = (1.0 - alpha) * max(0.10, 1.0 - lex_coverage)
    neural_weight = 1.0 - lexical_weight
    fused_scores = neural_weight * neural_norm + lexical_weight * lexical_norm

    return {
        "neural_scores": neural_scores_arr,
        "lexical_scores": lexical_scores_arr,
        "neural_norm": neural_norm,
        "lexical_norm": lexical_norm,
        "lexical_weight": lexical_weight,
        "neural_weight": neural_weight,
        "fused_scores": fused_scores,
        "lex_coverage": lex_coverage,
        "alpha": alpha,
    }


def compute_rerank_fusion(
    base_scores: Sequence[float],
    rerank_raw: Sequence[float],
    rerank_weight: float = 0.75,
) -> Dict[str, Any]:
    """Blend the base retrieval score with the cross-encoder rerank score."""
    rerank_weight = float(np.clip(rerank_weight, 0.0, 1.0))
    base_scores_arr = np.asarray(base_scores, dtype=float)
    max_base = float(np.max(base_scores_arr)) if base_scores_arr.size else 0.0
    if max_base > 0:
        base_norm = base_scores_arr / max_base
    else:
        base_norm = np.zeros_like(base_scores_arr)

    rerank_raw_arr = np.asarray(rerank_raw, dtype=float)
    rerank_norm = sigmoid(rerank_raw_arr)
    final_scores = (1.0 - rerank_weight) * base_norm + rerank_weight * rerank_norm

    return {
        "base_scores": base_scores_arr,
        "base_norm": base_norm,
        "rerank_raw": rerank_raw_arr,
        "rerank_norm": rerank_norm,
        "rerank_weight": rerank_weight,
        "final_scores": final_scores,
    }


def stable_rank_indices(scores: Sequence[float], tie_breaker: Sequence[float] = None, top_k: int = 10) -> List[int]:
    """Return indices sorted by score desc with deterministic tie_breaker.

    tie_breaker: higher is better. If None, uses index ordering.
    """
    n = len(scores)
    if tie_breaker is None:
        tie = list(range(n))
    else:
        tie = list(tie_breaker)
    idx = list(range(n))
    idx.sort(key=lambda i: (scores[i], tie[i]), reverse=True)
    return idx[:min(top_k, n)]


def build_positioning_tie_breaker(
    query: str,
    docs: Sequence[Dict[str, Any]],
    original_positions: Sequence[int],
) -> List[float]:
    """Build a deterministic tie-breaker using genre match, metadata completeness and original position."""
    query_genre_tokens = extract_query_genre_tokens(query, docs)
    genre_scores = [genre_match_score(query_genre_tokens, doc) for doc in docs]
    metadata_scores = [metadata_completeness_score(doc) for doc in docs]
    alignment_scores = [metadata_match_score(query, doc) for doc in docs]
    actor_scores = [actor_alignment_score(query, doc) for doc in docs]
    season_scores = [season_alignment_score(query, doc) for doc in docs]
    episode_scores = [episode_alignment_score(query, doc) for doc in docs]
    year_scores = [year_alignment_score(query, doc) for doc in docs]
    country_scores = [country_alignment_score(query, doc) for doc in docs]

    positions = list(original_positions)
    return [
        m * 1e2
        + g
        + (a * 14.0)
        + (ac * 16.0)
        + (s * 18.0)
        + (e * 14.0)
        + (y * 12.0)
        + (c * 10.0)
        - (pos * 1e-6)
        for m, g, a, ac, s, e, y, c, pos in zip(metadata_scores, genre_scores, alignment_scores, actor_scores, season_scores, episode_scores, year_scores, country_scores, positions)
    ]


def rank_by_positioning(
    primary_scores: Sequence[float],
    query: str,
    docs: Sequence[Dict[str, Any]],
    original_positions: Sequence[int],
    top_k: int | None = None,
) -> List[int]:
    """Rank items using the project's final positioning strategy."""
    tie_breaker = build_positioning_tie_breaker(query, docs, original_positions)
    return stable_rank_indices(primary_scores, tie_breaker=tie_breaker, top_k=top_k or len(primary_scores))


def bucket_for_ui(docs: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    """Simple bucketing by `type` for UI placement. Returns mapping type->list of doc indices."""
    buckets = {}
    for i, d in enumerate(docs):
        t = str(d.get("type", "desconocido"))
        buckets.setdefault(t, []).append(i)
    return buckets


def sort_by_name(results: List[Any], ascending: bool = True) -> List[Any]:
    """Sort results by title/name.
    
    Args:
        results: list of SearchResult objects
        ascending: True for A-Z, False for Z-A
    
    Returns:
        sorted list
    """
    return sorted(results, key=lambda r: (r.title or "").lower(), reverse=not ascending)


def sort_by_year(results: List[Any], ascending: bool = False, doc_lookup=None) -> List[Any]:
    """Sort results by year (freshness).

    Null/missing years are always placed at the end.

    Args:
        results: list of SearchResult or document dicts
        ascending: True for oldest first, False for newest first (default)
        doc_lookup: optional callable(item) -> document dict used to fetch `year`

    Returns:
        sorted list
    """
    def get_year(item):
        doc = doc_lookup(item) if callable(doc_lookup) else None
        year = None
        if isinstance(item, dict):
            year = item.get('year')
        elif doc is not None:
            year = doc.get('year')
        elif hasattr(item, '__dict__'):
            year = getattr(item, 'year', None)

        if year in (None, '', 'null'):
            return (1, 0)

        try:
            if isinstance(year, (int, float)):
                return (0, int(year))
            text = str(year).strip()
            value = int(text[:4]) if len(text) >= 4 and text[:4].isdigit() else int(text)
            return (0, value)
        except Exception:
            return (1, 0)

    valid_items = []
    null_items = []
    for item in results:
        is_null, value = get_year(item)
        if is_null:
            null_items.append(item)
        else:
            valid_items.append((value, item))

    valid_items.sort(key=lambda x: x[0], reverse=not ascending)
    return [item for _, item in valid_items] + null_items


def sort_by_rating(results: List[Any], ascending: bool = False, doc_lookup=None) -> List[Any]:
    """Sort results by rating.

    Null/missing ratings are always placed at the end.

    Args:
        results: list of SearchResult or document dicts
        ascending: True for worst first, False for best first (default)
        doc_lookup: optional callable(item) -> document dict used to fetch `rating`

    Returns:
        sorted list
    """
    def get_rating(item):
        doc = doc_lookup(item) if callable(doc_lookup) else None
        rating = None
        if isinstance(item, dict):
            rating = item.get('rating')
        elif doc is not None:
            rating = doc.get('rating')
        elif hasattr(item, '__dict__'):
            rating = getattr(item, 'rating', None)

        if rating in (None, '', 'null'):
            return (1, 0.0)

        try:
            if isinstance(rating, (int, float)):
                return (0, float(rating))
            text = str(rating).strip().replace(',', '.')
            return (0, float(text))
        except Exception:
            return (1, 0.0)

    valid_items = []
    null_items = []
    for item in results:
        is_null, value = get_rating(item)
        if is_null:
            null_items.append(item)
        else:
            valid_items.append((value, item))

    valid_items.sort(key=lambda x: x[0], reverse=not ascending)
    return [item for _, item in valid_items] + null_items


def sort_by_relevance(results: List[Any], ascending: bool = False) -> List[Any]:
    """Sort results by relevance score (default ordering).
    
    Args:
        results: list of SearchResult objects with `final_score` or `score` attribute
        ascending: True for worst first, False for best first (default)
    
    Returns:
        sorted list
    """
    def _relevance_value(item: Any) -> float:
        if isinstance(item, dict):
            value = item.get("final_score", item.get("score", 0.0))
        else:
            value = getattr(item, "final_score", getattr(item, "score", 0.0))
        try:
            return float(value or 0.0)
        except Exception:
            return 0.0

    return sorted(results, key=_relevance_value, reverse=not ascending)
