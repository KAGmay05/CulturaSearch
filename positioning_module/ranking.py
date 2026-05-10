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
import numpy as np

from index import indexer


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

    positions = list(original_positions)
    return [m * 1e2 + g - (pos * 1e-6) for m, g, pos in zip(metadata_scores, genre_scores, positions)]


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
        results: list of SearchResult objects with `score` attribute
        ascending: True for worst first, False for best first (default)
    
    Returns:
        sorted list
    """
    return sorted(results, key=lambda r: float(r.score or 0.0), reverse=not ascending)
