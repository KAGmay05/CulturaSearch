from __future__ import annotations

import html

import streamlit as st

from neural_based_model.neural_retriever import NeuralRetriever, SearchResult


st.set_page_config(
    page_title="CulturaSearch",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

EXAMPLE_QUERIES = [
    "películas de ciencia ficción con viajes en el tiempo",
    "series policiales con detectives inteligentes",
    "comedias románticas modernas",
    "dramas históricos con mucho conflicto",
]


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

        * { box-sizing: border-box; }
        html, body, .stApp { font-family: 'Inter', system-ui, sans-serif; }

        /* ── Fondo ─────────────────────────────────────────────── */
        .stApp { background: #0d1117; min-height: 100vh; }

        .block-container {
            padding: 2rem 2rem 4rem !important;
            max-width: 1280px !important;
        }

        /* ── Texto blanco en toda la app ──────────────────────── */
        .stApp p, .stApp div, .stApp span, .stApp label,
        .stApp .stMarkdown p { color: #e6edf3 !important; }
        .stApp h1, .stApp h2, .stApp h3, .stApp h4 { color: #ffffff !important; }

        /* ── Sidebar ──────────────────────────────────────────── */
        [data-testid="stSidebar"] {
            background: #161b22 !important;
            border-right: 1px solid #30363d !important;
        }
        [data-testid="stSidebar"] * { color: #e6edf3 !important; }
        [data-testid="stSidebarContent"] { padding: 1.5rem 1rem !important; }

        /* ── Botón primary (Buscar) ───────────────────────────── */
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 700 !important;
            font-size: 1rem !important;
            padding: 0.75rem 1.5rem !important;
            box-shadow: 0 4px 20px rgba(31,111,235,0.35) !important;
            transition: all 0.2s ease !important;
        }
        .stButton > button[kind="primary"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 30px rgba(31,111,235,0.55) !important;
        }

        /* ── Botones secundarios ─────────────────────────────── */
        .stButton > button:not([kind="primary"]) {
            background: #21262d !important;
            color: #e6edf3 !important;
            border: 1px solid #30363d !important;
            border-radius: 8px !important;
            font-weight: 500 !important;
            transition: all 0.2s ease !important;
        }
        .stButton > button:not([kind="primary"]):hover {
            background: #30363d !important;
            border-color: #8b949e !important;
        }

        /* ── Text area ───────────────────────────────────────── */
        .stTextArea textarea {
            background: #161b22 !important;
            color: #e6edf3 !important;
            border: 2px solid #30363d !important;
            border-radius: 12px !important;
            font-size: 1rem !important;
            padding: 1rem !important;
            transition: border-color 0.2s ease !important;
        }
        .stTextArea textarea:focus {
            border-color: #1f6feb !important;
            box-shadow: 0 0 0 3px rgba(31,111,235,0.15) !important;
        }
        .stTextArea textarea::placeholder { color: #8b949e !important; }
        .stTextArea label, .stSlider label, .stToggle label {
            color: #e6edf3 !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
        }
        .stToggle p { color: #e6edf3 !important; }

        /* ── Hero ────────────────────────────────────────────── */
        .cs-hero {
            background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
            border: 1px solid #30363d;
            border-radius: 20px;
            padding: 2.5rem 2.5rem 2rem;
            margin-bottom: 2rem;
            position: relative;
            overflow: hidden;
        }
        .cs-hero::before {
            content: "";
            position: absolute;
            top: -60px; right: -60px;
            width: 250px; height: 250px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(31,111,235,0.15), transparent 70%);
            pointer-events: none;
        }
        .cs-hero-badge {
            display: inline-block;
            background: rgba(31,111,235,0.15);
            border: 1px solid rgba(56,139,253,0.4);
            color: #58a6ff !important;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            padding: 0.3rem 0.8rem;
            border-radius: 50px;
            margin-bottom: 1rem;
        }
        .cs-hero h1 {
            margin: 0 0 0.75rem 0 !important;
            font-size: clamp(1.8rem, 3.5vw, 2.75rem) !important;
            font-weight: 800 !important;
            line-height: 1.2 !important;
            color: #ffffff !important;
        }
        .cs-hero h1 em {
            font-style: normal;
            background: linear-gradient(90deg, #58a6ff, #bc8cff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .cs-hero p {
            color: #8b949e !important;
            font-size: 1.05rem !important;
            line-height: 1.7 !important;
            max-width: 70ch !important;
            margin: 0 !important;
        }

        /* ── Métricas ────────────────────────────────────────── */
        .cs-metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 0.75rem;
            margin: 1.5rem 0;
        }
        .cs-metric {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 14px;
            padding: 1rem 1.25rem;
        }
        .cs-metric-label {
            font-size: 0.76rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #8b949e !important;
            margin-bottom: 0.4rem;
        }
        .cs-metric-value {
            font-size: 1.4rem !important;
            font-weight: 800 !important;
            color: #ffffff !important;
            line-height: 1;
        }

        /* ── Tarjeta de resultado ────────────────────────────── */
        .cs-card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 16px;
            padding: 1.5rem;
            margin-top: 1.25rem;
            transition: border-color 0.25s, box-shadow 0.25s;
        }
        .cs-card:hover {
            border-color: #388bfd;
            box-shadow: 0 8px 32px rgba(31,111,235,0.18);
        }
        .cs-card-top {
            display: flex;
            align-items: flex-start;
            gap: 1rem;
        }
        .cs-rank {
            flex-shrink: 0;
            width: 48px; height: 48px;
            border-radius: 12px;
            background: linear-gradient(135deg, #1f6feb, #bc8cff);
            display: flex; align-items: center; justify-content: center;
            font-size: 1.1rem; font-weight: 900; color: #fff;
            box-shadow: 0 4px 14px rgba(31,111,235,0.4);
        }
        .cs-card-title {
            font-size: 1.2rem !important;
            font-weight: 700 !important;
            color: #ffffff !important;
            margin: 0 0 0.4rem 0 !important;
            line-height: 1.3 !important;
        }
        .cs-card-badges {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.5rem;
            margin-top: 0.25rem;
        }
        .cs-badge {
            display: inline-block;
            font-size: 0.78rem;
            font-weight: 600;
            padding: 0.3rem 0.7rem;
            border-radius: 50px;
            border: 1px solid;
        }
        .cs-badge.movie {
            background: rgba(245,158,11,0.12);
            border-color: rgba(245,158,11,0.35);
            color: #f59e0b !important;
        }
        .cs-badge.series {
            background: rgba(139,92,246,0.12);
            border-color: rgba(139,92,246,0.35);
            color: #a78bfa !important;
        }
        .cs-badge.other {
            background: rgba(148,163,184,0.1);
            border-color: rgba(148,163,184,0.25);
            color: #94a3b8 !important;
        }
        .cs-relevance { font-size: 0.82rem !important; font-weight: 600 !important; }
        .cs-score {
            margin-left: auto;
            flex-shrink: 0;
            font-size: 1.6rem !important;
            font-weight: 900 !important;
            line-height: 1;
        }
        .cs-bar {
            margin: 1rem 0;
            height: 6px;
            background: #21262d;
            border-radius: 50px;
            overflow: hidden;
        }
        .cs-bar-fill { height: 100%; border-radius: 50px; }
        .cs-synopsis {
            color: #8b949e !important;
            font-size: 0.97rem !important;
            line-height: 1.75 !important;
            margin: 0 0 1rem 0 !important;
        }
        .cs-link {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            color: #58a6ff !important;
            font-size: 0.9rem;
            font-weight: 600;
            text-decoration: none;
            padding: 0.45rem 1rem;
            background: rgba(31,111,235,0.1);
            border: 1px solid rgba(56,139,253,0.25);
            border-radius: 8px;
            transition: all 0.2s ease;
        }
        .cs-link:hover {
            background: rgba(31,111,235,0.2);
            border-color: #388bfd;
            color: #79c0ff !important;
        }

        /* ── Estado vacío ────────────────────────────────────── */
        .cs-empty {
            text-align: center;
            padding: 3rem 2rem;
            border: 2px dashed #30363d;
            border-radius: 20px;
            background: #161b22;
            margin-top: 2rem;
        }
        .cs-empty-title {
            font-size: 1.4rem !important;
            font-weight: 700 !important;
            color: #ffffff !important;
            margin-bottom: 0.5rem !important;
        }
        .cs-empty-sub {
            color: #8b949e !important;
            font-size: 1rem !important;
        }

        /* ── Sidebar labels ──────────────────────────────────── */
        .cs-sidebar-section {
            font-size: 0.76rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #8b949e !important;
            margin: 1.25rem 0 0.6rem 0 !important;
        }

        /* ── Responsive ──────────────────────────────────────── */
        @media (max-width: 768px) {
            .cs-card-top { flex-wrap: wrap; }
            .cs-score { margin-left: 0; }
            .cs-hero { padding: 1.5rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_retriever() -> NeuralRetriever:
    retriever = NeuralRetriever()
    retriever.ensure_ready(force_rebuild=False)
    return retriever


def use_example(value: str) -> None:
    st.session_state["search_query"] = value


def clear_search() -> None:
    st.session_state["search_query"] = ""
    st.session_state["last_results"] = []
    st.session_state["last_query"] = ""


def score_label(score: float) -> tuple[str, str]:
    if score >= 0.80:
        return "Excelente", "#22c55e"
    if score >= 0.65:
        return "Alta", "#3b82f6"
    if score >= 0.45:
        return "Media", "#f59e0b"
    return "Baja", "#ef4444"


def content_kind(media_type: str) -> str:
    t = (media_type or "").lower()
    if "serie" in t:
        return "series"
    if "pel" in t or "film" in t:
        return "movie"
    return "other"


def snippet_text(text: str, limit: int = 380) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return "Sin sinopsis disponible."
    return cleaned if len(cleaned) <= limit else cleaned[:limit].rstrip() + "\u2026"


def render_metric(label: str, value: str) -> str:
    return (
        '<div class="cs-metric">'
        f'<div class="cs-metric-label">{html.escape(label)}</div>'
        f'<div class="cs-metric-value">{html.escape(value)}</div>'
        "</div>"
    )


def render_result_card(item: SearchResult, rank: int) -> None:
    title = html.escape(item.title or "Sin título")
    kind = content_kind(item.media_type)
    summary = html.escape(snippet_text(item.plot))
    score_pct = max(0.0, min(100.0, float(item.score) * 100.0))
    label, color = score_label(item.score)
    url = (item.url or "").strip()

    type_icon = "🎬" if kind == "movie" else "📺" if kind == "series" else "🎭"
    type_label = (
        "Película" if kind == "movie"
        else "Serie" if kind == "series"
        else (item.media_type or "").capitalize()
    )

    source_html = (
        f'<a class="cs-link" href="{html.escape(url)}" target="_blank" rel="noopener noreferrer">'
        "🔗 Ver fuente original</a>"
        if url else ""
    )

    st.markdown(
        f"""
        <div class="cs-card">
            <div class="cs-card-top">
                <div class="cs-rank">{rank}</div>
                <div style="flex:1;min-width:0;">
                    <div class="cs-card-title">{title}</div>
                    <div class="cs-card-badges">
                        <span class="cs-badge {kind}">{type_icon} {type_label}</span>
                        <span class="cs-relevance" style="color:{color}">● {label}</span>
                    </div>
                </div>
                <div class="cs-score" style="color:{color}">{score_pct:.0f}%</div>
            </div>
            <div class="cs-bar">
                <div class="cs-bar-fill" style="width:{score_pct:.1f}%;background:linear-gradient(90deg,{color},{color}88);"></div>
            </div>
            <p class="cs-synopsis">{summary}</p>
            {source_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def run_query(
    query: str,
    use_web_expansion: bool,
    top_k: int,
    candidate_k: int,
    alpha: float,
    rerank_weight: float,
) -> list[SearchResult]:
    retriever = get_retriever()
    if use_web_expansion:
        return retriever.search_with_web_expansion(
            query=query,
            top_k=top_k,
            candidate_k=candidate_k,
            alpha=alpha,
            rerank_weight=rerank_weight,
        )
    return retriever.search_advanced(
        query=query,
        top_k=top_k,
        candidate_k=candidate_k,
        alpha=alpha,
        rerank_weight=rerank_weight,
    )


def main() -> None:
    inject_styles()

    # Inicializar estado de sesión
    for key, default in [
        ("search_query", ""),
        ("last_results", []),
        ("last_query", ""),
        ("last_web_mode", False),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ── Sidebar ────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown('<div class="cs-sidebar-section">⚙️ Parámetros</div>', unsafe_allow_html=True)

        use_web_expansion = st.toggle(
            "Expansión web",
            value=True,
            help="Amplía la búsqueda con resultados de internet",
        )
        top_k = st.slider("Resultados", min_value=3, max_value=10, value=5, step=1)
        candidate_k = st.slider("Candidatos internos", min_value=10, max_value=100, value=50, step=5)
        alpha = st.slider(
            "Peso semántico",
            min_value=0.0, max_value=1.0, value=0.9, step=0.05,
            help="1.0 = solo semántico · 0.0 = solo léxico",
        )
        rerank_weight = st.slider(
            "Peso re-ranker",
            min_value=0.0, max_value=1.0, value=0.75, step=0.05,
        )

        st.divider()
        st.markdown('<div class="cs-sidebar-section">💡 Ejemplos</div>', unsafe_allow_html=True)
        for q in EXAMPLE_QUERIES:
            st.button(q, use_container_width=True, on_click=use_example, args=(q,))

    # ── Hero ───────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="cs-hero">
            <div class="cs-hero-badge">🎬 CulturaSearch &nbsp;·&nbsp; Motor de Recuperación IA</div>
            <h1>Encuentra <em>películas y series</em><br>con lenguaje natural</h1>
            <p>Escribe lo que buscas como si se lo dijeras a un amigo. El motor combina búsqueda semántica, léxica y re-ranking para darte los resultados más relevantes.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Buscador ───────────────────────────────────────────────────────
    query: str = st.text_area(
        "¿Qué quieres ver hoy?",
        key="search_query",
        placeholder="Ej: series de suspenso psicológico con giros inesperados al final...",
        height=110,
    )

    col_btn, col_clear = st.columns([5, 1])
    with col_btn:
        do_search = st.button("🔍  Buscar", type="primary", use_container_width=True)
    with col_clear:
        st.button("✕ Limpiar", use_container_width=True, on_click=clear_search)

    # ── Ejecutar búsqueda ──────────────────────────────────────────────
    if do_search:
        if not (query or "").strip():
            st.error("Escribe algo antes de buscar.")
        else:
            with st.spinner("Buscando los mejores resultados para ti..."):
                try:
                    res = run_query(
                        query=query.strip(),
                        use_web_expansion=use_web_expansion,
                        top_k=top_k,
                        candidate_k=candidate_k,
                        alpha=alpha,
                        rerank_weight=rerank_weight,
                    )
                    st.session_state["last_results"] = res
                    st.session_state["last_query"] = query.strip()
                    st.session_state["last_web_mode"] = use_web_expansion
                    if not res:
                        st.warning("No se encontraron resultados. Intenta con otros términos.")
                except Exception as exc:
                    st.error(f"Error al buscar: {exc}")
                    st.session_state["last_results"] = []

    # ── Mostrar resultados ─────────────────────────────────────────────
    results: list[SearchResult] = st.session_state.get("last_results", [])
    last_query: str = st.session_state.get("last_query", "")
    last_web_mode: bool = st.session_state.get("last_web_mode", False)

    if results:
        retriever = get_retriever()
        top_result = results[0]
        type_counts: dict[str, int] = {}
        for item in results:
            k = (item.media_type or "desconocido")
            type_counts[k] = type_counts.get(k, 0) + 1
        dominant = max(type_counts, key=type_counts.get)
        mode_tag = "🌐 Web + Local" if last_web_mode else "💾 Solo local"

        metrics_html = "".join([
            render_metric("Resultados", str(len(results))),
            render_metric("Mejor relevancia", f"{top_result.score:.0%}"),
            render_metric("Tipo dominante", dominant.capitalize()),
            render_metric("Docs indexados", str(len(retriever.documents))),
            render_metric("Modo", mode_tag),
        ])

        st.markdown(
            f"""
            <div style="margin-top:2rem;">
                <div style="font-size:0.8rem;text-transform:uppercase;letter-spacing:0.08em;
                            color:#8b949e;font-weight:700;margin-bottom:0.75rem;">
                    Resultados para: <span style="color:#58a6ff;">"{html.escape(last_query)}"</span>
                </div>
                <div class="cs-metrics">{metrics_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for rank, item in enumerate(results, start=1):
            render_result_card(item, rank)

    else:
        # Estado vacío
        st.markdown(
            """
            <div class="cs-empty">
                <div style="font-size:3.5rem;margin-bottom:0.75rem;">🎬</div>
                <div class="cs-empty-title">Empieza a descubrir contenido</div>
                <div class="cs-empty-sub">Escribe tu búsqueda arriba o usa los ejemplos del panel lateral</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div style='margin-top:1rem;color:#8b949e;font-size:0.95rem;'>💡 Los ejemplos están disponibles en el panel lateral izquierdo.</div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
