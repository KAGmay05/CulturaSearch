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
            :root {
                --text: #f6f7fb;
                --muted: rgba(246, 247, 251, 0.72);
                --accent: #52d6c5;
                --accent-2: #ffb547;
                --accent-3: #7cdb8f;
                --line: rgba(255, 255, 255, 0.08);
                --shadow: 0 24px 90px rgba(0, 0, 0, 0.42);
            }

            .stApp {
                background:
                    radial-gradient(circle at 10% 20%, rgba(82, 214, 197, 0.12), transparent 26%),
                    radial-gradient(circle at 85% 15%, rgba(255, 181, 71, 0.12), transparent 22%),
                    radial-gradient(circle at 85% 80%, rgba(124, 219, 143, 0.10), transparent 24%),
                    linear-gradient(145deg, #050b14 0%, #07111f 45%, #0d1729 100%);
                color: var(--text);
            }

            .block-container {
                padding-top: 1.6rem;
                padding-bottom: 2rem;
                max-width: 1300px;
            }

            h1, h2, h3, h4, h5, h6, p, label, div, span {
                color: var(--text);
            }

            .hero {
                position: relative;
                overflow: hidden;
                border: 1px solid var(--line);
                border-radius: 28px;
                padding: 1.75rem 1.75rem 1.35rem;
                background: linear-gradient(180deg, rgba(12, 19, 35, 0.94), rgba(10, 16, 29, 0.80));
                box-shadow: var(--shadow);
            }

            .hero::after {
                content: "";
                position: absolute;
                inset: auto -12% -50% auto;
                width: 320px;
                height: 320px;
                border-radius: 50%;
                background: radial-gradient(circle, rgba(82, 214, 197, 0.24), transparent 65%);
                pointer-events: none;
            }

            .eyebrow {
                display: inline-flex;
                align-items: center;
                padding: 0.35rem 0.75rem;
                border-radius: 999px;
                background: rgba(82, 214, 197, 0.10);
                border: 1px solid rgba(82, 214, 197, 0.25);
                color: var(--accent);
                font-size: 0.84rem;
                letter-spacing: 0.02em;
                margin-bottom: 0.9rem;
            }

            .hero h1 {
                margin: 0;
                font-size: clamp(2.2rem, 4vw, 4.25rem);
                line-height: 1.02;
                letter-spacing: -0.04em;
            }

            .hero p {
                max-width: 74ch;
                color: var(--muted);
                font-size: 1.02rem;
                line-height: 1.65;
                margin-top: 0.9rem;
            }

            .control-panel {
                border: 1px solid var(--line);
                border-radius: 22px;
                padding: 1rem 1rem 0.8rem;
                background: rgba(10, 18, 34, 0.84);
                backdrop-filter: blur(18px);
                box-shadow: var(--shadow);
            }

            .control-title {
                margin: 0 0 0.55rem 0;
                font-size: 1rem;
                font-weight: 700;
            }

            .control-note {
                color: var(--muted);
                font-size: 0.92rem;
                line-height: 1.5;
            }

            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 0.9rem;
                margin-top: 1rem;
            }

            .metric-card {
                border: 1px solid var(--line);
                border-radius: 18px;
                padding: 0.95rem 1rem;
                background: linear-gradient(180deg, rgba(16, 27, 49, 0.90), rgba(10, 16, 29, 0.82));
            }

            .metric-label {
                color: var(--muted);
                font-size: 0.83rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
            }

            .metric-value {
                margin-top: 0.35rem;
                font-size: 1.05rem;
                font-weight: 800;
                line-height: 1.25;
            }

            .result-card {
                margin-top: 1rem;
                border: 1px solid var(--line);
                border-radius: 24px;
                padding: 1.05rem 1.1rem;
                background: linear-gradient(180deg, rgba(17, 27, 48, 0.88), rgba(9, 15, 28, 0.82));
                box-shadow: 0 16px 44px rgba(0, 0, 0, 0.24);
            }

            .result-head {
                display: flex;
                gap: 0.9rem;
                justify-content: space-between;
                align-items: flex-start;
            }

            .rank-badge {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                min-width: 46px;
                height: 46px;
                border-radius: 16px;
                background: linear-gradient(135deg, rgba(82, 214, 197, 0.25), rgba(255, 181, 71, 0.18));
                border: 1px solid rgba(255, 255, 255, 0.10);
                font-weight: 800;
                font-size: 1.02rem;
                color: var(--text);
            }

            .result-title {
                margin: 0;
                font-size: 1.16rem;
                line-height: 1.2;
            }

            .result-meta {
                color: var(--muted);
                font-size: 0.92rem;
                margin-top: 0.22rem;
            }

            .score-pill {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                padding: 0.44rem 0.74rem;
                border-radius: 999px;
                background: rgba(82, 214, 197, 0.12);
                border: 1px solid rgba(82, 214, 197, 0.25);
                color: var(--accent);
                font-weight: 800;
                font-size: 0.86rem;
                white-space: nowrap;
            }

            .progress-track {
                margin-top: 0.85rem;
                width: 100%;
                height: 10px;
                border-radius: 999px;
                background: rgba(255, 255, 255, 0.07);
                overflow: hidden;
            }

            .progress-fill {
                height: 100%;
                border-radius: inherit;
                background: linear-gradient(90deg, var(--accent-2), var(--accent), var(--accent-3));
            }

            .result-body {
                display: grid;
                grid-template-columns: minmax(0, 1.8fr) minmax(240px, 1fr);
                gap: 1rem;
                margin-top: 0.95rem;
                align-items: start;
            }

            .snippet {
                color: var(--text);
                opacity: 0.92;
                line-height: 1.72;
                font-size: 0.96rem;
            }

            .chip-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.45rem;
            }

            .chip {
                display: inline-flex;
                align-items: center;
                gap: 0.4rem;
                padding: 0.37rem 0.68rem;
                border-radius: 999px;
                border: 1px solid rgba(255, 255, 255, 0.10);
                background: rgba(255, 255, 255, 0.05);
                color: var(--text);
                font-size: 0.80rem;
            }

            .chip.movie {
                border-color: rgba(255, 181, 71, 0.22);
                background: rgba(255, 181, 71, 0.10);
            }

            .chip.series {
                border-color: rgba(82, 214, 197, 0.22);
                background: rgba(82, 214, 197, 0.10);
            }

            .source-link {
                display: inline-flex;
                margin-top: 0.35rem;
                color: #9be7db;
                text-decoration: none;
                font-weight: 600;
            }

            .source-link:hover {
                text-decoration: underline;
            }

            @media (max-width: 860px) {
                .result-body {
                    grid-template-columns: 1fr;
                }

                .result-head {
                    flex-direction: column;
                }
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


def set_example_query(value: str) -> None:
    st.session_state.search_query = value


def score_class(score: float) -> str:
    if score >= 0.8:
        return "Excelente"
    if score >= 0.65:
        return "Alta"
    if score >= 0.45:
        return "Media"
    return "Baja"


def content_kind(media_type: str) -> str:
    normalized = media_type.lower()
    if "serie" in normalized:
        return "series"
    if "pel" in normalized or "film" in normalized:
        return "movie"
    return "other"


def snippet_text(text: str, limit: int = 300) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return "No hay sinopsis disponible para este resultado."
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "..."


def render_metric(label: str, value: str) -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-label">{html.escape(label)}</div>
        <div class="metric-value">{html.escape(value)}</div>
    </div>
    """


def render_result_card(item: SearchResult) -> None:
    title = html.escape(item.title or "Sin título")
    media_type = html.escape(item.media_type or "desconocido")
    kind = content_kind(item.media_type)
    summary = html.escape(snippet_text(item.plot))
    score_pct = max(0.0, min(100.0, float(item.score) * 100.0))
    score_value = f"{item.score:.3f}"
    neural_value = f"{item.neural_score:.3f}"
    lexical_value = f"{item.lexical_score:.2f}"
    rerank_value = f"{item.rerank_score:.3f}"
    url = item.url.strip()
    source_link = (
        f'<a class="source-link" href="{html.escape(url)}" target="_blank" rel="noopener noreferrer">Abrir fuente original</a>'
        if url
        else ""
    )

    st.markdown(
        f"""
        <article class="result-card">
            <div class="result-head">
                <div style="display:flex; gap:0.9rem; align-items:flex-start; min-width:0;">
                    <div class="rank-badge">#{item.rank}</div>
                    <div style="min-width:0;">
                        <h3 class="result-title">{title}</h3>
                        <div class="result-meta">{media_type} · Relevancia {html.escape(score_class(item.score))}</div>
                    </div>
                </div>
                <div class="score-pill">Score final {score_value}</div>
            </div>

            <div class="progress-track" aria-label="Nivel de relevancia">
                <div class="progress-fill" style="width:{score_pct:.1f}%"></div>
            </div>

            <div class="result-body">
                <div>
                    <div class="snippet">{summary}</div>
                    {source_link}
                </div>
                <div>
                    <div class="chip-row">
                        <span class="chip {kind}">Tipo: {media_type}</span>
                        <span class="chip">Fused: {score_value}</span>
                        <span class="chip">Neural: {neural_value}</span>
                        <span class="chip">Lexical: {lexical_value}</span>
                        <span class="chip">Rerank: {rerank_value}</span>
                    </div>
                </div>
            </div>
        </article>
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

    if "search_query" not in st.session_state:
        st.session_state.search_query = ""

    st.markdown(
        """
        <section class="hero">
            <div class="eyebrow">Interfaz visual de recuperación · CulturaSearch</div>
            <h1>Busca películas y series con una interfaz clara, moderna y orientada a relevancia.</h1>
            <p>
                Escribe consultas en lenguaje natural, activa expansión web cuando quieras ampliar el corpus y revisa
                los resultados en tarjetas ordenadas por importancia. Cada resultado muestra el score final, la señal
                semántica, la cobertura léxica y el re-ranking para que la decisión sea fácil de interpretar.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown('<div class="control-panel">', unsafe_allow_html=True)
        st.markdown('<div class="control-title">Panel de consulta</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="control-note">Ajusta el balance entre búsqueda semántica, re-ranking y expansión web.</div>',
            unsafe_allow_html=True,
        )

        use_web_expansion = st.toggle("Expansión web automática", value=True)
        top_k = st.slider("Resultados a mostrar", min_value=3, max_value=10, value=5, step=1)
        candidate_k = st.slider("Candidatos internos", min_value=10, max_value=100, value=50, step=5)
        alpha = st.slider("Peso semántico", min_value=0.0, max_value=1.0, value=0.9, step=0.05)
        rerank_weight = st.slider("Peso del re-ranker", min_value=0.0, max_value=1.0, value=0.75, step=0.05)

        st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 1rem 0;'>", unsafe_allow_html=True)
        st.markdown('<div class="control-title">Sugerencias rápidas</div>', unsafe_allow_html=True)
        for example in EXAMPLE_QUERIES:
            st.button(example, use_container_width=True, on_click=set_example_query, args=(example,))

        st.markdown('</div>', unsafe_allow_html=True)

    query = st.text_area(
        "Consulta en lenguaje natural",
        key="search_query",
        placeholder="Ej.: ¿Qué series policiales con detectives me recomiendas?",
        height=120,
        label_visibility="visible",
    )

    launch_search = st.button("Buscar", type="primary", use_container_width=True)

    if launch_search:
        if not query.strip():
            st.warning("Escribe una consulta antes de buscar.")
        else:
            with st.spinner("Recuperando y ordenando resultados..."):
                results = run_query(
                    query=query,
                    use_web_expansion=use_web_expansion,
                    top_k=top_k,
                    candidate_k=candidate_k,
                    alpha=alpha,
                    rerank_weight=rerank_weight,
                )

            st.session_state.last_query = query
            st.session_state.last_results = results
            st.session_state.last_web_mode = use_web_expansion

    results = st.session_state.get("last_results", [])
    last_query = st.session_state.get("last_query", "")
    last_web_mode = st.session_state.get("last_web_mode", False)

    if results:
        retriever = get_retriever()
        top_result = results[0]
        type_counts = {}
        for item in results:
            kind = item.media_type or "desconocido"
            type_counts[kind] = type_counts.get(kind, 0) + 1

        dominant_type = max(type_counts, key=type_counts.get) if type_counts else "Desconocido"
        metric_html = "".join(
            [
                render_metric("Resultados", str(len(results))),
                render_metric("Mejor score", f"{top_result.score:.3f}"),
                render_metric("Tipo dominante", dominant_type),
                render_metric("Corpus activo", f"{len(retriever.documents)} docs"),
            ]
        )

        st.markdown(
            f"""
            <section style="margin-top: 1rem;">
                <div class="control-title" style="margin-bottom:0.4rem;">Resumen de la consulta</div>
                <div class="control-note">{html.escape(last_query)}</div>
                <div class="metrics-grid">{metric_html}</div>
            </section>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div style="margin-top: 1rem; border: 1px solid rgba(255,255,255,0.08); border-radius: 20px; padding: 0.95rem 1rem; background: rgba(255,255,255,0.03);">
                <strong>Orden de presentación:</strong> los elementos aparecen rankeados por el score final del motor.
                {'La expansión web automática está activa para ampliar el alcance cuando la confianza local baja.' if last_web_mode else 'La búsqueda se resolvió con el corpus local y el re-ranking del motor.'}
            </div>
            """,
            unsafe_allow_html=True,
        )

        for item in results:
            render_result_card(item)
    else:
        st.info("Ingresa una consulta y pulsa Buscar para ver resultados rankeados.")
        st.markdown(
            """
            <div class="metrics-grid">
                <div class="metric-card"><div class="metric-label">Flujo</div><div class="metric-value">Consulta → ranking → navegación</div></div>
                <div class="metric-card"><div class="metric-label">Interacción</div><div class="metric-value">Natural y directa</div></div>
                <div class="metric-card"><div class="metric-label">Presentación</div><div class="metric-value">Tarjetas y métricas</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()