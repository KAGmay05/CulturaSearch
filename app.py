from __future__ import annotations

import html

import streamlit as st

from neural_based_model.neural_retriever import NeuralRetriever, SearchResult
from recommendation_module.profile_store import load_user, save_user
from recommendation_module.user_profile import User
from recommendation_module.recommendation import RecommendationEngine, recommend_for_user
from auth import authenticate


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

        .stApp { background: #0d1117; min-height: 100vh; }

        .block-container {
            padding: 2rem 2rem 4rem !important;
            max-width: 1280px !important;
        }

        .stApp p, .stApp div, .stApp span, .stApp label,
        .stApp .stMarkdown p { color: #e6edf3 !important; }
        .stApp h1, .stApp h2, .stApp h3, .stApp h4 { color: #ffffff !important; }

        [data-testid="stSidebar"] {
            background: #161b22 !important;
            border-right: 1px solid #30363d !important;
        }
        [data-testid="stSidebar"] * { color: #e6edf3 !important; }
        [data-testid="stSidebarContent"] { padding: 1.5rem 1rem !important; }

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

        .stButton > button:not([kind="primary"]) {
            background: #21262d !important;
            color: #e6edf3 !important;
            border: 1px solid #30363d !important;
            border-radius: 8px !important;
            font-weight: 500 !important;
            transition: all 0.2s ease !important;
        }

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

        .cs-metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 0.75rem; margin: 1.5rem 0; }
        .cs-metric { background: #161b22; border: 1px solid #30363d; border-radius: 14px; padding: 1rem 1.25rem; }
        .cs-metric-label { font-size: 0.76rem !important; font-weight: 700 !important; color: #8b949e !important; margin-bottom: 0.4rem; }
        .cs-metric-value { font-size: 1.4rem !important; font-weight: 800 !important; color: #ffffff !important; }

        .cs-card { background: #161b22; border: 1px solid #30363d; border-radius: 16px; padding: 1.5rem; margin-top: 1.25rem; }
        .cs-card-top { display:flex; align-items:flex-start; gap:1rem }
        .cs-rank { width:48px; height:48px; border-radius:12px; background:linear-gradient(135deg,#1f6feb,#bc8cff); display:flex;align-items:center;justify-content:center;color:#fff;font-weight:900 }
        .cs-card-title { font-size:1.2rem !important; font-weight:700 !important; color:#fff !important }
        .cs-bar { margin:1rem 0; height:6px; background:#21262d; border-radius:50px; overflow:hidden }
        .cs-bar-fill { height:100% }
        .cs-synopsis { color:#8b949e !important }

        .cs-empty { text-align:center; padding:3rem 2rem; border:2px dashed #30363d; border-radius:20px; background:#161b22; margin-top:2rem }
        .cs-empty-title { font-size:1.4rem !important; font-weight:700 !important; color:#fff !important }

        @media (max-width: 768px) { .cs-card-top { flex-wrap:wrap } .cs-hero { padding:1.5rem } }
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
    return cleaned if len(cleaned) <= limit else cleaned[:limit].rstrip() + "…"


def render_metric(label: str, value: str) -> str:
    return (
        '<div class="cs-metric">'
        f'<div class="cs-metric-label">{html.escape(label)}</div>'
        f'<div class="cs-metric-value">{html.escape(value)}</div>'
        '</div>'
    )


def render_result_card(item: SearchResult, rank: int) -> None:
    title = html.escape(item.title or "Sin título")
    kind = content_kind(item.media_type)
    summary = html.escape(snippet_text(item.plot or getattr(item, 'description', '') or ''))
    score_pct = max(0.0, min(100.0, float(item.score) * 100.0))
    label, color = score_label(float(item.score or 0.0))
    url = (item.url or "").strip()

    type_icon = "🎬" if kind == "movie" else "📺" if kind == "series" else "🎭"
    type_label = ("Película" if kind == "movie" else "Serie" if kind == "series" else (item.media_type or "").capitalize())

    source_html = (
        f'<a class="cs-link" href="{html.escape(url)}" target="_blank" rel="noopener noreferrer">🔗 Ver fuente original</a>'
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

    col1, col2 = st.columns([1, 4])
    with col1:
        rank_val = getattr(item, 'rank', rank)
        key_id = f"like_{rank_val}_{abs(hash(url))}" if url else f"like_{rank_val}_{abs(hash(title))}"

        if st.button("👍 Me gustó", key=key_id, use_container_width=True):
            user_profile = st.session_state.get("user_profile")
            if user_profile is None:
                st.warning("Inicia sesión para guardar tus preferencias.")
            else:
                try:
                    if not url:
                        st.warning("No hay URL disponible para este resultado")
                    else:
                        # Registrar como vista/like en el perfil
                        user_profile.register_view(url)
                        try:
                            retriever = get_retriever()
                            doc_meta = retriever._resolve_result_document(item)
                        except Exception:
                            doc_meta = None

                        doc_type = (doc_meta.get('type') or doc_meta.get('media_type') or item.media_type) if doc_meta else item.media_type
                        media_type_lower = (doc_type or '').lower()
                        if "serie" in media_type_lower:
                            user_profile.add_type_preference("serie")
                        elif "pel" in media_type_lower or "film" in media_type_lower:
                            user_profile.add_type_preference("película")

                        genres = []
                        if doc_meta:
                            genres = doc_meta.get('genres') or []
                        if not genres:
                            genres = getattr(item, 'genres', []) or []

                        if isinstance(genres, str):
                            genres = [g.strip() for g in genres.split(',') if g.strip()]

                        for genre in genres:
                            try:
                                user_profile.add_genre_preference(genre)
                            except Exception:
                                continue

                        save_user(user_profile)
                        st.success("✅ Gracias — añadido a tus gustos")
                except Exception as e:
                    st.error(f"Error guardando preferencia: {e}")


def track_viewed_results(results: list[SearchResult]) -> None:
    user_profile = st.session_state.get("user_profile")
    if user_profile is None:
        return

    changed = False
    for item in results:
        url = (getattr(item, 'url', '') or '').strip()
        if not url:
            continue
        try:
            before = len(user_profile.viewed_urls)
            user_profile.register_view(url)
            if len(user_profile.viewed_urls) > before:
                changed = True
        except Exception:
            continue

    if changed:
        try:
            save_user(user_profile)
        except Exception:
            pass


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
        results = retriever.search_with_web_expansion(
            query=query,
            top_k=top_k,
            candidate_k=candidate_k,
            alpha=alpha,
            rerank_weight=rerank_weight,
        )
    else:
        results = retriever.search_advanced(
            query=query,
            top_k=top_k,
            candidate_k=candidate_k,
            alpha=alpha,
            rerank_weight=rerank_weight,
        )

    user_profile = st.session_state.get("user_profile")
    if user_profile is not None and results:
        try:
            recommender = RecommendationEngine()
            personalized = recommender.personalize_results(user_profile, results, top_k=top_k)
            if personalized:
                results = personalized
        except Exception:
            pass

    return results


def show_login_page() -> None:
    inject_styles()

    st.markdown(
        '''
        <div class="cs-hero" style="max-width: 500px; margin: 4rem auto; text-align: center;">
            <div class="cs-hero-badge">CulturaSearch</div>
            <h1 style="font-size: 2.2rem; margin-top: 1rem;">Ingresa a tu cuenta</h1>
            <p style="margin: 0 auto;">Accede para personalizar tus búsquedas y recibir recomendaciones adaptadas a tu gusto.</p>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        user_id = st.text_input("👤 Usuario", placeholder="ej: alice", key="login_user_id")
        password = st.text_input("🔑 Contraseña", placeholder="ej: 1234", type="password", key="login_password")

        if st.button("Ingresar", type="primary", use_container_width=True):
            if not user_id or not password:
                st.error("⚠️ Por favor completa usuario y contraseña")
            else:
                success, user_name = authenticate(user_id, password)
                if success:
                    st.session_state.user_id = user_id
                    st.session_state.user_name = user_name
                    st.session_state.user_profile = load_user(user_id)
                    if st.session_state.user_profile is None:
                        st.session_state.user_profile = User(user_id)
                    save_user(st.session_state.user_profile)
                    st.success(f"✅ ¡Bienvenido, {user_name}!")
                    st.rerun()
                else:
                    st.error("❌ Usuario o contraseña inválidos")


def show_app() -> None:
    user_name = st.session_state.get("user_name", "Usuario")
    user_id = st.session_state.get("user_id", "")

    for key, default in [("search_query", ""), ("last_results", []), ("last_query", ""), ("last_web_mode", False)]:
        if key not in st.session_state:
            st.session_state[key] = default

    with st.sidebar:
        st.markdown('<div class="cs-sidebar-section">⚙️ Parámetros</div>', unsafe_allow_html=True)

        use_web_expansion = st.toggle("Expansión web", value=True, help="Amplía la búsqueda con resultados de internet")
        top_k = st.slider("Resultados", min_value=3, max_value=10, value=5, step=1)
        candidate_k = st.slider("Candidatos internos", min_value=10, max_value=100, value=50, step=5)
        alpha = st.slider("Peso semántico", min_value=0.0, max_value=1.0, value=0.9, step=0.05,
                          help="1.0 = solo semántico · 0.0 = solo léxico")
        rerank_weight = st.slider("Peso re-ranker", min_value=0.0, max_value=1.0, value=0.75, step=0.05)

        st.divider()
        st.markdown('<div class="cs-sidebar-section">💡 Ejemplos</div>', unsafe_allow_html=True)
        for q in EXAMPLE_QUERIES:
            st.button(q, use_container_width=True, on_click=use_example, args=(q,))

    st.markdown(
        '''
        <div class="cs-hero">
            <div class="cs-hero-badge">🎬 CulturaSearch · Motor de Recuperación IA</div>
            <h1>Encuentra <em>películas y series</em><br>con lenguaje natural</h1>
            <p>Escribe lo que buscas como si se lo dijeras a un amigo. El motor combina búsqueda semántica, léxica y re-ranking para darte los resultados más relevantes.</p>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    # (recomendaciones movidas debajo del buscador)

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

    # Mostrar botón de recomendaciones para el usuario justo debajo del buscador
    user_profile = st.session_state.get("user_profile")
    if user_profile is not None:
        st.markdown(
            '''
            <div style="background: #161b22; border: 1px solid #30363d; border-radius: 16px; padding: 0.9rem; margin: 1rem 0 1.25rem; display:flex; align-items:center; gap:1rem;">
                <div style="flex:1;">
                    <div style="color:#ffffff; font-weight:700;">¿No sabes qué ver?</div>
                    <div style="color:#8b949e; font-size:0.92rem;">Pulsa para obtener recomendaciones basadas en tus interacciones.</div>
                </div>
            ''',
            unsafe_allow_html=True,
        )
        if st.button("✨ Ver recomendaciones para mí", key="show_personal_recs"):
            with st.spinner("Generando recomendaciones personalizadas..."):
                recs = recommend_for_user(user_profile, top_k=top_k)
                track_viewed_results(recs)
                st.session_state["last_query"] = "Recomendaciones para ti"
                st.session_state["last_results"] = recs
                st.session_state["last_web_mode"] = False
        st.markdown('</div>', unsafe_allow_html=True)

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

                    try:
                        if user_profile is not None:
                            user_profile.register_search(query.strip())
                            save_user(user_profile)
                    except Exception:
                        pass

                    track_viewed_results(res)

                    st.session_state["last_results"] = res
                    st.session_state["last_query"] = query.strip()
                    st.session_state["last_web_mode"] = use_web_expansion

                    if not res:
                        st.warning("No se encontraron resultados. Intenta con otros términos.")
                except Exception as exc:
                    st.error(f"Error al buscar: {exc}")
                    st.session_state["last_results"] = []

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

        metrics_html = "".join([
            render_metric("Resultados", str(len(results))),
            render_metric("Mejor relevancia", f"{top_result.score:.0%}"),
            render_metric("Tipo dominante", dominant.capitalize()),
            render_metric("Docs indexados", str(len(retriever.documents))),
        ])

        st.markdown(
            f'''
            <div style="margin-top:2rem;">
                <div style="font-size:0.8rem;text-transform:uppercase;letter-spacing:0.08em;
                            color:#8b949e;font-weight:700;margin-bottom:0.75rem;">
                    Resultados para: <span style="color:#58a6ff;">"{html.escape(last_query)}"</span>
                </div>
                <div class="cs-metrics">{metrics_html}</div>
            </div>
            ''',
            unsafe_allow_html=True,
        )

        for rank, item in enumerate(results, start=1):
            render_result_card(item, rank)

    else:
        st.markdown(
            '''
            <div class="cs-empty">
                <div style="font-size:3.5rem;margin-bottom:0.75rem;">🎬</div>
                <div class="cs-empty-title">Empieza a descubrir contenido</div>
                <div class="cs-empty-sub">Escribe tu búsqueda arriba o usa los ejemplos del panel lateral</div>
            </div>
            ''',
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div style='margin-top:1rem;color:#8b949e;font-size:0.95rem;'>💡 Los ejemplos están disponibles en el panel lateral izquierdo.</div>",
            unsafe_allow_html=True,
        )


def main() -> None:
    inject_styles()
    if "user_id" not in st.session_state:
        show_login_page()
    else:
        show_app()


if __name__ == "__main__":
    main()
