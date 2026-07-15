"""
app.py
==================================================================
FRONTEND for the AI Season RAG Chatbot (Task 3) — neumorphic UI,
v2: theme toggle (dark/light) + animated ambient background.

Zero pipeline logic lives here. Every chunking, retrieval, and
generation function is imported directly from rag_chatbot.py.

Run with:
    streamlit run app.py
"""

import streamlit as st

from rag_chatbot import (
    load_document,
    build_all_indexes,
    get_llm,
    run_all_combinations,
    is_mock_mode,
    DATA_FILE,
    EMBEDDING_MODEL,
    GROQ_MODEL,
    TOP_K,
    CHUNKING_METHODS,
)

st.set_page_config(page_title="AI Season RAG Console", page_icon="🧠", layout="wide")

# ------------------------------------------------------------------
# THEME DEFINITIONS
# ------------------------------------------------------------------
THEMES = {
    "dark": {
        "bg": "#12151d",
        "card_bg": "#171b25",
        "inset_bg": "#0e1017",
        "shadow_dark": "rgba(0,0,0,0.65)",
        "shadow_light": "rgba(255,255,255,0.04)",
        "text_primary": "#e9ebf2",
        "text_secondary": "#8b93a3",
        "border": "rgba(255,255,255,0.06)",
        "accent_fixed": "#ff6b81",
        "accent_sentence": "#2dd4bf",
        "accent_recursive": "#8b93ff",
        "title_gradient": "linear-gradient(135deg,#8b93ff 0%, #2dd4bf 100%)",
        "orb1": "rgba(139,147,255,0.25)",
        "orb2": "rgba(45,212,191,0.20)",
        "orb3": "rgba(255,107,129,0.15)",
        "input_bg": "#0e1017",
    },
    "light": {
        "bg": "#eef1f7",
        "card_bg": "#eef1f7",
        "inset_bg": "#e1e5ee",
        "shadow_dark": "rgba(163,177,198,0.55)",
        "shadow_light": "rgba(255,255,255,0.85)",
        "text_primary": "#20242c",
        "text_secondary": "#5f6a78",
        "border": "rgba(0,0,0,0.04)",
        "accent_fixed": "#e0523a",
        "accent_sentence": "#0f9c86",
        "accent_recursive": "#4d55c9",
        "title_gradient": "linear-gradient(135deg,#4d55c9 0%, #0f9c86 100%)",
        "orb1": "rgba(77,85,201,0.18)",
        "orb2": "rgba(15,156,134,0.16)",
        "orb3": "rgba(224,82,58,0.12)",
        "input_bg": "#e1e5ee",
    },
}

RETRIEVAL_ICONS = {"dense": "◉", "sparse": "▤"}
METHOD_ICONS = {"fixed": "▦", "sentence": "❖", "recursive": "◆"}
METHOD_LABELS = {"fixed": "Fixed-size", "sentence": "Sentence-based", "recursive": "Recursive"}
RETRIEVAL_LABELS = {"dense": "Dense · Vector", "sparse": "Sparse · BM25"}

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

T = THEMES[st.session_state.theme]

# ------------------------------------------------------------------
# CSS — themed neumorphism + animated ambient background
# ------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    .stApp {{
        background: {T['bg']};
        position: relative;
        overflow-x: hidden;
    }}
    header[data-testid="stHeader"] {{ background: transparent; }}

    /* Animated ambient background orbs -- z-index: -1 guarantees these
       render BEHIND all page content regardless of Streamlit's internal
       class names, which change between versions and can't be relied on. */
    .orb {{
        position: fixed;
        border-radius: 50%;
        filter: blur(70px);
        z-index: -1;
        pointer-events: none;
    }}
    .orb1 {{
        width: 420px; height: 420px;
        background: {T['orb1']};
        top: -100px; left: -100px;
        animation: floatA 22s ease-in-out infinite alternate;
    }}
    .orb2 {{
        width: 380px; height: 380px;
        background: {T['orb2']};
        bottom: -120px; right: -80px;
        animation: floatB 26s ease-in-out infinite alternate;
    }}
    .orb3 {{
        width: 300px; height: 300px;
        background: {T['orb3']};
        top: 40%; right: 10%;
        animation: floatC 30s ease-in-out infinite alternate;
    }}
    @keyframes floatA {{
        0%   {{ transform: translate(0,0) scale(1); }}
        100% {{ transform: translate(80px,60px) scale(1.15); }}
    }}
    @keyframes floatB {{
        0%   {{ transform: translate(0,0) scale(1); }}
        100% {{ transform: translate(-70px,-50px) scale(1.1); }}
    }}
    @keyframes floatC {{
        0%   {{ transform: translate(0,0) scale(1); }}
        100% {{ transform: translate(-40px,40px) scale(0.9); }}
    }}

    .stApp > * {{ position: relative; z-index: 1; }}

    .neo-title-wrap {{ text-align: center; padding: 30px 10px 16px 10px; }}
    .neo-title {{
        font-size: 2.7rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: {T['title_gradient']};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }}
    .neo-subtitle {{ color: {T['text_secondary']}; font-size: 0.95rem; margin-top: 6px; }}

    .neo-panel {{
        background: {T['card_bg']};
        border: 1px solid {T['border']};
        border-radius: 22px;
        padding: 22px 24px;
        margin-bottom: 22px;
        box-shadow: 9px 9px 18px {T['shadow_dark']}, -9px -9px 18px {T['shadow_light']};
    }}

    .result-card {{
        background: {T['card_bg']};
        border: 1px solid {T['border']};
        border-radius: 20px;
        padding: 18px 20px;
        margin-bottom: 20px;
        box-shadow: 8px 8px 16px {T['shadow_dark']}, -8px -8px 16px {T['shadow_light']};
        height: 100%;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }}
    .result-card:hover {{
        transform: translateY(-4px);
        box-shadow: 12px 12px 22px {T['shadow_dark']}, -12px -12px 22px {T['shadow_light']};
    }}
    .card-header {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; }}
    .card-title {{ font-weight: 700; font-size: 0.98rem; }}
    .card-sub {{ font-size: 0.78rem; color: {T['text_secondary']}; font-weight: 500; }}

    .answer-box {{
        background: {T['inset_bg']};
        border-radius: 14px;
        padding: 12px 14px;
        font-size: 0.9rem;
        line-height: 1.5;
        color: {T['text_primary']};
        box-shadow: inset 3px 3px 8px {T['shadow_dark']}, inset -3px -3px 8px {T['shadow_light']};
        min-height: 90px;
    }}
    .token-row {{ display: flex; gap: 8px; margin-top: 10px; flex-wrap: wrap; }}
    .token-pill {{
        font-size: 0.72rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 999px;
        background: {T['card_bg']};
        color: {T['text_secondary']};
        box-shadow: 3px 3px 6px {T['shadow_dark']}, -3px -3px 6px {T['shadow_light']};
    }}
    .chunk-preview {{ font-size: 0.72rem; color: {T['text_secondary']}; font-family: 'SF Mono', monospace; margin: 3px 0; }}

    .stButton>button {{
        border-radius: 14px !important;
        background: {T['card_bg']} !important;
        color: {T['text_primary']} !important;
        border: 1px solid {T['border']} !important;
        font-weight: 700 !important;
        box-shadow: 6px 6px 12px {T['shadow_dark']}, -6px -6px 12px {T['shadow_light']} !important;
        padding: 10px 22px !important;
        transition: color 0.2s ease !important;
    }}
    .stButton>button:hover {{ color: {T['accent_recursive']} !important; }}

    div[data-testid="stTextInput"] input {{
        background: {T['input_bg']} !important;
        border: 1px solid {T['border']} !important;
        border-radius: 14px !important;
        box-shadow: inset 3px 3px 8px {T['shadow_dark']}, inset -3px -3px 8px {T['shadow_light']} !important;
        padding: 12px 16px !important;
        color: {T['text_primary']} !important;
    }}

    section[data-testid="stSidebar"] {{ background: {T['bg']}; border-right: 1px solid {T['border']}; }}
    h1, h2, h3, h4, p, span, label, .stMarkdown {{ color: {T['text_primary']}; }}
    </style>

    <div class="orb orb1"></div>
    <div class="orb orb2"></div>
    <div class="orb orb3"></div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def setup_pipelines():
    text = load_document()
    indexes = build_all_indexes(text, source=DATA_FILE)
    llm = get_llm()
    return text, indexes, llm


def token_pill(label: str, value) -> str:
    display = value if value is not None else "—"
    return f'<span class="token-pill">{label}: {display}</span>'


def method_accent(method: str) -> str:
    return {"fixed": T["accent_fixed"], "sentence": T["accent_sentence"], "recursive": T["accent_recursive"]}[method]


# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎨 Appearance")
    choice = st.radio("Theme", ["Dark", "Light"], index=0 if st.session_state.theme == "dark" else 1, horizontal=True)
    new_theme = choice.lower()
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()

    st.divider()
    st.markdown("### ⚙️ System status")
    if is_mock_mode():
        st.warning("**MOCK MODE** — no GROQ_API_KEY set. Showing raw retrieved chunks, no token usage.")
    else:
        st.success(f"**Live** — Groq `{GROQ_MODEL}`")
    st.caption(f"Embeddings: `{EMBEDDING_MODEL}`")
    st.caption(f"Knowledge base: `{DATA_FILE}`")
    st.caption(f"Top-k per pipeline: `{TOP_K}`")

    st.divider()
    st.markdown("### 📊 Session token usage")
    st.caption(f"Input: **{st.session_state.get('session_input_tokens', 0)}**")
    st.caption(f"Output: **{st.session_state.get('session_output_tokens', 0)}**")
    st.caption(f"Total: **{st.session_state.get('session_total_tokens', 0)}**")

    st.divider()
    if st.button("🔄 Rebuild knowledge base", use_container_width=True):
        st.cache_resource.clear()
        st.session_state.pop("history", None)
        st.rerun()


# ------------------------------------------------------------------
# TITLE
# ------------------------------------------------------------------
st.markdown(
    """
    <div class="neo-title-wrap">
        <div class="neo-title">🧠 AI SEASON — RAG CONSOLE</div>
        <div class="neo-subtitle">One question, six pipelines — 3 chunking methods × 2 retrieval techniques, compared live</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.spinner("Loading embedding model and building all 6 indexes..."):
    text, indexes, llm = setup_pipelines()

if "history" not in st.session_state:
    st.session_state.history = []
for key in ("session_input_tokens", "session_output_tokens", "session_total_tokens"):
    st.session_state.setdefault(key, 0)


# ------------------------------------------------------------------
# QUERY COMPARTMENT
# ------------------------------------------------------------------
st.markdown('<div class="neo-panel">', unsafe_allow_html=True)
st.markdown("**Ask a question about AI Season**")
col1, col2 = st.columns([5, 1])
with col1:
    query = st.text_input("query", label_visibility="collapsed",
                           placeholder="e.g. Who is the founder of AI Season? / What is the refund policy?")
with col2:
    run_clicked = st.button("Ask", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

if run_clicked and query.strip():
    with st.spinner("Running all 6 pipelines..."):
        results = run_all_combinations(query.strip(), indexes, llm)

    for r in results:
        st.session_state.session_input_tokens += r["input_tokens"] or 0
        st.session_state.session_output_tokens += r["output_tokens"] or 0
        st.session_state.session_total_tokens += r["total_tokens"] or 0

    st.session_state.history.insert(0, {"query": query.strip(), "results": results})
    st.rerun()

if not st.session_state.history:
    st.info("Ask a question above to see all 6 pipelines respond side by side.")


# ------------------------------------------------------------------
# RESULT COMPARTMENTS
# ------------------------------------------------------------------
def render_card(r: dict):
    method = r["chunking_method"]
    retrieval = r["retrieval_technique"]
    accent = method_accent(method)

    st.markdown(
        f"""
        <div class="result-card">
            <div class="card-header">
                <div class="card-title" style="color:{accent}">
                    {METHOD_ICONS[method]} {METHOD_LABELS[method]}
                </div>
                <div class="card-sub">{RETRIEVAL_ICONS[retrieval]} {RETRIEVAL_LABELS[retrieval]}</div>
            </div>
            <div class="answer-box">{r['answer']}</div>
            <div class="token-row">
                {token_pill("in", r['input_tokens'])}
                {token_pill("out", r['output_tokens'])}
                {token_pill("total", r['total_tokens'])}
                {token_pill("chunks", r['num_chunks_retrieved'])}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("View retrieved chunks"):
        for i, preview in enumerate(r["retrieved_previews"], start=1):
            st.markdown(f'<div class="chunk-preview">[{i}] {preview}</div>', unsafe_allow_html=True)


for entry_idx, entry in enumerate(st.session_state.history):
    is_latest = entry_idx == 0
    icon = "🟢" if is_latest else "⚪"
    st.markdown(f"#### {icon} \"{entry['query']}\"")

    results = entry["results"]
    for method in CHUNKING_METHODS:
        cols = st.columns(2)
        method_results = [r for r in results if r["chunking_method"] == method]
        for col, r in zip(cols, method_results):
            with col:
                render_card(r)

    st.divider()