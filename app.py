import streamlit as st
import time
import pandas as pd
from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InsightForge AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@500;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* ── Root Variables ── */
:root {
    --bg: #0a0a0f;
    --surface: #111118;
    --surface-2: #1a1a25;
    --border: #2a2a3a;
    --accent: #7c3aed;
    --accent-glow: #9f67ff;
    --accent-2: #06b6d4;
    --text: #e8e8f0;
    --text-muted: #7070a0;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.stApp {
    background: var(--bg) !important;
}

/* Animated grid background */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image:
        linear-gradient(rgba(124, 58, 237, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(124, 58, 237, 0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

/* ── Headings ── */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Syne', sans-serif !important;
    color: var(--text) !important;
}

/* ── Hero Title ── */
.hero-title {
    font-family: 'Inter', sans-serif;
    font-size: clamp(2.3rem, 4vw, 3.3rem);
    font-weight: 800;
    letter-spacing: -0.05em;
    line-height: 1;
    margin: 0;
    color: #f5f5ff;
    display: flex;
    align-items: center;
    gap: 0.65rem;
}

.hero-ai-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.25rem 0.55rem;
    border-radius: 999px;
    background: rgba(124,58,237,0.18);
    border: 1px solid rgba(124,58,237,0.35);
    color: var(--accent-glow);
}

.hero-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: var(--text-muted);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: 0.5rem;
}

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}

.card:hover {
    border-color: var(--accent);
}

.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, var(--accent), var(--accent-2));
}

.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.card-content {
    font-size: 0.875rem;
    line-height: 1.7;
    color: var(--text);
}

/* ── Accent Badge ── */
.badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.badge-purple { background: rgba(124,58,237,0.2); color: var(--accent-glow); border: 1px solid rgba(124,58,237,0.3); }
.badge-cyan   { background: rgba(6,182,212,0.15); color: var(--accent-2);    border: 1px solid rgba(6,182,212,0.3); }
.badge-green  { background: rgba(16,185,129,0.15); color: var(--success);    border: 1px solid rgba(16,185,129,0.3); }

/* ── Input & Buttons ── */
.stTextInput > div > div > input,
.stSelectbox > div > div {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(124,58,237,0.2) !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent), #5b21b6) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.875rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s !important;
    text-transform: uppercase !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(124,58,237,0.4) !important;
}

/* Secondary button */
.stButton > button[kind="secondary"] {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
}

/* ── Progress / Status ── */
.status-bar {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.75rem 1rem;
    background: var(--surface-2);
    border-radius: 8px;
    margin: 0.4rem 0;
    border: 1px solid var(--border);
    font-size: 0.8rem;
}

.status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}

.dot-active   { background: var(--accent-glow); box-shadow: 0 0 8px var(--accent-glow); animation: pulse 1.5s infinite; }
.dot-done     { background: var(--success); }
.dot-pending  { background: var(--border); }

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
}

/* ── Chat ── */
.chat-container {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem;
    max-height: 420px;
    overflow-y: auto;
    margin-bottom: 1rem;
}

.chat-msg {
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
}

.chat-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

.chat-bubble {
    display: inline-block;
    padding: 0.6rem 1rem;
    border-radius: 10px;
    font-size: 0.85rem;
    line-height: 1.6;
    max-width: 90%;
}

.user-label  { color: var(--accent-glow); }
.bot-label   { color: var(--accent-2); }

.user-bubble { background: rgba(124,58,237,0.15); border: 1px solid rgba(124,58,237,0.25); align-self: flex-end; }
.bot-bubble  { background: rgba(6,182,212,0.1);  border: 1px solid rgba(6,182,212,0.2);   align-self: flex-start; }

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.5rem 0 !important;
}

/* ── Transcript box ── */
.transcript-box {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.25rem;
    font-size: 0.82rem;
    line-height: 1.8;
    max-height: 300px;
    overflow-y: auto;
    color: var(--text-muted);
    white-space: pre-wrap;
    word-break: break-word;
}

/* ── Stale Streamlit elements ── */
.stProgress > div > div > div { background: var(--accent) !important; }
.stSpinner > div { border-top-color: var(--accent) !important; }
[data-testid="stMarkdownContainer"] p { color: var(--text) !important; }
label { color: var(--text-muted) !important; font-size: 0.8rem !important; }

/* scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

.panel-btn {
    width:100%;
    margin-bottom:0.5rem;
}

.section-anchor {
    scroll-margin-top: 80px;
}
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ──────────────────────────────────────────────────────────
for key, default in {
    "result": None,
    "chat_history": [],
    "processing": False,
    "pipeline_done": False,
    "pipeline_steps": {},
    "analytics": {
        "videos_processed": 0,
        "questions_asked": 0,
        "transcription_times": [],
        "pipeline_times": [],
        "question_times": [],
        "web_search_count": 0,
        "avg_rerank_scores": [],
    },
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Helpers ────────────────────────────────────────────────────────────────────
def step_status(steps: dict, key: str) -> str:
    s = steps.get(key, "pending")
    if s == "active":  return "dot-active"
    if s == "done":    return "dot-done"
    return "dot-pending"

def render_step_bar(label: str, key: str, icon: str):
    css = step_status(st.session_state.pipeline_steps, key)
    st.markdown(f"""
    <div class="status-bar">
        <div class="status-dot {css}"></div>
        <span>{icon} {label}</span>
    </div>""", unsafe_allow_html=True)

# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Branding
    st.markdown('''
<div class="hero-title" style="font-size:1.35rem;line-height:1.05">
    <span>InsightForge</span>
    <span class="hero-ai-badge">AI</span>
</div>
''', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Video Intelligence</div>', unsafe_allow_html=True)
    st.markdown("---")

    analytics = st.session_state.analytics

    st.markdown('<span class="badge badge-green">Analytics</span>', unsafe_allow_html=True)
    st.metric("Videos Processed", analytics["videos_processed"])
    st.metric("Questions Asked", analytics["questions_asked"])

    if analytics["pipeline_times"]:
        st.metric(
            "Avg Pipeline Time",
            f"{sum(analytics['pipeline_times']) / len(analytics['pipeline_times']):.1f}s"
        )

    if analytics["question_times"]:
        st.metric(
            "Avg Answer Time",
            f"{sum(analytics['question_times']) / len(analytics['question_times']):.2f}s"
        )

    if analytics["avg_rerank_scores"]:
        st.metric(
            "Avg Rerank Score",
            f"{sum(analytics['avg_rerank_scores']) / len(analytics['avg_rerank_scores']):.3f}"
        )

    if analytics["questions_asked"]:
        pct = (analytics["web_search_count"] / analytics["questions_asked"]) * 100
        st.metric("Web Search Usage", f"{pct:.1f}%")

    st.markdown("---")

    # Input controls removed here, keep analytics and pipeline status only.
    if st.session_state.pipeline_done:
        st.markdown("---")
        st.markdown('<span class="badge badge-green">Pipeline Status</span>', unsafe_allow_html=True)
        for step, icon, label in [
            ("audio",      "🔊", "Audio Processing"),
            ("transcript", "📝", "Transcription"),
            ("title",      "🏷️", "Title Generation"),
            ("summary",    "📋", "Summarisation"),
            ("extract",    "🔍", "Extraction"),
            ("rag",        "🧠", "RAG Engine"),
        ]:
            render_step_bar(label, step, icon)

# ─── Main Area ──────────────────────────────────────────────────────────────────
st.markdown('''
<div class="hero-title">
    <span>InsightForge</span>
    <span class="hero-ai-badge">AI</span>
</div>
''', unsafe_allow_html=True)
st.markdown('<div style="color:var(--accent-2);font-size:0.78rem;font-weight:700;letter-spacing:0.18em;text-transform:uppercase;margin-top:0.25rem">Video Intelligence Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">GPT-4o Transcription • Hybrid Retrieval • BM25 • RRF • Cohere Reranking • Web Search</div>', unsafe_allow_html=True)
st.markdown("---")

# --- Insert input controls in main area after hero
input_c1, input_c2, input_c3 = st.columns([6,2,2])

with input_c1:
    source = st.text_input(
        "Video Source",
        placeholder="Paste YouTube URL or local video path",
        label_visibility="collapsed"
    )

with input_c2:
    language = st.selectbox(
        "Language",
        ["English", "Hinglish"],
        label_visibility="collapsed"
    )

with input_c3:
    run_btn = st.button("⚡ Analyse", use_container_width=True)

st.markdown("---")

# ── Run Pipeline ────────────────────────────────────────────────────────────────
if run_btn:
    if not source.strip():
        st.error("Please enter a YouTube URL or file path.")
    else:
        st.session_state.pipeline_done = False
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.pipeline_steps = {}

        progress_placeholder = st.empty()

        def update_step(key, state):
            st.session_state.pipeline_steps[key] = state

        pipeline_start = time.perf_counter()
        try:
            with progress_placeholder.container():
                st.info("⚙️ InsightForge AI is processing your video. Running transcription, extraction, hybrid retrieval indexing, reranking setup and web-search augmentation...")

            update_step("audio", "active")
            chunks = process_input(source)
            update_step("audio", "done")

            update_step("transcript", "active")
            transcription_start = time.perf_counter()
            transcript = transcribe_all(chunks, language)
            transcription_elapsed = time.perf_counter() - transcription_start
            st.session_state.analytics["transcription_times"].append(transcription_elapsed)
            update_step("transcript", "done")

            update_step("title", "active")
            title = generate_title(transcript)
            update_step("title", "done")

            update_step("summary", "active")
            summary = summarize(transcript)
            update_step("summary", "done")

            update_step("extract", "active")
            action_items  = extract_action_items(transcript)
            decisions     = extract_key_decisions(transcript)
            questions     = extract_questions(transcript)
            update_step("extract", "done")

            update_step("rag", "active")
            rag_chain = build_rag_chain(transcript)
            update_step("rag", "done")

            pipeline_elapsed = time.perf_counter() - pipeline_start
            st.session_state.analytics["videos_processed"] += 1
            st.session_state.analytics["pipeline_times"].append(pipeline_elapsed)

            st.session_state.result = {
                "title": title,
                "transcript": transcript,
                "summary": summary,
                "action_items": action_items,
                "key_decisions": decisions,
                "open_questions": questions,
                "rag_chain": rag_chain,
            }
            st.session_state.pipeline_done = True
            progress_placeholder.success("✅ Analysis complete!")
            time.sleep(0.5)
            progress_placeholder.empty()
            st.rerun()

        except Exception as e:
            for k in ["audio","transcript","title","summary","extract","rag"]:
                if st.session_state.pipeline_steps.get(k) == "active":
                    st.session_state.pipeline_steps[k] = "pending"
            progress_placeholder.error(f"❌ Error: {e}")

# ── Results ──────────────────────────────────────────────────────────────────────
if st.session_state.result:
    r = st.session_state.result

    # Title banner
    st.markdown(f"""
    <div class="card">
        <div class="card-title">📌 Video Title</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:700;color:var(--text)">
            {r['title']}
        </div>
    </div>""", unsafe_allow_html=True)

    # KPI strip
    transcript_words = len(r["transcript"].split())
    pipeline_runtime = st.session_state.analytics["pipeline_times"][-1] if st.session_state.analytics["pipeline_times"] else 0
    chunk_count = st.session_state.get("chunk_count", "—")
    if chunk_count is None:
        chunk_count = "—"
    retrieval_method = "Hybrid RAG"

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Transcript Length", f"{transcript_words:,} words")
    with k2:
        st.metric("Chunks Created", chunk_count)
    with k3:
        st.metric("Processing Time", f"{pipeline_runtime:.2f}s")
    with k4:
        st.metric("Retrieval Method", retrieval_method)
        st.caption("Semantic • BM25 • RRF")

    st.markdown("---")

    chat_area, panel_area = st.columns([3,1], gap="large")

    with chat_area:
        st.markdown('<div style="font-family:\'Syne\',sans-serif;font-size:1.2rem;font-weight:700;margin-bottom:1rem">💬 Ask InsightForge AI</div>', unsafe_allow_html=True)
        chat_container = st.container()

    with panel_area:
        st.markdown("### Knowledge Panels")

        if "active_panel" not in st.session_state:
            st.session_state.active_panel = None

        if st.button("📊 Summary", use_container_width=True):
            st.session_state.active_panel = "summary"
            st.rerun()

        if st.button("📝 Transcript", use_container_width=True):
            st.session_state.active_panel = "transcript"
            st.rerun()

        if st.button("❓ Questions", use_container_width=True):
            st.session_state.active_panel = "questions"
            st.rerun()

        if st.button("📈 Analytics", use_container_width=True):
            st.session_state.active_panel = "analytics"
            st.rerun()

    if st.session_state.get("active_panel"):
        st.markdown("---")

        st.markdown('<div id="summary-section" class="section-anchor"></div>', unsafe_allow_html=True)
        if st.session_state.active_panel == "summary":
            st.markdown("## 📊 Executive Summary")
            st.markdown(f"""
            <div class='card'>
                <div class='card-title'>📋 Executive Summary</div>
                <div class='card-content'>{r['summary']}</div>
            </div>
            """, unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown(f"""
                <div class='card'>
                    <div class='card-title'>✅ Action Items</div>
                    <div class='card-content'>{r['action_items']}</div>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div class='card'>
                    <div class='card-title'>🔑 Key Decisions</div>
                    <div class='card-content'>{r['key_decisions']}</div>
                </div>
                """, unsafe_allow_html=True)

            with c3:
                st.markdown(f"""
                <div class='card'>
                    <div class='card-title'>❓ Open Questions</div>
                    <div class='card-content'>{r['open_questions']}</div>
                </div>
                """, unsafe_allow_html=True)

        elif st.session_state.active_panel == "transcript":
            st.markdown('<div id="transcript-section" class="section-anchor"></div>', unsafe_allow_html=True)
            st.markdown("## 📝 Full Transcript")
            st.markdown(f'<div class="transcript-box">{r["transcript"]}</div>', unsafe_allow_html=True)

        elif st.session_state.active_panel == "questions":
            st.markdown('<div id="questions-section" class="section-anchor"></div>', unsafe_allow_html=True)
            st.markdown("## ❓ Extracted Questions")
            st.markdown(f"""
            <div class='card'>
                <div class='card-title'>❓ Extracted Questions</div>
                <div class='card-content'>{r['open_questions']}</div>
            </div>
            """, unsafe_allow_html=True)


        elif st.session_state.active_panel == "analytics":
            st.markdown('<div id="analytics-section" class="section-anchor"></div>', unsafe_allow_html=True)
            st.markdown("## 📈 Analytics Overview")
            st.info("Detailed analytics dashboard is shown below.")

    # ── RAG Chat ──────────────────────────────────────────────────────────────

    # Chat history display
    with chat_container:
        if st.session_state.chat_history:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-container" style="margin-bottom:0.5rem">
                        <div class="chat-msg" style="align-items:flex-end">
                            <span class="chat-label user-label">You</span>
                            <div class="chat-bubble user-bubble">{msg['content']}</div>
                        </div>
                    </div>""", unsafe_allow_html=True)
                else:
                    tab1, tab2, tab3 = st.tabs([
                        "🤖 Answer",
                        "🔍 Transcript Sources",
                        "🌐 Web Sources"
                    ])

                    with tab1:
                        st.markdown(f"""
                        <div class="chat-container" style="margin-bottom:0.25rem">
                            <div class="chat-msg" style="align-items:flex-start">
                                <span class="chat-label bot-label">🤖 Assistant</span>
                                <div class="chat-bubble bot-bubble">{msg['content']}</div>
                            </div>
                        </div>""", unsafe_allow_html=True)

                        web_answer = msg.get("web_answer")
                        if web_answer:
                            st.markdown(f"""
                            <div class="chat-container" style="margin-bottom:0.25rem">
                                <div class="chat-msg" style="align-items:flex-start">
                                    <span class="chat-label bot-label">🌐 Web Research</span>
                                    <div class="chat-bubble bot-bubble">{web_answer}</div>
                                </div>
                            </div>""", unsafe_allow_html=True)

                    with tab2:
                        sources = msg.get("sources")

                        if sources:
                            for i, src in enumerate(sources):
                                found_by = src["found_by"]
                                rrf_score = src.get("rrf_score")
                                rerank_score = src.get("rerank_score")
                                chunk_idx = src["chunk_index"]

                                if found_by == "both":
                                    badge_class = "badge-green"
                                elif found_by == "bm25":
                                    badge_class = "badge-cyan"
                                else:
                                    badge_class = "badge-purple"

                                score_str = f"{rrf_score:.4f}" if rrf_score is not None else "—"
                                rerank_str = f"{rerank_score:.4f}" if rerank_score is not None else "—"

                                st.markdown(f"""
                                <div class="card" style="margin-bottom:0.5rem">
                                    <div class="card-title">
                                        Chunk #{chunk_idx}
                                        &nbsp;<span class="badge {badge_class}">{found_by}</span>
                                        &nbsp;<span style="color:var(--text-muted);font-size:0.65rem">
                                            RRF {score_str}
                                        </span>
                                        &nbsp;<span style="color:var(--success);font-size:0.65rem">
                                            Rerank {rerank_str}
                                        </span>
                                    </div>
                                    <div class="card-content" style="font-size:0.8rem;color:var(--text-muted)">
                                        {src['text']}
                                    </div>
                                </div>""", unsafe_allow_html=True)
                        else:
                            st.info("No transcript sources were used.")

                    with tab3:
                        web_sources = msg.get("web_sources")

                        if web_sources:
                            for src in web_sources:
                                title = src.get("title", "Untitled")
                                url = src.get("url", "")
                                st.markdown(f"- [{title}]({url})")
                        else:
                            st.info("No web sources available.")
        else:
            st.markdown("""
            <div class="card" style="text-align:center;padding:2rem">
                <div style="font-size:2rem;margin-bottom:0.5rem">💬</div>
                <div style="color:var(--text-muted);font-size:0.85rem">Ask anything about your video transcript</div>
            </div>""", unsafe_allow_html=True)

        # Chat input
        chat_col1, chat_col2 = st.columns([5, 1], gap="small")
        with chat_col1:
            user_input = st.text_input("Your question", placeholder="What were the main decisions made?", label_visibility="collapsed")
        with chat_col2:
            send_btn = st.button("Send →", use_container_width=True)

        if send_btn and user_input.strip():
            question_start = time.perf_counter()

            with st.spinner("Thinking…"):
                result = ask_question(
                    r["rag_chain"],
                    user_input.strip()
                )

            question_elapsed = time.perf_counter() - question_start
            st.session_state.analytics["questions_asked"] += 1
            st.session_state.analytics["question_times"].append(question_elapsed)

            st.session_state.chat_history.append(
                {
                    "role": "user",
                    "content": user_input.strip(),
                }
            )

            if result["mode"] == "web_only":
                st.session_state.analytics["web_search_count"] += 1
            else:
                transcript_sources = result.get("transcript_sources", [])
                scores = [
                    s.get("rerank_score")
                    for s in transcript_sources
                    if s.get("rerank_score") is not None
                ]
                if scores:
                    st.session_state.analytics["avg_rerank_scores"].append(
                        sum(scores) / len(scores)
                    )
            if result["mode"] == "web_only":
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": "Answer not found in transcript.",
                        "web_answer": result["web_answer"],
                        "sources": [],
                        "web_sources": result["web_sources"],
                    }
                )
            else:
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": result["rag_answer"],
                        "web_answer": result["web_answer"],
                        "sources": result["transcript_sources"],
                        "web_sources": result["web_sources"],
                    }
                )

            st.rerun()

        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat", type="secondary"):
                st.session_state.chat_history = []
                st.rerun()

    if st.session_state.get("active_panel") == "summary":
        st.markdown("""
<script>
setTimeout(function(){
    const el = document.getElementById('summary-section');
    if(el){el.scrollIntoView({behavior:'smooth', block:'start'});}
},100);
</script>
""", unsafe_allow_html=True)

    elif st.session_state.get("active_panel") == "transcript":
        st.markdown("""
<script>
setTimeout(function(){
    const el = document.getElementById('transcript-section');
    if(el){el.scrollIntoView({behavior:'smooth', block:'start'});}
},100);
</script>
""", unsafe_allow_html=True)

    elif st.session_state.get("active_panel") == "questions":
        st.markdown("""
<script>
setTimeout(function(){
    const el = document.getElementById('questions-section');
    if(el){el.scrollIntoView({behavior:'smooth', block:'start'});}
},100);
</script>
""", unsafe_allow_html=True)

    elif st.session_state.get("active_panel") == "analytics":
        st.markdown("""
<script>
setTimeout(function(){
    const el = document.getElementById('analytics-section');
    if(el){el.scrollIntoView({behavior:'smooth', block:'start'});}
},100);
</script>
""", unsafe_allow_html=True)

    if st.session_state.get("active_panel") == "analytics":
        st.markdown("---")
        st.markdown("## 📊 Analytics Dashboard")

        analytics = st.session_state.analytics

        avg_pipeline = (
            sum(analytics["pipeline_times"]) / len(analytics["pipeline_times"])
            if analytics["pipeline_times"]
            else 0
        )

        avg_question = (
            sum(analytics["question_times"]) / len(analytics["question_times"])
            if analytics["question_times"]
            else 0
        )

        avg_transcription = (
            sum(analytics["transcription_times"])
            / len(analytics["transcription_times"])
            if analytics["transcription_times"]
            else 0
        )

        avg_rerank = (
            sum(analytics["avg_rerank_scores"])
            / len(analytics["avg_rerank_scores"])
            if analytics["avg_rerank_scores"]
            else 0
        )

        web_pct = (
            analytics["web_search_count"]
            / analytics["questions_asked"]
            * 100
            if analytics["questions_asked"]
            else 0
        )

        k1, k2, k3, k4 = st.columns(4)

        with k1:
            st.metric("Videos Processed", analytics["videos_processed"])

        with k2:
            st.metric("Questions Asked", analytics["questions_asked"])

        with k3:
            st.metric("Avg Pipeline", f"{avg_pipeline:.2f}s")

        with k4:
            st.metric("Avg Answer", f"{avg_question:.2f}s")

        st.info(
            "This dashboard tracks transcription performance, retrieval quality, answer latency, and web-search fallback behaviour across the current session."
        )

        chart1, chart2 = st.columns(2)

        with chart1:
            st.subheader("⚡ Pipeline Processing Time")
            st.caption("Total time to process a video and build the RAG pipeline.")

            if analytics["pipeline_times"]:
                df_pipeline = pd.DataFrame({
                    "Run": range(1, len(analytics["pipeline_times"]) + 1),
                    "Seconds": analytics["pipeline_times"],
                }).set_index("Run")

                st.line_chart(df_pipeline)
            else:
                st.info("No pipeline runs recorded yet.")

        with chart2:
            st.subheader("💬 Question Latency")
            st.caption("End-to-end response time for user questions.")

            if analytics["question_times"]:
                df_question = pd.DataFrame({
                    "Question": range(1, len(analytics["question_times"]) + 1),
                    "Seconds": analytics["question_times"],
                }).set_index("Question")

                st.line_chart(df_question)
            else:
                st.info("No questions asked yet.")

        chart3, chart4 = st.columns(2)

        with chart3:
            st.subheader("🎯 Reranker Quality")
            st.caption("Average Cohere rerank score for retrieved chunks.")

            if analytics["avg_rerank_scores"]:
                df_rerank = pd.DataFrame({
                    "Query": range(1, len(analytics["avg_rerank_scores"]) + 1),
                    "Score": analytics["avg_rerank_scores"],
                }).set_index("Query")

                st.line_chart(df_rerank)
            else:
                st.info("No rerank data available yet.")

        with chart4:
            st.subheader("🌐 Retrieval Source Distribution")
            st.caption("Transcript answers versus web-search fallbacks.")

            if analytics["questions_asked"]:
                transcript_count = max(
                    analytics["questions_asked"] - analytics["web_search_count"],
                    0,
                )

                df_sources = pd.DataFrame(
                    {"Count": [transcript_count, analytics["web_search_count"]]},
                    index=["Transcript", "Web Search"],
                )

                st.bar_chart(df_sources)
            else:
                st.info("No query activity recorded yet.")

        with st.expander("📈 Detailed Metrics", expanded=False):
            detail_df = pd.DataFrame(
                [
                    ["Videos Processed", analytics["videos_processed"]],
                    ["Questions Asked", analytics["questions_asked"]],
                    ["Average Transcription Time", f"{avg_transcription:.2f}s"],
                    ["Average Pipeline Time", f"{avg_pipeline:.2f}s"],
                    ["Average Question Time", f"{avg_question:.2f}s"],
                    ["Average Rerank Score", f"{avg_rerank:.4f}"],
                    ["Web Search Usage", f"{web_pct:.1f}%"],
                ],
                columns=["Metric", "Value"],
            )

            st.dataframe(detail_df, use_container_width=True)

else:
    # Footer before empty state
    st.markdown("""
    <div style="text-align:center;margin-top:2rem;margin-bottom:2rem;">
        <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;color:var(--accent-glow);margin-bottom:0.2rem;">
            InsightForge <span class="hero-ai-badge">AI</span>
        </div>
        <div style="color:var(--text-muted);font-size:0.9rem;">
            GPT-4o Mini Transcribe • BM25 • RRF • Cohere Rerank • Web Search Augmentation
        </div>
    </div>
    """, unsafe_allow_html=True)
    # Empty state
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:5rem 2rem;text-align:center">
        <div style="font-size:4rem;margin-bottom:1rem">🎬</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:700;color:var(--text);margin-bottom:0.5rem">
            InsightForge <span class="hero-ai-badge">AI</span>
        </div>
        <div style="color:var(--text-muted);font-size:0.85rem;max-width:480px;line-height:1.7">
            GPT-4o transcription, hybrid retrieval (semantic + BM25), Reciprocal Rank Fusion, Cohere reranking, and web-augmented question answering for your videos.<br>
            Paste a YouTube URL or local file path above, choose your language, and hit <strong>Analyse</strong> to get started.
        </div>
        <div style="margin-top:2rem;display:flex;gap:1rem;flex-wrap:wrap;justify-content:center">
            <span class="badge badge-purple">GPT-4o Transcription</span>
            <span class="badge badge-cyan">Hybrid Retrieval · BM25 · RRF</span>
            <span class="badge badge-green">Cohere Rerank · Web QA</span>
        </div>
    </div>""", unsafe_allow_html=True)