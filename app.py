"""Resume Screening System — Streamlit Web App

Layout:
  Sidebar      : Resume source (samples / upload) + Run button
  Working area : [📝 Job Description] [📊 Ranking & Stats ◀default]
                 [🔍 Entities] [📈 Chart]
"""

import sys
import os
import base64
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.parsers.resume_parser import parse_resume, parse_uploaded_file
from src.ner.extractor import extract
from src.matching.similarity import rank_candidates

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Resume Screening System",
    page_icon="📄",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .block-container { padding-top: 3.5rem; padding-bottom: 1rem; }

  /* ── Profile badge — top right ── */
  #profile-badge {
    position: fixed; top: 3.5rem; right: 1rem;
    z-index: 2147483647;
    display: flex; align-items: center; gap: 0.45rem;
    pointer-events: auto;
  }
  #profile-badge img {
    width: 38px; height: 38px; border-radius: 50%;
    object-fit: cover; border: 2px solid #4a7fc1; flex-shrink: 0;
  }
  #profile-badge a.li-btn {
    display: inline-flex; align-items: center; gap: 0.35rem;
    background: #3b6fc4; color: white !important; text-decoration: none;
    padding: 0.28rem 0.75rem; border-radius: 8px; font-size: 0.82rem;
    font-weight: 600; line-height: 1; font-family: sans-serif;
  }
  #profile-badge a.li-btn:hover { background: #2d57a8; }
  hr { margin: 0.3rem 0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Profile badge ─────────────────────────────────────────────────────────────
_profile_path = ROOT / "assets" / "profile.png"
_profile_b64  = ""
if _profile_path.exists():
    with open(_profile_path, "rb") as _f:
        _profile_b64 = base64.b64encode(_f.read()).decode()

_li_svg = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" '
    'fill="white" viewBox="0 0 24 24" style="flex-shrink:0">'
    '<path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14'
    "c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11z"
    "m-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764"
    " 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604"
    "c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777"
    ' 7 2.476v6.759z"/></svg>'
)

_img_html = (
    f'<img src="data:image/png;base64,{_profile_b64}" alt="profile" />'
    if _profile_b64 else ""
)

st.markdown(
    f"""<div id="profile-badge">
  {_img_html}
  <a class="li-btn" href="https://www.linkedin.com/in/yuttapong-m/" target="_blank">
    {_li_svg} yuttapong-m
  </a>
</div>""",
    unsafe_allow_html=True,
)

# ── auto-select Ranking tab on first load (JS injection) ─────────────────────
if "tab_initialized" not in st.session_state:
    st.session_state.tab_initialized = True
    components.html("""
    <script>
    (function() {
        function clickRankingTab() {
            var tabs = window.parent.document.querySelectorAll('button[role="tab"]');
            if (tabs && tabs.length >= 2) {
                tabs[1].click();   // index 1 = Ranking & Stats
            } else {
                setTimeout(clickRankingTab, 150);
            }
        }
        setTimeout(clickRankingTab, 400);
    })();
    </script>
    """, height=0)

# ── session state defaults ────────────────────────────────────────────────────
if "jd_text" not in st.session_state:
    st.session_state.jd_text = ""
if "ranked" not in st.session_state:
    st.session_state.ranked = []
if "using_sample" not in st.session_state:
    st.session_state.using_sample = True   # show sample on first open

# ── sample data (cached) ──────────────────────────────────────────────────────
@st.cache_data(show_spinner="🔄 Loading sample results…")
def load_sample_results() -> tuple[list[dict], str]:
    raw_dir = ROOT / "data" / "raw"
    jd_path = raw_dir / "job_description.txt"
    if not jd_path.exists():
        return [], ""
    jd_text  = jd_path.read_text(encoding="utf-8")
    files    = sorted(raw_dir.glob("resume_*.txt"))
    candidates = []
    for f in files:
        text     = parse_resume(f)
        entities = extract(text)
        candidates.append({"name": f.stem, "text": text, "entities": entities})
    ranked = rank_candidates(candidates, jd_text, top_n=10)
    return ranked, jd_text

# ── helpers ───────────────────────────────────────────────────────────────────
def score_badge(score: float) -> str:
    if score >= 0.6: return "🟢"
    if score >= 0.4: return "🟡"
    return "🔴"

def ranking_df(ranked: list[dict]) -> pd.DataFrame:
    rows = []
    for i, r in enumerate(ranked):
        e = r.get("entities", {})
        skills = e.get("skills", [])
        rows.append({
            "Rank":      i + 1,
            "Candidate": r["name"],
            "Score":     r["score"],
            "Grade":     score_badge(r["score"]),
            "Skills":    ", ".join(skills[:5]) + ("…" if len(skills) > 5 else ""),
            "Companies": ", ".join(e.get("companies", [])[:3]),
            "Dates":     ", ".join(e.get("dates", [])[:3]),
        })
    return pd.DataFrame(rows)

def score_chart(ranked: list[dict], title: str = "") -> plt.Figure:
    names  = [r["name"] for r in ranked]
    scores = [r["score"] for r in ranked]
    colors = ["#2ecc71" if s >= 0.6 else "#f39c12" if s >= 0.4 else "#e74c3c"
              for s in scores]
    fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.45)))
    bars = ax.barh(names, scores, color=colors)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_xlim(0, 1.08)
    ax.axvline(0.6, color="green",  ls="--", lw=1, alpha=0.5, label="Strong ≥0.6")
    ax.axvline(0.4, color="orange", ls="--", lw=1, alpha=0.5, label="Moderate ≥0.4")
    ax.set_xlabel("Cosine Similarity Score")
    if title: ax.set_title(title)
    ax.invert_yaxis()
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig

def hist_chart(scores: list[float]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(scores, bins=10, range=(0, 1), color="steelblue", edgecolor="white")
    ax.axvline(np.mean(scores), color="red", ls="--",
               label=f"Mean = {np.mean(scores):.3f}")
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("📄 Resume Screening")
    st.caption("NER + Sentence Transformers")
    st.divider()

    # ── resume source ─────────────────────────────────────────────────────────
    st.markdown("#### 📂 Resume Source")
    source = st.radio(
        "source",
        ["📊 Use sample resumes (10)", "📤 Upload files"],
        label_visibility="collapsed",
    )
    using_upload = source.startswith("📤")

    uploaded_files = []
    if using_upload:
        uploaded_files = st.file_uploader(
            "Drop resumes here (max 10)",
            type=["pdf", "docx", "txt", "csv"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if len(uploaded_files) > 10:
            st.warning("⚠️ Max 10 files — using first 10")
            uploaded_files = uploaded_files[:10]
        if uploaded_files:
            for f in uploaded_files:
                st.caption(f"• `{f.name}` ({f.size/1024:.1f} KB)")

    st.divider()

    # ── run button ────────────────────────────────────────────────────────────
    jd_ready      = bool(st.session_state.jd_text.strip())
    resume_ready  = (not using_upload) or bool(uploaded_files)
    can_run       = jd_ready and resume_ready

    if not jd_ready:
        st.caption("⚠️ Paste or upload a Job Description first")
    if using_upload and not uploaded_files:
        st.caption("⚠️ Upload at least one resume")

    run = st.button("🚀 Run Screening", type="primary",
                    disabled=not can_run, use_container_width=True)

    # load sample on first open
    if not using_upload and not st.session_state.ranked:
        sample_ranked, sample_jd = load_sample_results()
        st.session_state.ranked     = sample_ranked
        st.session_state.using_sample = True
        if not st.session_state.jd_text:
            st.session_state.jd_text = sample_jd

    # run pipeline on button click
    if run:
        jd = st.session_state.jd_text.strip()
        with st.spinner("Parsing…"):
            if using_upload:
                candidates = []
                for uf in uploaded_files:
                    try:
                        candidates.extend(parse_uploaded_file(uf))
                    except Exception as e:
                        st.error(f"`{uf.name}`: {e}")
            else:
                raw_dir = ROOT / "data" / "raw"
                candidates = [
                    {"name": f.stem, "text": parse_resume(f)}
                    for f in sorted(raw_dir.glob("resume_*.txt"))
                ]

        with st.spinner("Extracting entities…"):
            for c in candidates:
                if "entities" not in c:
                    c["entities"] = extract(c["text"])

        with st.spinner("Ranking…"):
            st.session_state.ranked       = rank_candidates(candidates, jd, top_n=10)
            st.session_state.using_sample = False

        st.success(f"✅ {len(st.session_state.ranked)} candidates ranked")

# ══════════════════════════════════════════════════════════════════════════════
# WORKING AREA — TABS
# ══════════════════════════════════════════════════════════════════════════════
ranked = st.session_state.ranked
label  = "Sample data" if st.session_state.using_sample else "Results"

tab_jd, tab_rank, tab_ent, tab_chart = st.tabs([
    "📝 Job Description",
    f"📊 Ranking & Stats — {label}",
    "🔍 Entities",
    "📈 Chart",
])

# ── TAB 1: Job Description ────────────────────────────────────────────────────
with tab_jd:
    st.markdown("#### Job Description")
    jd_mode = st.radio("Input mode", ["✏️ Paste text", "📎 Upload file (PDF / DOCX / TXT)"],
                       horizontal=True, label_visibility="collapsed")

    if jd_mode.startswith("✏️"):
        jd_input = st.text_area(
            "Paste JD here",
            value=st.session_state.jd_text,
            height=420,
            placeholder="We are looking for a Senior Data Scientist with strong Python and NLP skills…",
            label_visibility="collapsed",
        )
        st.session_state.jd_text = jd_input

    else:
        jd_file = st.file_uploader("Upload JD file", type=["pdf", "docx", "txt"],
                                   label_visibility="collapsed")
        if jd_file:
            records = parse_uploaded_file(jd_file)
            if records:
                st.session_state.jd_text = records[0]["text"]
                st.success(f"✅ Loaded: `{jd_file.name}`")

        if st.session_state.jd_text:
            st.text_area("Loaded text (read-only)", st.session_state.jd_text,
                         height=350, disabled=True, label_visibility="collapsed")

# ── TAB 2: Ranking & Stats ────────────────────────────────────────────────────
with tab_rank:
    if not ranked:
        st.info("No results yet — use the sample data or run screening from the sidebar.")
    else:
        scores = [r["score"] for r in ranked]
        c1, c2, c3 = st.columns(3)
        c1.metric("Candidates", len(ranked))
        c2.metric("Top Score",  f"{max(scores):.3f}")
        c3.metric("Avg Score",  f"{np.mean(scores):.3f}")

        df = ranking_df(ranked)
        st.dataframe(
            df.style.background_gradient(subset=["Score"], cmap="RdYlGn", vmin=0, vmax=1),
            use_container_width=True,
            hide_index=True,
            height=380,
        )

# ── TAB 3: Entities ───────────────────────────────────────────────────────────
with tab_ent:
    if not ranked:
        st.info("No results yet.")
    else:
        selected = st.selectbox(
            "Select candidate",
            options=[r["name"] for r in ranked],
            format_func=lambda n: f"#{[r['name'] for r in ranked].index(n)+1} · {n}",
        )
        r = next(r for r in ranked if r["name"] == selected)
        e = r.get("entities", {})
        st.metric("Similarity Score", f"{r['score']:.4f}", delta=score_badge(r["score"]))
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**🛠 Skills**")
            st.write(", ".join(e.get("skills", [])) or "—")
            st.markdown("**🏢 Companies**")
            st.write(", ".join(e.get("companies", [])) or "—")
        with col_b:
            st.markdown("**📅 Dates**")
            st.write(", ".join(e.get("dates", [])) or "—")
            st.markdown("**📍 Locations**")
            st.write(", ".join(e.get("locations", [])) or "—")

        st.divider()
        with st.expander("📄 Full resume text"):
            st.text(r.get("text", ""))

# ── TAB 4: Chart ──────────────────────────────────────────────────────────────
with tab_chart:
    if not ranked:
        st.info("No results yet.")
    else:
        scores = [r["score"] for r in ranked]
        col_l, col_r = st.columns([3, 2])
        with col_l:
            st.markdown("**Score per Candidate**")
            fig_bar = score_chart(ranked, "Resume–JD Similarity")
            st.pyplot(fig_bar)
            plt.close(fig_bar)
        with col_r:
            st.markdown("**Distribution**")
            fig_hist = hist_chart(scores)
            st.pyplot(fig_hist)
            plt.close(fig_hist)
