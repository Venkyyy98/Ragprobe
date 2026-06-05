"""
dashboard.py — ragprobe Evaluation Dashboard

Run:
    pip install streamlit pandas
    streamlit run dashboard.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ragprobe Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styling ───────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.header-band {
    background: linear-gradient(120deg, #0a192f 0%, #112240 60%, #1a3a5c 100%);
    border: 1px solid #1e4d7b;
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.header-band::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(100,180,255,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.header-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    color: #e6f1ff;
    margin: 0;
    letter-spacing: -0.5px;
}
.header-sub {
    color: #7aadce;
    font-size: 0.9rem;
    margin: 0.4rem 0 0;
    font-weight: 300;
    letter-spacing: 0.3px;
}
.header-badge {
    display: inline-block;
    background: rgba(100,180,255,0.12);
    border: 1px solid rgba(100,180,255,0.25);
    color: #7aadce;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    margin-top: 0.8rem;
}

.verdict-pass {
    background: linear-gradient(135deg, #0d3320, #0a4a2a);
    border: 1px solid #2a7a4a;
    color: #4dbb7a;
    padding: 1.2rem 1.5rem;
    border-radius: 10px;
    text-align: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1rem;
    font-weight: 600;
    letter-spacing: 1px;
}
.verdict-reject {
    background: linear-gradient(135deg, #330d0d, #4a0a0a);
    border: 1px solid #7a2a2a;
    color: #e05c5c;
    padding: 1.2rem 1.5rem;
    border-radius: 10px;
    text-align: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1rem;
    font-weight: 600;
    letter-spacing: 1px;
}
.verdict-label {
    font-size: 0.72rem;
    opacity: 0.7;
    font-weight: 400;
    display: block;
    margin-bottom: 0.3rem;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.score-card {
    background: #0d1b2e;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1.2rem;
    text-align: center;
}
.score-label {
    font-size: 0.72rem;
    color: #6a90b0;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 0.5rem;
    font-family: 'IBM Plex Mono', monospace;
}
.score-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    color: #e6f1ff;
}
.score-delta-up   { color: #4dbb7a; font-size: 0.85rem; margin-top: 0.3rem; }
.score-delta-flat { color: #7aadce; font-size: 0.85rem; margin-top: 0.3rem; }
.score-delta-down { color: #e05c5c; font-size: 0.85rem; margin-top: 0.3rem; }

.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #4a90c4;
    text-transform: uppercase;
    letter-spacing: 2px;
    padding: 0.4rem 0;
    border-bottom: 1px solid #1e3a5f;
    margin: 1.5rem 0 1rem;
}

div[data-testid="stTabs"] button { font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; }
</style>
""", unsafe_allow_html=True)

# ── DB helpers ────────────────────────────────────────────────────────────────

try:
    from ragprobe.config import DB_PATH, FAITHFULNESS_THRESHOLD
except Exception:
    from pathlib import Path as _P
    DB_PATH = _P.home() / ".ragprobe" / "ragprobe.db"
    FAITHFULNESS_THRESHOLD = 0.75

THRESHOLD = float(FAITHFULNESS_THRESHOLD)


@st.cache_data(ttl=10)
def get_sessions() -> pd.DataFrame:
    if not Path(DB_PATH).exists():
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql(
            "SELECT * FROM sessions ORDER BY run_timestamp DESC", conn
        )


@st.cache_data(ttl=10)
def get_probes(session_id: str) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql(
            "SELECT * FROM probe_results WHERE session_id = ? ORDER BY id",
            conn, params=[session_id],
        )


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="header-band">
    <p class="header-title">⬡ ragprobe</p>
    <p class="header-sub">Adversarial Evaluation Dashboard · SEC 10-K Financial Document Q&A</p>
    <span class="header-badge">FE 524-B · Stevens Institute of Technology · Spring 2026</span>
</div>
""", unsafe_allow_html=True)

# ── Load sessions ─────────────────────────────────────────────────────────────

sessions = get_sessions()

if sessions.empty:
    st.info("No sessions in the database yet. Run `ragprobe run --mode baseline` to get started.")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_compare, tab_history, tab_probes = st.tabs([
    "📊  Comparison", "📋  Session History", "🔎  Probe Details"
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — COMPARISON
# ════════════════════════════════════════════════════════════════════════════

with tab_compare:
    baseline_rows = sessions[sessions["pipeline_mode"] == "baseline"]
    hardened_rows = sessions[sessions["pipeline_mode"] == "hardened"]

    if baseline_rows.empty or hardened_rows.empty:
        st.warning("Need at least one baseline **and** one hardened session to compare.")
        st.stop()

    # Session selectors
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        b_id = st.selectbox(
            "Baseline session",
            baseline_rows["session_id"].tolist(),
            format_func=lambda x: f"{x[:12]}…  ·  {baseline_rows[baseline_rows['session_id']==x]['run_timestamp'].values[0][:16]}",
        )
    with col_sel2:
        h_id = st.selectbox(
            "Hardened session",
            hardened_rows["session_id"].tolist(),
            format_func=lambda x: f"{x[:12]}…  ·  {hardened_rows[hardened_rows['session_id']==x]['run_timestamp'].values[0][:16]}",
        )

    b = baseline_rows[baseline_rows["session_id"] == b_id].iloc[0]
    h = hardened_rows[hardened_rows["session_id"] == h_id].iloc[0]

    bf = b["mean_faithfulness"] or 0.0
    hf = h["mean_faithfulness"] or 0.0
    br = b["mean_relevance"]    or 0.0
    hr = h["mean_relevance"]    or 0.0
    bc = b["mean_context_recall"] or 0.0
    hc = h["mean_context_recall"] or 0.0

    # ── Verdict banners ───────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Pipeline Verdict</p>', unsafe_allow_html=True)
    v1, v2 = st.columns(2)

    def verdict_html(label, mode, faith, threshold):
        passing = faith >= threshold
        cls  = "verdict-pass" if passing else "verdict-reject"
        icon = "✓ PASS" if passing else "✗ REJECT"
        return f"""
        <div class="{cls}">
            <span class="verdict-label">{mode} pipeline</span>
            {icon}<br>
            <span style="font-size:0.78rem;opacity:0.7;font-weight:400;">
                faithfulness {faith:.3f} / threshold {threshold:.2f}
            </span>
        </div>"""

    with v1:
        st.markdown(verdict_html("Baseline", "BASELINE", bf, THRESHOLD), unsafe_allow_html=True)
    with v2:
        st.markdown(verdict_html("Hardened", "HARDENED", hf, THRESHOLD), unsafe_allow_html=True)

    # ── Score cards ───────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Hardened Pipeline Scores  (vs Baseline)</p>', unsafe_allow_html=True)

    def delta_html(delta):
        if delta > 0.005:
            return f'<p class="score-delta-up">▲ +{delta:.3f} vs baseline</p>'
        elif delta < -0.005:
            return f'<p class="score-delta-down">▼ {delta:.3f} vs baseline</p>'
        else:
            return f'<p class="score-delta-flat">→ no change</p>'

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="score-card">
            <p class="score-label">Faithfulness</p>
            <p class="score-value">{hf:.3f}</p>
            {delta_html(hf - bf)}
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="score-card">
            <p class="score-label">Relevance</p>
            <p class="score-value">{hr:.3f}</p>
            {delta_html(hr - br)}
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="score-card">
            <p class="score-label">Context Recall</p>
            <p class="score-value">{hc:.3f}</p>
            {delta_html(hc - bc)}
        </div>""", unsafe_allow_html=True)

    # ── Bar chart ─────────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Score Breakdown</p>', unsafe_allow_html=True)

    chart_df = pd.DataFrame({
        "Metric": ["Faithfulness", "Relevance", "Context Recall"],
        "Baseline": [bf, br, bc],
        "Hardened": [hf, hr, hc],
    }).set_index("Metric")

    st.bar_chart(chart_df, height=280, color=["#2a5f8a", "#4dbb7a"])

    # ── Per-category failure rates ────────────────────────────────────────────
    st.markdown('<p class="section-header">Per-Category Failure Rates  (faithfulness &lt; 0.75)</p>', unsafe_allow_html=True)

    try:
        b_probes = get_probes(b_id)
        h_probes = get_probes(h_id)

        if not b_probes.empty and not h_probes.empty:
            cats = sorted(set(b_probes["prompt_category"].dropna()) |
                          set(h_probes["prompt_category"].dropna()))
            rows = []
            for cat in cats:
                bc_rows = b_probes[b_probes["prompt_category"] == cat]
                hc_rows = h_probes[h_probes["prompt_category"] == cat]
                b_fail  = (bc_rows["faithfulness"].fillna(0) < THRESHOLD).mean() * 100 if len(bc_rows) else 0
                h_fail  = (hc_rows["faithfulness"].fillna(0) < THRESHOLD).mean() * 100 if len(hc_rows) else 0
                rows.append({
                    "Category":         cat.replace("_", " ").title(),
                    "Baseline Fail %":  round(b_fail, 1),
                    "Hardened Fail %":  round(h_fail, 1),
                    "Δ Improvement":    round(b_fail - h_fail, 1),
                    "Baseline n":       len(bc_rows),
                    "Hardened n":       len(hc_rows),
                })

            cat_df = pd.DataFrame(rows).set_index("Category")
            st.dataframe(cat_df, use_container_width=True)
        else:
            st.info("Probe-level data not available for per-category breakdown.")
    except Exception as e:
        st.caption(f"Per-category data unavailable: {e}")

    # ── Safety flags ──────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Safety Flags</p>', unsafe_allow_html=True)

    try:
        b_probes = b_probes if "b_probes" in dir() else get_probes(b_id)
        h_probes = h_probes if "h_probes" in dir() else get_probes(h_id)

        safety_data = {
            "Flag": ["Injection Compliance", "Confidentiality Violation", "Refusal Evasion"],
            "Baseline": [
                int(b_probes["injection_compliance"].sum()),
                int(b_probes["confidentiality_violation"].sum()),
                int(b_probes["refusal_evasion"].sum()),
            ],
            "Hardened": [
                int(h_probes["injection_compliance"].sum()),
                int(h_probes["confidentiality_violation"].sum()),
                int(h_probes["refusal_evasion"].sum()),
            ],
        }
        st.dataframe(
            pd.DataFrame(safety_data).set_index("Flag"),
            use_container_width=True,
        )
    except Exception:
        st.caption("Safety flag data not available.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — SESSION HISTORY
# ════════════════════════════════════════════════════════════════════════════

with tab_history:
    st.markdown('<p class="section-header">All Probe Sessions</p>', unsafe_allow_html=True)

    disp = sessions[[
        "session_id", "run_timestamp", "pipeline_mode",
        "total_probes", "mean_faithfulness", "mean_relevance", "mean_context_recall"
    ]].copy()

    disp.columns = [
        "Session ID", "Timestamp", "Mode",
        "Probes", "Faithfulness", "Relevance", "Context Recall"
    ]

    disp["Verdict"] = disp["Faithfulness"].apply(
        lambda x: "✅ PASS" if (x or 0) >= THRESHOLD else "❌ REJECT"
    )

    st.dataframe(disp, use_container_width=True, hide_index=True)

    # Stat summary
    st.markdown('<p class="section-header">Summary</p>', unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    with s1:
        st.metric("Total Sessions", len(sessions))
    with s2:
        st.metric("Baseline Runs", len(sessions[sessions["pipeline_mode"] == "baseline"]))
    with s3:
        st.metric("Hardened Runs", len(sessions[sessions["pipeline_mode"] == "hardened"]))


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — PROBE DETAILS
# ════════════════════════════════════════════════════════════════════════════

with tab_probes:
    st.markdown('<p class="section-header">Individual Probe Results</p>', unsafe_allow_html=True)

    sel_id = st.selectbox(
        "Select session",
        sessions["session_id"].tolist(),
        format_func=lambda x: (
            f"{sessions[sessions['session_id']==x]['pipeline_mode'].values[0].upper()}"
            f"  ·  {sessions[sessions['session_id']==x]['run_timestamp'].values[0][:16]}"
            f"  ·  {x[:12]}…"
        ),
        key="probe_session_select",
    )

    probes = get_probes(sel_id)

    if probes.empty:
        st.info("No probe results found for this session.")
    else:
        cats = ["All"] + sorted(probes["prompt_category"].dropna().unique().tolist())
        cat_sel = st.selectbox("Filter by category", cats, key="cat_filter")

        filtered = probes if cat_sel == "All" else probes[probes["prompt_category"] == cat_sel]

        cols = ["prompt_category", "prompt_text", "faithfulness",
                "relevance", "context_recall", "injection_compliance",
                "confidentiality_violation", "refusal_evasion"]
        cols = [c for c in cols if c in filtered.columns]

        st.dataframe(
            filtered[cols].rename(columns={
                "prompt_category": "Category",
                "prompt_text": "Prompt",
                "faithfulness": "Faith",
                "relevance": "Rel",
                "context_recall": "Ctx Recall",
                "injection_compliance": "Injection",
                "confidentiality_violation": "Confidentiality",
                "refusal_evasion": "Refusal",
            }),
            use_container_width=True,
            hide_index=True,
        )

        # mini stats
        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Probes shown", len(filtered))
        with m2:
            st.metric("Mean Faithfulness", f"{filtered['faithfulness'].mean():.3f}")
        with m3:
            st.metric("Mean Relevance", f"{filtered['relevance'].mean():.3f}")
        with m4:
            fail_pct = (filtered["faithfulness"].fillna(0) < THRESHOLD).mean() * 100
            st.metric("Failure Rate", f"{fail_pct:.1f}%")

# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption("ragprobe · Adversarial evaluation of RAG pipelines · FE 524-B · Stevens Institute of Technology")
