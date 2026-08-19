"""
SpectrumLens — Evidence Panel UI  (Command 4)
=============================================
Streamlit interface that surfaces the full retrieval pipeline to the clinician
BEFORE the LLM answer appears. The doctor audits the evidence first.

Run:
    streamlit run app.py
"""

import streamlit as st
import time
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ─── Page Config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SpectrumLens — ASD Clinical Decision Support",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Noto+Naskh+Arabic:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.rtl { direction: rtl; text-align: right; font-family: 'Noto Naskh Arabic', 'Inter', sans-serif; }
.lang-badge-ar { display:inline-block; background:rgba(52,211,153,0.15); border:1px solid rgba(52,211,153,0.4); color:#34d399; font-size:0.7rem; font-weight:700; padding:0.1rem 0.4rem; border-radius:8px; margin-left:0.4rem; }
.lang-badge-en { display:inline-block; background:rgba(126,207,255,0.12); border:1px solid rgba(126,207,255,0.3); color:#7ecfff; font-size:0.7rem; font-weight:700; padding:0.1rem 0.4rem; border-radius:8px; margin-left:0.4rem; }

.hero-banner {
    background: linear-gradient(135deg,#0f2027 0%,#203a43 50%,#2c5364 100%);
    border-radius:16px; padding:2rem 2.5rem; margin-bottom:1.5rem;
    border:1px solid rgba(255,255,255,0.08);
}
.hero-title { font-size:2rem; font-weight:700; color:#e2f4ff; margin:0; }
.hero-sub   { color:#7ecfff; font-size:0.95rem; margin-top:0.4rem; }

.evidence-card {
    background:#1a2332; border:1px solid rgba(126,207,255,0.18);
    border-left:4px solid #7ecfff; border-radius:12px;
    padding:1.2rem 1.5rem; margin-bottom:0.9rem;
}
.evidence-card.top-pick { border-left-color:#00c875; background:#13231a; }
.evidence-rank  { font-size:0.72rem; font-weight:700; color:#7ecfff; text-transform:uppercase; letter-spacing:1px; }
.evidence-title { font-size:0.95rem; font-weight:600; color:#e2f4ff; margin-top:0.35rem; }
.evidence-meta  { font-size:0.78rem; color:#8ba6c0; margin-top:0.2rem; }
.evidence-excerpt {
    font-size:0.84rem; color:#c0d8ec; margin-top:0.8rem; line-height:1.6;
    border-top:1px solid rgba(255,255,255,0.06); padding-top:0.7rem;
}
.score-pill {
    display:inline-block; background:rgba(126,207,255,0.12);
    border:1px solid rgba(126,207,255,0.3); color:#7ecfff;
    font-size:0.75rem; font-weight:600; padding:0.15rem 0.55rem;
    border-radius:12px; margin-left:0.4rem;
}
.score-pill.high   { background:rgba(0,200,117,0.12); border-color:rgba(0,200,117,0.4); color:#00c875; }
.score-pill.medium { background:rgba(255,193,7,0.12); border-color:rgba(255,193,7,0.4); color:#ffc107; }
.score-pill.low    { background:rgba(224,82,82,0.12); border-color:rgba(224,82,82,0.4); color:#e05252; }

.answer-box {
    background:#0f1e2d; border:1px solid rgba(0,200,117,0.25);
    border-radius:12px; padding:1.5rem; line-height:1.7; color:#d4eaf8;
}
.stop-banner {
    background:#1f0e0e; border:1px solid rgba(224,82,82,0.4);
    border-radius:12px; padding:1.5rem; color:#f9c0c0;
}
.metric-strip { display:flex; gap:1rem; flex-wrap:wrap; margin:0.8rem 0; }
.metric-chip {
    background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.1);
    border-radius:8px; padding:0.4rem 0.9rem; font-size:0.82rem; color:#c0d8ec;
}
.metric-chip span { color:#7ecfff; font-weight:600; }
.citation-row {
    display:flex; gap:1rem; padding:0.5rem 0;
    border-bottom:1px solid rgba(255,255,255,0.05);
    font-size:0.83rem; color:#a8c8e0;
}
.citation-index { font-weight:700; color:#7ecfff; min-width:1.5rem; }
section[data-testid="stSidebar"] { background:#0d1928; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────────
def _score_class(score: float) -> str:
    return "high" if score >= 0.70 else ("medium" if score >= 0.45 else "low")


def _supabase_ok() -> bool:
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_SERVICE_KEY", "")
    return bool(url and key and "your-project" not in url)


@st.cache_resource(show_spinner="Loading retrieval pipeline…")
def load_pipeline():
    from day2_retrieval import VectorDBManager, ClinicalRetriever
    from day3_generation import CRAGOrchestrator
    db = VectorDBManager()
    return ClinicalRetriever(db), CRAGOrchestrator()


# ─── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Search Settings")
    search_mode = st.selectbox("Search Mode", ["hybrid","semantic","bm25"], index=0)
    top_k    = st.slider("Retrieve top K chunks", 5, 30, 20, 5)
    rerank_n = st.slider("Keep top N after reranking", 3, 10, 5, 1)
    target_doc = st.text_input("Filter by document (optional)", placeholder="e.g. peds.2019-3449") or None
    st.markdown("---")
    st.markdown("### 📚 Loaded Clinical Docs")
    for name, slug in [
        ("AAP Pediatrics 2020",     "peds.2019-3449"),
        ("NICE CG128",              "document"),
        ("ASD Identification 2020", "identificationevaluation…"),
        ("Eye-Tracking Biomarkers", "Eye-Tracking_Biomarkers…"),
        ("100-Day Toolkit",         "100_Day_Tool_Kit…"),
    ]:
        st.markdown(f"🗂️ **{name}**  \n`{slug}`")
    st.markdown("---")
    st.caption("SpectrumLens v1.0 · ASD CDSS · Research use only")

# ─── Hero ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🔬 SpectrumLens</div>
    <div class="hero-sub">
        Zero-hallucination ASD Clinical Decision Support &nbsp;·&nbsp;
        Corrective RAG &nbsp;·&nbsp; Evidence-First &nbsp;·&nbsp; Grounded in Clinical Guidelines
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Supabase guard ───────────────────────────────────────────────────────────────
if not _supabase_ok():
    st.error(
        "⚠️ **Supabase not configured.** Set `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` "
        "in `.env`, then run `python day2_retrieval.py --upload`.",
        icon="🔑",
    )
    st.stop()

# ─── Query input ─────────────────────────────────────────────────────────────────
col_q, col_btn = st.columns([5, 1])
with col_q:
    query = st.text_input(
        "Clinical Query",
        placeholder="e.g. At what age does the AAP recommend ASD-specific screening?",
        label_visibility="collapsed",
    )
with col_btn:
    run = st.button("🔍 Search", type="primary", use_container_width=True)

with st.expander("💡 Sample questions", expanded=False):
    SAMPLES = [
        "At what age does the AAP recommend universal ASD-specific screening?",
        "What are the two core DSM-5 symptom domains for ASD?",
        "Which FDA-approved medications treat irritability in ASD?",
        "What does NICE CG128 say about maximum wait time for ASD assessment?",
        "How does eye-tracking compare to M-CHAT-R/F for early ASD screening?",
        "What is the cure for autism spectrum disorder?",   # OOS safe-fail demo
    ]
    for s in SAMPLES:
        if st.button(s, key=f"smpl_{hash(s)}"):
            query = s
            run = True

# ─── Main flow ────────────────────────────────────────────────────────────────────
if run and query.strip():
    try:
        retriever, orchestrator = load_pipeline()
    except Exception as e:
        st.error(f"Pipeline load error: {e}")
        st.stop()

    # ── Step 1: Retrieve ─────────────────────────────────────────────────────────
    with st.spinner("🔎 Retrieving evidence from clinical guidelines…"):
        from day2_retrieval import ClinicalQuery
        t0 = time.perf_counter()
        cq = ClinicalQuery(text_query=query, target_document=target_doc, search_mode=search_mode)
        raw_chunks = retriever.retrieve_safe_context(cq, top_k=top_k)
        retrieval_ms = (time.perf_counter() - t0) * 1000

    # ── Step 2: Rerank ───────────────────────────────────────────────────────────
    with st.spinner("🏆 Re-ranking with cross-encoder…"):
        from reranker import ClinicalReranker
        t1 = time.perf_counter()
        reranked = ClinicalReranker(top_n=rerank_n).rerank(query, raw_chunks)
        rerank_ms = (time.perf_counter() - t1) * 1000

    # ════════════════════════════════════════════════════════════════════════════
    #  EVIDENCE PANEL — shown BEFORE the LLM answer
    # ════════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("## 📋 Evidence Panel")
    st.caption(
        f"Retrieved **{len(raw_chunks)}** candidates in **{retrieval_ms:.0f} ms** · "
        f"Re-ranked to top **{len(reranked)}** in **{rerank_ms:.0f} ms** · "
        f"Mode: `{search_mode}` · Embedding: `BAAI/bge-m3`"
    )

    if not reranked:
        st.warning("⚠️ No evidence above threshold — Safe Failure will be triggered.")
    else:
        for i, chunk in enumerate(reranked):
            cls = "evidence-card top-pick" if i == 0 else "evidence-card"
            top_lbl = " 🥇 Top Match" if i == 0 else ""
            sc = _score_class(chunk.vector_score)
            st.markdown(f"""
            <div class="{cls}">
              <div class="evidence-rank">Rank #{i+1}{top_lbl}</div>
              <div class="evidence-title">
                📄 {chunk.document_name}
                <span class="score-pill {sc}">vec {chunk.vector_score:.3f}</span>
                <span class="score-pill">rerank {chunk.rerank_score:+.2f}</span>
              </div>
              <div class="evidence-meta">
                📖 Section: <strong>{chunk.section_title}</strong> &nbsp;·&nbsp; 📃 Page: {chunk.page_number}
              </div>
              <div class="evidence-excerpt">
                {chunk.content[:480]}{"…" if len(chunk.content) > 480 else ""}
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Step 3: CRAG ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 🧠 CRAG Critic & Generated Answer")

    with st.spinner("🤖 Critic agent evaluating evidence quality via Groq…"):
        t2 = time.perf_counter()
        response = orchestrator.answer(query=query, target_document=target_doc, pre_reranked=reranked)
        gen_ms = (time.perf_counter() - t2) * 1000

    rep = response.context_report
    verdict_badge = (
        '<span style="background:#00c875;color:#000;padding:0.2rem 0.7rem;border-radius:12px;font-weight:700">✅ SUFFICIENT</span>'
        if response.verdict.value == "SUFFICIENT" else
        '<span style="background:#e05252;color:#fff;padding:0.2rem 0.7rem;border-radius:12px;font-weight:700">⛔ INSUFFICIENT</span>'
    )

    st.markdown(f"""
    <div class="metric-strip">
      <div class="metric-chip">Verdict: {verdict_badge}</div>
      <div class="metric-chip">Mean Relevance <span>{rep.mean_relevance:.3f}</span></div>
      <div class="metric-chip">Relevant <span>{rep.relevant_count}/{len(rep.chunk_evaluations)}</span></div>
      <div class="metric-chip">Latency <span>{gen_ms:.0f} ms</span></div>
    </div>
    """, unsafe_allow_html=True)

    # Detect language for RTL rendering
    is_arabic = getattr(response, "query_language", "en") == "ar"
    rtl_class = " rtl" if is_arabic else ""
    lang_badge = (
        '<span class="lang-badge-ar">🇸🇦 Arabic Response</span>'
        if is_arabic else
        '<span class="lang-badge-en">🇬🇧 English Response</span>'
    )

    if response.verdict.value == "INSUFFICIENT":
        st.markdown(f"""
        <div class="stop-banner{rtl_class}">
          <strong>⛔ Safe Failure — Answer Withheld for Clinical Safety</strong><br><br>
          {response.safe_failure_reason or "Evidence quality below clinical confidence threshold."}<br><br>
          <em>SpectrumLens refuses to answer when evidence is insufficient.</em>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"### 💬 Clinical Answer &nbsp; {lang_badge}", unsafe_allow_html=True)
        st.markdown(
            f'<div class="answer-box{rtl_class}">{response.answer}</div>',
            unsafe_allow_html=True
        )

        if response.citations:
            st.markdown("### 📚 Citations")
            for i, cit in enumerate(response.citations, 1):
                st.markdown(f"""
                <div class="citation-row{rtl_class}">
                  <div class="citation-index">[{i}]</div>
                  <div><strong>{cit.document_name}</strong> &nbsp;·&nbsp;
                  Section: {cit.section_title} &nbsp;·&nbsp; Page: {cit.page_number}</div>
                </div>
                """, unsafe_allow_html=True)

    # ── Debug expanders ───────────────────────────────────────────────────────────
    with st.expander("🔍 Critic Agent — Per-Chunk Evaluation", expanded=False):
        for ev in rep.chunk_evaluations:
            icon = "✅" if ev.is_relevant else "❌"
            st.markdown(f"**{icon} `{ev.chunk_id[:12]}…`** Score: `{ev.relevance_score:.3f}`\n> {ev.rationale}")
        if rep.evaluator_notes:
            st.info(f"**Critic:** {rep.evaluator_notes}")

    with st.expander("🏆 Reranker Scores", expanded=False):
        if response.rerank_scores:
            df = pd.DataFrame(response.rerank_scores)[
                ["document_name","page_number","vector_score","rerank_score"]
            ].rename(columns={
                "document_name":"Document","page_number":"Page",
                "vector_score":"Vector Score","rerank_score":"Rerank Score",
            }).sort_values("Rerank Score", ascending=False)
            st.dataframe(df, use_container_width=True, hide_index=True)

elif run and not query.strip():
    st.warning("Please enter a clinical query.")
