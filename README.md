# SpectrumLens

### Zero-Hallucination ASD Clinical Decision Support System

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit)
![AgentRouter](https://img.shields.io/badge/LLM-AgentRouter%20(GPT--5.6--sol)-purple)
![OpenRouter](https://img.shields.io/badge/Fallback-OpenRouter%20(Gemini)-blue)
![Groq](https://img.shields.io/badge/Fallback-Groq%20(Allam%2BGPT--OSS)-orange)
![Jina AI](https://img.shields.io/badge/Embeddings-Jina--AI--v5-green)
![License](https://img.shields.io/badge/License-Research%20Only-red)
![Live Demo](https://img.shields.io/badge/Demo-Live%20%F0%9F%9F%A2-brightgreen)

> **A bilingual (Arabic/English) Corrective RAG system that answers clinical questions about Autism Spectrum Disorder — grounded exclusively in official guidelines, with a built-in Critic Agent that refuses to answer when evidence is insufficient.**

---

## Quick Start

```bash
# 1. Clone & install
git clone <repo-url>
cd "Mediacal Rag System"
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Set API keys in .env
GROQ_API_KEY=gsk_...
AGENTROUTER_API_KEY=sk-...
OPENROUTER_API_KEY=sk-or-...
JINA_API_KEY=jina_...

# 3. Launch
streamlit run demo_app.py
```

> Precomputed embeddings load instantly (384-dim, 1,801 chunks). No model download needed.

> **Live Demo:** `http://156.203.241.128:8501` — running now

---

## Architecture

```
Clinical PDFs (23 documents, 1,801 chunks)
        |  day1_ingestion.py
        v
   PDF Parsing & Chunking
   + Adaptive section-header detection
   + Arabic/English normalization
   + Dual-field: original_text + normalized_text
   + Metadata: doc_name, page, section, chunk_id, source_url
        |
        v  Precomputed Embeddings (384-dim, instant load)
   SentenceTransformers (paraphrase-multilingual-MiniLM-L12-v2)
        |
   User Query (Arabic or English)
        |
        +-- Language Detection (ArabicPreprocessor)
        +-- Arabic → English runtime translation (Groq/Allam-2-7b)
        +-- Medical Acronym Expansion (AAP, DSM-5, M-CHAT-R/F, NICE...)
        +-- Hybrid Search: Semantic + BM25 + RRF → Top 50
        +-- ORG-Keyword Metadata Boost (+30% for org-matched docs)
        +-- Section-Title Similarity Boost (+20% for section overlap)
        +-- MAX-1-CHUNK-PER-DOCUMENT Deduplication
        +-- Jina Reranker v3.5 → Top 5
        |
        +-- Scope Check (GPT-5.6-sol / Allam-2-7b) ← TERNARY
        |       ALLOWED / NEEDS_CAUTION / REFUSE
        |
        +-- Critic Agent (GPT-5.6-sol / Allam-2-7b)
        |       SUFFICIENT / INSUFFICIENT (threshold: mean ≥ 6/10)
        |
        v
   Generator (AgentRouter → OpenRouter → Groq fallback)
   + Bilingual cited answer
   + Citation Verifier (structural + faithfulness)
   + Clinical disclaimer
        |
        v
   Streamlit UI
   + Evidence Panel (BEFORE answer)
   + Streaming token-by-token output
   + Chat history + RTL Arabic support
   + Provider selector (AgentRouter / OpenRouter / Groq)
```
   Bilingual answer + inline citations [SOURCE: doc, section, page]
        |
        +-- Citation Verifier (guardrails/citation_verifier.py)
        |   - Tier 1: [SOURCE: ...] pattern validity
        |   - Tier 2: Citation exists in retrieved chunks
        |   - Tier 3: Sentence-level faithfulness (token overlap)
        v
   Streamlit Evidence Panel
   (evidence shown BEFORE answer — required by hackathon guidelines)
```

---

## Hackathon Compliance Checklist

| Requirement | Status |
|---|---|
| Official public PDFs (WHO, CDC, NICE, AAP, DSM-5) | ✅ 23 documents |
| Metadata: doc_name, page, section, chunk_id, source_url | ✅ All fields present |
| Section-aware chunks (400–800 tokens) | ✅ Implemented |
| Hybrid search (Semantic + BM25 + Rerank) | ✅ RRF + Jina Reranker v3.5 |
| Evidence Panel shown BEFORE LLM answer | ✅ |
| Precision@3 and @5 reported | ✅ Live in eval tab |
| Ternary input classification (ALLOWED / NEEDS\_CAUTION / REFUSE) | ✅ |
| Retrieval confidence thresholds | ✅ Critic: mean ≥ 6/10 |
| Unsupported claim detection | ✅ guardrails/citation\_verifier.py |
| Citation accuracy measured | ✅ Live in eval tab |
| Faithfulness measured | ✅ Live in eval tab |
| 30-question eval set (factual/inferential/OOS/adversarial) | ✅ |
| Failure mode taxonomy | ✅ DUPLICATE\_CHUNKS, MISSING\_SOURCE, WRONG\_TOPIC\_SECTION |
| Bilingual Arabic/English | ✅ Full RT translation + RTL UI |
| Clinical disclaimer on every output | ✅ |
| "Fluent ≠ Safe" design | ✅ CRAG Critic + Safe Failure |

---

## Clinical Documents Indexed (23 PDFs)

| Category | Documents |
|---|---|
| ASD Screening | AAP Pediatrics 2020, ASD Identification & Evaluation |
| ASD Diagnosis | DSM-5-TR, NICE CG128, ICD-11 |
| Interventions | ABA Guidelines, Early Intensive Behavioral Intervention |
| Medications | Psychotropic Medication Guidelines, FDA Approved ASD Meds |
| Comorbidities | ADHD Pharmacotherapy, ASD Diagnosis Experiences |
| AI/Clinical | AI in ASD Diagnosis, ASD Community Report (CDC 2025) |
| WHO Reports | WHO ASD Progress Report, Global Autism Reports |

---

## Evaluation Results (v10 — Verified, 2026-08-19)

**Offline Retrieval (Jina 1024-dim + BM25 + RRF + Domain Boosts):**

> Verified fresh eval run with Jina 1024-dim embeddings, BM25 synonym expansion, and RRF fusion — same pipeline as live demo.

| Metric | Score | Target | Status |
|---|---|---|---|
| Precision@3 | **0.553** | ≥ 0.50 | ✅ Above target |
| Precision@5 | **0.416** | ≥ 0.40 | ✅ Above target |
| Recall@10 | **0.603** | ≥ 0.60 | ✅ At target |
| nDCG@5 | **0.836** | ≥ 0.80 | ✅ Near-perfect |
| Citation Coverage | **0.773** | — | ✅ |
| OOS Safe Failure | **100%** | ≥ 98% | ✅ Perfect |

**Key improvements over v9 baseline:**
- P@3: 0.293 → 0.553 (+89%) — Jina 1024-dim embeddings + BM25 synonym expansion
- nDCG@5: 0.518 → 0.836 (+61%) — RRF fusion with domain boosts
- OOS Refusal: 67% → 100% — All out-of-scope queries correctly refused
- `_embed_query` now uses Jina API (matches precomputed 1024-dim)
- Offline eval now uses hybrid BM25+RRF (same as live demo)

**Eval Dataset:** 50 questions across 4 categories (factual, inferential, OOS, adversarial) including 15 Arabic questions and 10 NICE-specific questions.

---

## Models Used

| Component | Model | Provider | Purpose |
|---|---|---|---|
| Embedding | Jina AI v5-text-small | Jina AI | 1024-dim cross-lingual embeddings |
| Reranker | Jina Reranker v3.5 | Jina AI | Cross-lingual precision boost |
| Scope Check | GPT-5.6-sol | AgentRouter | **Ternary** classification: ALLOWED / NEEDS\_CAUTION / REFUSE |
| Critic | GPT-5.6-sol | AgentRouter | Evidence relevance scoring (0–10) |
| Generator | GPT-5.6-sol | AgentRouter | Bilingual cited answer generation |
| Fallback LLM | Gemini 2.5 Flash | OpenRouter | Free-tier fallback for generation |
| Arabic Translation | Allam-2-7b | Groq | Runtime Arabic→English translation |
| Stream | GPT-OSS-120B | Groq | Token-by-token streaming output |

**LLM Provider Chain** (automatic fallback): AgentRouter → OpenRouter → Groq
- **AgentRouter**: GPT-5.6-sol, Claude Opus 5 ($125 credits) — premium quality
- **OpenRouter**: Gemini 2.5 Flash — free tier, fast
- **Groq**: Allam-2-7b (Arabic), GPT-OSS-120B (generation) — fast, free

---

## System Features

- **3-Provider LLM**: AgentRouter (GPT-5.6-sol) + OpenRouter (Gemini) + Groq (Allam/GPT-OSS) with automatic fallback
- **Evidence-First UI**: Clinical evidence panel displayed BEFORE the LLM answer
- **Zero-Hallucination Design**: CRAG critic refuses to answer when evidence is insufficient
- **Ternary Scope Check**: ALLOWED / NEEDS\_CAUTION / REFUSE (not binary)
- **Citation Verifier**: 3-tier post-generation check (structural + retrieval binding + faithfulness)
- **Bilingual**: Full Arabic/English support with runtime translation and RTL UI
- **Streaming Responses**: Real-time token-by-token LLM output (all 3 providers)
- **Chat History**: Conversational mode with query context
- **Model Comparison**: Side-by-side Semantic vs BM25 vs Hybrid RRF
- **50-Question Eval**: Precision@K, nDCG@5, Recall@10, failure mode taxonomy, Arabic + NICE questions
- **Clinical Disclaimer**: Mandatory on every output

---

## File Structure

| File | Purpose |
|---|---|
| `demo_app.py` | Streamlit demo (offline + online modes) |
| `app.py` | Full production app (Supabase required) |
| `day1_ingestion.py` | PDF parsing, chunking, metadata extraction |
| `day2_retrieval.py` | Hybrid search, RRF, reranking, MMR, deduplication |
| `day3_generation.py` | CRAG: scope check, critic, generator |
| `evaluate.py` | Evaluation harness (Precision@K, nDCG, failure modes) |
| `reranker.py` | Jina/Cohere/Local reranker abstraction |
| `arabic_preprocessor.py` | Arabic text normalization |
| `guardrails/citation_verifier.py` | 3-tier post-generation citation verification |
| `data/eval/eval_dataset.json` | 30-question evaluation dataset |
| `data/source_registry.json` | Document metadata (authority tiers, URLs) |

---

## License

Research use only. Not a medical device. All outputs must be validated by qualified healthcare professionals.
