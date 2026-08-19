# SpectrumLens — Hackathon Compliance Checklist

## Judging Criteria Mapping (CREATIVA / ITIDA / TIEC / Orange Digital Center)

### 1. Retrieval Quality (30%)

| Requirement | Status | Evidence |
|---|---|---|
| Semantic search | ✅ | Jina AI v5 (1024-dim, 100+ languages) |
| BM25 keyword search | ✅ | Supabase pgvector full-text search |
| Hybrid RRF fusion | ✅ | Reciprocal Rank Fusion combining semantic + BM25 |
| Top-K tuning (3,5,10) | ✅ | Configurable via sidebar slider |
| Precision@K reported | ✅ | P@3=0.580, P@5=0.460 (offline, v9); live Jina demo significantly higher |
| Cross-encoder reranking | ✅ | Jina Reranker v3.5 (API-based, multilingual) |
| Metadata transparency | ✅ | Document name, section, page, chunk ID shown |
| Failure mode analysis | ✅ | DUPLICATE_CHUNKS, WRONG_TOPIC_SECTION, MISSING_SOURCE documented |

### 2. Grounding & Citation (25%)

| Requirement | Status | Evidence |
|---|---|---|
| Inline citations | ✅ | [SOURCE: document, section, page] format |
| Evidence before answer | ✅ | Evidence Panel displayed BEFORE LLM generation |
| Citation verification | ✅ | guardrails/citation_verifier.py (structural + claim grounding) |
| Unsupported claim detection | ✅ | Post-generation citation check in day3_generation.py |
| Confidence levels | ✅ | HIGH / MEDIUM / LOW / INSUFFICIENT |
| Clinical disclaimer | ✅ | Mandatory on every output |

### 3. System Architecture (15%)

| Requirement | Status | Evidence |
|---|---|---|
| Modular pipeline | ✅ | day1 (ingestion) → day2 (retrieval) → day3 (generation) |
| Auditable pipeline | ✅ | Each stage logged with latency and results |
| CRAG pattern | ✅ | Scope Check → Critic → Generator / Safe Failure |
| Bilingual support | ✅ | Arabic/English with runtime translation |
| Offline mode | ✅ | Precomputed embeddings for instant demo |

### 4. Evaluation Depth (15%)

| Requirement | Status | Evidence |
|---|---|---|
| 20+ test cases | ✅ | 50 questions across 4 categories |
| Multiple metrics | ✅ | P@3, P@5, Recall@10, nDCG@5, citation coverage |
| Failure mode taxonomy | ✅ | 6 failure modes documented |
| Category breakdown | ✅ | Factual, Inferential, OOS, Adversarial |
| Difficulty levels | ✅ | Easy, Medium, Hard |
| Arabic eval questions | ✅ | 15 Arabic questions with ground truth |
| NICE-specific questions | ✅ | 10 NICE CG128 guideline questions |

### 5. Safety & UX (15%)

| Requirement | Status | Evidence |
|---|---|---|
| Ternary scope check | ✅ | ALLOWED / NEEDS_CAUTION / REFUSE |
| Safe failure (OOS) | ✅ | 100% safe failure rate (live demo) |
| Evidence-first UI | ✅ | Evidence Panel before answer |
| RTL Arabic support | ✅ | Full-page RTL when Arabic query detected |
| Streaming responses | ✅ | Token-by-token LLM output |
| Chat history | ✅ | Conversational mode with query context |
| Model comparison | ✅ | Side-by-side Semantic vs BM25 vs Hybrid |
| Authority tier badges | ✅ | 🏛️ Official / 📚 Peer-Reviewed / 📄 Toolkit |

## Architecture Summary

```
PDF Ingestion (23 PDFs, 1,801 chunks)
    ↓
Jina AI v5 Embedding (1024-dim, cross-lingual)
    ↓
Supabase pgvector (Semantic + BM25 + RRF)
    ↓
Section-Title Boost + ORG Metadata Boost
    ↓
Jina Reranker v3.5 (API-based)
    ↓
MMR Diversity Re-ranking
    ↓
Scope Check (GPT-5.6-sol / AgentRouter) → ALLOWED / NEEDS_CAUTION / REFUSE
    ↓
Critic Agent (GPT-5.6-sol / AgentRouter) → SUFFICIENT / INSUFFICIENT
    ↓
Generator (GPT-5.6-sol / AgentRouter) → Cited Answer
    ↓
Fallback: OpenRouter (Gemini 2.5 Flash) → Groq (GPT-OSS-120B)
    ↓
Citation Verifier → Post-check
    ↓
Streamlit Evidence Panel → Answer
```

## Models Used

| Component | Model | Provider | Why |
|---|---|---|---|
| Embedding | Jina AI v5-text-small | Jina AI | 1024-dim, 100+ languages, free API |
| Reranker | Jina Reranker v3.5 | Jina AI | Multilingual, Arabic+English, API-based |
| Scope Check | GPT-5.6-sol | AgentRouter | Best quality, $125 credits, structured JSON |
| Critic | GPT-5.6-sol | AgentRouter | Evidence relevance scoring |
| Generator | GPT-5.6-sol | AgentRouter | Bilingual, cited answers |
| Fallback | Gemini 2.5 Flash | OpenRouter | Free tier, fast, good Arabic |
| Arabic Translation | Allam-2-7B | Groq | Fast, native Arabic, free |
| Streaming | GPT-OSS-120B | Groq | Token-by-token output |

## How to Run

```bash
# Demo (instant, precomputed embeddings)
streamlit run demo_app.py
# → Sidebar: Select LLM provider (AgentRouter / OpenRouter / Groq)
# → Toggle: Enable LLM Generation

# Full evaluation (offline, no Supabase)
python evaluate.py --offline --no-generation --output-report eval_report.json

# Full evaluation with generation (requires API keys)
python evaluate.py --offline --output-report eval_report_gen.json
```

**API Keys Required** (in `.env`):
- `GROQ_API_KEY` — Free tier, required
- `AGENTROUTER_API_KEY` — $125 credits, optional (best quality)
- `OPENROUTER_API_KEY` — Free tier, optional (Gemini fallback)
