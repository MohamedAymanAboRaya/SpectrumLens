"""
SpectrumLens — Day 2: Multilingual Embeddings + Supabase Hybrid Retrieval
=========================================================================
Embedding:  Jina AI API (fast, free, 1024-dim, 100+ languages) [default]
            or BAAI/bge-m3 local (fallback, ~60 min for 453 chunks on CPU)
Vector DB:  Supabase pgvector
Search:     Hybrid (Semantic + BM25) with RRF fusion

Usage:
    python day2_retrieval.py --upload          # embed & push chunks to Supabase
    python day2_retrieval.py --query "..."     # test retrieval
    python day2_retrieval.py --query "..." --mode semantic | bm25 | hybrid
"""

import os
import json
import time
import logging
import argparse
import pickle
import re
import requests
import numpy as np
from typing import List, Dict, Any, Optional, Literal

from supabase import create_client, Client
from pydantic import BaseModel
from dotenv import load_dotenv

from arabic_preprocessor import ArabicPreprocessor, add_bge_query_prefix, detect_language
from reranker import ClinicalReranker

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("SpectrumLens-Retrieval")

# ─── Constants ───────────────────────────────────────────────────────────────────
TABLE_NAME       = "spectrumlens_clinical_chunks"
SEMANTIC_RPC     = "match_clinical_chunks"
HYBRID_RPC       = "hybrid_search_clinical_chunks"
SAFETY_THRESHOLD = 0.25
RRF_K            = 60
CANDIDATE_K      = 50  # two-stage: retrieve 50 → rerank → top 5
BATCH_SIZE       = 50
EMBED_DIM        = 1024

SearchMode = Literal["hybrid", "semantic", "bm25"]

# ─── Medical Acronym Expansion ─────────────────────────────────────────────────
_MEDICAL_ACRONYMS = {
    "AAP": "American Academy of Pediatrics",
    "DSM-5": "Diagnostic and Statistical Manual of Mental Disorders 5th edition",
    "DSM5": "Diagnostic and Statistical Manual of Mental Disorders 5th edition",
    "M-CHAT": "Modified Checklist for Autism in Toddlers",
    "M-CHAT-R/F": "Modified Checklist for Autism in Toddlers Revised with Follow-up",
    "ASD": "Autism Spectrum Disorder",
    "NICE": "National Institute for Health and Care Excellence",
    "FDA": "Food and Drug Administration",
    "WHO": "World Health Organization",
    "ABA": "Applied Behavior Analysis",
    "CBT": "Cognitive Behavioral Therapy",
    "ADHD": "Attention Deficit Hyperactivity Disorder",
    "GI": "gastrointestinal",
    "OAS": "occupational therapy",
    "SLP": "speech language pathology",
    "IEP": "Individualized Education Program",
    "CAMHS": "Child and Adolescent Mental Health Services",
    "PGT-A": "preimplantation genetic testing for aneuploidy",
}


def _expand_medical_acronyms(query: str) -> str:
    """Expand common medical acronyms in query text before embedding."""
    for acronym, expansion in _MEDICAL_ACRONYMS.items():
        query = query.replace(acronym, expansion)
    return query


import requests as _requests

def _runtime_arabic_to_english(query: str) -> str:
    """Translate Arabic medical query to English using LLM at runtime."""
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        return query
    try:
        resp = _requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
            json={
                "model": "allam-2-7b",
                "messages": [{"role": "user", "content": f"Translate this Arabic medical query to precise English for searching clinical guidelines. Output ONLY the English translation, nothing else.\n\nArabic: {query}\n\n/no_think"}],
                "temperature": 0,
                "max_tokens": 150,
            },
            timeout=15,
        )
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.warning(f"Runtime Arabic translation failed: {e}")
        return query


# ─── Query Model ─────────────────────────────────────────────────────────────────
class ClinicalQuery(BaseModel):
    text_query:      str
    target_document: Optional[str] = None
    search_mode:     SearchMode = "hybrid"


# ─── API Embedder (Jina AI — fast, free, 1024-dim) ──────────────────────────────
class APIEmbedder:
    """
    Uses Jina AI Embeddings API for fast, multilingual 1024-dim embeddings.
    Free tier: 1M tokens/month — no credit card needed.
    Sign up: https://jina.ai (get API key in dashboard)
    """

    API_URL = "https://api.jina.ai/v1/embeddings"
    MODEL   = "jina-embeddings-v5-text-small"

    def __init__(self):
        self.api_key = os.environ.get("JINA_API_KEY", "")
        if not self.api_key:
            raise EnvironmentError(
                "JINA_API_KEY not set. Get a free key at https://jina.ai\n"
                "  Add to .env: JINA_API_KEY=jina_..."
            )
        self._preprocessor = ArabicPreprocessor()
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })
        logger.info("API Embedder ready (Jina AI, 1024-dim) ✅")

    def _prepare_query(self, query: str) -> str:
        normalized = self._preprocessor.normalize_query(query)
        return add_bge_query_prefix(normalized)

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """Call Jina API with retry and rate-limit handling."""
        for attempt in range(3):
            try:
                resp = self._session.post(
                    self.API_URL,
                    json={
                        "model": self.MODEL,
                        "input": texts,
                        "dimensions": EMBED_DIM,
                    },
                    timeout=60,
                )
                if resp.status_code == 429:
                    wait = int(resp.headers.get("Retry-After", 5))
                    logger.warning(f"Rate limited, waiting {wait}s …")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                return [item["embedding"] for item in data["data"]]
            except Exception as e:
                if attempt < 2:
                    logger.warning(f"API error (attempt {attempt+1}): {e}")
                    time.sleep(2 ** attempt)
                else:
                    raise
        return []

    def embed(self, text: str, is_query: bool = True) -> List[float]:
        if is_query:
            text = self._prepare_query(text)
        result = self._call_api([text])
        if not result:
            logger.error("Embedder returned empty — rate limited or API error")
            return [0.0] * EMBED_DIM
        return result[0]

    def embed_batch(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        if is_query:
            texts = [self._prepare_query(t) for t in texts]
        # Jina supports batch — send in chunks of 100 with delay
        all_embeddings = []
        for i in range(0, len(texts), 100):
            batch = texts[i:i + 100]
            all_embeddings.extend(self._call_api(batch))
            if i + 100 < len(texts):
                time.sleep(1)  # rate limit buffer
        return all_embeddings


# ─── Local Fallback Embedder (BGE-M3) ──────────────────────────────────────────
class LocalEmbedder:
    """
    Fallback: BAAI/bge-m3 via sentence-transformers (local, free, ~60 min on CPU).
    Used only when JINA_API_KEY is not set.
    """

    MODEL_NAME = "BAAI/bge-m3"

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading local model: {self.MODEL_NAME} (~2.27 GB, slow on CPU) …")
        self._model = SentenceTransformer(self.MODEL_NAME)
        self._preprocessor = ArabicPreprocessor()
        logger.info("Local embedder ready ✅")

    def _prepare_query(self, query: str) -> str:
        normalized = self._preprocessor.normalize_query(query)
        return add_bge_query_prefix(normalized)

    def embed(self, text: str, is_query: bool = True) -> List[float]:
        if is_query:
            text = self._prepare_query(text)
        return self._model.encode(text, normalize_embeddings=True, show_progress_bar=False).tolist()

    def embed_batch(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        if is_query:
            texts = [self._prepare_query(t) for t in texts]
        return self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=True, batch_size=32
        ).tolist()


def get_embedder():
    """Factory: API embedder if JINA_API_KEY set, otherwise local BGE-M3."""
    if os.environ.get("JINA_API_KEY"):
        return APIEmbedder()
    logger.warning("No JINA_API_KEY — falling back to slow local BGE-M3 (~60 min)")
    return LocalEmbedder()


# ─── Vector DB Manager ───────────────────────────────────────────────────────────
class VectorDBManager:
    def __init__(self):
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_KEY")
        if not url or not key or "your-project" in url:
            raise EnvironmentError("Set SUPABASE_URL and SUPABASE_SERVICE_KEY in .env")
        self.db: Client = create_client(url, key)
        self.embedder = get_embedder()
        logger.info("VectorDBManager connected to Supabase.")

    def embed_and_store(self, json_file_path: str):
        with open(json_file_path, "r", encoding="utf-8") as f:
            chunks: List[Dict[str, Any]] = json.load(f)

        logger.info(f"Generating embeddings for {len(chunks)} chunks …")
        texts_to_embed = [c.get("normalized_text") or c.get("text", "") for c in chunks]
        embeddings = self.embedder.embed_batch(texts_to_embed, is_query=False)

        # Save embeddings to disk for instant demo load
        npz_path = "data/supabase_embeddings.npz"
        pkl_path = "data/embedding_index.pkl"
        os.makedirs("data", exist_ok=True)
        np.savez_compressed(npz_path, embeddings=np.array(embeddings, dtype="float32"))
        with open(pkl_path, "wb") as f:
            pickle.dump({
                "chunks": chunks,
                "embeddings": np.array(embeddings, dtype="float32"),
                "provider": "supabase_upload",
                "dim": len(embeddings[0]) if embeddings else 1024,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, f)
        logger.info(f"Saved embeddings to {npz_path} ({os.path.getsize(npz_path) / 1024 / 1024:.1f} MB)")

        records = [
            {
                "chunk_id":       c["chunk_id"],
                "document_name":  c["document_name"],
                "section_title":  c["section_title"],
                "page_number":    c["page_number"],
                "content":        c.get("original_text") or c["text"],
                "original_text":  c.get("original_text") or c["text"],
                "normalized_text": c.get("normalized_text", c.get("text", "")),
                "language":       c.get("language", "en"),
                "embedding":      emb,
            }
            for c, emb in zip(chunks, embeddings)
        ]

        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i: i + BATCH_SIZE]
            try:
                self.db.table(TABLE_NAME).upsert(batch, on_conflict="chunk_id").execute()
                logger.info(f"  Batch {i // BATCH_SIZE + 1} uploaded ({len(batch)} records)")
            except Exception as e:
                if "23505" in str(e) or "duplicate" in str(e).lower():
                    logger.info(f"  Batch {i // BATCH_SIZE + 1} skipped (already exists)")
                else:
                    raise

        logger.info(f"✅  {len(records)} chunks uploaded to Supabase (1024-dim).")


# ─── Clinical Retriever ───────────────────────────────────────────────────────────
class ClinicalRetriever:
    def __init__(self, db_manager: VectorDBManager):
        self.db = db_manager
        self.reranker = ClinicalReranker()

    def retrieve_safe_context(
        self, query: ClinicalQuery, top_k: int = CANDIDATE_K,
        language_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        mode = query.search_mode
        lang = detect_language(query.text_query)

        # ── Expand medical acronyms before embedding ─────────────────────────
        effective_query = _expand_medical_acronyms(query.text_query)

        # ── Arabic → English query translation for cross-lingual retrieval ─────
        if lang == "ar":
            effective_query = _runtime_arabic_to_english(effective_query)
            logger.info(f"  AR→EN (runtime): '{effective_query[:60]}'")

        # Use translated query for retrieval
        translated_query = ClinicalQuery(text_query=effective_query, search_mode=query.search_mode)

        logger.info(f"[{mode.upper()}][lang={lang}] '{effective_query[:80]}'")

        if mode == "hybrid":
            results = self._hybrid(translated_query, top_k, language_filter=language_filter)
        elif mode == "semantic":
            results = self._semantic(translated_query, top_k, language_filter=language_filter)
        elif mode == "bm25":
            results = self._bm25(translated_query, top_k, lang=lang, language_filter=language_filter)
        else:
            raise ValueError(f"Unknown search_mode: {mode}")

        results = self._deduplicate_chunks(results, top_k, max_per_doc=2)

        if not results:
            logger.warning("⚠️  No evidence above threshold — Safe Failure.")
        else:
            logger.info(f"✅  {len(results)} chunk(s) retrieved.")
        return results

    @staticmethod
    def _deduplicate_chunks(
        chunks: List[Dict[str, Any]], top_k: int, max_per_doc: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Deduplicate chunks by document_name and (document_name, section_title).
        - Keeps the highest-scoring chunk per document.
        - Allows at most `max_per_doc` chunks from the same document.
        - Skips chunks where the same (doc, section) combo is already in results.
        - Preserves ranking order by rrf_score / similarity.
        """
        doc_counts: Dict[str, int] = {}
        seen_doc_sections: set = set()
        deduplicated: List[Dict[str, Any]] = []

        for chunk in chunks:
            doc_name = chunk.get("document_name", "unknown")
            section_title = chunk.get("section_title", "unknown")
            current_count = doc_counts.get(doc_name, 0)

            if current_count >= max_per_doc:
                continue

            doc_section_key = (doc_name, section_title)
            if doc_section_key in seen_doc_sections:
                continue

            deduplicated.append(chunk)
            doc_counts[doc_name] = current_count + 1
            seen_doc_sections.add(doc_section_key)

            if len(deduplicated) >= top_k:
                break

        if len(deduplicated) < len(chunks):
            logger.info(
                f"  Dedup: {len(chunks)} → {len(deduplicated)} chunks "
                f"(max {max_per_doc}/doc, {len(doc_counts)} docs)"
            )

        return deduplicated

    def _hybrid(
        self, query: ClinicalQuery, top_k: int,
        language_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        emb = self.db.embedder.embed(query.text_query, is_query=True)
        f = self._filter(query, language_filter=language_filter)
        resp = self.db.db.rpc(HYBRID_RPC, {
            "query_text":      query.text_query,
            "query_embedding": emb,
            "match_threshold": SAFETY_THRESHOLD,
            "match_count":     CANDIDATE_K,
            "rrf_k":           RRF_K,
            "filter":          f,
        }).execute()
        candidates = self._normalise(resp.data or [], "hybrid")

        # Metadata boosting: boost score when query mentions org that matches doc
        _ORG_KEYWORDS = {
            "AAP": ["identificationevaluation", "peds.2019"],
            "NICE": ["2021-surveillance-of-autism-nice", "nice-guidelines"],
            "CDC": ["CDC_ASD"],
            "WHO": ["WHO_ASD"],
            "DSM": ["DSM5"],
            "FDA": ["psychotropic-medication"],
            "APA": ["DSM5_TR_Official"],
        }
        query_upper = query.text_query.upper()
        for chunk in candidates:
            doc_lower = chunk.get("document_name", "").lower()
            for org, doc_patterns in _ORG_KEYWORDS.items():
                if org in query_upper and any(p.lower() in doc_lower for p in doc_patterns):
                    current_score = chunk.get("rrf_score") or chunk.get("similarity") or 0.0
                    chunk["rrf_score"] = current_score * 1.30  # 30% boost for org match
                    break

        # Section-title similarity boost: chunks whose section title overlaps query
        query_words = set(query.text_query.lower().split())
        STOP_WORDS = {"the","a","an","is","are","was","were","of","in","on","at","for","to","and","or","what","how","when","where","which","who","does","do","can","should","would","could"}
        query_words -= STOP_WORDS
        if query_words:
            for chunk in candidates:
                section = (chunk.get("section_title") or "").lower()
                section_words = set(re.split(r'[\s\-_/]+', section)) - STOP_WORDS
                if section_words:
                    overlap = len(query_words & section_words) / max(len(query_words), 1)
                    if overlap >= 0.3:  # at least 30% query words in section title
                        current = chunk.get("rrf_score") or chunk.get("similarity") or 0.0
                        chunk["rrf_score"] = current * 1.35  # 35% boost for section match

        # Two-stage: rerank candidates with clinical cross-encoder
        # Set SPECTRUMLENS_NO_RERANK=1 to skip reranking (faster eval)
        if candidates and self.reranker and not os.environ.get("SPECTRUMLENS_NO_RERANK"):
            ranked = self.reranker.rerank(query.text_query, candidates)
            candidates = [r if isinstance(r, dict) else r.__dict__ for r in ranked]

        # MMR diversity re-ranking after reranking
        def _content_similarity(a: Dict, b: Dict) -> float:
            text_a = a.get("content") or a.get("text") or a.get("original_text") or ""
            text_b = b.get("content") or b.get("text") or b.get("original_text") or ""
            words_a = set(text_a.lower().split())
            words_b = set(text_b.lower().split())
            if not words_a or not words_b:
                return 0.0
            return len(words_a & words_b) / len(words_a | words_b)

        selected: List[Dict[str, Any]] = []
        remaining = list(candidates)
        lambda_param = 0.85  # 85% relevance, 15% diversity

        while remaining and len(selected) < top_k:
            best_idx = 0
            best_mmr = -float("inf")
            for i, chunk in enumerate(remaining):
                relevance = chunk.get("rrf_score") or chunk.get("similarity") or 0.0
                if selected:
                    max_sim = max(_content_similarity(chunk, s) for s in selected)
                else:
                    max_sim = 0.0
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = i
            selected.append(remaining.pop(best_idx))

        return selected

    def _semantic(
        self, query: ClinicalQuery, top_k: int,
        language_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        emb = self.db.embedder.embed(query.text_query, is_query=True)
        f = self._filter(query, language_filter=language_filter)
        resp = self.db.db.rpc(SEMANTIC_RPC, {
            "query_embedding": emb,
            "match_threshold": SAFETY_THRESHOLD,
            "match_count":     top_k,
            "filter":          f,
        }).execute()
        return self._normalise(resp.data or [], "semantic")

    def _bm25(
        self, query: ClinicalQuery, top_k: int, lang: str = "en",
        language_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        try:
            f = self._filter(query, language_filter=language_filter)
            resp = self.db.db.rpc("bm25_search_clinical_chunks", {
                "query_text": query.text_query,
                "match_count": top_k,
                "filter": f,
            }).execute()
            return self._normalise(resp.data or [], "bm25")
        except Exception:
            q = (
                self.db.db.table(TABLE_NAME)
                .select("chunk_id,document_name,section_title,page_number,content,language")
                .text_search("fts_vector", query.text_query, config="simple")
                .limit(top_k)
            )
            if query.target_document:
                q = q.eq("document_name", query.target_document)
            if language_filter:
                q = q.eq("language", language_filter)
            return self._normalise(q.execute().data or [], "bm25")

    @staticmethod
    def _filter(
        query: ClinicalQuery, language_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        f: Dict[str, Any] = {}
        if query.target_document:
            f["document_name"] = query.target_document
        if language_filter:
            f["language"] = language_filter
        return f

    @staticmethod
    def _normalise(rows: List[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
        return [{
            "chunk_id":         r.get("chunk_id", "unknown"),
            "document_name":    r.get("document_name", "unknown"),
            "section_title":    r.get("section_title", "unknown"),
            "page_number":      r.get("page_number", "unknown"),
            "content":          r.get("content", ""),
            "language":         r.get("language", "en"),
            "similarity":       r.get("similarity") or r.get("semantic_score") or r.get("bm25_score") or 0.0,
            "bm25_rank":        r.get("bm25_rank"),
            "rrf_score":        r.get("rrf_score"),
            "retrieval_source": source,
        } for r in rows]


# ─── CLI ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--query",  type=str)
    parser.add_argument("--doc",    type=str, default=None)
    parser.add_argument("--mode",   type=str, default="hybrid",
                        choices=["hybrid", "semantic", "bm25"])
    parser.add_argument("--top-k",  type=int, default=CANDIDATE_K)
    parser.add_argument("--lang-filter", type=str, default=None,
                        choices=["en", "ar"], help="Filter results by language")
    args = parser.parse_args()

    db_manager = VectorDBManager()

    if args.upload:
        json_path = "data/processed_chunks/day1_chunks_output.json"
        if not os.path.exists(json_path):
            logger.error("Run day1_ingestion.py first.")
        else:
            db_manager.embed_and_store(json_path)

    if args.query:
        retriever = ClinicalRetriever(db_manager)
        results   = retriever.retrieve_safe_context(
            ClinicalQuery(text_query=args.query, target_document=args.doc,
                          search_mode=args.mode),
            top_k=args.top_k,
            language_filter=args.lang_filter,
        )
        print(f"\n{'═'*68}\n  Mode={args.mode.upper()} | {len(results)} results\n{'═'*68}")
        for i, r in enumerate(results, 1):
            print(f"\n[{i}] {r['document_name']} | {r['section_title']} | p.{r['page_number']} [{r['language']}]")
            rrf = f"  RRF={r['rrf_score']:.5f}" if r.get("rrf_score") else ""
            print(f"     sim={r['similarity']:.4f}{rrf}")
            print(f"     {r['content'][:280]} …")
