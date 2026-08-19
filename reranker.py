"""
SpectrumLens — Reranker Module
==============================
Cross-encoder reranking layer between vector retrieval and CRAG critic.

Strategy: Retrieve MORE (top-50) → Rerank → Keep FEWER (top-5).

Three backends (auto-selected by priority):
    1. COHERE  — Cohere Rerank v3.5 (best quality, 100+ languages, Arabic+English)
    2. OPENROUTER — nvidia/llama-nemotron-rerank-vl-1b-v2 (FREE, multilingual)
    3. LOCAL   — sentence-transformers CrossEncoder (free, offline fallback)
"""

import os
import json
import logging
import requests as _requests
from typing import List, Dict, Any, Optional

from pydantic import BaseModel

logger = logging.getLogger("SpectrumLens-Reranker")


# ─── Output Model ────────────────────────────────────────────────────────────────
class RankedChunk(BaseModel):
    """A retrieved chunk augmented with its cross-encoder rerank score."""
    chunk_id: str
    document_name: str
    section_title: str
    page_number: str
    content: str
    vector_score: float     # cosine similarity from Supabase
    rerank_score: float     # cross-encoder score (higher = more relevant)


# ─── OpenRouter Reranker (PRIMARY) ────────────────────────────────────────────────
class OpenRouterReranker:
    """
    Uses OpenRouter's free rerank API: nvidia/llama-nemotron-rerank-vl-1b-v2:free
    Multilingual, 10K context, no cost.
    """

    API_URL = "https://openrouter.ai/api/v1/rerank"
    MODEL = "nvidia/llama-nemotron-rerank-vl-1b-v2:free"

    def __init__(self):
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise EnvironmentError("OPENROUTER_API_KEY not set — cannot use OpenRouter reranker")
        self._session = _requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })
        logger.info("OpenRouter Reranker initialised ✅ (nvidia/llama-nemotron-rerank-vl-1b-v2, FREE)")

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_n: int = 5,
    ) -> List[RankedChunk]:
        if not chunks:
            return []

        documents = [c.get("content") or c.get("text") or c.get("original_text", "") for c in chunks]

        data = None
        for attempt in range(3):
            try:
                import time
                time.sleep(0.5)  # rate limit buffer
                resp = self._session.post(
                    self.API_URL,
                    json={
                        "model": self.MODEL,
                        "query": query,
                        "documents": documents,
                        "top_n": min(top_n, len(documents)),
                    },
                    timeout=30,
                )
                if resp.status_code == 429:
                    wait = int(resp.headers.get("Retry-After", 5))
                    logger.warning(f"OpenRouter rerank rate limited, waiting {wait}s")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                if attempt < 2:
                    logger.warning(f"OpenRouter rerank attempt {attempt+1} failed: {e}")
                    import time
                    time.sleep(2)
                else:
                    logger.error(f"OpenRouter rerank failed after 3 attempts: {e}")

        # If all retries failed, return chunks without reranking
        if data is None:
            logger.warning("OpenRouter rerank failed — returning chunks without reranking")
            return [RankedChunk(
                chunk_id=c.get("chunk_id", "unknown"),
                document_name=c.get("document_name", "unknown"),
                section_title=c.get("section_title", "unknown"),
                page_number=c.get("page_number", "unknown"),
                content=c.get("content") or c.get("text") or "",
                vector_score=c.get("similarity", 0.0),
                rerank_score=c.get("similarity", 0.0),
            ) for c in chunks[:top_n]]

        ranked = []
        for result in data.get("results", []):
            idx = result.get("index", 0)
            score = result.get("relevance_score", 0.0)
            chunk = chunks[idx]
            ranked.append(RankedChunk(
                chunk_id=chunk.get("chunk_id", "unknown"),
                document_name=chunk.get("document_name", "unknown"),
                section_title=chunk.get("section_title", "unknown"),
                page_number=chunk.get("page_number", "unknown"),
                content=chunk.get("content") or chunk.get("text") or "",
                vector_score=chunk.get("similarity", 0.0),
                rerank_score=score,
            ))

        ranked.sort(key=lambda x: x.rerank_score, reverse=True)
        logger.info(f"OpenRouter reranked {len(chunks)} → {len(ranked)} chunks")
        return ranked


# ─── Local Cross-Encoder Backend (FALLBACK) ────────────────────────────────────
class LocalCrossEncoderReranker:
    """
    Uses sentence-transformers CrossEncoder locally.
    No API calls, no cost, no network dependency — ideal for hackathon demos.

    Model choice rationale:
        BAAI/bge-reranker-v2-m3 (primary)
        • Multilingual: handles Arabic + English natively — critical for
          bilingual medical corpora
        • Strong cross-lingual relevance scoring for clinical text
        • 568M parameters: more accurate than ms-marco on multilingual data

        cross-encoder/ms-marco-MiniLM-L-6-v2 (fallback)
        • Trained on MS MARCO passage ranking (strong general relevance)
        • 6-layer MiniLM: fast enough to rerank 20 chunks in ~200ms on CPU
        • Used only if bge-reranker-v2-m3 fails to load (network, OOM, etc.)
    """

    PRIMARY_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # fast, 80MB
    FALLBACK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self):
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for local reranking. "
                "Run: pip install sentence-transformers"
            )

        self.MODEL_NAME = self.PRIMARY_MODEL
        try:
            logger.info(f"Loading cross-encoder model: {self.PRIMARY_MODEL} …")
            self._model = CrossEncoder(self.PRIMARY_MODEL, max_length=512)
            logger.info("Cross-encoder loaded (bge-reranker-v2-m3) ✅")
        except Exception as e:
            logger.warning(
                f"Failed to load {self.PRIMARY_MODEL}: {e}. "
                f"Falling back to {self.FALLBACK_MODEL}."
            )
            self.MODEL_NAME = self.FALLBACK_MODEL
            self._model = CrossEncoder(self.FALLBACK_MODEL, max_length=512)
            logger.info("Cross-encoder loaded (ms-marco-MiniLM-L-6-v2 fallback) ✅")

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
    ) -> List[RankedChunk]:
        if not chunks:
            return []

        # Cross-encoder expects (query, passage) pairs
        pairs = [(query, c.get("content", "")) for c in chunks]
        scores: List[float] = self._model.predict(pairs).tolist()

        ranked = []
        for chunk, score in zip(chunks, scores):
            ranked.append(
                RankedChunk(
                    chunk_id=chunk.get("chunk_id", "unknown"),
                    document_name=chunk.get("document_name", "unknown"),
                    section_title=chunk.get("section_title", "unknown"),
                    page_number=chunk.get("page_number", "unknown"),
                    content=chunk.get("content", ""),
                    vector_score=chunk.get("similarity", 0.0),
                    rerank_score=score,
                )
            )

        # Sort descending by cross-encoder score
        ranked.sort(key=lambda x: x.rerank_score, reverse=True)
        return ranked


# ─── Cohere API Backend ───────────────────────────────────────────────────────────
class CohereReranker:
    """
    Uses the Cohere Rerank v3 API.
    Requires: pip install cohere  +  COHERE_API_KEY env var.

    Preferred over local when:
      • You have a Cohere API key
      • You want state-of-the-art relevance scoring
    """

    MODEL_NAME = "rerank-v3.5"  # multilingual, Arabic + English support

    def __init__(self):
        try:
            import cohere  # type: ignore
        except ImportError:
            raise ImportError(
                "cohere package is required for Cohere reranking. "
                "Run: pip install cohere"
            )
        api_key = os.environ.get("COHERE_API_KEY")
        if not api_key:
            raise EnvironmentError("COHERE_API_KEY environment variable is not set.")
        self._client = cohere.Client(api_key)
        logger.info("Cohere Reranker initialised ✅")

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
    ) -> List[RankedChunk]:
        if not chunks:
            return []

        documents = [c.get("content", "") for c in chunks]
        response = self._client.rerank(
            model=self.MODEL_NAME,
            query=query,
            documents=documents,
            return_documents=False,
        )

        # Map Cohere's index-based results back to our chunks
        ranked = []
        for result in response.results:
            chunk = chunks[result.index]
            ranked.append(
                RankedChunk(
                    chunk_id=chunk.get("chunk_id", "unknown"),
                    document_name=chunk.get("document_name", "unknown"),
                    section_title=chunk.get("section_title", "unknown"),
                    page_number=chunk.get("page_number", "unknown"),
                    content=chunk.get("content", ""),
                    vector_score=chunk.get("similarity", 0.0),
                    rerank_score=result.relevance_score,
                )
            )
        # Already sorted by Cohere, but sort explicitly for safety
        ranked.sort(key=lambda x: x.rerank_score, reverse=True)
        return ranked


# ─── Main Reranker (auto-selects backend) ────────────────────────────────────────
class ClinicalReranker:
    """
    Top-level reranker used by the CRAG orchestrator.

    Selection logic:
        COHERE_API_KEY set → CohereReranker    (best quality, 100+ langs, Arabic+English)
        OPENROUTER_API_KEY set → OpenRouterReranker (free, multilingual)
        otherwise → LocalCrossEncoderReranker  (free, offline, English-optimized)

    Args:
        rerank_threshold: Minimum rerank score to keep a chunk.
                          Chunks below this are discarded even if the
                          vector retriever returned them.
        top_n:            Maximum chunks to keep after reranking.
    """

    # Sensible defaults tuned for clinical use:
    # Local cross-encoder scores are logits (can be negative), so we use
    # a relative top-n cut rather than a hard threshold for the local backend.
    # For Cohere, scores are 0–1, so 0.5 is a good floor.

    def __init__(
        self,
        top_n: int = 5,
        rerank_threshold: Optional[float] = None,
    ):
        self.top_n = top_n
        self.rerank_threshold = rerank_threshold

        # Priority: Cohere (best) → OpenRouter (free) → Local (fallback)
        if os.environ.get("COHERE_API_KEY"):
            logger.info("Reranker backend: Cohere Rerank v3.5 (best quality)")
            self._backend = CohereReranker()
            if self.rerank_threshold is None:
                self.rerank_threshold = 0.40
        elif os.environ.get("OPENROUTER_API_KEY"):
            logger.info("Reranker backend: OpenRouter (nvidia/llama-nemotron-rerank-vl-1b-v2, FREE)")
            self._backend = OpenRouterReranker()
            if self.rerank_threshold is None:
                self.rerank_threshold = float("-inf")
            logger.info("Reranker backend: Cohere Rerank v3")
            self._backend = CohereReranker()
            if self.rerank_threshold is None:
                self.rerank_threshold = 0.40
        else:
            logger.info("Reranker backend: Local CrossEncoder (ms-marco-MiniLM-L-6-v2)")
            self._backend = LocalCrossEncoderReranker()
            if self.rerank_threshold is None:
                self.rerank_threshold = float("-inf")

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
    ) -> List[RankedChunk]:
        """
        Reranks retrieved chunks and returns the top_n most relevant ones
        above the rerank_threshold.

        Args:
            query:  The original clinical question.
            chunks: Raw retrieval results from ClinicalRetriever.

        Returns:
            List of RankedChunk, sorted by rerank_score descending,
            truncated to top_n and filtered by rerank_threshold.
        """
        if not chunks:
            logger.warning("Reranker received 0 chunks — nothing to rerank.")
            return []

        logger.info(f"Reranking {len(chunks)} chunks for query: '{query[:80]}…'")

        all_ranked = self._backend.rerank(query, chunks)

        # Apply threshold filter
        filtered = [r for r in all_ranked if r.rerank_score >= self.rerank_threshold]

        # Take top_n
        result = filtered[: self.top_n]

        logger.info(
            f"Reranking complete: {len(chunks)} → {len(result)} chunks kept "
            f"(threshold={self.rerank_threshold}, top_n={self.top_n})"
        )

        # Log the ranking table for transparency
        for i, r in enumerate(result, 1):
            logger.info(
                f"  Rank {i}: score={r.rerank_score:.4f} | "
                f"vector={r.vector_score:.4f} | "
                f"{r.document_name} | p.{r.page_number}"
            )

        return result
