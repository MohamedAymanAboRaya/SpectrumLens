"""
SpectrumLens — Single Pipeline Entry Point
==========================================
One function for demo, eval, and CLI. Ensures identical behavior
across all interfaces.

Usage:
    from pipeline import run_query
    result = run_query("AAP screening age for autism?")
"""

import os
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("SpectrumLens-Pipeline")

# Lazy imports to avoid circular dependencies
_retriever = None
_crager = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        from day2_retrieval import VectorDBManager, ClinicalRetriever
        db = VectorDBManager()
        _retriever = ClinicalRetriever(db)
    return _retriever


def _get_crager():
    global _crager
    if _crager is None:
        from day3_generation import CRAGOrchestrator
        _crager = CRAGOrchestrator()
    return _crager


def run_query(
    query: str,
    search_mode: str = "hybrid",
    top_k: int = 5,
    language_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run a complete query through the SpectrumLens pipeline.
    
    Returns dict with:
        - answer: generated response text
        - verdict: SAFE | CAUTION | REFUSE | INSUFFICIENT
        - retrieved_chunks: list of chunk dicts
        - citation_verification: citation accuracy metrics
        - scope_check: scope classification result
    """
    from day2_retrieval import ClinicalQuery
    
    retriever = _get_retriever()
    crager = _get_crager()
    
    # Build query
    cq = ClinicalQuery(text_query=query, search_mode=search_mode)
    
    # Retrieve
    chunks = retriever.retrieve_safe_context(cq, top_k=top_k, language_filter=language_filter)
    
    # Generate with CRAG
    response = crager.answer(query, chunks)
    
    return {
        "answer": response.get("answer", ""),
        "verdict": response.get("verdict", "INSUFFICIENT"),
        "retrieved_chunks": chunks,
        "scope_check": response.get("scope_check", {}),
        "critic_score": response.get("critic_score", 0),
        "confidence": response.get("confidence", "LOW"),
    }


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "AAP screening age for autism?"
    result = run_query(query)
    print(f"\nVerdict: {result['verdict']}")
    print(f"Confidence: {result['confidence']}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nChunks retrieved: {len(result['retrieved_chunks'])}")
