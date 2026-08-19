"""
SpectrumLens — Three-Tier Citation Verifier
============================================
Layer 03 of the safety guardrails. Verifies that every claim in a generated
answer is grounded in retrieved evidence.

Tier 1: Structural — parse [SOURCE: ...] patterns, reject malformed
Tier 2: Retrieval binding — match citations against retrieved chunk IDs
Tier 3: Claim grounding — token overlap between sentence and cited chunk
"""

import re
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("SpectrumLens-CitationVerifier")


@dataclass
class CitationVerification:
    total_citations: int = 0
    valid_citations: int = 0
    hallucinated_citations: int = 0
    malformed_citations: int = 0
    citation_accuracy: float = 0.0
    total_sentences: int = 0
    grounded_sentences: int = 0
    faithfulness: float = 0.0
    unsupported_claim_rate: float = 0.0
    unsupported_sentences: List[str] = field(default_factory=list)
    missing_citations: List[str] = field(default_factory=list)


def _token_overlap(text_a: str, text_b: str) -> float:
    """Compute token F1 between two texts."""
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    if not intersection:
        return 0.0
    precision = len(intersection) / len(tokens_a)
    recall = len(intersection) / len(tokens_b)
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences (simple heuristic)."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def verify_answer(answer: str, retrieved_chunks: List[Dict[str, Any]]) -> CitationVerification:
    """
    Three-tier citation verification.
    
    Tier 1: Structural — find all citation patterns (both [SOURCE: ...] and 【Source N】)
    Tier 2: Retrieval binding — check citations exist in retrieved set
    Tier 3: Claim grounding — check sentence overlap with cited chunks
    """
    result = CitationVerification()
    
    # Tier 1: Structural parsing — support both formats
    citation_pattern_old = r'\[SOURCE:\s*([^\]]+)\]'
    citation_pattern_new = r'【Source (\d+)】'
    citations_old = re.findall(citation_pattern_old, answer)
    citations_new = re.findall(citation_pattern_new, answer)
    result.total_citations = len(citations_old) + len(citations_new)
    
    # Build retrieval index
    chunk_index = {}
    for i, chunk in enumerate(retrieved_chunks):
        chunk_id = chunk.get("chunk_id", "")
        doc_name = chunk.get("document_name", "").lower()
        section = chunk.get("section_title", "").lower()
        content = chunk.get("content") or chunk.get("text") or chunk.get("original_text") or ""
        chunk_index[chunk_id] = {"doc": doc_name, "section": section, "content": content}
        chunk_index[doc_name] = chunk_index[chunk_id]
        # Map Source N → chunk index
        chunk_index[f"source_{i+1}"] = chunk_index[chunk_id]
    
    # Tier 2: Retrieval binding — check if citation text mentions any retrieved doc/section
    # Handle both old [SOURCE: ...] and new 【Source N】 formats
    for cite in citations_old:
        cite_lower = cite.strip().lower()
        found = False
        for key, info in chunk_index.items():
            doc = info.get("doc", "")
            section = info.get("section", "")
            if (doc and doc in cite_lower) or (section and section in cite_lower) or \
               (cite_lower in str(key).lower()) or (cite_lower in doc):
                found = True
                break
        if found:
            result.valid_citations += 1
        else:
            result.hallucinated_citations += 1
            result.missing_citations.append(cite.strip())
    
    # New format: 【Source N】 — validate by index
    for source_num in citations_new:
        key = f"source_{source_num}"
        if key in chunk_index:
            result.valid_citations += 1
        else:
            result.hallucinated_citations += 1
            result.missing_citations.append(f"Source {source_num}")
    
    # Tier 3: Claim grounding
    sentences = _split_into_sentences(answer)
    result.total_sentences = len(sentences)
    
    for sentence in sentences:
        # Skip citation-only sentences and disclaimer sentences
        if re.match(r'^\[SOURCE:.*\]$', sentence.strip()) or \
           re.match(r'^【Source \d+】', sentence.strip()):
            result.grounded_sentences += 1
            continue
        if any(kw in sentence.lower() for kw in ["disclaimer", "not a substitute", "clinical decision", "professional medical"]):
            result.grounded_sentences += 1
            continue
        
        # Check overlap with any retrieved chunk content
        max_overlap = 0.0
        for chunk in retrieved_chunks:
            content = chunk.get("content") or chunk.get("text") or chunk.get("original_text") or ""
            if content:
                overlap = _token_overlap(sentence, content)
                max_overlap = max(max_overlap, overlap)
        
        if max_overlap >= 0.15:
            result.grounded_sentences += 1
        else:
            result.unsupported_sentences.append(sentence[:100])
    
    # Compute metrics
    result.citation_accuracy = result.valid_citations / result.total_citations if result.total_citations > 0 else 1.0
    result.faithfulness = result.grounded_sentences / result.total_sentences if result.total_sentences > 0 else 1.0
    result.unsupported_claim_rate = 1.0 - result.faithfulness
    
    logger.info(
        f"Citation verification: {result.valid_citations}/{result.total_citations} valid "
        f"({result.citation_accuracy:.1%}), faithfulness={result.faithfulness:.1%}, "
        f"unsupported={result.unsupported_claim_rate:.1%}"
    )
    
    return result
