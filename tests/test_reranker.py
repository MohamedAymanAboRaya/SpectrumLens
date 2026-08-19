"""Tests for the reranker module."""

import pytest
from unittest.mock import MagicMock, patch
from reranker import ClinicalReranker, RankedChunk, LocalCrossEncoderReranker


SAMPLE_CHUNKS = [
    {
        "chunk_id": "chunk-001",
        "document_name": "peds.2019-3449.pdf",
        "section_title": "ASD Screening",
        "page_number": "5",
        "content": "The AAP recommends universal ASD screening at 18 and 24 months.",
        "similarity": 0.85,
    },
    {
        "chunk_id": "chunk-002",
        "document_name": "document.pdf",
        "section_title": "Diagnosis",
        "page_number": "12",
        "content": "NICE CG128 recommends maximum wait of 3 months for assessment.",
        "similarity": 0.72,
    },
    {
        "chunk_id": "chunk-003",
        "document_name": "irrelevant.pdf",
        "section_title": "Cooking",
        "page_number": "1",
        "content": "How to bake a chocolate cake from scratch.",
        "similarity": 0.30,
    },
]


class TestRankedChunk:
    def test_creation(self):
        chunk = RankedChunk(
            chunk_id="c1",
            document_name="doc.pdf",
            section_title="Section",
            page_number="1",
            content="test content",
            vector_score=0.8,
            rerank_score=0.9,
        )
        assert chunk.chunk_id == "c1"
        assert chunk.vector_score == 0.8
        assert chunk.rerank_score == 0.9

    def test_model_dump(self):
        chunk = RankedChunk(
            chunk_id="c1",
            document_name="doc.pdf",
            section_title="Section",
            page_number="1",
            content="test",
            vector_score=0.8,
            rerank_score=0.9,
        )
        d = chunk.model_dump()
        assert isinstance(d, dict)
        assert d["chunk_id"] == "c1"


class TestClinicalReranker:
    @patch("reranker.LocalCrossEncoderReranker")
    def test_empty_chunks(self, mock_local):
        reranker = ClinicalReranker(top_n=5)
        result = reranker.rerank("test query", [])
        assert result == []

    @patch("reranker.LocalCrossEncoderReranker")
    def test_rerank_returns_top_n(self, mock_local_cls):
        mock_backend = MagicMock()
        mock_backend.rerank.return_value = [
            RankedChunk(chunk_id=f"chunk-{i}", document_name="doc.pdf",
                        section_title="S", page_number=str(i), content="c",
                        vector_score=0.9 - i * 0.1, rerank_score=1.0 - i * 0.1)
            for i in range(10)
        ]
        mock_local_cls.return_value = mock_backend

        reranker = ClinicalReranker(top_n=3)
        result = reranker.rerank("test query", SAMPLE_CHUNKS)
        assert len(result) == 3
        assert result[0].rerank_score >= result[1].rerank_score

    @patch("reranker.LocalCrossEncoderReranker")
    def test_rerank_ordering(self, mock_local_cls):
        mock_backend = MagicMock()
        # Backend returns pre-sorted results (highest first)
        mock_backend.rerank.return_value = [
            RankedChunk(chunk_id="high", document_name="doc.pdf",
                        section_title="S", page_number="2", content="c",
                        vector_score=0.9, rerank_score=0.95),
            RankedChunk(chunk_id="low", document_name="doc.pdf",
                        section_title="S", page_number="1", content="c",
                        vector_score=0.5, rerank_score=0.2),
        ]
        mock_local_cls.return_value = mock_backend

        reranker = ClinicalReranker(top_n=5)
        result = reranker.rerank("test", SAMPLE_CHUNKS[:2])
        assert result[0].chunk_id == "high"
        assert result[1].chunk_id == "low"
