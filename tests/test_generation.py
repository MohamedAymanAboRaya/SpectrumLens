"""Tests for the CRAG generation module (data models and verdict logic)."""

import pytest
from day3_generation import (
    EvalVerdict,
    ChunkEvaluation,
    ContextEvalReport,
    Citation,
    ClinicalResponse,
)


class TestEvalVerdict:
    def test_sufficient_value(self):
        assert EvalVerdict.SUFFICIENT.value == "SUFFICIENT"

    def test_insufficient_value(self):
        assert EvalVerdict.INSUFFICIENT.value == "INSUFFICIENT"

    def test_enum_members(self):
        assert len(EvalVerdict) == 2


class TestChunkEvaluation:
    def test_relevant_chunk(self):
        ev = ChunkEvaluation(
            chunk_id="c1",
            relevance_score=0.85,
            is_relevant=True,
            rationale="Highly relevant to ASD screening.",
        )
        assert ev.is_relevant is True
        assert ev.relevance_score == 0.85

    def test_irrelevant_chunk(self):
        ev = ChunkEvaluation(
            chunk_id="c2",
            relevance_score=0.3,
            is_relevant=False,
            rationale="Not relevant to the query.",
        )
        assert ev.is_relevant is False


class TestContextEvalReport:
    def test_sufficient_report(self):
        evals = [
            ChunkEvaluation(chunk_id="c1", relevance_score=0.8, is_relevant=True, rationale="r1"),
            ChunkEvaluation(chunk_id="c2", relevance_score=0.7, is_relevant=True, rationale="r2"),
        ]
        report = ContextEvalReport(
            verdict=EvalVerdict.SUFFICIENT,
            chunk_evaluations=evals,
            mean_relevance=0.75,
            relevant_count=2,
            evaluator_notes="Adequate evidence.",
        )
        assert report.verdict == EvalVerdict.SUFFICIENT
        assert report.mean_relevance == 0.75

    def test_insufficient_report(self):
        report = ContextEvalReport(
            verdict=EvalVerdict.INSUFFICIENT,
            chunk_evaluations=[],
            mean_relevance=0.0,
            relevant_count=0,
            evaluator_notes="No relevant chunks.",
        )
        assert report.verdict == EvalVerdict.INSUFFICIENT


class TestCitation:
    def test_creation(self):
        cit = Citation(
            document_name="peds.pdf",
            section_title="Screening",
            page_number="5",
        )
        assert cit.document_name == "peds.pdf"
        assert cit.page_number == "5"


class TestClinicalResponse:
    def test_sufficient_response(self):
        report = ContextEvalReport(
            verdict=EvalVerdict.SUFFICIENT,
            chunk_evaluations=[],
            mean_relevance=0.8,
            relevant_count=3,
            evaluator_notes="Good evidence.",
        )
        resp = ClinicalResponse(
            query="ASD screening age",
            verdict=EvalVerdict.SUFFICIENT,
            answer="The AAP recommends screening at 18 months.",
            citations=[Citation(document_name="doc.pdf", section_title="S", page_number="1")],
            context_report=report,
        )
        assert resp.verdict == EvalVerdict.SUFFICIENT
        assert resp.answer is not None
        assert len(resp.citations) == 1

    def test_insufficient_response(self):
        report = ContextEvalReport(
            verdict=EvalVerdict.INSUFFICIENT,
            chunk_evaluations=[],
            mean_relevance=0.0,
            relevant_count=0,
            evaluator_notes="No evidence.",
        )
        resp = ClinicalResponse(
            query="cure for autism",
            verdict=EvalVerdict.INSUFFICIENT,
            safe_failure_reason="No evidence found.",
            context_report=report,
        )
        assert resp.verdict == EvalVerdict.INSUFFICIENT
        assert resp.answer is None
        assert resp.safe_failure_reason is not None

    def test_default_values(self):
        report = ContextEvalReport(
            verdict=EvalVerdict.SUFFICIENT,
            chunk_evaluations=[],
            mean_relevance=0.8,
            relevant_count=2,
            evaluator_notes="ok",
        )
        resp = ClinicalResponse(
            query="test",
            verdict=EvalVerdict.SUFFICIENT,
            context_report=report,
        )
        assert resp.query_language == "en"
        assert resp.rerank_scores == []
        assert resp.citations == []
