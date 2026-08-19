"""Tests for the evaluation harness precision@K and failure mode logic."""

import pytest
import json
from pathlib import Path
from demo_app import precision_at_k, failure_mode


class TestPrecisionAtK:
    def test_perfect_match(self):
        retrieved = [
            {"document_name": "doc_a.pdf"},
            {"document_name": "doc_b.pdf"},
            {"document_name": "doc_a.pdf"},
        ]
        gt = ["doc_a.pdf", "doc_b.pdf"]
        assert precision_at_k(retrieved, gt, 3) == 1.0

    def test_no_match(self):
        retrieved = [
            {"document_name": "other.pdf"},
            {"document_name": "other2.pdf"},
        ]
        gt = ["doc_a.pdf"]
        assert precision_at_k(retrieved, gt, 2) == 0.0

    def test_empty_gt(self):
        retrieved = [{"document_name": "doc.pdf"}]
        assert precision_at_k(retrieved, [], 1) == 0.0

    def test_partial_match(self):
        retrieved = [
            {"document_name": "doc_a.pdf"},
            {"document_name": "other.pdf"},
            {"document_name": "other2.pdf"},
        ]
        gt = ["doc_a.pdf"]
        p = precision_at_k(retrieved, gt, 3)
        assert abs(p - 1 / 3) < 1e-6

    def test_reverse_match(self):
        retrieved = [
            {"document_name": "doc_a.pdf"},
        ]
        gt = ["doc_a.pdf"]
        p = precision_at_k(retrieved, gt, 1)
        assert p == 1.0


class TestFailureMode:
    def test_no_results(self):
        fm = failure_mode([], ["doc.pdf"], 0.0)
        assert fm == "NO_RESULTS"

    def test_ok_when_no_gt(self):
        fm = failure_mode([{"document_name": "x.pdf"}], [], 0.8)
        assert fm is None

    def test_duplicate_detection(self):
        retrieved = [
            {"document_name": "doc.pdf"},
            {"document_name": "doc.pdf"},
            {"document_name": "doc.pdf"},
            {"document_name": "other.pdf"},
        ]
        fm = failure_mode(retrieved, ["target.pdf"], 0.5)
        assert "DUPLICATE" in fm

    def test_missing_source(self):
        retrieved = [
            {"document_name": "wrong.pdf"},
            {"document_name": "wrong2.pdf"},
        ]
        fm = failure_mode(retrieved, ["correct.pdf"], 0.8)
        assert "MISSING_SOURCE" in fm


class TestEvalDataset:
    def test_eval_file_exists(self):
        eval_path = Path("data/eval/eval_dataset.json")
        assert eval_path.exists(), f"Eval dataset not found at {eval_path}"

    def test_eval_dataset_structure(self):
        with open("data/eval/eval_dataset.json", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 30, f"Expected 30 questions, got {len(data)}"

    def test_eval_required_fields(self):
        with open("data/eval/eval_dataset.json", encoding="utf-8") as f:
            data = json.load(f)
        required = {"id", "question", "category", "difficulty", "ground_truth_sources"}
        for item in data:
            assert required.issubset(item.keys()), f"Missing fields in {item.get('id', '?')}"

    def test_eval_categories(self):
        with open("data/eval/eval_dataset.json", encoding="utf-8") as f:
            data = json.load(f)
        cats = {item["category"] for item in data}
        assert cats == {"factual", "inferential", "out_of_scope", "adversarial"}
