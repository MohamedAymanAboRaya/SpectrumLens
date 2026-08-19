"""
SpectrumLens — Day 1: Document Ingestion & Chunking Pipeline
============================================================
Reads clinical PDFs, strips noise (headers/footers), extracts medical
section headers, and produces semantically coherent chunks saved as JSON.

Arabic/Multilingual update:
  • Each chunk now carries BOTH original_text (for citations) and
    normalized_text (for BGE-M3 embedding + BM25 retrieval).
  • Language detected per chunk: "ar" | "en" | "mixed" | "unknown"

Usage:
    python day1_ingestion.py
Expects PDFs in:  data/raw_pdfs/
Writes JSON to:   data/processed_chunks/day1_chunks_output.json
"""

import os
import re
import json
import uuid
import logging
import unicodedata
from typing import List, Dict, Any

import fitz  # PyMuPDF
import tiktoken
from pydantic import BaseModel, Field
from langchain_text_splitters import RecursiveCharacterTextSplitter

from arabic_preprocessor import ArabicPreprocessor, detect_language


# ─── Source Registry Loader ─────────────────────────────────────────────────────
REGISTRY_PATH = os.path.join("data", "source_registry.json")


def load_source_registry() -> Dict[str, Dict[str, str]]:
    """Load source_registry.json → {document_name: {source_url, authority_tier, ...}}."""
    if not os.path.exists(REGISTRY_PATH):
        logger.warning(f"Source registry not found at '{REGISTRY_PATH}' — URLs will be empty.")
        return {}
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Support both list-of-dicts and dict-of-dicts formats
    if isinstance(raw, list):
        return {entry["document_name"]: entry for entry in raw if "document_name" in entry}
    return raw


def backfill_source_urls(chunks_path: str = None) -> int:
    """Retroactively add source_url and authority_tier to existing chunks.

    Reads the JSON file, enriches each chunk via the registry, writes it back,
    and returns the number of chunks updated.
    """
    if chunks_path is None:
        chunks_path = os.path.join("data", "processed_chunks", "day1_chunks_output.json")
    if not os.path.exists(chunks_path):
        logger.error(f"Chunks file not found: {chunks_path}")
        return 0

    registry = load_source_registry()
    if not registry:
        return 0

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    updated = 0
    for chunk in chunks:
        doc_name = chunk.get("document_name", "")
        entry = registry.get(doc_name)
        if entry:
            chunk["source_url"] = entry.get("source_url", "")
            chunk["authority_tier"] = entry.get("authority_tier", "")
            updated += 1

    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)

    logger.info(f"Backfill complete: {updated}/{len(chunks)} chunks enriched in {chunks_path}")
    return updated


# ─── Standalone Normalization Functions (per hackathon guidelines) ──────────────
def normalize_arabic(text: str) -> str:
    """Arabic normalization pipeline per hackathon guidelines."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'[\u0640]', '', text)
    text = re.sub(
        r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]',
        '', text,
    )
    text = re.sub(r'[أإآٱ]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = text.replace('ی', 'ي').replace('ك', 'ك')
    arabic_digits = '٠١٢٣٤٥٦٧٨٩'
    for i, d in enumerate(arabic_digits):
        text = text.replace(d, str(i))
    text = text.replace('،', ',').replace('؛', ';').replace('؟', '?')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_english(text: str) -> str:
    """Light English normalization."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_text(text: str, language: str) -> str:
    """Route to language-specific normalizer."""
    if language == "ar":
        return normalize_arabic(text)
    return normalize_english(text)

# ─── Logging ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SpectrumLens-Ingestion")


# ─── Data Model ──────────────────────────────────────────────────────────────────
class ChunkMetadata(BaseModel):
    document_name:   str
    page_number:     str
    section_title:   str
    chunk_id:        str = Field(default_factory=lambda: str(uuid.uuid4()))
    original_text:   str          # original text — NEVER modified (for citations/display)
    text:            str = ""     # backward-compat alias for original_text
    normalized_text: str = ""     # cleaned for embedding + BM25 retrieval
    language:        str = "en"   # "ar" | "en" | "mixed" | "unknown"


# ─── PDF Noise Removal ───────────────────────────────────────────────────────────
class TextCleaner:
    """Removes page headers, footers, and control characters."""

    def __init__(self, header_margin: float = 0.08, footer_margin: float = 0.08):
        self.header_margin = header_margin
        self.footer_margin = footer_margin

    def is_noise(self, bbox: fitz.Rect, page_rect: fitz.Rect) -> bool:
        y0, y1 = bbox.y0, bbox.y1
        if y1 < page_rect.height * self.header_margin:
            return True   # header region
        if y0 > page_rect.height * (1 - self.footer_margin):
            return True   # footer region
        return False

    def clean_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]", "", text).strip()


# ─── Document Parser ─────────────────────────────────────────────────────────────
class DiagnosticDocumentParser:
    """
    Parses a clinical PDF into structured text blocks with adaptive
    section-header detection.

    Key improvement over a fixed baseline:
        Each PDF has its own font-size conventions (6pt body for AAP journals,
        8pt for the 100-Day Kit, 10pt for NICE guidelines, etc.).
        This parser measures the actual median body-text font size from the
        document itself, then classifies any span >= HEADER_RATIO × body_size
        as a section header — making it robust across all 5 source PDFs.
    """

    HEADER_RATIO    = 1.25
    MIN_HEADER_SIZE = 9.0
    MAX_HEADER_SIZE = 30.0
    MAX_HEADER_CHARS = 120

    def __init__(self, cleaner: TextCleaner):
        self.cleaner = cleaner

    def _detect_body_font_size(self, doc: fitz.Document) -> float:
        import statistics
        sizes = []
        for page_num in range(min(10, len(doc))):
            page = doc[page_num]
            for block in page.get_text("dict").get("blocks", []):
                for line in block.get("lines", []):
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if len(text) >= 20:
                            sizes.append(span["size"])
        if not sizes:
            return 10.0
        median = statistics.median(sizes)
        logger.info(f"  Auto-detected body font size: {median:.1f}pt")
        return median

    def _is_section_header(self, span: Dict[str, Any], text: str, body_size: float) -> bool:
        """
        Returns True if the span looks like a section/subsection header.
        Handles three conventions found across the 5 clinical PDFs:

        1. Size ratio  — span is >=1.25x body size (e.g., NICE CG128 h1/h2)
        2. Bold font   — font name ends in '.B' or contains 'Bold'/'bold'
                         at body_size ± 1pt (AAP journal PDFs)
        3. ALL-CAPS    — short all-caps text at or near body size
                         (NICE section headers like 'YOUR RESPONSIBILITY')
        """
        font_size = span.get("size", 0)
        font_name = span.get("font", "")
        is_bold = (
            "bold" in font_name.lower()
            or font_name.endswith(".B")
            or font_name.endswith("-B")
        )

        size_ratio_hit = (
            font_size >= body_size * self.HEADER_RATIO
            and self.MIN_HEADER_SIZE <= font_size <= self.MAX_HEADER_SIZE
        )
        bold_body_hit = (
            is_bold
            and abs(font_size - body_size) <= 1.0
            and 5 < len(text) < self.MAX_HEADER_CHARS
        )
        allcaps_hit = (
            text.isupper()
            and abs(font_size - body_size) <= 1.5
            and 4 < len(text) < 60
        )
        return size_ratio_hit or bold_body_hit or allcaps_hit

    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        doc = fitz.open(file_path)
        parsed_blocks = []
        body_size = self._detect_body_font_size(doc)
        current_section = "General Overview"

        for page_num, page in enumerate(doc, start=1):
            page_rect = page.rect
            blocks = page.get_text("dict").get("blocks", [])

            for block in blocks:
                if "lines" not in block:
                    continue
                bbox = fitz.Rect(block["bbox"])
                if self.cleaner.is_noise(bbox, page_rect):
                    continue

                block_text = ""
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if not text:
                            continue
                        if (
                            self._is_section_header(span, text, body_size)
                            and len(text) <= self.MAX_HEADER_CHARS
                        ):
                            current_section = self.cleaner.clean_text(text)
                        else:
                            block_text += text + " "

                block_text = self.cleaner.clean_text(block_text)
                if block_text:
                    parsed_blocks.append({
                        "text": block_text,
                        "page_number": str(page_num),
                        "section_title": current_section,
                    })

        doc.close()
        return parsed_blocks


# ─── Section-Aware Chunker ───────────────────────────────────────────────────────
class SectionAwareChunker:
    """
    Merges parsed blocks into chunks that respect section boundaries
    and stay within [min_tokens, max_tokens].

    Arabic/Multilingual update:
        After chunking, each chunk's text is passed through ArabicPreprocessor
        to produce a normalized_text and language field.
    """

    def __init__(self, min_tokens: int = 300, max_tokens: int = 700):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.tokenizer  = tiktoken.get_encoding("cl100k_base")
        self.splitter   = RecursiveCharacterTextSplitter(
            chunk_size=max_tokens,
            chunk_overlap=50,
            length_function=lambda x: len(self.tokenizer.encode(x)),
            separators=["\n\n", "\n", ". ", " "],
        )
        self.preprocessor = ArabicPreprocessor()

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def chunk(
        self,
        parsed_blocks: List[Dict[str, Any]],
        document_name: str,
    ) -> List[Dict[str, Any]]:

        final_chunks: List[Dict[str, Any]] = []
        current_chunk_text  = ""
        current_chunk_pages: set = set()
        current_section     = None

        def flush_chunk():
            nonlocal current_chunk_text, current_chunk_pages
            if current_chunk_text:
                # ── Bilingual preprocessing ────────────────────────────────
                processed = self.preprocessor.process(current_chunk_text.strip())
                chunk_data = ChunkMetadata(
                    document_name=document_name,
                    page_number="-".join(sorted(current_chunk_pages)),
                    section_title=current_section or "Unknown Section",
                    original_text=processed.original_text,
                    normalized_text=processed.normalized_text,
                    language=processed.language,
                ).model_dump()
                # backward-compat alias: text == original_text
                chunk_data["text"] = chunk_data["original_text"]
                final_chunks.append(chunk_data)
            current_chunk_text  = ""
            current_chunk_pages = set()

        for block in parsed_blocks:
            b_text    = block["text"]
            b_section = block["section_title"]
            b_page    = block["page_number"]

            if b_section != current_section and current_section is not None:
                if self.count_tokens(current_chunk_text) >= self.min_tokens:
                    flush_chunk()
                current_section = b_section
            elif current_section is None:
                current_section = b_section

            pot_text = current_chunk_text + " " + b_text if current_chunk_text else b_text

            if self.count_tokens(pot_text) > self.max_tokens:
                if self.count_tokens(b_text) > self.max_tokens:
                    for split in self.splitter.split_text(b_text):
                        current_chunk_text = split
                        current_chunk_pages.add(b_page)
                        flush_chunk()
                else:
                    flush_chunk()
                    current_chunk_text = b_text
                    current_chunk_pages.add(b_page)
            else:
                current_chunk_text = pot_text
                current_chunk_pages.add(b_page)

            if self.min_tokens <= self.count_tokens(current_chunk_text) <= self.max_tokens:
                flush_chunk()

        if current_chunk_text:
            flush_chunk()

        return final_chunks


# ─── Orchestration Pipeline ──────────────────────────────────────────────────────
class SpectrumLensIngestionPipeline:
    """Top-level pipeline: parse → chunk → dual-field preprocessing for a single PDF."""

    def __init__(self):
        cleaner      = TextCleaner()
        self.parser  = DiagnosticDocumentParser(cleaner)
        self.chunker = SectionAwareChunker()
        self._registry = load_source_registry()

    def _inject_source_url(self, chunks: List[Dict[str, Any]]) -> None:
        """Attach source_url and authority_tier from the registry to each chunk."""
        for chunk in chunks:
            doc_name = chunk.get("document_name", "")
            entry = self._registry.get(doc_name)
            if entry:
                chunk["source_url"] = entry.get("source_url", "")
                chunk["authority_tier"] = entry.get("authority_tier", "")
            else:
                chunk["source_url"] = ""
                chunk["authority_tier"] = ""

    def process_document(
        self,
        file_path: str,
        document_name: str,
    ) -> List[Dict[str, Any]]:
        logger.info(f"Processing: {file_path}")
        parsed_blocks = self.parser.parse(file_path)
        logger.info(f"  → {len(parsed_blocks)} raw blocks extracted.")
        chunks = self.chunker.chunk(parsed_blocks, document_name)
        self._inject_source_url(chunks)
        lang_counts: Dict[str, int] = {}
        for c in chunks:
            l = c.get("language", "unknown")
            lang_counts[l] = lang_counts.get(l, 0) + 1
        lang_str = ", ".join(f"{k}={v}" for k, v in sorted(lang_counts.items()))
        logger.info(f"  → {len(chunks)} chunks produced ({lang_str}).")
        return chunks


# ─── Entry Point ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    RAW_DIR = "data/raw_pdfs"
    OUT_DIR = "data/processed_chunks"
    os.makedirs(OUT_DIR, exist_ok=True)

    pipeline   = SpectrumLensIngestionPipeline()
    all_chunks: List[Dict[str, Any]] = []

    if os.path.exists(RAW_DIR):
        pdf_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".pdf")]
        if not pdf_files:
            logger.warning(f"No PDFs found in '{RAW_DIR}'. Add clinical PDFs first.")
        for filename in pdf_files:
            doc_name = os.path.splitext(filename)[0]
            try:
                all_chunks.extend(
                    pipeline.process_document(os.path.join(RAW_DIR, filename), doc_name)
                )
            except Exception as e:
                logger.error(f"Failed to process {filename}: {e}")

        out_path = os.path.join(OUT_DIR, "day1_chunks_output.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=4)

        lang_counts: Dict[str, int] = {}
        for c in all_chunks:
            l = c.get("language", "unknown")
            lang_counts[l] = lang_counts.get(l, 0) + 1
        lang_str = ", ".join(f"{k}={v}" for k, v in sorted(lang_counts.items()))
        logger.info(
            f"Day 1 Complete! {len(all_chunks)} chunks saved → {out_path} ({lang_str})"
        )
    else:
        logger.error(f"Directory '{RAW_DIR}' not found.")
