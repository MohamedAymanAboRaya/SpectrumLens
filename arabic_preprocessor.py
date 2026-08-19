"""
SpectrumLens — Arabic & Multilingual Preprocessor
==================================================
Implements the retrieval-preserving normalization pipeline described in the
hackathon guidelines: Arabic and English preprocessing as a dual-field
architecture that improves lexical matching and embedding consistency
WITHOUT destroying original text needed for citations.

Architecture (per guidelines):
    Original Document
           │
           ├──────────────► original_text (citations / display — NEVER modified)
           │
           ▼
    ArabicPreprocessor.normalize()
           │
           ▼
    normalized_text  ──► Sparse / BM25 / FTS
                     └──► Embedding model (BGE-M3)

Usage:
    from arabic_preprocessor import ArabicPreprocessor, detect_language

    preprocessor = ArabicPreprocessor()
    result = preprocessor.process(text)
    # result.original_text    — untouched
    # result.normalized_text  — cleaned for retrieval
    # result.language         — "ar" | "en" | "mixed" | "unknown"
"""

import re
import unicodedata
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("SpectrumLens-Preprocessor")


# ─── Arabic Unicode ranges & characters ──────────────────────────────────────────
# Arabic diacritics (tashkeel / harakat)
ARABIC_DIACRITICS = re.compile(
    r"[\u064B\u064C\u064D\u064E\u064F\u0650"   # tanwin + kasra/fatha/damma
    r"\u0651\u0652\u0653\u0654\u0655"           # shadda, sukun, maddah
    r"\u0656\u0657\u0658\u0659\u065A"           # extended tashkeel
    r"\u065B\u065C\u065D\u065E\u065F"           # more tashkeel
    r"\u0670"                                    # alef wasla superscript
    r"\u06D6-\u06DC\u06DF-\u06E4"              # Quranic annotation signs
    r"\u06E7\u06E8\u06EA-\u06ED]"              # more annotation
)

# Arabic tatweel / kashida (ـ)
TATWEEL = "\u0640"

# Alef variants → plain Alef ا
ALEF_VARIANTS = str.maketrans({
    "\u0622": "\u0627",   # آ  → ا
    "\u0623": "\u0627",   # أ  → ا
    "\u0625": "\u0627",   # إ  → ا
    "\u0671": "\u0627",   # ٱ  → ا
    "\u0672": "\u0627",   # ٲ  → ا
    "\u0673": "\u0627",   # ٳ  → ا
})

# Ya / Alef Maqsura variants → Ya ي
YA_VARIANTS = str.maketrans({
    "\u0649": "\u064A",   # ى  → ي
    "\u06CC": "\u064A",   # ی  (Farsi Ya) → ي
})

# Persian / Arabic character variants
PERSIAN_ARABIC_VARIANTS = str.maketrans({
    "\u06A9": "\u0643",   # ک  (Farsi Kaf) → ك
    "\u06AF": "\u063A",   # گ  → غ  (no exact Arabic equiv, use closest)
    "\u0660": "0",        # ٠ Arabic-Indic 0 → ASCII
    "\u0661": "1",        # ١ → 1
    "\u0662": "2",        # ٢ → 2
    "\u0663": "3",        # ٣ → 3
    "\u0664": "4",        # ٤ → 4
    "\u0665": "5",        # ٥ → 5
    "\u0666": "6",        # ٦ → 6
    "\u0667": "7",        # ٧ → 7
    "\u0668": "8",        # ٨ → 8
    "\u0669": "9",        # ٩ → 9
    "\u06F0": "0",        # ۰ Extended Arabic-Indic 0 → ASCII
    "\u06F1": "1",
    "\u06F2": "2",
    "\u06F3": "3",
    "\u06F4": "4",
    "\u06F5": "5",
    "\u06F6": "6",
    "\u06F7": "7",
    "\u06F8": "8",
    "\u06F9": "9",
})

# Arabic text detection — a block of text is "Arabic" if it has enough Arabic chars
_ARABIC_CHAR_PATTERN = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")


# ─── Output type ─────────────────────────────────────────────────────────────────
@dataclass
class ProcessedText:
    """
    Dual representation as recommended by the guidelines.
    Always display/cite from original_text.
    Always embed/search with normalized_text.
    """
    original_text:   str
    normalized_text: str
    language:        str  # "ar" | "en" | "mixed" | "unknown"


# ─── Language Detection (free, offline) ──────────────────────────────────────────
def detect_language(text: str) -> str:
    """
    Lightweight language detection without any external library.

    Strategy:
    1. Count Arabic Unicode characters vs total alpha characters.
    2. >40% Arabic chars → "ar"
    3. <5%  Arabic chars → "en" (or unknown if very short)
    4. Otherwise         → "mixed"

    Falls back to langdetect if installed for better accuracy on short texts.
    """
    if not text or not text.strip():
        return "unknown"

    total_alpha = sum(1 for c in text if c.isalpha())
    arabic_chars = len(_ARABIC_CHAR_PATTERN.findall(text))

    if total_alpha == 0:
        return "unknown"

    arabic_ratio = arabic_chars / total_alpha

    if arabic_ratio >= 0.40:
        return "ar"
    elif arabic_ratio <= 0.05:
        # Try langdetect for better accuracy on ambiguous English text
        try:
            from langdetect import detect as ld_detect  # type: ignore
            lang = ld_detect(text)
            return lang if lang else "en"
        except Exception:
            return "en"
    else:
        return "mixed"


# ─── Arabic Normalizer ────────────────────────────────────────────────────────────
class ArabicNormalizer:
    """
    Applies the full Arabic normalization pipeline from the hackathon guidelines.
    Each step is a separate method for testability and composability.
    """

    def __init__(self, normalize_ta_marbuta: bool = False):
        """
        Args:
            normalize_ta_marbuta: If True, normalize ة → ه.
                Set False by default — Ta Marbuta carries morphological
                information important for Arabic medical terminology.
                Enable only if your BM25 engine doesn't handle it.
        """
        self.normalize_ta_marbuta = normalize_ta_marbuta

    # ── Step A: Unicode normalization ─────────────────────────────────────────────
    @staticmethod
    def unicode_normalize(text: str) -> str:
        """NFKC: handles characters with multiple Unicode representations."""
        return unicodedata.normalize("NFKC", text)

    # ── Step B: Remove Tatweel ────────────────────────────────────────────────────
    @staticmethod
    def remove_tatweel(text: str) -> str:
        """Remove Arabic tatweel/kashida (ـ) — purely decorative elongation."""
        return text.replace(TATWEEL, "")

    # ── Step C: Remove Tashkeel (diacritics) ─────────────────────────────────────
    @staticmethod
    def remove_tashkeel(text: str) -> str:
        """
        Remove Arabic diacritics (harakat).
        Improves BM25 matching — Arabic speakers rarely type diacritics in queries.
        """
        return ARABIC_DIACRITICS.sub("", text)

    # ── Step D: Normalize Alef variants ──────────────────────────────────────────
    @staticmethod
    def normalize_alef(text: str) -> str:
        """Normalize أ إ آ ٱ → ا (plain Alef)."""
        return text.translate(ALEF_VARIANTS)

    # ── Step E: Normalize Ya / Alef Maqsura ──────────────────────────────────────
    @staticmethod
    def normalize_ya(text: str) -> str:
        """Normalize ى (Alef Maqsura) and Farsi ی → ي (Arabic Ya)."""
        return text.translate(YA_VARIANTS)

    # ── Step F: Normalize Ta Marbuta (optional) ───────────────────────────────────
    @staticmethod
    def normalize_ta_marbuta_fn(text: str) -> str:
        """Normalize ة → ه. Enable only when BM25 needs exact stem matching."""
        return text.replace("\u0629", "\u0647")   # ة → ه

    # ── Step G: Persian/Arabic character & digit variants ─────────────────────────
    @staticmethod
    def normalize_persian_variants(text: str) -> str:
        """Normalize Farsi Kaf/Gaf, Arabic-Indic digits → ASCII."""
        return text.translate(PERSIAN_ARABIC_VARIANTS)

    # ── Step H: Whitespace normalization ─────────────────────────────────────────
    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Collapse multiple whitespace (incl. non-breaking space) into single space."""
        text = re.sub(r"[\u00A0\u200B\u200C\u200D\uFEFF]", " ", text)  # special spaces
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # ── Step I: Punctuation normalization ─────────────────────────────────────────
    @staticmethod
    def normalize_punctuation(text: str) -> str:
        """Normalize Arabic punctuation to ASCII equivalents for BM25."""
        text = text.replace("،", ",").replace("؛", ";").replace("؟", "?")
        text = text.replace("«", '"').replace("»", '"')
        return text

    # ── Full pipeline ─────────────────────────────────────────────────────────────
    def normalize(self, text: str) -> str:
        """
        Apply the complete Arabic normalization pipeline.
        Order matters — tashkeel removal must come AFTER unicode normalization.
        """
        text = self.unicode_normalize(text)
        text = self.remove_tatweel(text)
        text = self.remove_tashkeel(text)
        text = self.normalize_alef(text)
        text = self.normalize_ya(text)
        if self.normalize_ta_marbuta:
            text = self.normalize_ta_marbuta_fn(text)
        text = self.normalize_persian_variants(text)
        text = self.normalize_punctuation(text)
        text = self.normalize_whitespace(text)
        return text


# ─── English Normalizer ───────────────────────────────────────────────────────────
class EnglishNormalizer:
    """
    Light normalization for English clinical text.
    Aggressive cleaning is avoided to preserve medical terminology,
    clinical codes (DSM-5, ICD-10), and numeric values.
    """

    @staticmethod
    def normalize(text: str) -> str:
        # Unicode normalization
        text = unicodedata.normalize("NFKC", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove control characters (keep printable ASCII + extended)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        return text.strip()


# ─── Unified Preprocessor ────────────────────────────────────────────────────────
class ArabicPreprocessor:
    """
    Top-level preprocessor that:
    1. Detects the language of input text.
    2. Routes to the appropriate normalization pipeline.
    3. Returns a ProcessedText with both original and normalized representations.

    This is the only class you need to import in other modules.
    """

    def __init__(self, normalize_ta_marbuta: bool = False):
        self._arabic   = ArabicNormalizer(normalize_ta_marbuta=normalize_ta_marbuta)
        self._english  = EnglishNormalizer()

    def process(self, text: str) -> ProcessedText:
        """
        Process text and return dual representation.

        The original_text is NEVER modified — it is preserved for
        display and citation rendering as required by the guidelines.
        """
        if not text or not text.strip():
            return ProcessedText(
                original_text=text,
                normalized_text=text,
                language="unknown",
            )

        lang = detect_language(text)

        if lang == "ar":
            normalized = self._arabic.normalize(text)
        elif lang == "mixed":
            # Mixed Arabic/English: apply Arabic pipeline (safe for English too)
            normalized = self._arabic.normalize(text)
        else:
            # English or unknown
            normalized = self._english.normalize(text)

        return ProcessedText(
            original_text=text,
            normalized_text=normalized,
            language=lang,
        )

    def normalize_query(self, query: str) -> str:
        """
        Convenience method for normalizing a user query before embedding.
        Detects language and applies appropriate normalization.
        """
        result = self.process(query)
        return result.normalized_text


# ─── BGE-M3 query prefix helper ──────────────────────────────────────────────────
def add_bge_query_prefix(query: str, lang: str = "en") -> str:
    """
    BGE-M3 recommendation: prepend instruction prefix to queries for better retrieval.
    Documents (during indexing) do NOT get this prefix — only queries at search time.

    Uses language-specific prefixes for better cross-lingual recall:
      - Arabic: Arabic instruction prefix
      - English/default: English instruction prefix
    """
    if lang == "ar":
        return f"searchify: {query}"
    return f"Represent this sentence for searching relevant passages: {query}"


# ─── Standalone test / CLI ────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    preprocessor = ArabicPreprocessor()

    test_cases = [
        # Arabic clinical text with diacritics and Alef variants
        "اضطرابُ طيفِ التَّوحُّد (أو إضطراب التوحد) هو اضطراب عصبي تطوري.",
        # Arabic with tatweel
        "الأعـــراض الرئيسية للتوحد تشمل صعوبات في التواصل الاجتماعي.",
        # Arabic with Arabic-Indic digits
        "تشخيص التوحد يبدأ في سن ١٨ شهرًا وفقًا لـ AAP.",
        # English medical text
        "ASD screening: M-CHAT-R/F at 18 and 24 months (AAP 2020 guidelines).",
        # Mixed Arabic-English
        "DSM-5 معايير التشخيص تشمل Criterion A و Criterion B.",
    ]

    print("\n" + "═" * 68)
    print("  ArabicPreprocessor — Test Results")
    print("═" * 68)

    for i, text in enumerate(test_cases, 1):
        result = preprocessor.process(text)
        print(f"\n[Test {i}]")
        print(f"  Lang     : {result.language}")
        print(f"  Original : {result.original_text[:80]}")
        print(f"  Normalized: {result.normalized_text[:80]}")

    print("\n" + "═" * 68)
    print("✅  All preprocessing tests complete.")
    print("═" * 68 + "\n")
