"""Tests for the Arabic/multilingual preprocessor module."""

import pytest
from arabic_preprocessor import (
    ArabicPreprocessor,
    ArabicNormalizer,
    EnglishNormalizer,
    detect_language,
    add_bge_query_prefix,
    ProcessedText,
)


class TestDetectLanguage:
    def test_english_text(self):
        assert detect_language("ASD screening at 18 months") == "en"

    def test_arabic_text(self):
        assert detect_language("اضطراب طيف التوحد") == "ar"

    def test_mixed_text(self):
        # "DSM-5 معايير التشخيص" — Arabic chars dominate ratio, detected as "ar"
        result = detect_language("DSM-5 معايير التشخيص")
        assert result in ("ar", "mixed")  # depends on char ratio threshold

    def test_empty_text(self):
        assert detect_language("") == "unknown"

    def test_none_text(self):
        assert detect_language(None) == "unknown"

    def test_numbers_only(self):
        assert detect_language("12345") == "unknown"


class TestArabicNormalizer:
    def setup_method(self):
        self.normalizer = ArabicNormalizer()

    def test_remove_tatweel(self):
        text = "الأعـــراض"
        result = self.normalizer.remove_tatweel(text)
        assert "ـ" not in result

    def test_remove_tashkeel(self):
        text = "اضطرابُ طيفِ التَّوحُّد"
        result = self.normalizer.remove_tashkeel(text)
        assert "\u064B" not in result  # tanwin
        assert "\u064E" not in result  # fatha

    def test_normalize_alef(self):
        text = "أ إ آ"
        result = self.normalizer.normalize_alef(text)
        assert result == "ا ا ا"

    def test_normalize_ya(self):
        text = "ى ی"
        result = self.normalizer.normalize_ya(text)
        assert result == "ي ي"

    def test_normalize_arabic_digits(self):
        text = "١٢٣٤٥"
        result = self.normalizer.normalize_persian_variants(text)
        assert result == "12345"

    def test_normalize_punctuation(self):
        text = "، ؛ ؟"
        result = self.normalizer.normalize_punctuation(text)
        assert result == ", ; ?"


class TestEnglishNormalizer:
    def test_basic_normalization(self):
        text = "  ASD   screening   at   18   months  "
        result = EnglishNormalizer.normalize(text)
        assert result == "ASD screening at 18 months"

    def test_remove_control_chars(self):
        text = "ASD\x00screening"
        result = EnglishNormalizer.normalize(text)
        assert "\x00" not in result


class TestArabicPreprocessor:
    def setup_method(self):
        self.preprocessor = ArabicPreprocessor()

    def test_english_processing(self):
        result = self.preprocessor.process("ASD screening at 18 months")
        assert result.language == "en"
        assert result.original_text == "ASD screening at 18 months"
        assert result.normalized_text is not None

    def test_arabic_processing(self):
        result = self.preprocessor.process("اضطراب طيف التوحد")
        assert result.language == "ar"
        assert result.original_text == "اضطراب طيف التوحد"

    def test_empty_text(self):
        result = self.preprocessor.process("")
        assert result.language == "unknown"

    def test_normalize_query(self):
        query = self.preprocessor.normalize_query("ASD screening")
        assert query is not None
        assert len(query) > 0


class TestBGEQueryPrefix:
    def test_english_prefix(self):
        result = add_bge_query_prefix("ASD screening", lang="en")
        assert result.startswith("Represent this sentence for searching relevant passages:")

    def test_arabic_prefix(self):
        result = add_bge_query_prefix("توحد", lang="ar")
        assert result.startswith("searchify:")

    def test_default_prefix(self):
        result = add_bge_query_prefix("test query")
        assert result.startswith("Represent this sentence for searching relevant passages:")
