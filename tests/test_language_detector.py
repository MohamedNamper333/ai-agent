"""Tests for core.language_detector module"""
import pytest
from core.language_detector import detect_language, get_language_name, get_supported_languages, get_response_language


class TestDetectLanguage:
    def test_english(self):
        assert detect_language("Hello, how are you?") == "en"

    def test_arabic(self):
        assert detect_language("مرحبا، كيف حالك؟") == "ar"

    def test_spanish(self):
        assert detect_language("Hola, como estas?") == "es"

    def test_french(self):
        assert detect_language("Bonjour, comment allez-vous?") == "fr"

    def test_german(self):
        assert detect_language("Guten Tag, wie geht es Ihnen?") == "de"

    def test_italian(self):
        assert detect_language("Buongiorno, come stai?") == "it"

    def test_portuguese(self):
        assert detect_language("Ola, como voce esta?") == "pt"

    def test_russian(self):
        assert detect_language("Привет, как дела?") == "ru"

    def test_chinese(self):
        assert detect_language("你好，你好吗？") == "zh"

    def test_japanese(self):
        assert detect_language("こんにちは、お元気ですか？") == "ja"

    def test_korean(self):
        assert detect_language("안녕하세요, 어떻게 지내세요?") == "ko"

    def test_hindi(self):
        assert detect_language("नमस्ते, आप कैसे हैं?") == "hi"

    def test_thai(self):
        assert detect_language("สวัสดีครับ สบายดีไหม?") == "th"

    def test_hebrew(self):
        assert detect_language("שלום, מה שלומך?") == "he"

    def test_empty_string(self):
        assert detect_language("") == "en"

    def test_none_input(self):
        assert detect_language(None) == "en"

    def test_numbers_only(self):
        assert detect_language("12345 !@#$%") == "en"

    def test_mixed_english_words(self):
        text = "I have been working on this project for a long time"
        assert detect_language(text) == "en"


class TestGetLanguageName:
    def test_english(self):
        assert get_language_name("en") == "English"

    def test_arabic(self):
        assert get_language_name("ar") == "Arabic"

    def test_spanish(self):
        assert get_language_name("es") == "Spanish"

    def test_unknown(self):
        assert get_language_name("xx") == "Unknown"


class TestGetSupportedLanguages:
    def test_returns_dict(self):
        langs = get_supported_languages()
        assert isinstance(langs, dict)

    def test_contains_english(self):
        langs = get_supported_languages()
        assert "en" in langs

    def test_contains_arabic(self):
        langs = get_supported_languages()
        assert "ar" in langs


class TestGetResponseLanguage:
    def test_english_input(self):
        assert get_response_language("Write a Python function") == "en"

    def test_arabic_input(self):
        assert get_response_language("اكتب لي دالة بايثون") == "ar"
