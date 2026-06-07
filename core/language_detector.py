"""Core language detection module for AI Agent"""
import re
from typing import Dict


LANGUAGE_PATTERNS: Dict[str, str] = {
    "ar": r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]',
    "he": r'[\u0590-\u05FF\uFB1D-\uFB4F]',
    "ja": r'[\u3040-\u309F\u30A0-\u30FF]',
    "ko": r'[\uAC00-\uD7AF\u1100-\u11FF]',
    "hi": r'[\u0900-\u097F]',
    "th": r'[\u0E00-\u0E7F]',
    "ru": r'[\u0400-\u04FF]',
    "zh": r'[\u4E00-\u9FFF\u3400-\u4DBF]',
}

LANGUAGE_NAMES: Dict[str, str] = {
    "en": "English",
    "ar": "Arabic",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "hi": "Hindi",
    "th": "Thai",
    "he": "Hebrew",
    "tr": "Turkish",
    "nl": "Dutch",
    "sv": "Swedish",
    "pl": "Polish",
}

LATIN_WORD_LISTS = {
    "en": {'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
           'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can',
           'shall', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us',
           'them', 'my', 'your', 'his', 'its', 'our', 'their', 'this', 'that', 'these', 'those',
           'what', 'which', 'when', 'where', 'who', 'how', 'not', 'no', 'but', 'and', 'or',
           'write', 'read', 'create', 'function', 'code', 'python', 'file', 'test', 'run'},
    "es": {'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'de', 'del', 'en', 'con',
           'por', 'para', 'como', 'pero', 'mas', 'este', 'esta', 'esto', 'fue', 'ser', 'estar',
           'haber', 'tiene', 'hacer', 'puede', 'hay', 'muy', 'bien', 'todo', 'cada', 'otro'},
    "fr": {'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'en', 'dans', 'avec', 'pour',
           'comme', 'mais', 'plus', 'ce', 'cette', 'ces', 'qui', 'que', 'est', 'sont', 'avoir',
           'fait', 'peut', 'tout', 'bien', 'aussi', 'entre', 'apres', 'avant', 'encore', 'sans',
           'comment', 'allez', 'bonjour', 'merci', 'oui', 'non', 'je', 'tu', 'nous', 'vous',
           'bonjour', 'soir', 'jour', 'nuit', 'homme', 'femme', 'enfant', 'monde', 'temps'},
    "de": {'der', 'die', 'das', 'ein', 'eine', 'einem', 'einen', 'von', 'in', 'mit', 'auf',
           'fur', 'aber', 'auch', 'als', 'noch', 'nur', 'schon', 'wenn', 'ich', 'nicht', 'sich',
           'ist', 'hat', 'kann', 'wird', 'bei', 'nach', 'wie', 'was', 'oder', 'so', 'alle',
           'guten', 'tag', 'danke', 'ja', 'nein', 'wir', 'ihr', 'haben', 'sein', 'machen'},
    "it": {'il', 'lo', 'la', 'i', 'gli', 'le', 'un', 'uno', 'una', 'di', 'in', 'con', 'per',
           'ma', 'che', 'non', 'questo', 'questa', 'questo', 'sono', 'essere', 'avere', 'fare',
           'anche', 'come', 'piu', 'giu', 'gia', 'cosa', 'dove', 'quando', 'perche', 'ogni',
           'buongiorno', 'grazie', 'si', 'no', 'io', 'tu', 'noi', 'voi', 'buona', 'tempo'},
    "pt": {'o', 'a', 'os', 'as', 'um', 'uma', 'uns', 'umas', 'de', 'em', 'com', 'para', 'por',
           'mas', 'que', 'nao', 'se', 'tem', 'este', 'esta', 'isso', 'ser', 'ter', 'fazer',
           'tambem', 'ainda', 'muito', 'bem', 'todo', 'cada', 'outro', 'quando', 'onde', 'porque',
           'ola', 'obrigado', 'sim', 'eu', 'voce', 'nos', 'eles', 'bom', 'dia', 'noite'},
}


def detect_language(text: str) -> str:
    """Detect the primary language of text"""
    if not text or not text.strip():
        return "en"

    for lang, pattern in LANGUAGE_PATTERNS.items():
        if re.search(pattern, text):
            return lang

    text_lower = text.lower()
    words = set(re.findall(r'\b\w+\b', text_lower))

    if not words:
        return "en"

    scores = {}
    for lang, word_list in LATIN_WORD_LISTS.items():
        count = len(words & word_list)
        scores[lang] = count

    best_lang = max(scores, key=scores.get)
    best_count = scores[best_lang]

    if best_count == 0:
        return "en"

    return best_lang


def get_language_name(code: str) -> str:
    """Get full language name from ISO code"""
    return LANGUAGE_NAMES.get(code, "Unknown")


def get_supported_languages() -> Dict[str, str]:
    """Get all supported languages"""
    return LANGUAGE_NAMES.copy()


def get_response_language(user_text: str) -> str:
    """Determine the language to respond in based on user input"""
    detected = detect_language(user_text)
    return detected
