"""Text front-end: native-orthography clean-up, then africa-g2p universal IPA (the symbols the expanded bases understand)."""
import re, unicodedata

# lookalike characters that stand in for the real orthographic letter in scraped text
HOMOGLYPHS = str.maketrans({"ε": "ɛ", "ԑ": "ɛ", "ð": "ɖ", "đ": "ɖ", "ô": "o"})


def clean(text: str) -> str:
    """NFC, lowercase, lookalike letters, punctuation removed. Hyphen -> space, apostrophes dropped."""
    t = unicodedata.normalize("NFC", text).lower().translate(HOMOGLYPHS)
    for a in "’‘ʼ`'":
        t = t.replace(a, "")
    t = t.replace("-", " ").replace("‐", " ").replace("‑", " ")
    t = re.sub(r"[^\w\s]", " ", t).replace("_", " ")
    return re.sub(r"\s+", " ", t).strip()


def to_ipa(text: str, lang: str) -> str:
    """africa-g2p IPA, word by word (words joined by a space)."""
    from africa_g2p import convert_to_ipa
    words = []
    for w in clean(text).split():
        try:
            words.append("".join(ch for ch in convert_to_ipa(w, lang) if not ch.isspace()))
        except Exception:
            words.append(w)
    return " ".join(x for x in words if x)


def normalise(text: str, lang: str, mode: str = "ipa") -> str:
    """mode 'ipa' (default, what the expanded bases are built for) or 'clean' (native orthography only)."""
    return to_ipa(text, lang) if mode == "ipa" else clean(text)
