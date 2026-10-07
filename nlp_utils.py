import re


def clean_text(text: str) -> str:
    """Normalise a user question so training and prediction see identical text."""
    t = str(text).lower()
    t = re.sub(r"e\s*&\s*tc", "entc", t)      # e&tc / e & tc  ->  entc
    t = re.sub(r"a\s*&\s*r\b", "ar", t)       # a&r            ->  ar
    t = re.sub(r"[^a-z0-9 ]", " ", t)         # drop punctuation
    return re.sub(r"\s+", " ", t).strip()
