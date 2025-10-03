# ocr_text_cleanup.py
import re
from typing import List
from spellchecker import SpellChecker  # optional

# --------------------------
# Text Cleanup
# --------------------------
def clean_text(text: str) -> str:
    """Fix common OCR artifacts and normalize whitespace."""
    text = re.sub(r'\|', '', text)            # remove unwanted characters
    text = re.sub(r'\s+', ' ', text)          # normalize spaces
    return text.strip()

# --------------------------
# Optional Spell Correction
# --------------------------
def fix_spelling(text: str) -> str:
    """Fix spelling using a generic dictionary (no custom words)."""
    spell = SpellChecker()
    words = text.split()
    corrected = [spell.correction(word) if spell.unknown([word]) else word for word in words]
    return ' '.join(corrected)