"""
ocr_processor.py
----------------
Handles OCR processing for static images and live video frames.
- Supports multiple engines (Tesseract now, EasyOCR later)
- Preprocessing, deskewing, thresholding
- Optional text cleanup and spell correction
- Bounding box extraction for overlay
"""

from typing import Union, List, Dict
from PIL import Image
import numpy as np
import cv2
import pytesseract
import re

# Optional spellchecker (only for static/demo mode)
try:
    from spellchecker import SpellChecker
    SPELLCHECKER_AVAILABLE = True
except ImportError:
    SPELLCHECKER_AVAILABLE = False

# Optional EasyOCR import
try:
    import easyocr
except ImportError:
    easyocr = None


# --------------------------
# Image Loading (for static images)
# --------------------------
def load_image(path: str) -> Image.Image:
    """Load an image from a file path and convert to RGB."""
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception as e:
        raise IOError(f"Could not load image from {path}: {e}")


# --------------------------
# Preprocessing
# --------------------------
def preprocess_image(img: Union[Image.Image, np.ndarray], max_width: int = 1200) -> np.ndarray:
    """Convert image to grayscale, blur, threshold, resize, and deskew."""
    if isinstance(img, Image.Image):
        img = np.array(img)

    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Gaussian blur
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Otsu threshold
    _, preprocessed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Resize if too large
    h, w = preprocessed.shape
    if w > max_width:
        scale = max_width / w
        preprocessed = cv2.resize(preprocessed, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)

    # Deskew
    coords = np.column_stack(np.where(preprocessed > 0))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        M = cv2.getRotationMatrix2D((preprocessed.shape[1]//2, preprocessed.shape[0]//2), angle, 1.0)
        preprocessed = cv2.warpAffine(preprocessed, M, (preprocessed.shape[1], preprocessed.shape[0]),
                                      flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return preprocessed


# --------------------------
# Text cleanup
# --------------------------
def clean_text(text: str) -> str:
    """Fix OCR artifacts and normalize whitespace."""
    text = re.sub(r'\|', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def fix_spelling(text: str) -> str:
    """Optional spell correction (slow, for static images only)."""
    if not SPELLCHECKER_AVAILABLE:
        return text
    spell = SpellChecker()
    words = text.split()
    corrected = [spell.correction(word) if spell.unknown([word]) else word for word in words]
    return ' '.join(corrected)


# --------------------------
# OCR Interface
# --------------------------
def extract_text_from_image(image: np.ndarray, engine: str = "tesseract") -> str:
    """
    Extract text from an image or frame.
    Engine can be "tesseract" or "easyocr".
    """
    if engine == "tesseract":
        return _extract_with_tesseract(image)
    elif engine == "easyocr":
        return _extract_with_easyocr(image)
    else:
        raise ValueError(f"Unsupported OCR engine: {engine}")


# --------------------------
# Engine implementations
# --------------------------
def _extract_with_tesseract(image: np.ndarray) -> str:
    """OCR using pytesseract."""
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
    preprocessed = preprocess_image(image)
    text = pytesseract.image_to_string(Image.fromarray(preprocessed), config="--oem 3 --psm 6")
    text = clean_text(text)
    return text


def _extract_with_easyocr(image: np.ndarray) -> str:
    """OCR using EasyOCR."""
    if easyocr is None:
        raise ImportError("EasyOCR not installed")
    reader = easyocr.Reader(["en"], gpu=False)
    results = reader.readtext(image)
    text = " ".join([res[1] for res in results])
    text = clean_text(text)
    return text


# --------------------------
# Optional: bounding box extraction
# --------------------------
def extract_text_with_boxes(image: np.ndarray, engine: str = "tesseract") -> List[Dict]:
    """
    Returns a list of dicts: [{'text': str, 'bbox': (x,y,w,h)}].
    Useful for overlay in video.
    """
    preprocessed = preprocess_image(image)
    boxes = []

    if engine == "tesseract":
        data = pytesseract.image_to_data(Image.fromarray(preprocessed), output_type=pytesseract.Output.DICT)
        for i, text in enumerate(data["text"]):
            if text.strip():
                boxes.append({
                    "text": text,
                    "bbox": (data["left"][i], data["top"][i], data["width"][i], data["height"][i])
                })

    elif engine == "easyocr":
        if easyocr is None:
            raise ImportError("EasyOCR not installed")
        reader = easyocr.Reader(["en"], gpu=False)
        results = reader.readtext(image)
        for res in results:
            # res = [bbox, text, confidence]
            (top_left, top_right, bottom_right, bottom_left) = res[0]
            x = int(top_left[0])
            y = int(top_left[1])
            w = int(top_right[0] - top_left[0])
            h = int(bottom_left[1] - top_left[1])
            boxes.append({"text": res[1], "bbox": (x, y, w, h)})

    return boxes