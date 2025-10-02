"""
ocr_processor.py

Handles OCR processing for static images and live video frames.
Uses pytesseract and OpenCV/Pillow for text extraction.
"""

from typing import List, Dict, Union
import cv2
from PIL import Image
import pytesseract
import numpy as np


# --------------------------
# Image Loading
# --------------------------
def load_image(path: str) -> Image.Image:
    """
    Load an image from the given file path.

    Args:
        path (str): Path to the image file.

    Returns:
        PIL.Image.Image: Loaded image.
    """
    try:
        img = Image.open(path)
        img = img.convert('RGB')
        return img
    except Exception as e:
        raise IOError(f"Could not load image from {path}: {e}")


def preprocess_image(img: Union[Image.Image, "ndarray"]) -> "ndarray":
    """
    Preprocess image for OCR:
    - Convert to grayscale
    - Resize to improve OCR accuracy
    - Apply adaptive thresholding for better contrast
    Args:
        img: PIL Image or OpenCV ndarray
    Returns:
        ndarray: preprocessed image ready for OCR
    """

    # Step 1: Convert PIL Image to NumPy array if needed
    if isinstance(img, Image.Image):
        img = img.convert("RGB")      # ensure consistent format
        img = np.array(img)

    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Step 3: Resize (scale up small text)
    height, width = gray.shape
    scale_factor = 2
    gray = cv2.resize(gray, (width * scale_factor, height * scale_factor), interpolation=cv2.INTER_LINEAR)

    # Step 4: Adaptive thresholding (improves OCR on uneven lighting)
    preprocessed = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )

    # Optional: denoising can be added here if needed
    # preprocessed = cv2.medianBlur(preprocessed, 3)

    return preprocessed
    
# --------------------------
# Text Extraction
# --------------------------
def extract_text(img: "ndarray", lang: str = 'eng', config: str = '') -> str:
    """
    Extract text from a preprocessed image using pytesseract.

    Args:
        image (ndarray): Preprocessed image.
        lang (str): Language code (default: 'eng').
        config (str): Tesseract config string (optional).

    Returns:
        str: Detected text as a string.
    """
    img = pytesseract.image_to_string(img, lang=lang, config=config)


def extract_text_with_boxes(image: "ndarray", lang: str = 'eng', config: str = '') -> List[Dict]:
    """
    Extract text along with bounding box information.

    Args:
        image (ndarray): Preprocessed image.
        lang (str): Language code (default: 'eng').
        config (str): Tesseract config string (optional).

    Returns:
        List[Dict]: List of dictionaries with keys: text, x, y, w, h
    """
    # TODO: Implement text + bounding box extraction
    pass


# --------------------------
# Convenience Pipeline Functions
# --------------------------
def process_image(path: str) -> str:
    """
    Complete pipeline for static image: load → preprocess → OCR.

    Args:
        path (str): Path to image file.

    Returns:
        str: Extracted text.
    """
    # TODO: Combine load_image, preprocess_image, extract_text
    pass


def process_frame(frame: "ndarray") -> str:
    """
    Pipeline for a single video frame.

    Args:
        frame (ndarray): OpenCV video frame.

    Returns:
        str: Extracted text.
    """
    # TODO: Preprocess frame and extract text
    pass


# --------------------------
# Optional Utilities
# --------------------------
def save_text_to_file(text: str, path: str):
    """
    Save OCR output text to a file.

    Args:
        text (str): Text to save.
        path (str): Path to output file.
    """
    # TODO: Implement file saving
    pass


def draw_boxes(image: "ndarray", boxes: List[Dict]) -> "ndarray":
    """
    Draw bounding boxes around detected words on the image (for debugging/visualization).

    Args:
        image (ndarray): Original image.
        boxes (List[Dict]): List of bounding boxes.

    Returns:
        ndarray: Image with boxes drawn.
    """
    # TODO: Implement drawing boxes
    pass
