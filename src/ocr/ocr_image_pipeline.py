
"""
Handles OCR processing for static images and live video frames.
Uses pytesseract and OpenCV/Pillow for text extraction.
"""

from typing import Union
from PIL import Image
import pytesseract
import numpy as np
import cv2


# --------------------------
# Image Loading
# --------------------------
def load_image(path: str) -> Image.Image:
    """
    Load an image from a file path and convert to RGB.
    """
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception as e:
        raise IOError(f"Could not load image from {path}: {e}")


# --------------------------
# Preprocessing
# --------------------------
def preprocess_image(img: Union[Image.Image, np.ndarray], max_width: int = 1200) -> np.ndarray:
    """
    Preprocess image for OCR:
    - Grayscale conversion
    - Gaussian blur denoising
    - Otsu's thresholding
    - Resize to reasonable width
    - Deskew to straighten text
    """
    # Convert PIL to NumPy array if needed
    if isinstance(img, Image.Image):
        img = np.array(img)

    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Otsu thresholding
    _, preprocessed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Resize if width is too large
    h, w = preprocessed.shape
    if w > max_width:
        scale = max_width / w
        preprocessed = cv2.resize(preprocessed, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)

    # Deskew
    coords = np.column_stack(np.where(preprocessed > 0))
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
# Text Extraction
# --------------------------
def extract_text(img: np.ndarray, lang: str = "eng", config: str = "--psm 3") -> str:
    """
    Extract text from a preprocessed image using pytesseract.
    Automatically converts NumPy array to PIL Image.
    """
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    text = pytesseract.image_to_string(img, lang=lang, config=config)
    return text


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
