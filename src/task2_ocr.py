"""
Task 2 — Student Information Extraction [15 points]
Locates and extracts student Name and Registration Number from the quiz sheet
using OCR with robust preprocessing for handwritten text.

Evaluation criteria:
  - Correct field region detection              : 5 pts
  - Accurate OCR on printed text               : 5 pts
  - Reasonable accuracy on handwritten text    : +5 pts
"""

import cv2
import numpy as np
import re
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)

# ── OCR backend selection ────────────────────────────────────────────────────
# Try EasyOCR first (better for handwriting), fall back to pytesseract
try:
    import easyocr
    _EASYOCR_AVAILABLE = True
except ImportError:
    _EASYOCR_AVAILABLE = False
    logger.warning("EasyOCR not available")

try:
    import pytesseract
    _TESSERACT_AVAILABLE = True
except ImportError:
    _TESSERACT_AVAILABLE = False
    logger.warning("pytesseract not available")

# Lazy-initialized EasyOCR reader (expensive to create)
_easyocr_reader = None


def _get_reader():
    global _easyocr_reader
    if _easyocr_reader is None and _EASYOCR_AVAILABLE:
        _easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _easyocr_reader


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StudentInfo:
    name: str = "Unknown"
    reg_no: str = "Unknown"
    class_name: str = ""
    subject: str = ""
    confidence: float = 0.0  # average OCR confidence [0–1]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_student_info(image_input) -> StudentInfo:
    """
    Extract student name and registration number from a quiz sheet image.

    Args:
        image_input: file path (str) or numpy BGR array

    Returns:
        StudentInfo dataclass with name, reg_no (and optionally class/subject)
    """
    img = _load_image(image_input)
    if img is None:
        return StudentInfo()

    # ── Step 1: Find field regions using label detection ─────────────────────
    name_region, reg_region = _find_field_regions(img)

    name = ""
    reg_no = ""
    confidences = []

    # ── Step 2: OCR each region ───────────────────────────────────────────────
    if name_region is not None and name_region.size > 0:
        name, conf = _ocr_region(name_region, field_type="name")
        name = _clean_name(name)
        confidences.append(conf)
        logger.info(f"Name OCR: '{name}' (conf={conf:.2f})")

    if reg_region is not None and reg_region.size > 0:
        reg_no, conf = _ocr_region(reg_region, field_type="reg_no")
        reg_no = _clean_reg_no(reg_no)
        confidences.append(conf)
        logger.info(f"Reg No OCR: '{reg_no}' (conf={conf:.2f})")

    # ── Step 3: Fallback — full-page OCR with pattern matching ───────────────
    if not name or name == "Unknown" or not reg_no or reg_no == "Unknown":
        full_results = _ocr_full_page(img)
        if not name or name == "Unknown":
            name = _find_name_in_results(full_results)
        if not reg_no or reg_no == "Unknown":
            reg_no = _find_reg_in_results(full_results)

    return StudentInfo(
        name=name or "Unknown",
        reg_no=reg_no or "Unknown",
        confidence=sum(confidences) / len(confidences) if confidences else 0.0
    )


# ─────────────────────────────────────────────────────────────────────────────
# Field region detection  (5 pts)
# ─────────────────────────────────────────────────────────────────────────────

def _find_field_regions(img: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Locate the Name and Registration # field boxes on the quiz sheet.
    Uses two strategies:
      A) Label-anchor: find 'Name' / 'Reg' text labels → extract the region to their right/below
      B) Layout heuristic: assume standard printed quiz format
    """
    h, w = img.shape[:2]

    # ── Strategy A: label-anchored detection ─────────────────────────────────
    name_box, reg_box = _label_anchor_detection(img)

    if name_box is not None and reg_box is not None:
        return name_box, reg_box

    # ── Strategy B: layout heuristic (standard quiz layout) ──────────────────
    # Name field: typically ~8–18% from top, left ~15% to 70% of width
    # Reg No:    typically ~18–28% from top, same horizontal span
    name_box = img[int(h * 0.07):int(h * 0.17), int(w * 0.12):int(w * 0.72)]
    reg_box  = img[int(h * 0.17):int(h * 0.27), int(w * 0.12):int(w * 0.72)]

    return name_box, reg_box


def _label_anchor_detection(img: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Find label text, then extract the writing area adjacent to it."""
    h, w = img.shape[:2]

    # Only search the top 40% of the page for headers
    search_area = img[0:int(h * 0.40), 0:int(w * 0.80)]
    results = _ocr_full_page(search_area)

    name_y = None
    reg_y  = None
    name_x_end = 0
    reg_x_end  = 0

    for (bbox, text, conf) in results:
        tl = bbox[0]
        br = bbox[2]
        cx = int((tl[0] + br[0]) / 2)
        cy = int((tl[1] + br[1]) / 2)
        txt_lower = text.lower().strip().rstrip(':')

        if txt_lower in ('name', 'student name', 'naam') and name_y is None:
            name_y = cy
            name_x_end = int(br[0])

        if any(kw in txt_lower for kw in ('reg', 'registration', 'roll', 'id')) and reg_y is None:
            reg_y = cy
            reg_x_end = int(br[0])

    if name_y is None or reg_y is None:
        return None, None

    pad = max(5, int(h * 0.01))
    row_h = max(30, int(h * 0.05))

    name_box = img[
        max(0, name_y - pad): min(h, name_y + row_h + pad),
        name_x_end: min(w, int(w * 0.75))
    ]

    reg_box = img[
        max(0, reg_y - pad): min(h, reg_y + row_h + pad),
        reg_x_end: min(w, int(w * 0.75))
    ]

    return (name_box if name_box.size > 0 else None,
            reg_box  if reg_box.size  > 0 else None)


# ─────────────────────────────────────────────────────────────────────────────
# OCR engine  (5 pts printed + 5 pts handwritten)
# ─────────────────────────────────────────────────────────────────────────────

def _ocr_region(region: np.ndarray, field_type: str = "text") -> Tuple[str, float]:
    """
    Run OCR on a region with preprocessing optimized for handwritten text.
    Returns (text, confidence).
    """
    preprocessed_variants = _preprocess_for_ocr(region)

    best_text = ""
    best_conf = 0.0

    for variant in preprocessed_variants:
        text, conf = _run_ocr(variant, field_type)
        text = text.strip()
        if text and conf > best_conf:
            best_text = text
            best_conf = conf

    return best_text, best_conf


def _preprocess_for_ocr(region: np.ndarray) -> List[np.ndarray]:
    """Generate preprocessing variants optimised for handwriting accuracy."""
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region.copy()

    # Upscale to at least 60px tall (Tesseract/EasyOCR need reasonable resolution)
    scale = max(1, 80 // max(1, gray.shape[0]))
    if scale > 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    variants = [gray]

    # Denoise (ink smudges)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    variants.append(denoised)

    # Adaptive threshold (best for handwritten ink)
    at = cv2.adaptiveThreshold(denoised, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 15)
    variants.append(at)

    # Otsu threshold
    _, ot = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(ot)

    # CLAHE → threshold (handles varying lighting)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    _, ot2 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(ot2)

    # Dilation (thickens thin strokes)
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(ot, kernel, iterations=1)
    variants.append(dilated)

    return variants


def _run_ocr(img: np.ndarray, field_type: str = "text") -> Tuple[str, float]:
    """Run the best available OCR engine and return (text, confidence)."""
    # ── EasyOCR (handles handwriting better) ─────────────────────────────────
    reader = _get_reader()
    if reader is not None:
        try:
            results = reader.readtext(img, detail=1, paragraph=False)
            if results:
                texts = [r[1] for r in results if r[2] > 0.2]
                confs = [r[2] for r in results if r[2] > 0.2]
                if texts:
                    return ' '.join(texts), sum(confs) / len(confs)
        except Exception as e:
            logger.debug(f"EasyOCR error: {e}")

    # ── Tesseract fallback ────────────────────────────────────────────────────
    if _TESSERACT_AVAILABLE:
        try:
            config = (
                '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-/ '
                if field_type == "reg_no"
                else '--psm 7'
            )
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=config)
            texts = [t for t, c in zip(data['text'], data['conf'])
                     if t.strip() and int(c) > 30]
            confs = [int(c) / 100 for c, t in zip(data['conf'], data['text'])
                     if t.strip() and int(c) > 30]
            if texts:
                return ' '.join(texts), sum(confs) / len(confs)
        except Exception as e:
            logger.debug(f"Tesseract error: {e}")

    return "", 0.0


def _ocr_full_page(img: np.ndarray) -> List[Tuple]:
    """OCR the full page; returns list of (bbox, text, confidence)."""
    reader = _get_reader()
    if reader is not None:
        try:
            return reader.readtext(img, detail=1, paragraph=False)
        except Exception as e:
            logger.debug(f"Full-page EasyOCR error: {e}")

    if _TESSERACT_AVAILABLE:
        try:
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            results = []
            for i, text in enumerate(data['text']):
                if not text.strip():
                    continue
                conf = int(data['conf'][i]) / 100
                x, y, w2, h2 = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                bbox = [[x, y], [x + w2, y], [x + w2, y + h2], [x, y + h2]]
                results.append((bbox, text, conf))
            return results
        except Exception as e:
            logger.debug(f"Full-page Tesseract error: {e}")

    return []


# ─────────────────────────────────────────────────────────────────────────────
# Text cleaners
# ─────────────────────────────────────────────────────────────────────────────

def _clean_name(text: str) -> str:
    # Strip common OCR label artifacts
    text = re.sub(r'^(name|student|naam|nom)[:\s\-_]*', '', text, flags=re.IGNORECASE)
    # Remove non-name characters
    text = re.sub(r'[^a-zA-Z\s\.]', '', text)
    text = ' '.join(text.split())
    return text.title() if text else "Unknown"


def _clean_reg_no(text: str) -> str:
    text = re.sub(r'^(reg|registration|roll|no|number|#|id)[:\s\-_#]*', '', text, flags=re.IGNORECASE)
    # Keep alphanumeric, hyphens, slashes
    text = re.sub(r'[^a-zA-Z0-9\-/]', '', text)
    # Correct common OCR misreads: O→0, I→1 in numeric segments
    text = re.sub(r'(?<=\d)O(?=\d)', '0', text)
    return text.upper() if text else "Unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Fallback pattern matchers on full-page OCR results
# ─────────────────────────────────────────────────────────────────────────────

def _find_name_in_results(results: List[Tuple]) -> str:
    """Find the student name by looking for a label followed by a value."""
    for i, (_, text, conf) in enumerate(results):
        if re.search(r'\bname\b', text, re.IGNORECASE) and i + 1 < len(results):
            candidate = results[i + 1][1]
            # Sanity check: name should be mostly alphabetic
            alpha_ratio = sum(c.isalpha() for c in candidate) / max(1, len(candidate))
            if alpha_ratio > 0.6 and len(candidate) > 2:
                return _clean_name(candidate)

    # Second pass: look for runs of capitalized words (likely a name)
    for (_, text, conf) in results:
        if conf > 0.5 and re.match(r'^[A-Z][a-z]+ [A-Z][a-z]+', text):
            if not any(kw in text.lower() for kw in ['quiz', 'exam', 'part', 'reg']):
                return text.strip()

    return "Unknown"


def _find_reg_in_results(results: List[Tuple]) -> str:
    """Find registration number by pattern matching."""
    # Direct pattern: YYYY-XXXX-NNN or similar
    reg_patterns = [
        r'\b\d{4}[-/][A-Z]{2,6}[-/]\d{3,4}\b',   # 2021-BSCS-001
        r'\b[A-Z]{2,4}\d{4}\b',                     # CS2021
        r'\b\d{5,10}\b',                             # pure numeric
        r'\b[A-Z]\d{2,3}[-/]\d{4}\b',               # L18-1234
    ]

    full_text = ' '.join(t for (_, t, _) in results)
    for pat in reg_patterns:
        match = re.search(pat, full_text)
        if match:
            return match.group(0)

    # Look for label
    for i, (_, text, conf) in enumerate(results):
        if re.search(r'\b(reg|registration|roll)\b', text, re.IGNORECASE) and i + 1 < len(results):
            return _clean_reg_no(results[i + 1][1])

    return "Unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _load_image(image_input) -> Optional[np.ndarray]:
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            logger.error(f"Could not read image: {image_input}")
        return img
    elif isinstance(image_input, np.ndarray):
        return image_input.copy()
    return None


# ─────────────────────────────────────────────────────────────────────────────
# CLI demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "samples/quiz_sample.jpg"
    info = extract_student_info(path)
    print("=" * 40)
    print("STUDENT INFO EXTRACTION")
    print("=" * 40)
    print(f"Name:       {info.name}")
    print(f"Reg No:     {info.reg_no}")
    print(f"Class:      {info.class_name}")
    print(f"Confidence: {info.confidence:.2%}")
