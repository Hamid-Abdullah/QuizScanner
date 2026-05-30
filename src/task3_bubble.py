"""
Task 3 — Bubble Sheet Reading [20 points]
Detects and interprets filled bubbles in the answer grid for Part-I and Part-II.

Evaluation criteria:
  - Reads all 16 bubbles on a clean scan              : 12 pts
  - Handles moderate image tilt/warp                  : +4 pts
  - Flags invalid (multi-filled / unattempted) bubbles : +4 pts
"""

import cv2
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

QUESTIONS = [f"Q{i:02d}" for i in range(1, 9)]   # Q01 … Q08
OPTIONS   = ['A', 'B', 'C', 'D']
FILL_THRESHOLD = 0.42    # ratio of filled pixels → bubble is marked
PARTIAL_THRESHOLD = 0.20 # ratio → bubble is partially filled (flagged)


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BubbleStatus:
    option: Optional[str]  # 'A'/'B'/'C'/'D' or None
    fill_ratio: float
    status: str  # 'filled' | 'empty' | 'partial' | 'invalid'


@dataclass
class StudentAnswers:
    part1: Dict[str, Optional[str]] = field(default_factory=lambda: {q: None for q in QUESTIONS})
    part2: Dict[str, Optional[str]] = field(default_factory=lambda: {q: None for q in QUESTIONS})
    invalid: List[str]     = field(default_factory=list)   # "part1_Q03" etc.
    unattempted: List[str] = field(default_factory=list)
    # Detailed per-bubble info (optional, for UI overlay)
    detail: Dict[str, List[BubbleStatus]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "part1": self.part1,
            "part2": self.part2,
            "invalid": self.invalid,
            "unattempted": self.unattempted,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def read_bubble_sheet(image_input) -> StudentAnswers:
    """
    Detect and read all filled bubbles from a quiz answer sheet.

    Pipeline:
      1. Perspective correction  (handles tilt/warp → +4 pts)
      2. Find answer grid region
      3. Detect individual bubbles
      4. Classify fill state
      5. Map to part → question → answer
      6. Flag invalid / unattempted  (→ +4 pts)
    """
    img = _load_image(image_input)
    if img is None:
        return StudentAnswers()

    # ── Step 1: Perspective correction ───────────────────────────────────────
    corrected = _perspective_correction(img)

    # ── Step 2: Preprocessing ────────────────────────────────────────────────
    gray  = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # ── Step 3 + 4 + 5: Detect bubbles and extract answers ───────────────────
    answers = _extract_answers(corrected, thresh)
    return answers


# ─────────────────────────────────────────────────────────────────────────────
# Perspective correction  (+4 pts for tilt/warp)
# ─────────────────────────────────────────────────────────────────────────────

def _perspective_correction(img: np.ndarray) -> np.ndarray:
    """Deskew the quiz sheet using contour-based perspective transform."""
    orig_h, orig_w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 30, 120)

    # Dilate edges to close small gaps
    kernel = np.ones((3, 3), np.uint8)
    edges  = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img

    # Sort by area descending; look for the document boundary
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours[:5]:
        area = cv2.contourArea(cnt)
        if area < (orig_h * orig_w * 0.1):   # skip tiny contours
            continue

        peri  = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
            pts = _order_points(pts)

            (tl, tr, br, bl) = pts
            max_w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
            max_h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

            if max_w < 100 or max_h < 100:
                continue

            dst = np.array([[0, 0], [max_w - 1, 0],
                             [max_w - 1, max_h - 1], [0, max_h - 1]],
                            dtype=np.float32)

            M       = cv2.getPerspectiveTransform(pts, dst)
            warped  = cv2.warpPerspective(img, M, (max_w, max_h))
            logger.info(f"Perspective correction applied: {orig_w}×{orig_h} → {max_w}×{max_h}")
            return warped

    return img  # no suitable quad found


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


# ─────────────────────────────────────────────────────────────────────────────
# Bubble detection & answer extraction  (12 pts clean scan)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_answers(img: np.ndarray, thresh: np.ndarray) -> StudentAnswers:
    """Find all bubble contours and map them to answers."""
    h, w = img.shape[:2]

    bubble_contours = _find_bubble_contours(thresh)
    logger.info(f"Found {len(bubble_contours)} bubble candidates")

    if len(bubble_contours) >= 24:
        # Enough contours found; use contour-based approach
        return _contour_based_extraction(bubble_contours, thresh, h, w)
    else:
        # Fall back to grid-based cell analysis
        logger.info("Falling back to grid-based extraction")
        return _grid_based_extraction(thresh, h, w)


def _find_bubble_contours(thresh: np.ndarray) -> list:
    """Find contours that look like bubbles (circles)."""
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bubbles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 80 or area > 6000:
            continue

        peri = cv2.arcLength(cnt, True)
        if peri < 1:
            continue

        circularity = (4 * np.pi * area) / (peri * peri)
        if circularity < 0.50:   # too irregular
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / max(bh, 1)
        if not (0.6 < aspect < 1.4):   # too elongated
            continue

        bubbles.append({
            'cnt': cnt, 'x': x, 'y': y,
            'w': bw, 'h': bh,
            'cx': x + bw // 2, 'cy': y + bh // 2,
            'area': area, 'circularity': circularity,
        })

    return bubbles


def _contour_based_extraction(bubbles: list, thresh: np.ndarray,
                               img_h: int, img_w: int) -> StudentAnswers:
    """
    Map detected bubble contours to (part, question, option).
    Assumes standard layout:
      - Left half → Part-I, Right half → Part-II
      - 8 rows per part, 4 columns (A B C D)
    """
    answers = StudentAnswers()
    mid_x = img_w * 0.50   # divide at 50%

    left_bubbles  = [b for b in bubbles if b['cx'] < mid_x]
    right_bubbles = [b for b in bubbles if b['cx'] >= mid_x]

    for part_name, part_bubbles in [('part1', left_bubbles), ('part2', right_bubbles)]:
        if not part_bubbles:
            continue

        # Cluster rows by Y coordinate
        rows = _cluster_by_y(part_bubbles, tolerance=img_h // 40)

        q_idx = 0
        for _, row_bubbles in sorted(rows.items()):
            if q_idx >= 8:
                break

            # Sort row left-to-right
            row_bubbles = sorted(row_bubbles, key=lambda b: b['cx'])
            if len(row_bubbles) < 2:
                continue   # skip noise rows

            q_key = f"Q{q_idx + 1:02d}"
            filled = []
            bubble_statuses = []

            # Take up to 4 bubbles per row
            for opt_idx, bubble in enumerate(row_bubbles[:4]):
                ratio = _fill_ratio(bubble['cnt'], thresh)
                if ratio >= FILL_THRESHOLD:
                    filled.append(OPTIONS[min(opt_idx, 3)])
                    status = 'filled'
                elif ratio >= PARTIAL_THRESHOLD:
                    status = 'partial'
                else:
                    status = 'empty'

                bubble_statuses.append(BubbleStatus(
                    option=OPTIONS[min(opt_idx, 3)],
                    fill_ratio=ratio,
                    status=status
                ))

            detail_key = f"{part_name}_{q_key}"
            answers.detail[detail_key] = bubble_statuses

            part_dict = getattr(answers, part_name)
            if len(filled) == 0:
                part_dict[q_key] = None
                answers.unattempted.append(detail_key)
            elif len(filled) == 1:
                part_dict[q_key] = filled[0]
            else:
                # Multi-filled: flag as invalid, record first answer
                part_dict[q_key] = filled[0]
                answers.invalid.append(detail_key)
                logger.warning(f"Invalid bubble at {detail_key}: {filled}")

            q_idx += 1

    return answers


def _fill_ratio(cnt, thresh: np.ndarray) -> float:
    """Compute fraction of pixels inside contour that are foreground (filled)."""
    mask = np.zeros(thresh.shape, dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)

    total = cv2.countNonZero(mask)
    if total == 0:
        return 0.0

    intersection = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
    return intersection / total


def _grid_based_extraction(thresh: np.ndarray, h: int, w: int) -> StudentAnswers:
    """
    Fallback: divide the image into a fixed grid and measure fill per cell.
    Assumes:
      - Answer grid occupies rows 20%–85% of image height
      - Part-I: columns  5%–47%, Part-II: columns 53%–95%
      - 8 question rows, 4 option columns per part
    """
    answers = StudentAnswers()

    grid_top    = int(h * 0.20)
    grid_bottom = int(h * 0.85)
    grid_height = grid_bottom - grid_top

    row_h = grid_height // 8

    parts = [
        ('part1', int(w * 0.05), int(w * 0.47)),
        ('part2', int(w * 0.53), int(w * 0.95)),
    ]

    for part_name, px1, px2 in parts:
        col_w = (px2 - px1) // 4

        for q_idx in range(8):
            q_key = f"Q{q_idx + 1:02d}"
            ry1 = grid_top + q_idx * row_h
            ry2 = ry1 + row_h

            filled = []
            bubble_statuses = []

            for opt_idx, opt in enumerate(OPTIONS):
                cx1 = px1 + opt_idx * col_w
                cx2 = cx1 + col_w

                # Shrink cell by 20% on each side to ignore grid lines
                mx = int(col_w * 0.20)
                my = int(row_h * 0.20)
                cell = thresh[ry1 + my:ry2 - my, cx1 + mx:cx2 - mx]

                if cell.size == 0:
                    ratio = 0.0
                else:
                    ratio = cv2.countNonZero(cell) / cell.size

                if ratio >= FILL_THRESHOLD:
                    filled.append(opt)
                    status = 'filled'
                elif ratio >= PARTIAL_THRESHOLD:
                    status = 'partial'
                else:
                    status = 'empty'

                bubble_statuses.append(BubbleStatus(option=opt, fill_ratio=ratio, status=status))

            detail_key = f"{part_name}_{q_key}"
            answers.detail[detail_key] = bubble_statuses

            part_dict = getattr(answers, part_name)
            if len(filled) == 0:
                part_dict[q_key] = None
                answers.unattempted.append(detail_key)
            elif len(filled) == 1:
                part_dict[q_key] = filled[0]
            else:
                part_dict[q_key] = filled[0]
                answers.invalid.append(detail_key)

    return answers


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cluster_by_y(bubbles: list, tolerance: int = 15) -> Dict[int, list]:
    """Group bubbles into rows by proximity of Y center."""
    if not bubbles:
        return {}

    sorted_b = sorted(bubbles, key=lambda b: b['cy'])
    clusters: Dict[int, list] = {}
    current_center = sorted_b[0]['cy']
    current_group  = [sorted_b[0]]

    for b in sorted_b[1:]:
        if abs(b['cy'] - current_center) <= tolerance:
            current_group.append(b)
            current_center = int(np.mean([x['cy'] for x in current_group]))
        else:
            clusters[current_center] = current_group
            current_group  = [b]
            current_center = b['cy']

    if current_group:
        clusters[current_center] = current_group

    return clusters


def _load_image(image_input) -> Optional[np.ndarray]:
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            logger.error(f"Cannot read image: {image_input}")
        return img
    elif isinstance(image_input, np.ndarray):
        return image_input.copy()
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Annotated output (for web UI / demo)
# ─────────────────────────────────────────────────────────────────────────────

def annotate_image(img: np.ndarray, student_answers: StudentAnswers,
                   answer_key=None) -> np.ndarray:
    """
    Draw colored circles over each bubble:
      Green  = correct
      Red    = incorrect
      Yellow = unattempted
      Orange = invalid
    """
    annotated = img.copy()
    h, w = annotated.shape[:2]

    # Re-detect bubbles to get coordinates
    gray   = cv2.cvtColor(annotated, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bubbles = _find_bubble_contours(thresh)

    mid_x = w * 0.50
    left_bubbles  = sorted([b for b in bubbles if b['cx'] < mid_x], key=lambda b: (b['cy'], b['cx']))
    right_bubbles = sorted([b for b in bubbles if b['cx'] >= mid_x], key=lambda b: (b['cy'], b['cx']))

    rows_l = _cluster_by_y(left_bubbles,  h // 40)
    rows_r = _cluster_by_y(right_bubbles, h // 40)

    for part_name, rows, key_dict in [
        ('part1', rows_l, answer_key.part1 if answer_key else {}),
        ('part2', rows_r, answer_key.part2 if answer_key else {}),
    ]:
        for q_idx, (_, row_bubbles) in enumerate(sorted(rows.items())):
            if q_idx >= 8:
                break
            q_key = f"Q{q_idx + 1:02d}"
            row_bubbles = sorted(row_bubbles, key=lambda b: b['cx'])
            student_ans = getattr(student_answers, part_name).get(q_key)
            correct_ans = key_dict.get(q_key)
            flag_key    = f"{part_name}_{q_key}"
            is_invalid  = flag_key in student_answers.invalid

            for opt_idx, b in enumerate(row_bubbles[:4]):
                opt   = OPTIONS[opt_idx]
                cx, cy = b['cx'], b['cy']
                r = max(b['w'], b['h']) // 2 + 3

                if is_invalid:
                    color = (0, 165, 255)  # orange
                    thick = 3
                elif opt == student_ans and answer_key:
                    color = (0, 255, 0) if opt == correct_ans else (0, 0, 255)
                    thick = -1
                elif opt == correct_ans and answer_key:
                    color = (0, 255, 0)
                    thick = 2
                else:
                    continue

                cv2.circle(annotated, (cx, cy), r, color, thick)

    return annotated


# ─────────────────────────────────────────────────────────────────────────────
# CLI demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "samples/quiz_sample.jpg"
    sa   = read_bubble_sheet(path)

    print("=" * 50)
    print("BUBBLE SHEET READING RESULTS")
    print("=" * 50)

    for part_name in ('part1', 'part2'):
        label = 'Part-I' if part_name == 'part1' else 'Part-II'
        print(f"\n{label}:")
        part = getattr(sa, part_name)
        for q in QUESTIONS:
            ans = part.get(q, None)
            flag = ""
            key  = f"{part_name}_{q}"
            if key in sa.invalid:
                flag = " ⚠ INVALID (multi-filled)"
            elif key in sa.unattempted:
                flag = " — UNATTEMPTED"
            print(f"  {q}: {ans or '—'}{flag}")

    print(f"\nUnattempted : {len(sa.unattempted)}")
    print(f"Invalid     : {len(sa.invalid)}")
