"""
generate_sample.py — Creates synthetic quiz sheet images for testing.
Run this once to populate samples/ before running the batch demo.

Usage:
    python generate_sample.py          # generates 5 samples
    python generate_sample.py 10       # generates 10 samples
"""

import sys
import os
import random
import string
import numpy as np
import cv2
import qrcode
from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'samples')
os.makedirs(OUTPUT_DIR, exist_ok=True)

OPTIONS = ['A', 'B', 'C', 'D']
QUESTIONS = [f'Q{i}' for i in range(1, 9)]

ANSWER_KEYS = {
    'A': {'Part-I': dict(Q1='D', Q2='A', Q3='B', Q4='A', Q5='D', Q6='A', Q7='A', Q8='B'),
           'Part-II': dict(Q1='C', Q2='D', Q3='D', Q4='D', Q5='C', Q6='C', Q7='C', Q8='B')},
    'B': {'Part-I': dict(Q1='B', Q2='C', Q3='A', Q4='D', Q5='B', Q6='C', Q7='D', Q8='A'),
           'Part-II': dict(Q1='A', Q2='B', Q3='C', Q4='A', Q5='B', Q6='D', Q7='A', Q8='C')},
    'C': {'Part-I': dict(Q1='C', Q2='D', Q3='D', Q4='B', Q5='A', Q6='D', Q7='B', Q8='C'),
           'Part-II': dict(Q1='B', Q2='A', Q3='B', Q4='C', Q5='D', Q6='A', Q7='D', Q8='A')},
}

NAMES = ['Ahmad Ali', 'Fatima Khan', 'Usman Malik', 'Ayesha Raza', 'Hassan Siddiqui',
         'Zainab Baig', 'Bilal Chaudhry', 'Mariam Iqbal', 'Saad Tariq', 'Hira Javed']

REG_TEMPLATE = '2021-BSE-{:03d}'


def make_qr_payload(set_id: str) -> str:
    key = ANSWER_KEYS[set_id]
    p1 = ' '.join(f'Q{i+1}={v}' for i, v in enumerate(key['Part-I'].values()))
    p2 = ' '.join(f'Q{i+1}={v}' for i, v in enumerate(key['Part-II'].values()))
    return f"AI Quiz SP2026 Set-{set_id} | Part-I: {p1} | Part-II: {p2}"


def generate_quiz_image(student_index: int,
                        set_id: str = 'A',
                        add_noise: bool = False,
                        add_rotation: float = 0.0) -> str:
    """Generate one synthetic quiz sheet image and save it."""

    W, H = 850, 1100
    img = Image.new('RGB', (W, H), color='white')
    draw = ImageDraw.Draw(img)

    try:
        font_lg = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 18)
        font_md = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
        font_sm = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 11)
    except (IOError, OSError):
        font_lg = font_md = font_sm = ImageFont.load_default()

    # ── QR code (top-right corner) ──────────────────────────────────────
    payload = make_qr_payload(set_id)
    qr = qrcode.QRCode(version=2, error_correction=qrcode.constants.ERROR_CORRECT_H, box_size=4, border=2)
    qr.add_data(payload)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color='black', back_color='white').convert('RGB')
    qr_w, qr_h = qr_img.size
    qr_x = W - qr_w - 20
    qr_y = 20
    img.paste(qr_img, (qr_x, qr_y))

    # ── Header ──────────────────────────────────────────────────────────
    draw.rectangle([(20, 20), (qr_x - 10, 120)], outline='black', width=2)
    draw.text((30, 28), 'CAPITAL UNIVERSITY OF SCIENCE & TECHNOLOGY', font=font_lg, fill='black')
    draw.text((30, 52), f'Department of CS & SE  |  AI Quiz SP2026  |  Set-{set_id}', font=font_md, fill='black')
    draw.text((30, 72), 'Course: Artificial Intelligence  |  BSE-4A  |  Time: 20 min  |  Total Marks: 16', font=font_sm, fill='black')
    draw.text((30, 92), f'Date: May 30, 2026', font=font_sm, fill='black')

    # ── Student info fields ──────────────────────────────────────────────
    draw.rectangle([(20, 130), (W - 20, 200)], outline='black', width=1)
    name = NAMES[student_index % len(NAMES)]
    reg  = REG_TEMPLATE.format(student_index + 1)

    draw.text((30, 140), 'Name:', font=font_md, fill='black')
    draw.text((100, 140), name, font=font_md, fill='#1a1a1a')
    draw.line([(100, 162), (500, 162)], fill='black', width=1)

    draw.text((520, 140), 'Reg #:', font=font_md, fill='black')
    draw.text((580, 140), reg, font=font_md, fill='#1a1a1a')
    draw.line([(580, 162), (830, 162)], fill='black', width=1)

    draw.text((30, 172), 'Class: BSE-4A', font=font_sm, fill='black')
    draw.text((200, 172), 'Subject: Artificial Intelligence', font=font_sm, fill='black')

    # ── Instructions ─────────────────────────────────────────────────────
    draw.rectangle([(20, 210), (W - 20, 250)], outline='black', width=1, fill='#f5f5f5')
    draw.text((30, 220), 'INSTRUCTIONS: Fill the bubble completely. Use dark ink/pencil. No overwriting.',
              font=font_sm, fill='black')
    draw.text((30, 235), 'Negative marking: 0 marks for wrong / unattempted.',
              font=font_sm, fill='black')

    # ── Bubble grid ──────────────────────────────────────────────────────
    # Two parts side by side
    answer_key = ANSWER_KEYS[set_id]

    def draw_bubble_section(start_x, part_label, key_dict):
        y = 280
        # Header
        draw.rectangle([(start_x, y), (start_x + 370, y + 30)], fill='#e8e8e8', outline='black', width=1)
        draw.text((start_x + 130, y + 8), f'{part_label}  (Q01–Q08)', font=font_md, fill='black')
        y += 35

        # Column headers A B C D
        draw.text((start_x + 60, y), 'Q', font=font_sm, fill='black')
        for j, opt in enumerate('ABCD'):
            draw.text((start_x + 100 + j * 60, y), opt, font=font_md, fill='black')
        y += 24

        # Rows Q01–Q08
        student_answers = {}
        for i, q in enumerate(QUESTIONS):
            correct_ans = key_dict.get(q, 'A')
            # Student answers: 85% accuracy
            if random.random() < 0.85:
                chosen = correct_ans
            elif random.random() < 0.10:
                chosen = None   # unattempted
            else:
                chosen = random.choice([o for o in OPTIONS if o != correct_ans])

            student_answers[q] = chosen

            row_y = y + i * 45
            # Q label
            draw.text((start_x + 55, row_y + 10), q, font=font_sm, fill='black')

            # Draw 4 bubbles
            for j, opt in enumerate(OPTIONS):
                cx = start_x + 112 + j * 60
                cy = row_y + 16
                r  = 14
                # Bubble circle
                draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], outline='black', width=2)
                # Fill if chosen
                if chosen == opt:
                    inner_r = r - 3
                    draw.ellipse([(cx - inner_r, cy - inner_r),
                                   (cx + inner_r, cy + inner_r)],
                                  fill='black')

        return student_answers

    draw_bubble_section(25,  'PART-I',  answer_key['Part-I'])
    draw_bubble_section(430, 'PART-II', answer_key['Part-II'])

    # ── Border ───────────────────────────────────────────────────────────
    draw.rectangle([(5, 5), (W - 5, H - 5)], outline='black', width=3)

    # ── Noise (optional) ─────────────────────────────────────────────────
    img_np = np.array(img)
    if add_noise:
        noise = np.random.normal(0, 8, img_np.shape).astype(np.int16)
        img_np = np.clip(img_np.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # ── Rotation (optional) ──────────────────────────────────────────────
    if add_rotation != 0:
        h_cv, w_cv = img_np.shape[:2]
        M = cv2.getRotationMatrix2D((w_cv // 2, h_cv // 2), add_rotation, 1.0)
        img_np = cv2.warpAffine(img_np, M, (w_cv, h_cv), borderValue=(255, 255, 255))

    # ── Save ─────────────────────────────────────────────────────────────
    fname    = f"quiz_sample_{student_index + 1:02d}_set{set_id}.jpg"
    out_path = os.path.join(OUTPUT_DIR, fname)
    cv2.imwrite(out_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 92])

    print(f"  Generated: {fname}")
    return out_path


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    print(f"Generating {n} sample quiz images in '{OUTPUT_DIR}/'...")

    sets = ['A', 'B', 'C']
    for i in range(n):
        set_id      = sets[i % len(sets)]
        add_noise   = (i % 3 == 2)                   # every 3rd has noise
        add_rotation= random.uniform(-3, 3) if i > 2 else 0.0  # slight rotation

        generate_quiz_image(
            student_index=i,
            set_id=set_id,
            add_noise=add_noise,
            add_rotation=add_rotation,
        )

    print(f"\nDone! {n} images saved to samples/")
    print("You can now run: python src/task5_batch.py samples/ output/ 'AI Quiz SP2026'")


if __name__ == '__main__':
    main()
