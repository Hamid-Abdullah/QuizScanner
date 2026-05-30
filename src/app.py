"""
Flask Web Application — Automated Quiz Scanner & Grading System
Provides REST API endpoints for all tasks + a modern web UI.
"""

import os
import sys
import json
import uuid
import logging
import threading
from datetime import datetime
from pathlib import Path

from flask import (Flask, request, jsonify, render_template,
                   send_file, url_for)
from werkzeug.utils import secure_filename

# ── Add src to path ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from task1_qr_decoder import decode_answer_key
from task2_ocr         import extract_student_info
from task3_bubble      import read_bubble_sheet, annotate_image
from task4_grading     import grade_quiz, format_grade_report, generate_html_report
from task5_batch       import process_batch

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB max upload

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'uploads')
OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'output')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'pdf'}

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger(__name__)

# In-memory batch job tracker
_batch_jobs: dict = {}
_batch_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def save_upload(file, sub_folder: str = "") -> str:
    """Save an uploaded file and return its absolute path."""
    folder = os.path.join(UPLOAD_FOLDER, sub_folder)
    os.makedirs(folder, exist_ok=True)
    filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    path = os.path.join(folder, filename)
    file.save(path)
    return path


def _load_cv2_image(path: str):
    import cv2
    img = cv2.imread(path)
    if img is None and path.lower().endswith('.pdf'):
        try:
            import fitz
            doc = fitz.open(path)
            pix = doc[0].get_pixmap(matrix=fitz.Matrix(2, 2))
            import numpy as np
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.error(f"PDF load failed: {e}")
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Routes — UI
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


# ─────────────────────────────────────────────────────────────────────────────
# Routes — Task 1: QR Decode
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/decode-qr', methods=['POST'])
def api_decode_qr():
    """Task 1 — Decode QR code from uploaded image."""
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image provided'}), 400

    file = request.files['image']
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Unsupported file type'}), 400

    path = save_upload(file, 'qr')
    try:
        key = decode_answer_key(path)
        if key is None:
            return jsonify({'success': False, 'error': 'QR code not found or unreadable'})

        return jsonify({
            'success':    True,
            'quiz_set':   key.quiz_set,
            'subject':    key.subject,
            'semester':   key.semester,
            'part1':      key.part1,
            'part2':      key.part2,
            'negative_marking': key.negative_marking,
            'raw_payload': key.raw_payload,
        })
    except Exception as e:
        logger.exception("QR decode error")
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        try:
            os.remove(path)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Routes — Task 2: OCR
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/extract-info', methods=['POST'])
def api_extract_info():
    """Task 2 — Extract student info via OCR."""
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image provided'}), 400

    file = request.files['image']
    path = save_upload(file, 'ocr')
    try:
        info = extract_student_info(path)
        return jsonify({
            'success':    True,
            'name':       info.name,
            'reg_no':     info.reg_no,
            'class_name': info.class_name,
            'confidence': round(info.confidence, 3),
        })
    except Exception as e:
        logger.exception("OCR error")
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        try:
            os.remove(path)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Routes — Task 3: Bubble Sheet
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/read-bubbles', methods=['POST'])
def api_read_bubbles():
    """Task 3 — Read bubble sheet answers."""
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image provided'}), 400

    file = request.files['image']
    path = save_upload(file, 'bubble')
    try:
        sa = read_bubble_sheet(path)
        return jsonify({
            'success':     True,
            'part1':       sa.part1,
            'part2':       sa.part2,
            'invalid':     sa.invalid,
            'unattempted': sa.unattempted,
        })
    except Exception as e:
        logger.exception("Bubble read error")
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        try:
            os.remove(path)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Routes — Task 4: Full grade (QR + OCR + Bubble + Grade)
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/grade', methods=['POST'])
def api_grade():
    """Full pipeline: Tasks 1-4 on a single image."""
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image provided'}), 400

    file = request.files['image']
    path = save_upload(file, 'grade')
    try:
        img = _load_cv2_image(path)
        if img is None:
            return jsonify({'success': False, 'error': 'Could not read image'}), 400

        # Run all tasks
        key    = decode_answer_key(img)
        info   = extract_student_info(img)
        sa     = read_bubble_sheet(img)

        if key is None:
            # No QR found — return partial results
            return jsonify({
                'success':       True,
                'qr_found':      False,
                'student_name':  info.name,
                'reg_no':        info.reg_no,
                'part1':         sa.part1,
                'part2':         sa.part2,
                'invalid':       sa.invalid,
                'unattempted':   sa.unattempted,
                'report':        None,
                'html_report':   '<p class="warning">QR code not found — grading unavailable.</p>',
            })

        report = grade_quiz(sa, key, info.name, info.reg_no)

        # Annotated image
        annotated = annotate_image(img, sa, key)
        ann_path  = path.replace('.', '_annotated.')
        import cv2
        cv2.imwrite(ann_path, annotated)

        return jsonify({
            'success':       True,
            'qr_found':      True,
            'student_name':  info.name,
            'reg_no':        info.reg_no,
            'quiz_set':      key.quiz_set,
            'subject':       key.subject,
            'part1':         sa.part1,
            'part2':         sa.part2,
            'answer_key_p1': key.part1,
            'answer_key_p2': key.part2,
            'invalid':       sa.invalid,
            'unattempted':   sa.unattempted,
            'report':        report.to_dict(),
            'html_report':   generate_html_report(report),
            'text_report':   format_grade_report(report),
        })
    except Exception as e:
        logger.exception("Grade pipeline error")
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        try:
            os.remove(path)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Routes — Task 5: Batch Processing
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/batch', methods=['POST'])
def api_batch():
    """
    Task 5 — Batch processing.
    Accepts multiple files OR a zip archive.
    Returns a job_id; poll /api/batch/<job_id> for status.
    """
    files = request.files.getlist('images')
    if not files:
        return jsonify({'success': False, 'error': 'No images provided'}), 400

    quiz_title = request.form.get('quiz_title', 'AI Quiz SP2026')
    class_name = request.form.get('class_name', 'BSE-4A')

    # Save all files to a temp folder
    job_id     = uuid.uuid4().hex
    job_folder = os.path.join(UPLOAD_FOLDER, 'batch', job_id)
    os.makedirs(job_folder, exist_ok=True)

    saved = 0
    for f in files:
        if f and allowed_file(f.filename):
            fname = secure_filename(f.filename)
            f.save(os.path.join(job_folder, fname))
            saved += 1

    if saved == 0:
        return jsonify({'success': False, 'error': 'No valid images found'}), 400

    # Track job
    with _batch_lock:
        _batch_jobs[job_id] = {
            'status':    'queued',
            'total':     saved,
            'processed': 0,
            'current':   '',
            'result':    None,
            'error':     None,
        }

    # Run batch in background thread
    def _run():
        def cb(cur, tot, fname, status):
            with _batch_lock:
                _batch_jobs[job_id]['processed'] = cur
                _batch_jobs[job_id]['current']   = fname
                _batch_jobs[job_id]['status']    = 'running'

        try:
            result = process_batch(
                input_folder=job_folder,
                output_folder=OUTPUT_FOLDER,
                quiz_title=quiz_title,
                class_name=class_name,
                progress_callback=cb,
            )
            with _batch_lock:
                _batch_jobs[job_id]['status'] = 'done'
                _batch_jobs[job_id]['result'] = {
                    'excel_path': result['excel_path'],
                    'csv_path':   result['csv_path'],
                    'summary':    result['summary'],
                    'excel_name': os.path.basename(result['excel_path']),
                    'csv_name':   os.path.basename(result['csv_path']),
                }
        except Exception as e:
            logger.exception("Batch error")
            with _batch_lock:
                _batch_jobs[job_id]['status'] = 'error'
                _batch_jobs[job_id]['error']  = str(e)

    threading.Thread(target=_run, daemon=True).start()

    return jsonify({'success': True, 'job_id': job_id, 'files_queued': saved})


@app.route('/api/batch/<job_id>', methods=['GET'])
def api_batch_status(job_id: str):
    """Poll batch job status."""
    with _batch_lock:
        job = _batch_jobs.get(job_id)

    if job is None:
        return jsonify({'success': False, 'error': 'Job not found'}), 404

    return jsonify({'success': True, **job})


@app.route('/api/batch/<job_id>/download/<fmt>', methods=['GET'])
def api_batch_download(job_id: str, fmt: str):
    """Download Excel or CSV result for a batch job."""
    with _batch_lock:
        job = _batch_jobs.get(job_id)

    if not job or job['status'] != 'done' or not job['result']:
        return jsonify({'error': 'Not ready'}), 404

    key  = 'excel_path' if fmt == 'xlsx' else 'csv_path'
    path = job['result'].get(key)

    if not path or not os.path.exists(path):
        return jsonify({'error': 'File not found'}), 404

    return send_file(path, as_attachment=True,
                     download_name=os.path.basename(path))


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'true').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)
