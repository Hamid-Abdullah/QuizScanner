"""
Task 4 — Quiz Grading [20 points]
Compares student answers against the answer key and computes a grade report.

Evaluation criteria:
  - Correct score calculation                   : 10 pts
  - Per-question breakdown display              : 6 pts
  - Handles negative marking if in QR payload   : +4 pts
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)

# Standard grading scale (percentage → letter)
GRADE_SCALE: List[tuple] = [
    (90, 'A+'), (85, 'A'), (80, 'A-'),
    (75, 'B+'), (70, 'B'), (65, 'B-'),
    (60, 'C+'), (55, 'C'), (50, 'C-'),
    (45, 'D'),  (0,  'F'),
]

SYMBOLS = {
    'correct':     '✓',
    'incorrect':   '✗',
    'unattempted': '—',
    'invalid':     '⚠',
}


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QuestionResult:
    question:       str              # 'Q01'
    part:           str              # 'Part-I' | 'Part-II'
    student_answer: Optional[str]    # 'A' / 'B' / 'C' / 'D' / None
    correct_answer: Optional[str]
    status:         str              # 'correct' | 'incorrect' | 'unattempted' | 'invalid'
    marks:          float            # points earned/lost for this question
    symbol:         str              # ✓ ✗ — ⚠


@dataclass
class GradeReport:
    # Identity
    student_name:    str   = ""
    reg_no:          str   = ""
    quiz_set:        str   = ""
    subject:         str   = ""
    # Counts
    total_questions: int   = 16
    correct:         int   = 0
    incorrect:       int   = 0
    unattempted:     int   = 0
    invalid:         int   = 0
    # Scores
    total_marks:     float = 0.0
    max_marks:       float = 16.0
    negative_marking:float = 0.0
    percentage:      float = 0.0
    letter_grade:    str   = "F"
    # Per-question breakdown
    breakdown:       List[QuestionResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "student_name":    self.student_name,
            "reg_no":          self.reg_no,
            "quiz_set":        self.quiz_set,
            "subject":         self.subject,
            "correct":         self.correct,
            "incorrect":       self.incorrect,
            "unattempted":     self.unattempted,
            "invalid":         self.invalid,
            "total_marks":     self.total_marks,
            "max_marks":       self.max_marks,
            "percentage":      round(self.percentage, 2),
            "letter_grade":    self.letter_grade,
            "negative_marking":self.negative_marking,
            "breakdown": [
                {
                    "question":       r.question,
                    "part":           r.part,
                    "student_answer": r.student_answer,
                    "correct_answer": r.correct_answer,
                    "status":         r.status,
                    "marks":          r.marks,
                    "symbol":         r.symbol,
                }
                for r in self.breakdown
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def grade_quiz(student_answers, answer_key,
               student_name: str = "",
               reg_no: str = "") -> GradeReport:
    """
    Grade a quiz by comparing student_answers against answer_key.

    Args:
        student_answers : StudentAnswers (from task3_bubble)
        answer_key      : AnswerKey      (from task1_qr_decoder)
        student_name    : str (from task2_ocr)
        reg_no          : str (from task2_ocr)

    Returns:
        GradeReport with full per-question breakdown and final score
    """
    from task3_bubble import QUESTIONS

    neg = answer_key.negative_marking   # marks deducted per wrong answer (+4 pts)

    report = GradeReport(
        student_name=student_name,
        reg_no=reg_no,
        quiz_set=answer_key.quiz_set,
        subject=answer_key.subject,
        negative_marking=neg,
        max_marks=float(len(QUESTIONS) * 2),  # 8+8 = 16 questions
    )

    # ── Per-question evaluation ───────────────────────────────────────────────
    part_map = [
        ('Part-I',  'part1', answer_key.part1),
        ('Part-II', 'part2', answer_key.part2),
    ]

    for part_label, attr, key_dict in part_map:
        student_part = getattr(student_answers, attr)

        for q in QUESTIONS:
            student_ans = student_part.get(q)
            correct_ans = key_dict.get(q)
            flag_key    = f"{attr}_{q}"

            is_invalid   = flag_key in student_answers.invalid
            is_unattempted = (student_ans is None) and not is_invalid

            # ── Determine status & marks ──────────────────────────────────────
            if is_invalid:
                status = 'invalid'
                marks  = 0.0
                report.invalid += 1

            elif is_unattempted:
                status = 'unattempted'
                marks  = 0.0
                report.unattempted += 1

            elif student_ans == correct_ans:
                status = 'correct'
                marks  = 1.0
                report.correct += 1

            else:
                status = 'incorrect'
                marks  = -neg   # 0 if no negative marking, or -0.25, etc. (+4 pts)
                report.incorrect += 1

            report.breakdown.append(QuestionResult(
                question=q,
                part=part_label,
                student_answer=student_ans,
                correct_answer=correct_ans,
                status=status,
                marks=marks,
                symbol=SYMBOLS.get(status, '?'),
            ))

    # ── Aggregate score ───────────────────────────────────────────────────────
    report.total_marks = max(0.0, sum(r.marks for r in report.breakdown))
    report.percentage  = (report.total_marks / report.max_marks * 100
                          if report.max_marks > 0 else 0.0)
    report.letter_grade = _letter_grade(report.percentage)

    logger.info(
        f"Graded {student_name} ({reg_no}): "
        f"{report.total_marks}/{report.max_marks} = {report.percentage:.1f}% [{report.letter_grade}]"
    )

    return report


# ─────────────────────────────────────────────────────────────────────────────
# Grade lookup
# ─────────────────────────────────────────────────────────────────────────────

def _letter_grade(percentage: float) -> str:
    for threshold, grade in GRADE_SCALE:
        if percentage >= threshold:
            return grade
    return 'F'


# ─────────────────────────────────────────────────────────────────────────────
# Formatted text report  (6 pts for per-question breakdown display)
# ─────────────────────────────────────────────────────────────────────────────

def format_grade_report(report: GradeReport) -> str:
    """Pretty-print a grade report to a string."""
    lines = [
        "=" * 58,
        "  AUTOMATED QUIZ GRADING REPORT",
        "=" * 58,
        f"  Student : {report.student_name}",
        f"  Reg No  : {report.reg_no}",
        f"  Subject : {report.subject}",
        f"  Set     : {report.quiz_set}",
    ]

    if report.negative_marking > 0:
        lines.append(f"  Neg Mrk : -{report.negative_marking} per wrong answer")

    lines += [
        "-" * 58,
        f"  {'Q':<6} {'Your':<8} {'Key':<8} {'Status':<14} {'Marks':>5}",
        "-" * 58,
    ]

    current_part = None
    for r in report.breakdown:
        if r.part != current_part:
            current_part = r.part
            lines.append(f"\n  ── {current_part} ──")

        student = r.student_answer or '—'
        correct = r.correct_answer or '—'
        mark_str = f"{r.marks:+.2f}" if r.marks != 0 else " 0.00"
        lines.append(
            f"  {r.question:<6} {student:<8} {correct:<8} "
            f"{r.symbol} {r.status:<12} {mark_str:>6}"
        )

    lines += [
        "\n" + "-" * 58,
        f"  Correct      : {report.correct}",
        f"  Incorrect    : {report.incorrect}",
        f"  Unattempted  : {report.unattempted}",
        f"  Invalid      : {report.invalid}",
        "=" * 58,
        f"  SCORE    : {report.total_marks:.2f} / {report.max_marks:.0f}",
        f"  PERCENT  : {report.percentage:.1f}%",
        f"  GRADE    : {report.letter_grade}",
        "=" * 58,
    ]

    return '\n'.join(lines)


def generate_html_report(report: GradeReport) -> str:
    """Generate an HTML version of the grade report for web display."""
    grade_colors = {
        'A+': '#22c55e', 'A': '#22c55e', 'A-': '#4ade80',
        'B+': '#84cc16', 'B': '#a3e635', 'B-': '#bef264',
        'C+': '#eab308', 'C': '#facc15', 'C-': '#fde047',
        'D':  '#f97316', 'F': '#ef4444',
    }
    status_colors = {
        'correct':     '#22c55e',
        'incorrect':   '#ef4444',
        'unattempted': '#9ca3af',
        'invalid':     '#f97316',
    }

    grade_color = grade_colors.get(report.letter_grade, '#6b7280')

    rows_html = ""
    for r in report.breakdown:
        sc = status_colors.get(r.status, '#6b7280')
        student = r.student_answer or '—'
        correct = r.correct_answer or '—'
        bg = 'rgba(34,197,94,0.08)' if r.status == 'correct' else \
             'rgba(239,68,68,0.08)'  if r.status == 'incorrect' else ''
        rows_html += f"""
        <tr style="background:{bg}">
            <td>{r.part}</td>
            <td><strong>{r.question}</strong></td>
            <td style="font-weight:bold;color:#3b82f6">{student}</td>
            <td style="font-weight:bold;color:#22c55e">{correct}</td>
            <td><span style="color:{sc};font-size:1.2em">{r.symbol}</span></td>
            <td style="color:{sc}">{r.status.capitalize()}</td>
            <td style="font-weight:bold">{r.marks:+.2f}</td>
        </tr>"""

    neg_info = (f"<p>⚠ Negative marking: <strong>-{report.negative_marking}</strong> per wrong answer</p>"
                if report.negative_marking > 0 else "")

    return f"""
<div class="grade-report">
    <div class="report-header">
        <div class="student-info">
            <h3>📋 Grade Report</h3>
            <table class="info-table">
                <tr><td>Student</td><td><strong>{report.student_name}</strong></td></tr>
                <tr><td>Reg No</td><td>{report.reg_no}</td></tr>
                <tr><td>Subject</td><td>{report.subject}</td></tr>
                <tr><td>Set</td><td>{report.quiz_set}</td></tr>
            </table>
        </div>
        <div class="score-badge" style="border-color:{grade_color}">
            <div class="grade-letter" style="color:{grade_color}">{report.letter_grade}</div>
            <div class="score-text">{report.total_marks:.0f} / {report.max_marks:.0f}</div>
            <div class="percent-text">{report.percentage:.1f}%</div>
        </div>
    </div>
    {neg_info}
    <div class="stats-row">
        <span class="stat correct">✓ Correct: {report.correct}</span>
        <span class="stat incorrect">✗ Incorrect: {report.incorrect}</span>
        <span class="stat unattempted">— Unattempted: {report.unattempted}</span>
        <span class="stat invalid">⚠ Invalid: {report.invalid}</span>
    </div>
    <table class="breakdown-table">
        <thead>
            <tr>
                <th>Part</th><th>Question</th><th>Your Answer</th>
                <th>Key</th><th></th><th>Status</th><th>Marks</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>
</div>"""


# ─────────────────────────────────────────────────────────────────────────────
# CLI demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from task1_qr_decoder import decode_answer_key
    from task2_ocr import extract_student_info
    from task3_bubble import read_bubble_sheet

    path = sys.argv[1] if len(sys.argv) > 1 else "samples/quiz_sample.jpg"

    key  = decode_answer_key(path)
    info = extract_student_info(path)
    sa   = read_bubble_sheet(path)

    if key:
        report = grade_quiz(sa, key, info.name, info.reg_no)
        print(format_grade_report(report))
    else:
        print("Could not decode answer key from image.")
