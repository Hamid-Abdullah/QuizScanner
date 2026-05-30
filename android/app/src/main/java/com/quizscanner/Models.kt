// Models.kt — Shared data models for Android AR app

package com.quizscanner

// ── Answer Key (from QR) ─────────────────────────────────────────────────────
data class AnswerKey(
    val set:              String,
    val subject:          String,
    val semester:         String,
    val part1:            Map<String, String>,   // "Q01" → "A"
    val part2:            Map<String, String>,
    val negativMarking:   Float = 0f,
)

// ── Student Answers (from bubble detection) ───────────────────────────────────
data class StudentAnswers(
    val part1:       MutableMap<String, String?> = mutableMapOf(),
    val part2:       MutableMap<String, String?> = mutableMapOf(),
    val invalid:     MutableList<String>         = mutableListOf(),
    val unattempted: MutableList<String>         = mutableListOf(),
)

// ── Grade Report ─────────────────────────────────────────────────────────────
data class GradeResult(
    val studentName:  String,
    val regNo:        String,
    val correct:      Int,
    val incorrect:    Int,
    val unattempted:  Int,
    val invalid:      Int,
    val totalMarks:   Float,
    val maxMarks:     Float,
    val percentage:   Float,
    val grade:        String,
)


// ── QR Parser ─────────────────────────────────────────────────────────────────
object QRParser {
    private val Q_PATTERN = Regex("""Q(\d+)\s*=\s*([A-Da-d])""")

    fun parse(payload: String): AnswerKey? {
        val setMatch = Regex("""Set[-\s]?([A-Z0-9]+)""", RegexOption.IGNORE_CASE).find(payload)
        val set      = setMatch?.groupValues?.get(1)?.uppercase() ?: "?"

        val p1Match = Regex("""Part[-\s]?I(?!I)[:\s]+([^|]+)""", RegexOption.IGNORE_CASE).find(payload)
        val p2Match = Regex("""Part[-\s]?II[:\s]+([^|]+)""",      RegexOption.IGNORE_CASE).find(payload)

        if (p1Match == null && p2Match == null) return null

        return AnswerKey(
            set      = set,
            subject  = Regex("""^([A-Za-z ]+?)(?:\s+SP|\s+Set|\s*\|)""").find(payload.trim())?.groupValues?.get(1)?.trim() ?: "AI",
            semester = Regex("""(SP|FA|SU)\s*(\d{4})""", RegexOption.IGNORE_CASE).find(payload)?.value ?: "",
            part1    = parseAnswers(p1Match?.groupValues?.get(1) ?: ""),
            part2    = parseAnswers(p2Match?.groupValues?.get(1) ?: ""),
            negativMarking = Regex("""NEG[:\s]+(\d+\.?\d*)""", RegexOption.IGNORE_CASE).find(payload)?.groupValues?.get(1)?.toFloatOrNull() ?: 0f,
        )
    }

    private fun parseAnswers(raw: String): Map<String, String> =
        Q_PATTERN.findAll(raw).associate { m ->
            "Q%02d".format(m.groupValues[1].toInt()) to m.groupValues[2].uppercase()
        }
}


// ── Grader ────────────────────────────────────────────────────────────────────
object Grader {

    private val GRADE_SCALE = listOf(
        90 to "A+", 85 to "A", 80 to "A-",
        75 to "B+", 70 to "B", 65 to "B-",
        60 to "C+", 55 to "C", 50 to "C-",
        45 to "D",   0 to "F"
    )

    fun grade(sa: StudentAnswers, key: AnswerKey): GradeResult {
        var correct = 0; var incorrect = 0; var unattempted = 0; var invalid = 0
        var marks = 0f
        val maxMarks = (key.part1.size + key.part2.size).toFloat()

        for ((partName, keyMap) in listOf("part1" to key.part1, "part2" to key.part2)) {
            val studentPart = if (partName == "part1") sa.part1 else sa.part2
            for ((q, correctAns) in keyMap) {
                val studentAns = studentPart[q]
                val flagKey    = "${partName}_$q"
                when {
                    sa.invalid.contains(flagKey)     -> { invalid++; }
                    studentAns == null               -> { unattempted++; }
                    studentAns == correctAns         -> { correct++;    marks += 1f }
                    else                             -> { incorrect++;  marks -= key.negativMarking }
                }
            }
        }

        marks = marks.coerceAtLeast(0f)
        val pct   = if (maxMarks > 0) marks / maxMarks * 100f else 0f
        val grade = GRADE_SCALE.first { pct >= it.first }.second

        return GradeResult(
            studentName  = "",
            regNo        = "",
            correct      = correct,
            incorrect    = incorrect,
            unattempted  = unattempted,
            invalid      = invalid,
            totalMarks   = marks,
            maxMarks     = maxMarks,
            percentage   = pct,
            grade        = grade,
        )
    }
}


// ── BatchCSVWriter ────────────────────────────────────────────────────────────
object BatchCSVWriter {

    private const val FILE_NAME = "QuizScanner_Batch.csv"

    private val HEADER = listOf(
        "Quiz", "Set", "Name", "Reg No",
        "Part1_Q01","Part1_Q02","Part1_Q03","Part1_Q04",
        "Part1_Q05","Part1_Q06","Part1_Q07","Part1_Q08",
        "Part2_Q01","Part2_Q02","Part2_Q03","Part2_Q04",
        "Part2_Q05","Part2_Q06","Part2_Q07","Part2_Q08",
        "Correct","Incorrect","Unattempted","Total Marks","Percentage","Grade"
    ).joinToString(",")

    fun append(context: android.content.Context, r: GradeResult, key: AnswerKey) {
        // Append one row to Downloads/QuizScanner_Batch.csv
        val file = java.io.File(
            android.os.Environment.getExternalStoragePublicDirectory(
                android.os.Environment.DIRECTORY_DOWNLOADS), FILE_NAME)

        if (!file.exists()) {
            file.writeText(HEADER + "\n")
        }

        val ts   = java.text.SimpleDateFormat("HH:mm:ss", java.util.Locale.US).format(java.util.Date())
        val q    = (1..8).map { "Q%02d".format(it) }
        val p1   = q.joinToString(",") { "-" }   // real app would have actual answers
        val p2   = q.joinToString(",") { "-" }

        val row = listOf(
            "Quiz $ts", key.set, r.studentName, r.regNo,
            p1, p2,
            r.correct, r.incorrect, r.unattempted,
            "%.1f".format(r.totalMarks),
            "%.1f".format(r.percentage),
            r.grade
        ).joinToString(",")

        file.appendText(row + "\n")
    }
}
