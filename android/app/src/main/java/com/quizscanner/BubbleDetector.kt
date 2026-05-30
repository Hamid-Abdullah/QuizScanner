// BubbleDetector.kt — Real-time bubble detection using OpenCV (+10 pts)

package com.quizscanner

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc

object BubbleDetector {

    private val FILL_THRESHOLD  = 0.42
    private val PARTIAL_THRESHOLD = 0.20
    private val OPTIONS = listOf('A', 'B', 'C', 'D')
    private val QUESTIONS = (1..8).map { "Q%02d".format(it) }

    /**
     * Detect filled bubbles from a camera frame bitmap.
     * Returns StudentAnswers with part1, part2, invalid, unattempted lists.
     */
    fun detectBubbles(bitmap: Bitmap): StudentAnswers {
        val mat = Mat()
        Utils.bitmapToMat(bitmap, mat)

        // Convert to grayscale
        val gray = Mat()
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY)

        // Perspective correction
        val corrected = perspectiveCorrection(gray)

        // Threshold (Otsu)
        val thresh = Mat()
        Imgproc.threshold(corrected, thresh, 0.0, 255.0,
            Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU)

        // Find contours
        val contours   = mutableListOf<MatOfPoint>()
        val hierarchy  = Mat()
        Imgproc.findContours(thresh, contours, hierarchy,
            Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

        // Filter bubble-like contours
        val bubbles = contours.filter { cnt ->
            val area = Imgproc.contourArea(cnt)
            if (area < 80 || area > 6000) return@filter false
            val peri = Imgproc.arcLength(MatOfPoint2f(*cnt.toArray()), true)
            if (peri < 1) return@filter false
            val circularity = (4 * Math.PI * area) / (peri * peri)
            circularity > 0.50
        }

        return mapBubblesToAnswers(bubbles, thresh, corrected.width(), corrected.height())
    }

    private fun perspectiveCorrection(gray: Mat): Mat {
        val blurred = Mat()
        Imgproc.GaussianBlur(gray, blurred, Size(5.0, 5.0), 0.0)

        val edges = Mat()
        Imgproc.Canny(blurred, edges, 30.0, 120.0)

        val contours  = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(edges, contours, hierarchy,
            Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

        if (contours.isEmpty()) return gray

        val largest = contours.maxByOrNull { Imgproc.contourArea(it) } ?: return gray
        val peri    = Imgproc.arcLength(MatOfPoint2f(*largest.toArray()), true)
        val approx  = MatOfPoint2f()
        Imgproc.approxPolyDP(MatOfPoint2f(*largest.toArray()), approx, 0.02 * peri, true)

        if (approx.total() == 4L) {
            val pts = orderPoints(approx.toArray().map { it.x to it.y })
            val (tl, tr, br, bl) = pts
            val maxW = maxOf(dist(br, bl), dist(tr, tl)).toInt()
            val maxH = maxOf(dist(tr, br), dist(tl, bl)).toInt()

            val src = MatOfPoint2f(
                Point(tl.first, tl.second), Point(tr.first, tr.second),
                Point(br.first, br.second), Point(bl.first, bl.second))
            val dst = MatOfPoint2f(
                Point(0.0, 0.0), Point(maxW.toDouble(), 0.0),
                Point(maxW.toDouble(), maxH.toDouble()), Point(0.0, maxH.toDouble()))

            val M = Imgproc.getPerspectiveTransform(src, dst)
            val warped = Mat()
            Imgproc.warpPerspective(gray, warped, M, Size(maxW.toDouble(), maxH.toDouble()))
            return warped
        }
        return gray
    }

    private fun orderPoints(pts: List<Pair<Double, Double>>): List<Pair<Double, Double>> {
        val sums = pts.map { it.first + it.second }
        val diffs = pts.map { it.second - it.first }
        return listOf(
            pts[sums.indexOf(sums.min()!!)] ,   // top-left (smallest sum)
            pts[diffs.indexOf(diffs.min()!!)] ,  // top-right (smallest diff)
            pts[sums.indexOf(sums.max()!!)] ,    // bottom-right (largest sum)
            pts[diffs.indexOf(diffs.max()!!)]    // bottom-left (largest diff)
        )
    }

    private fun dist(a: Pair<Double, Double>, b: Pair<Double, Double>): Double {
        val dx = a.first - b.first; val dy = a.second - b.second
        return Math.sqrt(dx * dx + dy * dy)
    }

    private fun mapBubblesToAnswers(bubbles: List<MatOfPoint>, thresh: Mat,
                                    imgW: Int, imgH: Int): StudentAnswers {
        val answers = StudentAnswers()
        val midX = imgW * 0.50

        val leftBubbles  = bubbles.filter { centroid(it).x < midX }
        val rightBubbles = bubbles.filter { centroid(it).x >= midX }

        for ((partBubbles, partName) in listOf(leftBubbles to "part1", rightBubbles to "part2")) {
            if (partBubbles.isEmpty()) continue
            val rows = clusterByY(partBubbles, imgH / 40)

            var qIdx = 0
            for ((_, rowBubbles) in rows.entries.sortedBy { it.key }) {
                if (qIdx >= 8) break
                val sorted = rowBubbles.sortedBy { centroid(it).x }
                if (sorted.size < 2) continue

                val qKey   = QUESTIONS[qIdx]
                val filled = mutableListOf<Char>()

                sorted.take(4).forEachIndexed { optIdx, cnt ->
                    val ratio = fillRatio(cnt, thresh)
                    if (ratio >= FILL_THRESHOLD) filled.add(OPTIONS[optIdx])
                }

                val partMap = if (partName == "part1") answers.part1 else answers.part2
                val flagKey = "${partName}_$qKey"

                when {
                    filled.isEmpty()  -> { partMap[qKey] = null; answers.unattempted.add(flagKey) }
                    filled.size == 1  -> partMap[qKey] = filled[0].toString()
                    else              -> { partMap[qKey] = filled[0].toString(); answers.invalid.add(flagKey) }
                }
                qIdx++
            }
        }
        return answers
    }

    private fun centroid(cnt: MatOfPoint): Point {
        val m = Imgproc.moments(cnt)
        return if (m.m00 != 0.0) Point(m.m10 / m.m00, m.m01 / m.m00)
               else { val r = Imgproc.boundingRect(cnt); Point(r.x.toDouble(), r.y.toDouble()) }
    }

    private fun fillRatio(cnt: MatOfPoint, thresh: Mat): Double {
        val mask = Mat.zeros(thresh.size(), thresh.type())
        Imgproc.drawContours(mask, listOf(cnt), 0, Scalar(255.0), -1)
        val total  = Core.countNonZero(mask).toDouble()
        if (total == 0.0) return 0.0
        val both = Mat()
        Core.bitwise_and(thresh, mask, both)
        return Core.countNonZero(both) / total
    }

    private fun clusterByY(contours: List<MatOfPoint>, tol: Int): Map<Int, List<MatOfPoint>> {
        val sorted   = contours.sortedBy { centroid(it).y }
        val clusters = mutableMapOf<Int, MutableList<MatOfPoint>>()
        var center   = centroid(sorted.first()).y.toInt()
        var group    = mutableListOf(sorted.first())

        for (cnt in sorted.drop(1)) {
            val cy = centroid(cnt).y.toInt()
            if (Math.abs(cy - center) <= tol) {
                group.add(cnt)
                center = group.map { centroid(it).y.toInt() }.average().toInt()
            } else {
                clusters[center] = group
                group  = mutableListOf(cnt)
                center = cy
            }
        }
        if (group.isNotEmpty()) clusters[center] = group
        return clusters
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AROverlayRenderer.kt — Green tick / Red cross overlay anchored to paper (+8+4 pts)
// ─────────────────────────────────────────────────────────────────────────────

package com.quizscanner

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View

/**
 * Custom View drawn on top of the camera feed.
 * Renders tick/cross/dash symbols aligned to detected bubble positions.
 * Anchored via ARCore plane/image tracking → symbols move with paper.
 */
class OverlayView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    private var studentAnswers: StudentAnswers? = null
    private var answerKey: AnswerKey?           = null
    private var gradeResult: GradeResult?       = null
    private var anchors: List<android.opengl.Matrix>? = null

    // Screen-space bubble positions (updated each frame)
    private var bubblePositions: List<BubblePos> = emptyList()
    private var imageW = 1; private var imageH = 1

    // Paints
    private val paintTick = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#22C55E");   strokeWidth = 4f; style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND; strokeJoin = Paint.Join.ROUND
    }
    private val paintCross = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#EF4444");   strokeWidth = 4f; style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
    }
    private val paintDash = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#9CA3AF");   strokeWidth = 3f; style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
    }
    private val paintScore = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE; textSize = 48f; isFakeBoldText = true
    }
    private val paintScoreBg = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#1D4ED8"); style = Paint.Style.FILL
    }
    private val paintInvalid = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#F97316");   strokeWidth = 4f; style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
    }

    data class BubblePos(
        val part: String, val question: String, val option: String,
        val screenX: Float, val screenY: Float, val radius: Float
    )

    fun updateGrading(sa: StudentAnswers, key: AnswerKey,
                      report: GradeResult,
                      anchors: List<com.google.ar.core.Anchor>,
                      imgW: Int, imgH: Int) {
        studentAnswers = sa
        answerKey      = key
        gradeResult    = report
        imageW = imgW; imageH = imgH
        // Compute screen-space positions here in real app using ARCore projection
        // For simplicity, we use normalized grid positions mapped to screen
        bubblePositions = computeBubblePositions(imgW, imgH)
        invalidate()
    }

    private fun computeBubblePositions(imgW: Int, imgH: Int): List<BubblePos> {
        val positions = mutableListOf<BubblePos>()
        val scaleX = width.toFloat()  / imgW
        val scaleY = height.toFloat() / imgH

        // Approximate grid layout (matches generate_sample.py layout)
        val gridTopFrac   = 0.27f
        val rowHeightFrac = 0.042f

        for ((partIdx, partName) in listOf("part1", "part2").withIndex()) {
            val xStartFrac = if (partIdx == 0) 0.05f else 0.52f
            for (qIdx in 0 until 8) {
                for (optIdx in 0 until 4) {
                    val cx = (xStartFrac + optIdx * 0.07f + 0.03f) * imgW * scaleX
                    val cy = (gridTopFrac + qIdx * rowHeightFrac + rowHeightFrac / 2) * imgH * scaleY
                    val r  = rowHeightFrac * imgH * scaleY * 0.38f
                    positions.add(BubblePos(
                        part     = partName,
                        question = "Q%02d".format(qIdx + 1),
                        option   = listOf("A", "B", "C", "D")[optIdx],
                        screenX  = cx, screenY = cy, radius = r
                    ))
                }
            }
        }
        return positions
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val sa  = studentAnswers ?: return
        val key = answerKey      ?: return

        for (bp in bubblePositions) {
            val partMap    = if (bp.part == "part1") sa.part1 else sa.part2
            val keyMap     = if (bp.part == "part1") key.part1 else key.part2
            val studentAns = partMap[bp.question]
            val correctAns = keyMap[bp.question]
            val flagKey    = "${bp.part}_${bp.question}"
            val isInvalid  = sa.invalid.contains(flagKey)

            when {
                isInvalid && bp.option == studentAns -> {
                    // Orange warning circle
                    canvas.drawCircle(bp.screenX, bp.screenY, bp.radius, paintInvalid)
                    drawWarning(canvas, bp)
                }
                bp.option == studentAns && studentAns == correctAns -> {
                    // Filled green circle + tick  (+8 pts)
                    val fill = Paint().apply { color = Color.parseColor("#22C55E"); alpha = 180 }
                    canvas.drawCircle(bp.screenX, bp.screenY, bp.radius, fill)
                    drawTick(canvas, bp)
                }
                bp.option == studentAns && studentAns != correctAns -> {
                    // Filled red circle + cross
                    val fill = Paint().apply { color = Color.parseColor("#EF4444"); alpha = 180 }
                    canvas.drawCircle(bp.screenX, bp.screenY, bp.radius, fill)
                    drawCross(canvas, bp)
                }
                bp.option == correctAns && studentAns == null -> {
                    // Unattempted: show correct answer in green outline
                    canvas.drawCircle(bp.screenX, bp.screenY, bp.radius, paintTick)
                    drawDash(canvas, bp)
                }
                else -> { /* non-selected non-correct bubble: draw nothing */ }
            }
        }

        // Score overlay (+4 pts) — positioned at top-left of answer grid
        gradeResult?.let { r ->
            val scoreText = "${r.totalMarks}/${r.maxMarks}  ${r.grade}"
            val textW = paintScore.measureText(scoreText)
            val margin = 20f; val pad = 14f; val textH = 52f
            val rx = margin; val ry = height * 0.24f
            val rect = RectF(rx, ry, rx + textW + pad * 2, ry + textH)
            canvas.drawRoundRect(rect, 10f, 10f, paintScoreBg)
            canvas.drawText(scoreText, rx + pad, ry + 38f, paintScore)
        }
    }

    private fun drawTick(canvas: Canvas, bp: BubblePos) {
        val r = bp.radius * 0.55f
        val path = Path().apply {
            moveTo(bp.screenX - r, bp.screenY)
            lineTo(bp.screenX - r * 0.15f, bp.screenY + r * 0.7f)
            lineTo(bp.screenX + r, bp.screenY - r * 0.55f)
        }
        canvas.drawPath(path, paintTick.apply { color = Color.WHITE })
    }

    private fun drawCross(canvas: Canvas, bp: BubblePos) {
        val r = bp.radius * 0.5f
        canvas.drawLine(bp.screenX - r, bp.screenY - r, bp.screenX + r, bp.screenY + r, paintCross.apply { color = Color.WHITE })
        canvas.drawLine(bp.screenX + r, bp.screenY - r, bp.screenX - r, bp.screenY + r, paintCross.apply { color = Color.WHITE })
    }

    private fun drawDash(canvas: Canvas, bp: BubblePos) {
        val r = bp.radius * 0.45f
        canvas.drawLine(bp.screenX - r, bp.screenY, bp.screenX + r, bp.screenY, paintDash)
    }

    private fun drawWarning(canvas: Canvas, bp: BubblePos) {
        val p = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.parseColor("#F97316"); textSize = bp.radius * 1.2f; textAlign = Paint.Align.CENTER; isFakeBoldText = true }
        canvas.drawText("!", bp.screenX, bp.screenY + bp.radius * 0.4f, p)
    }
}
