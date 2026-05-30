// MainActivity.kt — Bonus AR Grading App (+30 pts)
// Stack: Kotlin + ARCore + ML Kit (QR) + OpenCV (bubbles)
//
// Bonus points:
//   Real-time QR decoding through camera    : 8 pts
//   Real-time bubble detection through camera: 10 pts
//   AR tick/cross overlay anchored to paper : 8 pts
//   Score display overlay                   : 4 pts

package com.quizscanner

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.ar.core.*
import com.google.ar.core.exceptions.*
import com.google.mlkit.vision.barcode.BarcodeScanning
import com.google.mlkit.vision.barcode.common.Barcode
import com.google.mlkit.vision.common.InputImage
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : AppCompatActivity() {

    private lateinit var arSession: Session
    private lateinit var arView: com.google.ar.core.ArView  // custom GLSurfaceView
    private lateinit var renderer: AROverlayRenderer
    private lateinit var overlay: OverlayView

    private lateinit var tvStatus: TextView
    private lateinit var tvScore: TextView
    private lateinit var btnSave: Button
    private lateinit var btnFlash: ToggleButton

    private val TAG = "QuizScannerAR"
    private val CAMERA_PERMISSION_CODE = 101

    // State
    private var answerKey: AnswerKey? = null
    private var gradeResult: GradeResult? = null
    private val isProcessingFrame = AtomicBoolean(false)
    private var frameCount = 0
    private val PROCESS_EVERY_N_FRAMES = 8   // process QR/bubbles every 8 frames

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Initialize OpenCV
        if (!OpenCVLoader.initDebug()) {
            Log.e(TAG, "OpenCV init failed")
        }

        bindViews()
        requestCameraPermission()
    }

    private fun bindViews() {
        tvStatus  = findViewById(R.id.tv_status)
        tvScore   = findViewById(R.id.tv_score)
        btnSave   = findViewById(R.id.btn_save)
        btnFlash  = findViewById(R.id.btn_flash)
        overlay   = findViewById(R.id.ar_overlay)

        btnSave.setOnClickListener   { saveCurrentResult() }
        btnFlash.setOnCheckedChangeListener { _, isOn -> toggleFlash(isOn) }
    }

    // ── Camera permission ─────────────────────────────────────────────
    private fun requestCameraPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED) {
            initAR()
        } else {
            ActivityCompat.requestPermissions(this,
                arrayOf(Manifest.permission.CAMERA), CAMERA_PERMISSION_CODE)
        }
    }

    override fun onRequestPermissionsResult(code: Int, perms: Array<String>, grants: IntArray) {
        super.onRequestPermissionsResult(code, perms, grants)
        if (code == CAMERA_PERMISSION_CODE && grants.getOrNull(0) == PackageManager.PERMISSION_GRANTED) {
            initAR()
        } else {
            tvStatus.text = "Camera permission required"
        }
    }

    // ── ARCore setup ──────────────────────────────────────────────────
    private fun initAR() {
        try {
            when (ArCoreApk.getInstance().requestInstall(this, true)) {
                ArCoreApk.InstallStatus.INSTALL_REQUESTED -> return
                ArCoreApk.InstallStatus.INSTALLED -> { /* proceed */ }
            }

            arSession = Session(this)
            val config = Config(arSession).apply {
                focusMode      = Config.FocusMode.AUTO
                updateMode     = Config.UpdateMode.LATEST_CAMERA_IMAGE
                augmentedImageDatabase = buildImageDatabase()
            }
            arSession.configure(config)

            renderer = AROverlayRenderer(this, arSession) { frame ->
                onNewARFrame(frame)
            }

            tvStatus.text = "Point camera at quiz sheet"

        } catch (e: UnavailableException) {
            tvStatus.text = "ARCore unavailable: ${e.message}"
            Log.e(TAG, "ARCore init failed", e)
        }
    }

    private fun buildImageDatabase(): AugmentedImageDatabase {
        // In production: add reference images of the quiz sheet corners/border
        // for stable anchor tracking
        val db = AugmentedImageDatabase(arSession)
        try {
            assets.open("quiz_reference.jpg").use { stream ->
                val bmp = BitmapFactory.decodeStream(stream)
                db.addImage("quiz_sheet", bmp, 0.21f)  // A4 width in metres
            }
        } catch (e: Exception) {
            Log.w(TAG, "No reference image found, using plane detection instead")
        }
        return db
    }

    // ── AR frame callback ─────────────────────────────────────────────
    private fun onNewARFrame(frame: Frame) {
        frameCount++
        if (frameCount % PROCESS_EVERY_N_FRAMES != 0) return
        if (!isProcessingFrame.compareAndSet(false, true))  return

        try {
            val camera = frame.camera
            if (camera.trackingState != TrackingState.TRACKING) return

            val bitmap = frameToBitmap(frame) ?: return

            // Task 1 in AR: decode QR  (+8 pts)
            if (answerKey == null) {
                decodeQRFromBitmap(bitmap) { key ->
                    answerKey = key
                    runOnUiThread { tvStatus.text = "✅ QR decoded — Set ${key.set}" }
                }
            }

            // Task 3 in AR: detect bubbles  (+10 pts)
            answerKey?.let { key ->
                val studentAnswers = BubbleDetector.detectBubbles(bitmap)

                // Task 4 in AR: grade
                val report = Grader.grade(studentAnswers, key)
                gradeResult = report

                // Task AR overlay: tick/cross  (+8 pts) + score  (+4 pts)
                val anchors = findPaperAnchors(frame)
                overlay.updateGrading(studentAnswers, key, report, anchors, bitmap.width, bitmap.height)

                runOnUiThread {
                    tvScore.text = "${report.totalMarks}/${report.maxMarks}  ${report.grade}"
                    tvScore.visibility = View.VISIBLE
                    btnSave.visibility = View.VISIBLE
                    tvStatus.text = "Grading: ${report.percentage.toInt()}% — ${report.grade}"
                }
            }
        } finally {
            isProcessingFrame.set(false)
        }
    }

    // ── QR decode via ML Kit  (+8 pts) ────────────────────────────────
    private fun decodeQRFromBitmap(bitmap: Bitmap, callback: (AnswerKey) -> Unit) {
        val image    = InputImage.fromBitmap(bitmap, 0)
        val scanner  = BarcodeScanning.getClient()

        scanner.process(image)
            .addOnSuccessListener { barcodes ->
                for (barcode in barcodes) {
                    if (barcode.format == Barcode.FORMAT_QR_CODE) {
                        val payload = barcode.rawValue ?: continue
                        val key     = QRParser.parse(payload)
                        if (key != null) {
                            callback(key)
                            return@addOnSuccessListener
                        }
                    }
                }
            }
            .addOnFailureListener { Log.e(TAG, "QR scan failed", it) }
    }

    // ── Find paper plane in AR scene ─────────────────────────────────
    private fun findPaperAnchors(frame: Frame): List<Anchor> {
        val anchors = mutableListOf<Anchor>()

        // Check for augmented image tracking (preferred — most stable)
        for (ai in frame.getUpdatedTrackables(AugmentedImage::class.java)) {
            if (ai.trackingState == TrackingState.TRACKING) {
                anchors.add(ai.createAnchor(ai.centerPose))
                return anchors
            }
        }

        // Fallback: first horizontal plane hit
        for (plane in frame.getUpdatedTrackables(Plane::class.java)) {
            if (plane.trackingState == TrackingState.TRACKING &&
                plane.type == Plane.Type.HORIZONTAL_UPWARD_FACING) {
                try {
                    anchors.add(plane.createAnchor(plane.centerPose))
                } catch (e: Exception) { /* skip */ }
                break
            }
        }
        return anchors
    }

    // ── Convert AR frame to Bitmap ────────────────────────────────────
    private fun frameToBitmap(frame: Frame): Bitmap? {
        return try {
            val image  = frame.acquireCameraImage()
            val planes = image.planes
            val yBuf   = planes[0].buffer
            val uBuf   = planes[1].buffer
            val vBuf   = planes[2].buffer

            val ySize = yBuf.remaining()
            val uSize = uBuf.remaining()
            val vSize = vBuf.remaining()

            val nv21 = ByteArray(ySize + uSize + vSize)
            yBuf.get(nv21, 0, ySize)
            vBuf.get(nv21, ySize, vSize)
            uBuf.get(nv21, ySize + vSize, uSize)

            val yuvImage = android.graphics.YuvImage(
                nv21, android.graphics.ImageFormat.NV21,
                image.width, image.height, null)
            image.close()

            val out = java.io.ByteArrayOutputStream()
            yuvImage.compressToJpeg(android.graphics.Rect(0, 0, image.width, image.height), 85, out)
            val bytes = out.toByteArray()
            BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        } catch (e: Exception) {
            Log.e(TAG, "Frame to bitmap failed", e)
            null
        }
    }

    // ── Save to CSV ───────────────────────────────────────────────────
    private fun saveCurrentResult() {
        val r = gradeResult ?: return
        val key = answerKey ?: return
        BatchCSVWriter.append(applicationContext, r, key)
        Toast.makeText(this, "Saved to batch CSV ✅", Toast.LENGTH_SHORT).show()
    }

    private fun toggleFlash(on: Boolean) {
        // Camera flash toggle via CameraManager
        val camManager = getSystemService(android.hardware.camera2.CameraManager::class.java)
        try { camManager.setTorchMode(camManager.cameraIdList[0], on) }
        catch (e: Exception) { /* no flash */ }
    }

    override fun onResume() {
        super.onResume()
        try { arSession.resume() } catch (e: Exception) { Log.e(TAG, "Session resume", e) }
    }

    override fun onPause() {
        super.onPause()
        arSession.pause()
    }

    override fun onDestroy() {
        super.onDestroy()
        arSession.close()
    }
}
