package com.example.microzleepz

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.gson.Gson
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors
import android.widget.LinearLayout
import android.util.Size


class DetectionActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var tvLabel: TextView
    private lateinit var tvOutput: TextView
    private val REQUEST_CAMERA_PERMISSION = 10

    // Ganti dengan URL ngrok Anda yang TERBARU
    private val BASE_URL_API = "https://cf9c-114-122-8-17.ngrok-free.app"
    private val client = OkHttpClient()
    private val gson = Gson()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_detection)

        previewView = findViewById(R.id.previewView)
        tvLabel = findViewById(R.id.tvLabel)
        tvOutput = findViewById(R.id.tvOutput)

        val btnSelesai = findViewById<Button>(R.id.btnSelesai)
        btnSelesai.setOnClickListener {
            finish()
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.CAMERA),
                REQUEST_CAMERA_PERMISSION
            )
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            val imageAnalysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(640, 480)) // Tetap gunakan ini untuk kontrol resolusi utama
                // .setTargetAspectRatio(AspectRatio.RATIO_4_3) // Ini yang menyebabkan konflik dan deprecated, hapus
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setTargetRotation(previewView.display.rotation)
                .build()
                .also {
                    it.setAnalyzer(Executors.newSingleThreadExecutor(), ImageAnalyzer { bitmap ->
                        sendImageToFastAPI(bitmap)
                    })
                }

            val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalysis
                )
            } catch (e: Exception) {
                Log.e("DetectionActivity", "Kamera gagal dimulai: ${e.message}")
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun sendImageToFastAPI(bitmap: Bitmap) {
        val stream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 90, stream)
        val imageBytes = stream.toByteArray()

        val requestBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("file", "image.jpg", imageBytes.toRequestBody("image/jpeg".toMediaTypeOrNull()))
            .build()

        val request = Request.Builder()
            .url("$BASE_URL_API/predict-microsleep/")
            .post(requestBody)
            .build()

        client.newCall(request).enqueue(object : okhttp3.Callback {
            override fun onFailure(call: okhttp3.Call, e: java.io.IOException) {
                Log.e("DetectionActivity", "Error calling API: ${e.message}")
                runOnUiThread {
                    Toast.makeText(this@DetectionActivity, "Error koneksi API", Toast.LENGTH_SHORT).show()
                }
            }

            override fun onResponse(call: okhttp3.Call, response: okhttp3.Response) {
                val responseBody = response.body?.string()
                if (response.isSuccessful && responseBody != null) {
                    try {
                        val predictionResponse = gson.fromJson(responseBody, PredictionResponse::class.java)
                        runOnUiThread {
                            tvLabel.text = predictionResponse.label
                            tvOutput.text = "Prediksi: ${String.format("%.2f", predictionResponse.prediction)}, EAR: ${String.format("%.2f", predictionResponse.ear_value)}"
                            if (predictionResponse.label == "MICROSLEEP") {
                                findViewById<LinearLayout>(R.id.outputBox).setBackgroundColor(ContextCompat.getColor(this@DetectionActivity, android.R.color.holo_red_light))
                            } else {
                                findViewById<LinearLayout>(R.id.outputBox).setBackgroundColor(ContextCompat.getColor(this@DetectionActivity, android.R.color.holo_green_light))
                            }
                        }
                    } catch (e: Exception) {
                        Log.e("DetectionActivity", "Error parsing JSON: ${e.message}")
                        runOnUiThread {
                            Toast.makeText(this@DetectionActivity, "Error memproses data", Toast.LENGTH_SHORT).show()
                        }
                    }
                } else {
                    Log.e("DetectionActivity", "API response not successful: ${response.code}, ${responseBody}")
                    runOnUiThread {
                        if (response.code == 400 && responseBody != null) {
                            try {
                                val errorDetail = gson.fromJson(responseBody, ErrorResponse::class.java)
                                Toast.makeText(this@DetectionActivity, "API Error 400: ${errorDetail.detail}", Toast.LENGTH_LONG).show()
                                tvLabel.text = "Tidak Terdeteksi"
                                tvOutput.text = errorDetail.detail
                                findViewById<LinearLayout>(R.id.outputBox).setBackgroundColor(ContextCompat.getColor(this@DetectionActivity, android.R.color.darker_gray))
                            } catch (e: Exception) {
                                Toast.makeText(this@DetectionActivity, "API Error: ${response.code} (Gagal parse error detail)", Toast.LENGTH_SHORT).show()
                            }
                        } else {
                            Toast.makeText(this@DetectionActivity, "API error: ${response.code}", Toast.LENGTH_SHORT).show()
                        }
                    }
                }
            }
        })
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CAMERA_PERMISSION) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startCamera()
            } else {
                Toast.makeText(this, "Izin kamera ditolak", Toast.LENGTH_SHORT).show()
            }
        }
    }
}

data class PredictionResponse(
    val prediction: Float,
    val label: String,
    val ear_value: Float
)

data class ErrorResponse(val detail: String)

class ImageAnalyzer(private val listener: (Bitmap) -> Unit) : ImageAnalysis.Analyzer {
    override fun analyze(image: androidx.camera.core.ImageProxy) {
        val bitmap = image.toBitmap()
        listener(bitmap)
        image.close()
    }
}

fun androidx.camera.core.ImageProxy.toBitmap(): Bitmap {
    val yBuffer = planes[0].buffer
    val uBuffer = planes[1].buffer
    val vBuffer = planes[2].buffer

    val ySize = yBuffer.remaining()
    val uSize = uBuffer.remaining()
    val vSize = vBuffer.remaining()

    val nv21 = ByteArray(ySize + uSize + vSize)

    yBuffer.get(nv21, 0, ySize)
    vBuffer.get(nv21, ySize, vSize)
    uBuffer.get(nv21, ySize + vSize, uSize)

    val yuvImage = YuvImage(nv21, ImageFormat.NV21, this.width, this.height, null)
    val out = ByteArrayOutputStream()
    yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 90, out)
    val imageBytes = out.toByteArray()
    return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
}