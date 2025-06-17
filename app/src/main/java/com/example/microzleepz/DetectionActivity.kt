package com.example.microzleepz

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.AudioManager
import android.media.ToneGenerator
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
// import com.google.gson.Gson // Hapus jika tidak digunakan
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors
import kotlin.math.sqrt
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import org.json.JSONObject


// Data class untuk hasil prediksi dari model ON-DEVICE
data class PredictionResult(
    val prediction: Float, // Hasil confidence
    val label: String, // Label "NORMAL" atau "MICROSLEEP"
    val ear_value: Float // Nilai EAR
)

// Data class untuk error jika API digunakan (tidak langsung relevan untuk on-device)
// data class ErrorResponse(val detail: String) // Hapus jika tidak digunakan

// ImageAnalyzer untuk CameraX
class ImageAnalyzer(private val listener: (Bitmap) -> Unit) : ImageAnalysis.Analyzer {
    override fun analyze(image: ImageProxy) {
        val bitmap = image.toBitmap()
        Log.d("ImageAnalyzer", "Bitmap created from camera: ${bitmap.width}x${bitmap.height}")
        listener(bitmap)
        image.close()
    }
}

// Ekstensi ImageProxy -> Bitmap
fun ImageProxy.toBitmap(): Bitmap {
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
    yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 100, out) // Kualitas 100%
    val imageBytes = out.toByteArray()
    return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
}

class DetectionActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    // Menggunakan ID yang ada di XML: tvLabel dan tvOutput
    private lateinit var tvLabel: TextView // ID di XML untuk "Status Deteksi"
    private lateinit var tvOutput: TextView // ID di XML untuk "Initializing..."
    // Menambahkan variabel tambahan untuk output yang lebih spesifik jika diperlukan
    private lateinit var tvEARValue: TextView // Akan diisi dengan nilai EAR (dari tvOutput jika ingin digabung)
    private lateinit var tvPrediction: TextView // Akan diisi dengan label prediksi (dari tvLabel jika ingin digabung)

    private val REQUEST_CAMERA_PERMISSION = 10

    // Model TFLite dan Scaler
    private var tfliteInterpreter: Interpreter? = null
    private var scalerMean: FloatArray? = null
    private var scalerScale: FloatArray? = null
    private val MODEL_INPUT_IMAGE_SIZE = 128 // Ukuran input gambar untuk model CNN
    private val MODEL_PATH = "model_microsleep.tflite" // Nama file model di assets
    private val SCALER_PARAMS_PATH = "ear_scaler_params.json" // Nama file scaler params di assets

    // MediaPipe FaceLandmarker
    private var faceLandmarker: FaceLandmarker? = null
    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private var lastBitmap: Bitmap? = null

    // Landmark mata untuk perhitungan EAR
    private val LEFT_EYE_LANDMARKS = listOf(362, 385, 387, 263, 373, 380)
    private val RIGHT_EYE_LANDMARKS = listOf(33, 160, 158, 133, 153, 144)

    private var toneGenerator: ToneGenerator? = null

    // Flag untuk menandakan apakah activity sedang aktif dan siap untuk prediksi
    @Volatile private var isActivityResumedAndReady = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_detection) // Mengacu ke activity_prediction.xml

        // Inisialisasi UI components - Sesuaikan dengan ID di activity_prediction.xml
        previewView = findViewById(R.id.previewView)
        tvLabel = findViewById(R.id.tvLabel) // Mengacu ke ID tvLabel di XML
        tvOutput = findViewById(R.id.tvOutput) // Mengacu ke ID tvOutput di XML
        // Inisialisasi tvPrediction dan tvEARValue untuk menghindari UninitializedPropertyAccessException
        // Meskipun ID-nya tidak ada di XML, kita bisa mengarahkannya ke tvLabel dan tvOutput
        // Atau, yang lebih baik, ubah saja XML-nya agar ID-nya ada.
        // Untuk sekarang, saya akan arahkan ke tvLabel/tvOutput untuk menghindari crash
        tvPrediction = tvLabel // Arahkan ke tvLabel untuk menampilkan prediksi
        tvEARValue = tvOutput // Arahkan ke tvOutput untuk menampilkan EAR

        val btnSelesai = findViewById<Button>(R.id.btnSelesai)
        btnSelesai.setOnClickListener { finish() }

        toneGenerator = ToneGenerator(AudioManager.STREAM_ALARM, 100)

        // Inisialisasi TFLite Interpreter dan Scaler
        // Inisialisasi MediaPipe FaceLandmarker
        // Dipanggil di onResume() untuk memastikan sumber daya siap saat Activity aktif
    }

    override fun onResume() {
        super.onResume()
        isActivityResumedAndReady = true // Set flag true saat Activity aktif

        // Muat ulang model dan scaler jika belum dimuat atau ditutup
        if (tfliteInterpreter == null) {
            loadTFLiteModel()
        }
        if (scalerMean == null || scalerScale == null) {
            loadScalerParameters()
        }
        // Inisialisasi FaceLandmarker dan mulai kamera jika belum atau ditutup
        if (faceLandmarker == null) {
            setupFaceLandmarker()
        }
        startCamera() // Pastikan kamera dimulai kembali saat Activity di-resume
        Log.d("DetectionActivity", "Activity resumed.")
    }

    override fun onPause() {
        super.onPause()
        isActivityResumedAndReady = false // Set flag false saat Activity di-pause

        // Tutup interpreter dan faceLandmarker untuk membebaskan sumber daya
        tfliteInterpreter?.close()
        tfliteInterpreter = null // Set ke null setelah ditutup
        faceLandmarker?.close()
        faceLandmarker = null // Set ke null setelah ditutup
        Log.d("DetectionActivity", "TFLite Interpreter and FaceLandmarker closed in onPause.")

        // Hentikan ImageAnalysis agar tidak ada lagi frame yang masuk setelah onPause
        ProcessCameraProvider.getInstance(this).get().unbindAll()
    }

    private fun loadTFLiteModel() {
        try {
            val tfliteModel = FileUtil.loadMappedFile(this, MODEL_PATH)
            val options = Interpreter.Options()
            tfliteInterpreter = Interpreter(tfliteModel, options)
            Log.i("DetectionActivity", "TFLite model loaded successfully: $MODEL_PATH")
        } catch (e: Exception) {
            Log.e("DetectionActivity", "Error loading TFLite model: ${e.message}", e)
            Toast.makeText(this, "Failed to load TFLite model: ${e.message}", Toast.LENGTH_LONG).show()
            tfliteInterpreter = null
        }
    }

    private fun loadScalerParameters() {
        try {
            val jsonString = assets.open(SCALER_PARAMS_PATH).bufferedReader().use { it.readText() }
            val jsonObject = JSONObject(jsonString)
            scalerMean = (jsonObject.getJSONArray("mean")).let { array ->
                FloatArray(array.length()) { array.getDouble(it).toFloat() }
            }
            scalerScale = (jsonObject.getJSONArray("scale")).let { array -> // Menggunakan "scale" dari Python
                FloatArray(array.length()) { array.getDouble(it).toFloat() }
            }
            Log.i("DetectionActivity", "Scaler parameters loaded successfully.")
        } catch (e: Exception) {
            Log.e("DetectionActivity", "Error loading scaler parameters: ${e.message}", e)
            Toast.makeText(this, "Failed to load scaler parameters: ${e.message}", Toast.LENGTH_LONG).show()
            scalerMean = null
            scalerScale = null
        }
    }

    private fun setupFaceLandmarker() {
        try {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath("face_landmarker.task")
                .setDelegate(Delegate.CPU)
                .build()

            val options = FaceLandmarker.FaceLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.LIVE_STREAM)
                .setNumFaces(1)
                .setResultListener(this::onResults)
                .setErrorListener(this::onError)
                .build()

            faceLandmarker = FaceLandmarker.createFromOptions(this, options)
            Log.i("DetectionActivity", "Face Landmarker initialized successfully.")
        } catch (e: Exception) {
            Log.e("DetectionActivity", "Error initializing Face Landmarker: ${e.message}", e)
            Toast.makeText(this, "Failed to initialize Face Landmarker: ${e.message}", Toast.LENGTH_LONG).show()
            faceLandmarker = null
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
                .setTargetResolution(Size(320, 240))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setTargetRotation(previewView.display.rotation)
                .build()
                .also {
                    it.setAnalyzer(Executors.newSingleThreadExecutor(), ImageAnalyzer { bitmap ->
                        // HANYA proses jika activity sedang aktif dan tidak sedang di-pause
                        if (isActivityResumedAndReady) {
                            processBitmapForPrediction(bitmap)
                        } else {
                            bitmap.recycle() // Penting: recycle bitmap jika tidak digunakan
                        }
                    })
                }

            val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis)
                Log.d("DetectionActivity", "Kamera terhubung ke siklus hidup berhasil.")
            } catch (e: Exception) {
                Log.e("DetectionActivity", "Kamera gagal dimulai: ${e.message}", e)
                Toast.makeText(this, "Kamera gagal dimulai", Toast.LENGTH_SHORT).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun processBitmapForPrediction(bitmap: Bitmap) {
        lastBitmap = bitmap

        val mpImage = BitmapImageBuilder(bitmap).build()
        val frameTime = System.currentTimeMillis()
        faceLandmarker?.detectAsync(mpImage, frameTime) // Ini akan memanggil onResults ketika selesai
    }

    // onResults dari MediaPipe FaceLandmarker
    private fun onResults(resultBundle: FaceLandmarkerResult, inputImage: MPImage) {
        // Guard condition di awal onResults juga
        if (!isActivityResumedAndReady) {
            // Log.d("DetectionActivity", "Skipping onResults: Activity not resumed.")
            return
        }

        runOnUiThread {
            if (resultBundle.faceLandmarks().isNotEmpty()) {
                val faceLandmarks = resultBundle.faceLandmarks()[0]
                val landmarksList = faceLandmarks // Dapatkan daftar NormalizedLandmark

                if (landmarksList.size >= 478) { // Cek apakah semua landmark terdeteksi
                    val ear = calculateEar(landmarksList, LEFT_EYE_LANDMARKS, RIGHT_EYE_LANDMARKS)
                    tvEARValue.text = String.format("EAR: %.2f", ear) // Tampilkan EAR di UI

                    // Lakukan Inferensi TFLite di sini
                    if (tfliteInterpreter != null && scalerMean != null && scalerScale != null) {
                        lastBitmap?.let { bitmap ->
                            performTFLiteInference(bitmap, ear)
                        } ?: run {
                            tvPrediction.text = "Status: Bitmap is null"
                            Log.e("DetectionActivity", "lastBitmap is null before TFLite inference.")
                        }
                    } else {
                        tvPrediction.text = "Status: Model/Scaler Not Ready"
                    }
                } else {
                    tvEARValue.text = "EAR: Landmark count mismatch (${landmarksList.size})"
                    tvPrediction.text = "Status: Eye landmarks not fully detected"
                }
            } else {
                tvEARValue.text = "EAR: No face detected" // Ini akan terupdate dari onResults FaceLandmarker
                tvPrediction.text = "Status: No face detected"
            }
        }
    }

    private fun onError(error: Exception) {
        // Guard condition di awal onError juga
        if (!isActivityResumedAndReady) {
            return
        }
        runOnUiThread {
            Log.e("DetectionActivity", "MediaPipe Face Landmarker Error: ${error.message}", error)
            Toast.makeText(this, "Face Landmarker Error: ${error.message}", Toast.LENGTH_SHORT).show()
            tvPrediction.text = "Status: Face Landmarker Error"
        }
    }

    // Fungsi untuk menghitung EAR
    private fun calculateEar(landmarks: List<NormalizedLandmark>, leftEyeIndices: List<Int>, rightEyeIndices: List<Int>): Float {
        // Pindahkan validasi ukuran dan indeks ke awal fungsi
        if (leftEyeIndices.size != 6 || leftEyeIndices.any { it < 0 || it >= landmarks.size } ||
            rightEyeIndices.size != 6 || rightEyeIndices.any { it < 0 || it >= landmarks.size }) {
            Log.e("EARCalculation", "Invalid eyeIndices size or out-of-bound for one or both eyes.")
            return 0.0f
        }

        fun calculateSingleEar(eyeIndices: List<Int>): Float { // Fungsi pembantu untuk menghitung EAR satu mata
            val p1 = landmarks[eyeIndices[0]].x() to landmarks[eyeIndices[0]].y()
            val p2 = landmarks[eyeIndices[1]].x() to landmarks[eyeIndices[1]].y()
            val p3 = landmarks[eyeIndices[2]].x() to landmarks[eyeIndices[2]].y()
            val p4 = landmarks[eyeIndices[3]].x() to landmarks[eyeIndices[3]].y()
            val p5 = landmarks[eyeIndices[4]].x() to landmarks[eyeIndices[4]].y()
            val p6 = landmarks[eyeIndices[5]].x() to landmarks[eyeIndices[5]].y()

            val vertical1 = euclideanDistance(p2, p6)
            val vertical2 = euclideanDistance(p3, p5)
            val horizontal = euclideanDistance(p1, p4)

            if (horizontal == 0f) return 0.0f
            return (vertical1 + vertical2) / (2.0f * horizontal)
        }

        val leftEar = calculateSingleEar(leftEyeIndices)
        val rightEar = calculateSingleEar(rightEyeIndices)
        return (leftEar + rightEar) / 2.0f
    }

    private fun euclideanDistance(p1: Pair<Float, Float>, p2: Pair<Float, Float>): Float {
        return sqrt((p2.first - p1.first).square() + (p2.second - p1.second).square())
    }

    private fun Float.square(): Float = this * this

    // Fungsi untuk melakukan inferensi TFLite
    private fun performTFLiteInference(bitmap: Bitmap, earValue: Float) {
        if (!isActivityResumedAndReady || tfliteInterpreter == null || scalerMean == null || scalerScale == null) {
            Log.e("DetectionActivity", "TFLite Interpreter or scaler not ready for inference.")
            tvPrediction.text = "Status: Model not ready"
            return
        }

        try {
            // 1. Pra-pemrosesan Gambar untuk Model CNN
            val resizedBitmap = Bitmap.createScaledBitmap(bitmap, MODEL_INPUT_IMAGE_SIZE, MODEL_INPUT_IMAGE_SIZE, false)

            // **PERBAIKAN KRUSIAL DI SINI:** Konversi Bitmap ke ByteBuffer FLOAT32 secara manual
            val inputImageBuffer: ByteBuffer = ByteBuffer.allocateDirect(
                MODEL_INPUT_IMAGE_SIZE * MODEL_INPUT_IMAGE_SIZE * 3 * 4 // Width * Height * Channels * Bytes_per_float
            ).order(ByteOrder.nativeOrder())

            // Ambil piksel dari bitmap dan konversi ke float32 (0-1)
            val intValues = IntArray(MODEL_INPUT_IMAGE_SIZE * MODEL_INPUT_IMAGE_SIZE)
            resizedBitmap.getPixels(intValues, 0, resizedBitmap.width, 0, 0, resizedBitmap.width, resizedBitmap.height)

            for (pixelValue in intValues) {
                // Ambil nilai RGB dari pixel (ARGB_8888) dan normalisasi ke 0-1
                inputImageBuffer.putFloat(((pixelValue shr 16) and 0xFF) / 255.0f) // Red
                inputImageBuffer.putFloat(((pixelValue shr 8) and 0xFF) / 255.0f)  // Green
                inputImageBuffer.putFloat((pixelValue and 0xFF) / 255.0f)        // Blue
            }
            inputImageBuffer.rewind() // Pastikan buffer di-rewind

            Log.d("InferenceDebug", "Image buffer capacity: ${inputImageBuffer.capacity()} bytes.") // DEBUGGING

            // 2. Pra-pemrosesan EAR
            val earScaledValue = (earValue - scalerMean!![0]) / scalerScale!![0]
            val earInputBuffer: ByteBuffer = ByteBuffer.allocateDirect(4).order(ByteOrder.nativeOrder())
            earInputBuffer.putFloat(earScaledValue)
            earInputBuffer.rewind()

            Log.d("InferenceDebug", "EAR raw: %.2f, Scaled EAR: %.2f".format(earValue, earScaledValue))
            Log.d("InferenceDebug", "EAR buffer ready.")

            // Dapatkan detail input tensor dari interpreter (sudah benar)
            val earInputTensorIndex = tfliteInterpreter!!.getInputIndex("serving_default_ear_input:0")
            val imgInputTensorIndex = tfliteInterpreter!!.getInputIndex("serving_default_img_input:0")

            Log.d("InferenceDebug", "Input tensor indexes: ear_input=$earInputTensorIndex, img_input=$imgInputTensorIndex")

            if (earInputTensorIndex == -1 || imgInputTensorIndex == -1) {
                Log.e("DetectionActivity", "Input tensor names (serving_default_ear_input:0 or serving_default_img_input:0) not found in TFLite model. Check model conversion.")
                runOnUiThread { tvPrediction.text = "Status: TFLite Input Error" }
                return
            }

            // Siapkan map input untuk runForMultipleInputsOutputs
            val inputs = arrayOf<Any>(
                earInputBuffer,    // Input untuk ear_input (indeks 0)
                inputImageBuffer   // Input untuk img_input (indeks 1)
            )

            // Siapkan map output
            val outputBuffer: TensorBuffer = TensorBuffer.createFixedSize(tfliteInterpreter!!.getOutputTensor(0).shape(), tfliteInterpreter!!.getOutputTensor(0).dataType())
            val outputs = mutableMapOf<Int, Any>()
            outputs[0] = outputBuffer.buffer

            // Jalankan inferensi
            tfliteInterpreter!!.runForMultipleInputsOutputs(inputs, outputs)

            val confidence = outputBuffer.getFloatArray()[0]
            val label = if (confidence > 0.5) "NORMAL" else "MICROSLEEP" // Logika sudah benar berdasarkan validasi Python

            runOnUiThread {
                tvPrediction.text = "Pred: %s, Conf: %.2f".format(label, confidence)
                val outputBoxLayout = findViewById<LinearLayout>(R.id.outputBox)
                if (label == "MICROSLEEP") {
                    outputBoxLayout.setBackgroundColor(ContextCompat.getColor(this, android.R.color.holo_red_light))
                    toneGenerator?.startTone(ToneGenerator.TONE_CDMA_ABBR_ALERT, 500)
                } else {
                    outputBoxLayout.setBackgroundColor(ContextCompat.getColor(this, android.R.color.holo_green_light))
                }
            }

        } catch (e: Exception) {
            Log.e("DetectionActivity", "TFLite inference error: ${e.message}", e)
            runOnUiThread {
                tvPrediction.text = "Status: Inference Error"
            }
        }
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

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        faceLandmarker?.close()
        tfliteInterpreter?.close() // Penting: tutup interpreter TFLite
        toneGenerator?.release()
    }
}