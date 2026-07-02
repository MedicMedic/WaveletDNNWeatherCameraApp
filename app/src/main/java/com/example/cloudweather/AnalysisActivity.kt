package com.example.cloudweather

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.exifinterface.media.ExifInterface
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class AnalysisActivity : AppCompatActivity() {
    private val TAG = "AnalysisActivity"
    private var tfliteInterpreter: Interpreter? = null
    private var originalBitmap: Bitmap? = null

    // Cloud type mapping (order = training class order: sorted folder names)
    private val cloudTypes = arrayOf("Ac", "As", "Cc", "Cs", "Ci", "Cb", "Cu", "Ns", "Sc", "St")
    private val cloudNames = mapOf(
        "Ac" to "ALTOCUMULUS",
        "As" to "ALTOSTRATUS",
        "Cc" to "CIRROCUMULUS",
        "Cs" to "CIRROSTRATUS",
        "Ci" to "CIRRUS",
        "Cb" to "CUMULONIMBUS",
        "Cu" to "CUMULUS",
        "Ns" to "NIMBOSTRATUS",
        "Sc" to "STRATOCUMULUS",
        "St" to "STRATUS"
    )

    private val precipClasses = arrayOf(1, 1, 0, 0, 0, 3, 1, 2, 1, 1)
    private val precipDescriptions = mapOf(
        0 to "NO PRECIPITATION",
        1 to "LIGHT PRECIPITATION",
        2 to "MODERATE PRECIPITATION",
        3 to "HEAVY PRECIPITATION"
    )

    // StandardScaler parameters exported from training (one value per feature)
    private var featureMeans = DoubleArray(0)
    private var featureStds = DoubleArray(0)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_analysis)

        // Initialize OpenCV
        if (!OpenCVLoader.initDebug()) {
            Log.e("OpenCV", "Initialization failed!")
        } else {
            Log.d("OpenCV", "OpenCV initialized successfully!")
        }

        // Load normalization values first
        loadNormalizationValues()

        // Load TFLite model
        try {
            tfliteInterpreter = Interpreter(loadModelFile())

            val inputTensor = tfliteInterpreter?.getInputTensor(0)
            val outputTensor = tfliteInterpreter?.getOutputTensor(0)
            Log.d(TAG, "Model input shape: ${inputTensor?.shape()?.contentToString()}")
            Log.d(TAG, "Model output shape: ${outputTensor?.shape()?.contentToString()}")

            Log.d(TAG, "TFLite model loaded successfully!")
            Toast.makeText(this, "Cloud model loaded successfully!", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Log.e(TAG, "Error loading TFLite model", e)
            Toast.makeText(this, "Error loading model: ${e.message}", Toast.LENGTH_LONG).show()
        }

        val imageUriString = intent.getStringExtra("image_uri")
        val imageUri = Uri.parse(imageUriString)

        val imageView: ImageView = findViewById(R.id.capturedImage)
        originalBitmap = loadBitmapFromUri(imageUri)
        // Display the original image
        imageView.setImageBitmap(originalBitmap)

        val analyzeCloudButton: Button = findViewById(R.id.read_notes_button)
        val progressLoader: ProgressBar = findViewById(R.id.progress_loader)
        val resultBox: View = findViewById(R.id.result_box)
        val precipitationResultTextView: TextView = findViewById(R.id.precipitation_result)
        val cloudTypeResultTextView: TextView = findViewById(R.id.cloud_type_result)

        analyzeCloudButton.setOnClickListener {
            progressLoader.visibility = ProgressBar.VISIBLE
            analyzeCloudButton.isEnabled = false

            Handler(Looper.getMainLooper()).postDelayed({
                // Preprocess exactly like training: grayscale, resize to
                // 256x256, scale to [0,1], Gaussian blur 3x3 sigma 1
                val pixels = originalBitmap?.let { preprocessImage(it) }

                // Wavelet features identical to the training notebook
                val features = pixels?.let { WaveletFeatureExtractor.extract(it) }

                Log.d(TAG, "Feature vector size: ${features?.size ?: 0}")

                if (features == null || tfliteInterpreter == null) {
                    Log.e(TAG, "Features empty or interpreter null")
                    precipitationResultTextView.text = "ERROR"
                    cloudTypeResultTextView.text = "ERROR"
                } else {
                    val result = classifyCloud(features)
                    if (result.precipClass < 0) {
                        precipitationResultTextView.text = "ERROR"
                        cloudTypeResultTextView.text = "ERROR"
                    } else {
                        val cloudName = cloudNames[result.cloudType] ?: result.cloudType
                        val precipDesc = precipDescriptions[result.precipClass] ?: "Unknown"
                        precipitationResultTextView.text = precipDesc
                        cloudTypeResultTextView.text = cloudName
                    }
                }

                resultBox.visibility = View.VISIBLE
                progressLoader.visibility = ProgressBar.GONE
                analyzeCloudButton.text = "Return to Camera"
                analyzeCloudButton.isEnabled = true

                analyzeCloudButton.setOnClickListener {
                    // Simply return to camera without saving the image
                    finish()
                }
            }, 1000)
        }
    }

    private fun classifyCloud(features: DoubleArray): CloudPrediction {
        Log.d(TAG, "Starting cloud classification with ${features.size} features")
        Log.d(TAG, "Feature stats: min=${features.minOrNull()}, max=${features.maxOrNull()}")

        val modelInputSize = tfliteInterpreter?.getInputTensor(0)?.shape()?.get(1) ?: 0
        if (features.size != modelInputSize ||
            featureMeans.size != modelInputSize || featureStds.size != modelInputSize) {
            // A size mismatch means the extraction no longer matches training —
            // truncating/padding would silently feed the model garbage
            Log.e(TAG, "Size mismatch: features=${features.size}, " +
                    "means=${featureMeans.size}, stds=${featureStds.size}, " +
                    "model expects $modelInputSize")
            return CloudPrediction("Error", -1, 0f)
        }

        // sklearn StandardScaler.transform: (x - mean) / std, no clamping
        val inputFeatures = Array(1) {
            FloatArray(modelInputSize) { i ->
                ((features[i] - featureMeans[i]) / featureStds[i]).toFloat()
            }
        }

        val outputSize = tfliteInterpreter?.getOutputTensor(0)?.shape()?.get(1) ?: 10
        val outputProbabilities = Array(1) { FloatArray(outputSize) }

        try {
            Log.d(TAG, "Running TFLite inference...")
            tfliteInterpreter?.run(inputFeatures, outputProbabilities)
            Log.d(TAG, "Raw output: ${outputProbabilities[0].contentToString()}")

            var maxIdx = 0
            var maxProb = outputProbabilities[0][0]
            for (i in outputProbabilities[0].indices) {
                Log.d(TAG, "Class $i (${if (i < cloudTypes.size) cloudTypes[i] else "unknown"}): ${outputProbabilities[0][i]}")
                if (outputProbabilities[0][i] > maxProb) {
                    maxProb = outputProbabilities[0][i]
                    maxIdx = i
                }
            }

            Log.d(TAG, "Selected class: $maxIdx (${cloudTypes[maxIdx]}) with confidence $maxProb")

            return CloudPrediction(
                cloudType = cloudTypes[maxIdx],
                precipClass = precipClasses[maxIdx],
                confidence = maxProb
            )
        } catch (e: Exception) {
            Log.e(TAG, "Inference error: ${e.message}", e)
            return CloudPrediction("Error", -1, 0f)
        }
    }

    data class CloudPrediction(
        val cloudType: String,
        val precipClass: Int,
        val confidence: Float
    )

    private fun loadModelFile(): MappedByteBuffer {
        val assetManager = assets
        val modelPath = "cloud_model.tflite"
        val fileDescriptor = assetManager.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun loadBitmapFromUri(uri: Uri): Bitmap? {
        return try {
            val inputStream = contentResolver.openInputStream(uri)
            val bitmap = BitmapFactory.decodeStream(inputStream)
            inputStream?.close()

            val input = contentResolver.openInputStream(uri)
            val exif = ExifInterface(input!!)
            val orientation = exif.getAttributeInt(
                ExifInterface.TAG_ORIENTATION,
                ExifInterface.ORIENTATION_NORMAL
            )

            when (orientation) {
                ExifInterface.ORIENTATION_ROTATE_90 -> rotateBitmap(bitmap, 90f)
                ExifInterface.ORIENTATION_ROTATE_180 -> rotateBitmap(bitmap, 180f)
                ExifInterface.ORIENTATION_ROTATE_270 -> rotateBitmap(bitmap, 270f)
                else -> bitmap
            }
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }

    private fun rotateBitmap(bitmap: Bitmap, degrees: Float): Bitmap {
        val matrix = Matrix()
        matrix.postRotate(degrees)
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    /**
     * Training preprocessing (notebook process_image): grayscale -> resize to
     * 256x256 -> scale to [0,1] -> GaussianBlur (3,3) sigma=1, all in float.
     */
    private fun preprocessImage(bitmap: Bitmap): Array<DoubleArray> {
        val rgba = Mat()
        Utils.bitmapToMat(bitmap, rgba)  // Android bitmaps convert to RGBA
        val gray = Mat()
        Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY)

        val grayFloat = Mat()
        gray.convertTo(grayFloat, CvType.CV_32F, 1.0 / 255.0)

        val resized = Mat()
        // INTER_AREA is the closest OpenCV match to PIL's antialiased downscale
        Imgproc.resize(grayFloat, resized, Size(256.0, 256.0), 0.0, 0.0, Imgproc.INTER_AREA)

        val blurred = Mat()
        Imgproc.GaussianBlur(resized, blurred, Size(3.0, 3.0), 1.0)

        val data = FloatArray(256 * 256)
        blurred.get(0, 0, data)

        rgba.release()
        gray.release()
        grayFloat.release()
        resized.release()
        blurred.release()

        return Array(256) { r -> DoubleArray(256) { c -> data[r * 256 + c].toDouble() } }
    }

    private fun loadNormalizationValues() {
        try {
            featureMeans = readValues("feature_means.txt")
            featureStds = readValues("feature_stds.txt")
            Log.d(TAG, "Loaded ${featureMeans.size} means, ${featureStds.size} stds")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load normalization values", e)
        }
    }

    private fun readValues(assetName: String): DoubleArray {
        assets.open(assetName).bufferedReader().use { reader ->
            return reader.readLines()
                .filter { it.isNotBlank() }
                .map { it.trim().toDouble() }
                .toDoubleArray()
        }
    }

    override fun onDestroy() {
        tfliteInterpreter?.close()
        super.onDestroy()
    }
}
