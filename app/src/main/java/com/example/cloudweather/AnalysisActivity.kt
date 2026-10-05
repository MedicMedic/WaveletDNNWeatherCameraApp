package com.example.cloudweather

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
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
import java.util.concurrent.Executors

class AnalysisActivity : AppCompatActivity() {
    private val TAG = "AnalysisActivity"
    private var tfliteInterpreter: Interpreter? = null
    private var originalBitmap: Bitmap? = null
    private val analysisExecutor = Executors.newSingleThreadExecutor()

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
            Toast.makeText(this, R.string.model_loaded, Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Log.e(TAG, "Error loading TFLite model", e)
            Toast.makeText(this, getString(R.string.model_load_error, e.message), Toast.LENGTH_LONG).show()
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
        val confidenceTextView: TextView = findViewById(R.id.confidence_result)
        val lowConfidenceNote: View = findViewById(R.id.low_confidence_note)

        analyzeCloudButton.setOnClickListener {
            progressLoader.visibility = ProgressBar.VISIBLE
            analyzeCloudButton.isEnabled = false

            // Feature extraction + inference are heavy: keep them off the UI thread
            analysisExecutor.execute {
                // Preprocess exactly like training: grayscale, resize to
                // 256x256, scale to [0,1], Gaussian blur 3x3 sigma 1
                val pixels = originalBitmap?.let { preprocessImage(it) }

                // Wavelet features identical to the training notebook
                val features = pixels?.let { WaveletFeatureExtractor.extract(it) }

                Log.d(TAG, "Feature vector size: ${features?.size ?: 0}")

                val result = features?.let { classifyCloud(it) }

                runOnUiThread {
                    if (isDestroyed) return@runOnUiThread
                    if (result == null) {
                        precipitationResultTextView.setText(R.string.result_error)
                        cloudTypeResultTextView.setText(R.string.result_error)
                    } else {
                        precipitationResultTextView.setText(precipStringRes(result.precipClass))
                        cloudTypeResultTextView.text = cloudName(result.cloudType)
                        confidenceTextView.text = getString(
                            R.string.confidence_format, (result.confidence * 100).toInt()
                        )
                        lowConfidenceNote.visibility =
                            if (result.isLowConfidence) View.VISIBLE else View.GONE
                    }

                    resultBox.visibility = View.VISIBLE
                    progressLoader.visibility = ProgressBar.GONE
                    analyzeCloudButton.setText(R.string.return_to_camera)
                    analyzeCloudButton.isEnabled = true

                    analyzeCloudButton.setOnClickListener {
                        // Simply return to camera without saving the image
                        finish()
                    }
                }
            }
        }
    }

    private fun precipStringRes(precipClass: Int) = when (precipClass) {
        0 -> R.string.precip_0
        1 -> R.string.precip_1
        2 -> R.string.precip_2
        else -> R.string.precip_3
    }

    private fun cloudName(code: String): String {
        val id = resources.getIdentifier("cloud_$code", "string", packageName)
        return if (id != 0) getString(id) else code
    }

    private fun classifyCloud(features: DoubleArray): CloudClassifier.Prediction? {
        val interpreter = tfliteInterpreter ?: return null
        val modelInputSize = interpreter.getInputTensor(0).shape()[1]
        if (features.size != modelInputSize ||
            featureMeans.size != modelInputSize || featureStds.size != modelInputSize) {
            // A size mismatch means the extraction no longer matches training -
            // truncating/padding would silently feed the model garbage
            Log.e(TAG, "Size mismatch: features=${features.size}, " +
                    "means=${featureMeans.size}, stds=${featureStds.size}, " +
                    "model expects $modelInputSize")
            return null
        }

        val input = arrayOf(CloudClassifier.standardize(features, featureMeans, featureStds))
        val output = Array(1) { FloatArray(interpreter.getOutputTensor(0).shape()[1]) }
        return try {
            interpreter.run(input, output)
            Log.d(TAG, "Raw output: ${output[0].contentToString()}")
            CloudClassifier.predict(output[0])
        } catch (e: Exception) {
            Log.e(TAG, "Inference error: ${e.message}", e)
            null
        }
    }

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
            // Bounds pass first so multi-megapixel photos are downsampled
            // instead of decoded at full size (the model only sees 256x256)
            val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, bounds) }
            var sample = 1
            while (bounds.outWidth / (sample * 2) >= MAX_DECODE_SIDE &&
                bounds.outHeight / (sample * 2) >= MAX_DECODE_SIDE) sample *= 2
            val options = BitmapFactory.Options().apply { inSampleSize = sample }
            val bitmap = contentResolver.openInputStream(uri)?.use {
                BitmapFactory.decodeStream(it, null, options)
            } ?: return null

            val orientation = contentResolver.openInputStream(uri)?.use {
                ExifInterface(it).getAttributeInt(
                    ExifInterface.TAG_ORIENTATION,
                    ExifInterface.ORIENTATION_NORMAL
                )
            } ?: ExifInterface.ORIENTATION_NORMAL

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
        // Let any in-flight analysis finish before freeing the interpreter it uses
        analysisExecutor.execute { tfliteInterpreter?.close() }
        analysisExecutor.shutdown()
        super.onDestroy()
    }

    companion object {
        // Decode no smaller than this per side; plenty for the 256x256 model input
        private const val MAX_DECODE_SIDE = 1024
    }
}
