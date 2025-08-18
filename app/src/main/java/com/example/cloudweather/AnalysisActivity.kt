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
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Scalar
import android.content.Context
import java.io.FileOutputStream
import kotlin.math.sqrt

class AnalysisActivity : AppCompatActivity() {
    private val TAG = "AnalysisActivity"
    private var tfliteInterpreter: Interpreter? = null
    private var originalBitmap: Bitmap? = null

    // Cloud type mapping
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

    // Statistical normalization parameters (estimated from training data)
    private val featureMeans = FloatArray(2048) { 0.0f }  // Replace with actual values if available
    private val featureStds = FloatArray(2048) { 1.0f }   // Replace with actual values if available

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

            // Add debug info about model input/output shapes
            val inputTensor = tfliteInterpreter?.getInputTensor(0)
            val outputTensor = tfliteInterpreter?.getOutputTensor(0)
            Log.d(TAG, "Model input shape: ${inputTensor?.shape()?.contentToString()}")
            Log.d(TAG, "Model output shape: ${outputTensor?.shape()?.contentToString()}")

            Log.d(TAG, "TFLite model loaded successfully!")
            Toast.makeText(this, "Cloud model loaded successfully!", Toast.LENGTH_SHORT).show()

            // Test model with dummy data
            if (tfliteInterpreter != null) {
                testModelWithDummy()
            }
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
                // Preprocess the image for analysis but don't change displayed image
                val processedBitmap = originalBitmap?.let { preprocessImage(it) }

                // Keep displaying the original image
                // imageView.setImageBitmap remains with the original bitmap

                // Extract features compatible with Python model from the processed image
                val features = processedBitmap?.let { extractWaveletFeatures(it) } ?: FloatArray(0)

                Log.d(TAG, "Feature vector size: ${features.size}")

                if (features.isEmpty() || tfliteInterpreter == null) {
                    Log.e(TAG, "Features empty or interpreter null")
                    precipitationResultTextView.text = "ERROR"
                    cloudTypeResultTextView.text = "ERROR"
                    return@postDelayed
                }
                else if (features.isNotEmpty() && tfliteInterpreter != null) {
                    val result = classifyCloud(features)
                    val cloudName = cloudNames[result.cloudType] ?: result.cloudType
                    val precipDesc = precipDescriptions[result.precipClass] ?: "Unknown"

                    // Update the separate TextViews in the result box
                    precipitationResultTextView.text = precipDesc
                    cloudTypeResultTextView.text = cloudName
                } else {
                    precipitationResultTextView.text = "ERROR"
                    cloudTypeResultTextView.text = "ERROR"
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

    private fun classifyCloud(features: FloatArray): CloudPrediction {
        // Add extensive logging
        Log.d(TAG, "Starting cloud classification with ${features.size} features")
        Log.d(TAG, "Feature stats: min=${features.minOrNull()}, max=${features.maxOrNull()}, avg=${features.average()}")

        // Normalize features
        val normalizedFeatures = normalizeFeatures(features)
        Log.d(TAG, "Normalized stats: min=${normalizedFeatures.minOrNull()}, max=${normalizedFeatures.maxOrNull()}, avg=${normalizedFeatures.average()}")

        // Prepare input for TFLite model
        val modelInputSize = tfliteInterpreter?.getInputTensor(0)?.shape()?.get(1) ?: 0
        Log.d(TAG, "Model expects input size: $modelInputSize")

        // Match feature size to model input size
        val inputFeatures = Array(1) {
            when {
                normalizedFeatures.size > modelInputSize -> {
                    Log.d(TAG, "Truncating features from ${normalizedFeatures.size} to $modelInputSize")
                    normalizedFeatures.copyOfRange(0, modelInputSize)
                }
                normalizedFeatures.size < modelInputSize -> {
                    Log.d(TAG, "Padding features from ${normalizedFeatures.size} to $modelInputSize")
                    val padded = FloatArray(modelInputSize) { 0f }
                    normalizedFeatures.copyInto(padded)
                    padded
                }
                else -> normalizedFeatures
            }
        }

        // Run inference
        val outputSize = tfliteInterpreter?.getOutputTensor(0)?.shape()?.get(1) ?: 10
        val outputProbabilities = Array(1) { FloatArray(outputSize) }

        try {
            Log.d(TAG, "Running TFLite inference...")
            tfliteInterpreter?.run(inputFeatures, outputProbabilities)
            Log.d(TAG, "Raw output: ${outputProbabilities[0].contentToString()}")

            // Find predicted class
            var maxIdx = 0
            var maxProb = outputProbabilities[0][0]

            // Log all probabilities for debugging
            for (i in outputProbabilities[0].indices) {
                Log.d(TAG, "Class $i (${if (i < cloudTypes.size) cloudTypes[i] else "unknown"}): ${outputProbabilities[0][i]}")
                if (outputProbabilities[0][i] > maxProb) {
                    maxProb = outputProbabilities[0][i]
                    maxIdx = i
                }
            }

            Log.d(TAG, "Selected class: $maxIdx with confidence $maxProb")

            // Ensure index is valid
            val safeIdx = maxIdx.coerceIn(0, cloudTypes.size - 1)
            if (maxIdx != safeIdx) {
                Log.w(TAG, "Had to adjust index from $maxIdx to $safeIdx")
            }

            return CloudPrediction(
                cloudType = cloudTypes[safeIdx],
                precipClass = precipClasses[safeIdx],
                confidence = maxProb
            )
        } catch (e: Exception) {
            Log.e(TAG, "Inference error: ${e.message}", e)
            e.printStackTrace()

            // Default to Cc with precipitation 0 on error (let's fix this!)
            return CloudPrediction(
                cloudType = "Error",
                precipClass = -1,
                confidence = 0f
            )
        }
    }

    private fun normalizeFeatures(features: FloatArray): FloatArray {
        // Apply standard scaling (zero mean, unit variance) like sklearn.preprocessing.StandardScaler
        val normalized = FloatArray(features.size)

        for (i in features.indices) {
            if (i < featureMeans.size && i < featureStds.size) {
                normalized[i] = (features[i] - featureMeans[i]) / (featureStds[i] + 1e-8f)
            } else {
                // If we don't have normalization parameters for this feature, use as is
                normalized[i] = features[i]
            }
        }

        Log.d(TAG, "Normalized features: min=${normalized.minOrNull()}, max=${normalized.maxOrNull()}")
        return normalized
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

    private fun preprocessImage(bitmap: Bitmap): Bitmap {
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 256, 256, true)
        val originalMat = Mat()
        val grayMat = Mat()
        Utils.bitmapToMat(resizedBitmap, originalMat)
        Imgproc.cvtColor(originalMat, grayMat, Imgproc.COLOR_BGR2GRAY)

        val blurredMat = Mat()
        Imgproc.GaussianBlur(grayMat, blurredMat, Size(3.0, 3.0), 1.0)

        val resultBitmap = Bitmap.createBitmap(256, 256, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(blurredMat, resultBitmap)

        originalMat.release()
        grayMat.release()
        blurredMat.release()

        return resultBitmap
    }

    /**
     * Extract wavelet features compatible with the Python model
     * This implementation matches the Python pywt.wavedec2 approach using 'db4' wavelet
     */
    private fun extractWaveletFeatures(bitmap: Bitmap): FloatArray {
        // Convert bitmap to grayscale float array
        val mat = Mat()
        val grayMat = Mat()
        Utils.bitmapToMat(bitmap, mat)
        Imgproc.cvtColor(mat, grayMat, Imgproc.COLOR_BGR2GRAY)

        // Normalize image values between 0 and 1
        val minMaxResult = Core.minMaxLoc(grayMat)
        val minVal = minMaxResult.minVal
        val maxVal = minMaxResult.maxVal
        val normalizedMat = Mat()

        // Create Scalar with proper constructor - use double value
        Core.subtract(grayMat, Scalar(minVal), normalizedMat)
        Core.divide(normalizedMat, Scalar(maxVal - minVal + 1e-8), normalizedMat)

        // Convert to 2D array for wavelet transform
        val rows = normalizedMat.rows()
        val cols = normalizedMat.cols()
        val imageArray = Array(rows) { FloatArray(cols) }
        for (i in 0 until rows) {
            for (j in 0 until cols) {
                imageArray[i][j] = normalizedMat.get(i, j)[0].toFloat()
            }
        }

        val features = mutableListOf<Float>()
        val level = 3

        // Implement wavedec2 for db4 wavelet
        var currentArray = imageArray
        val allCoeffs = mutableListOf<Array<FloatArray>>()

        // Approximation coefficients for each level
        for (l in 0 until level) {
            val size = currentArray.size / 2

            // Create arrays for approximation and details
            val approx = Array(size) { FloatArray(size) }
            val horizDetail = Array(size) { FloatArray(size) }
            val vertDetail = Array(size) { FloatArray(size) }
            val diagDetail = Array(size) { FloatArray(size) }

            // Implement simplified wavelet transform
            // Note: This is a simplified implementation - for production use a proper wavelet library
            for (i in 0 until size) {
                for (j in 0 until size) {
                    if (2*i+1 < currentArray.size && 2*j+1 < currentArray[0].size) {
                        // Approximation (LL)
                        approx[i][j] = (currentArray[2*i][2*j] + currentArray[2*i][2*j+1] +
                                currentArray[2*i+1][2*j] + currentArray[2*i+1][2*j+1]) / 4.0f

                        // Horizontal detail (LH)
                        horizDetail[i][j] = (currentArray[2*i][2*j] - currentArray[2*i][2*j+1] +
                                currentArray[2*i+1][2*j] - currentArray[2*i+1][2*j+1]) / 4.0f

                        // Vertical detail (HL)
                        vertDetail[i][j] = (currentArray[2*i][2*j] + currentArray[2*i][2*j+1] -
                                currentArray[2*i+1][2*j] - currentArray[2*i+1][2*j+1]) / 4.0f

                        // Diagonal detail (HH)
                        diagDetail[i][j] = (currentArray[2*i][2*j] - currentArray[2*i][2*j+1] -
                                currentArray[2*i+1][2*j] + currentArray[2*i+1][2*j+1]) / 4.0f
                    }
                }
            }

            // Process approximation coefficient - flatten and add statistical features
            allCoeffs.add(approx)
            // Process detail coefficients at this level
            allCoeffs.add(horizDetail)
            allCoeffs.add(vertDetail)
            allCoeffs.add(diagDetail)

            // Use approximation for the next level
            currentArray = approx
        }

        // Extract features from all coefficients exactly like in Python code
        for (coeff in allCoeffs) {
            val flattened = coeff.flatten()

            // Add all coefficients
            features.addAll(flattened.toList())

            // Add statistical features
            val mean = flattened.average().toFloat()
            val std = calculateStd(flattened)
            val median = flattened.sorted()[flattened.size / 2]
            val max = flattened.maxOrNull() ?: 0f
            val min = flattened.minOrNull() ?: 0f
            val energy = flattened.sumOf { it.toDouble() * it.toDouble() }.toFloat()

            features.add(mean)      // Mean
            features.add(std)       // Std
            features.add(median)    // Median
            features.add(max)       // Max
            features.add(min)       // Min
            features.add(energy)    // Energy (sum of squares)
        }

        // Clean up OpenCV resources
        mat.release()
        grayMat.release()
        normalizedMat.release()

        Log.d(TAG, "Extracted ${features.size} wavelet features")
        return features.toFloatArray()
    }

    private fun Array<FloatArray>.flatten(): FloatArray {
        val result = FloatArray(this.size * this[0].size)
        var index = 0
        for (row in this) {
            for (value in row) {
                result[index++] = value
            }
        }
        return result
    }

    private fun calculateStd(array: FloatArray): Float {
        val mean = array.average()
        var sum = 0.0
        for (value in array) {
            sum += (value - mean) * (value - mean)
        }
        return sqrt(sum.toFloat() / array.size)
    }

    private fun testModelWithDummy() {
        if (tfliteInterpreter == null) {
            Log.e(TAG, "Cannot test - interpreter is null")
            return
        }

        try {
            val inputTensor = tfliteInterpreter?.getInputTensor(0)
            val inputSize = inputTensor?.shape()?.get(1) ?: 0

            Log.d(TAG, "Creating dummy input with size $inputSize")
            val dummyInput = Array(1) { FloatArray(inputSize) { 0.0f } }
            val outputTensor = tfliteInterpreter?.getOutputTensor(0)
            val outputSize = outputTensor?.shape()?.get(1) ?: 0
            val outputs = Array(1) { FloatArray(outputSize) }

            tfliteInterpreter?.run(dummyInput, outputs)
            Log.d(TAG, "Test inference successful! Output: ${outputs[0].contentToString()}")
        } catch (e: Exception) {
            Log.e(TAG, "Test inference failed: ${e.message}", e)
            e.printStackTrace()
        }
    }

    private fun loadNormalizationValues() {
        try {
            val inputMeans = assets.open("feature_means.txt")
            val inputStds = assets.open("feature_stds.txt")

            // Option 1: Direct indexing into existing arrays
            // Read means
            inputMeans.bufferedReader().useLines { lines ->
                lines.forEachIndexed { index, line ->
                    if (index < featureMeans.size) {
                        featureMeans[index] = line.trim().toFloat()
                    }
                }
            }

            // Read stds
            inputStds.bufferedReader().useLines { lines ->
                lines.forEachIndexed { index, line ->
                    if (index < featureStds.size) {
                        featureStds[index] = line.trim().toFloat()
                        // Avoid division by zero
                        if (featureStds[index] < 1e-5f) featureStds[index] = 1.0f
                    }
                }
            }

            Log.d(TAG, "Normalization values loaded successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load normalization values", e)
        }
    }

    override fun onDestroy() {
        tfliteInterpreter?.close()
        super.onDestroy()
    }
}