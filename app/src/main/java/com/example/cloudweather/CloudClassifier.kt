package com.example.cloudweather

/** Pure (Android-free) classification helpers so they can be unit tested on the JVM. */
object CloudClassifier {
    /** Order = training class order (sorted folder names). */
    val cloudTypes = arrayOf("Ac", "As", "Cc", "Cs", "Ci", "Cb", "Cu", "Ns", "Sc", "St")

    /** Precipitation level per cloud type, same order as [cloudTypes]. */
    val precipClasses = intArrayOf(1, 1, 0, 0, 0, 3, 1, 2, 1, 1)

    /** Below this top-class probability the result is shown as a low-confidence guess. */
    const val LOW_CONFIDENCE = 0.5f

    data class Prediction(val cloudType: String, val precipClass: Int, val confidence: Float) {
        val isLowConfidence get() = confidence < LOW_CONFIDENCE
    }

    /** sklearn StandardScaler.transform: (x - mean) / std, no clamping. */
    fun standardize(features: DoubleArray, means: DoubleArray, stds: DoubleArray): FloatArray =
        FloatArray(features.size) { i -> ((features[i] - means[i]) / stds[i]).toFloat() }

    /** Picks the most probable class; null if the output size doesn't match the labels. */
    fun predict(probabilities: FloatArray): Prediction? {
        if (probabilities.size != cloudTypes.size) return null
        val best = probabilities.indices.maxByOrNull { probabilities[it] } ?: return null
        return Prediction(cloudTypes[best], precipClasses[best], probabilities[best])
    }
}
