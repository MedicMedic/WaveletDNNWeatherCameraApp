package com.example.cloudweather

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * Verifies the Kotlin wavelet feature extraction against fixtures generated
 * with the training pipeline itself (PIL + cv2 + pywt.wavedec2 db4 level 3).
 *
 * Fixture *_pixels.txt holds the preprocessed 256x256 image (row-major) and
 * *_features.txt the expected 71602-value feature vector produced by pywt.
 */
class WaveletFeatureExtractorTest {

    private fun loadResource(name: String): DoubleArray =
        javaClass.classLoader!!.getResourceAsStream(name)!!.bufferedReader().useLines { lines ->
            lines.filter { it.isNotBlank() }.map { it.trim().toDouble() }.toList().toDoubleArray()
        }

    private fun loadAsset(name: String): DoubleArray {
        var f = File("src/main/assets/$name")
        if (!f.exists()) f = File("app/src/main/assets/$name")
        return f.readLines().filter { it.isNotBlank() }.map { it.trim().toDouble() }.toDoubleArray()
    }

    private fun checkPhoto(name: String) {
        val flat = loadResource("${name}_pixels.txt")
        assertEquals(256 * 256, flat.size)
        val image = Array(256) { r -> DoubleArray(256) { c -> flat[r * 256 + c] } }

        val actual = WaveletFeatureExtractor.extract(image)
        val expected = loadResource("${name}_features.txt")
        assertEquals("feature vector length", expected.size, actual.size)

        // What matters to the model is the StandardScaler-transformed value,
        // so measure error in scaled units
        val stds = loadAsset("feature_stds.txt")
        assertEquals(expected.size, stds.size)

        var maxScaledErr = 0.0
        var maxRawErr = 0.0
        var worstIdx = -1
        for (i in expected.indices) {
            val raw = Math.abs(actual[i] - expected[i])
            val scaled = raw / stds[i]
            if (scaled > maxScaledErr) {
                maxScaledErr = scaled
                worstIdx = i
            }
            if (raw > maxRawErr) maxRawErr = raw
        }
        println("$name: maxRawErr=$maxRawErr maxScaledErr=$maxScaledErr at idx $worstIdx")
        assertTrue(
            "max scaled error $maxScaledErr at index $worstIdx exceeds 0.05",
            maxScaledErr < 0.05
        )
    }

    @Test
    fun featuresMatchPywtForAltostratusPhoto() = checkPhoto("photo_34")

    @Test
    fun featuresMatchPywtForAltocumulusPhoto() = checkPhoto("photo_24")
}
