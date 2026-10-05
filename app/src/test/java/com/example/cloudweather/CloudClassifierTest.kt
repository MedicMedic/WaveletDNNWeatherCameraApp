package com.example.cloudweather

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class CloudClassifierTest {
    @Test
    fun standardize_matchesStandardScaler() {
        val out = CloudClassifier.standardize(
            doubleArrayOf(3.0, -1.0), doubleArrayOf(1.0, 1.0), doubleArrayOf(2.0, 0.5)
        )
        assertEquals(1.0f, out[0], 1e-6f)
        assertEquals(-4.0f, out[1], 1e-6f)  // not clamped
    }

    @Test
    fun predict_picksArgmaxAndMapsPrecipitation() {
        val probs = FloatArray(10).also { it[5] = 0.9f; it[0] = 0.1f }
        val p = CloudClassifier.predict(probs)!!
        assertEquals("Cb", p.cloudType)
        assertEquals(3, p.precipClass)  // cumulonimbus = heavy
        assertFalse(p.isLowConfidence)
    }

    @Test
    fun predict_flagsLowConfidence() {
        val probs = FloatArray(10) { 0.1f }.also { it[2] = 0.19f }
        assertTrue(CloudClassifier.predict(probs)!!.isLowConfidence)
    }

    @Test
    fun predict_rejectsWrongOutputSize() {
        assertNull(CloudClassifier.predict(FloatArray(3)))
    }

    @Test
    fun labelTablesStayAligned() {
        assertEquals(CloudClassifier.cloudTypes.size, CloudClassifier.precipClasses.size)
    }
}
