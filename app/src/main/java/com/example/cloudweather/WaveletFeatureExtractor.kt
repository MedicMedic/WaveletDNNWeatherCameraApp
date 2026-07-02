package com.example.cloudweather

/**
 * Faithful port of the training pipeline's feature extraction:
 * pywt.wavedec2(image, wavelet='db4', level=3, mode='symmetric') followed by
 * the notebook's extract_wavelet_features() feature assembly.
 *
 * Pure Kotlin (no Android/OpenCV dependencies) so it can be verified on the
 * JVM against fixtures generated with pywt itself.
 */
object WaveletFeatureExtractor {

    // pywt db4 decomposition filters (full double precision)
    private val DEC_LO = doubleArrayOf(
        -0.010597401785069032, 0.0328830116668852, 0.030841381835560764,
        -0.18703481171909309, -0.027983769416859854, 0.6308807679298589,
        0.7148465705529157, 0.2303778133088965
    )
    private val DEC_HI = doubleArrayOf(
        -0.2303778133088965, 0.7148465705529157, -0.6308807679298589,
        -0.027983769416859854, 0.18703481171909309, 0.030841381835560764,
        -0.0328830116668852, -0.010597401785069032
    )
    private const val FILTER_LEN = 8
    private const val LEVELS = 3

    /**
     * Single-level 1-D DWT matching pywt's downsampling convolution with
     * 'symmetric' (half-point, edge-repeating) signal extension:
     *   out[i] = sum_j ext[2*i + 8 - j] * filt[j],  outLen = (n + 7) / 2
     */
    private fun dwt1d(x: DoubleArray, filt: DoubleArray): DoubleArray {
        val n = x.size
        val outLen = (n + FILTER_LEN - 1) / 2
        val out = DoubleArray(outLen)
        for (i in 0 until outLen) {
            var s = 0.0
            for (j in 0 until FILTER_LEN) {
                // index into the symmetrically extended signal, shifted back
                // by the pad width (FILTER_LEN - 1) to original coordinates
                var k = 2 * i + FILTER_LEN - j - (FILTER_LEN - 1)
                if (k < 0) k = -k - 1
                if (k >= n) k = 2 * n - k - 1
                s += x[k] * filt[j]
            }
            out[i] = s
        }
        return out
    }

    /** Filter each row with [filt] and downsample: (rows x cols) -> (rows x outLen). */
    private fun filterRows(a: Array<DoubleArray>, filt: DoubleArray): Array<DoubleArray> =
        Array(a.size) { r -> dwt1d(a[r], filt) }

    /** Filter each column with [filt] and downsample: (rows x cols) -> (outLen x cols). */
    private fun filterCols(a: Array<DoubleArray>, filt: DoubleArray): Array<DoubleArray> {
        val rows = a.size
        val cols = a[0].size
        val col = DoubleArray(rows)
        val outLen = (rows + FILTER_LEN - 1) / 2
        val out = Array(outLen) { DoubleArray(cols) }
        for (c in 0 until cols) {
            for (r in 0 until rows) col[r] = a[r][c]
            val f = dwt1d(col, filt)
            for (r in 0 until outLen) out[r][c] = f[r]
        }
        return out
    }

    data class Dwt2Result(
        val cA: Array<DoubleArray>,  // LL
        val cH: Array<DoubleArray>,  // rows: lo, cols: hi  (pywt "horizontal")
        val cV: Array<DoubleArray>,  // rows: hi, cols: lo  (pywt "vertical")
        val cD: Array<DoubleArray>   // HH
    )

    /** Single-level 2-D DWT with pywt's (cA, (cH, cV, cD)) orientation. */
    private fun dwt2(a: Array<DoubleArray>): Dwt2Result {
        val rowLo = filterRows(a, DEC_LO)
        val rowHi = filterRows(a, DEC_HI)
        return Dwt2Result(
            cA = filterCols(rowLo, DEC_LO),
            cH = filterCols(rowLo, DEC_HI),
            cV = filterCols(rowHi, DEC_LO),
            cD = filterCols(rowHi, DEC_HI)
        )
    }

    /**
     * Feature vector identical to the notebook's extract_wavelet_features():
     * min-max normalize, wavedec2 db4 level 3, then coarsest-first
     * [cA3 | stats, (cH,cV,cD) per level 3->1, each | stats].
     * Stats per block: mean, std (population), median (numpy), max, min, energy.
     * For a 256x256 input this yields exactly 71602 values.
     */
    fun extract(image: Array<DoubleArray>): DoubleArray {
        // Normalize image values between 0 and 1 (as in the notebook)
        var mn = Double.POSITIVE_INFINITY
        var mx = Double.NEGATIVE_INFINITY
        for (row in image) for (v in row) {
            if (v < mn) mn = v
            if (v > mx) mx = v
        }
        val denom = mx - mn + 1e-8
        val norm = Array(image.size) { r ->
            DoubleArray(image[r].size) { c -> (image[r][c] - mn) / denom }
        }

        // wavedec2: iterate LL; details collected finest-first, emitted coarsest-first
        var cur = norm
        val detailLevels = ArrayList<Dwt2Result>(LEVELS)
        repeat(LEVELS) {
            val res = dwt2(cur)
            detailLevels.add(res)
            cur = res.cA
        }

        val features = ArrayList<Double>(72000)
        appendBlock(features, cur)  // final approximation cA3
        for (lvl in LEVELS - 1 downTo 0) {
            val d = detailLevels[lvl]
            appendBlock(features, d.cH)
            appendBlock(features, d.cV)
            appendBlock(features, d.cD)
        }
        return features.toDoubleArray()
    }

    private fun appendBlock(features: ArrayList<Double>, block: Array<DoubleArray>) {
        val n = block.size * block[0].size
        val flat = DoubleArray(n)
        var idx = 0
        for (row in block) for (v in row) flat[idx++] = v

        for (v in flat) features.add(v)

        var sum = 0.0
        var mx = flat[0]
        var mn = flat[0]
        var energy = 0.0
        for (v in flat) {
            sum += v
            if (v > mx) mx = v
            if (v < mn) mn = v
            energy += v * v
        }
        val mean = sum / n
        var varSum = 0.0
        for (v in flat) {
            val d = v - mean
            varSum += d * d
        }
        val std = kotlin.math.sqrt(varSum / n)

        val sorted = flat.copyOf()
        sorted.sort()
        // numpy median: average the two middle values for even-length arrays
        val median = if (n % 2 == 1) sorted[n / 2]
                     else (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0

        features.add(mean)
        features.add(std)
        features.add(median)
        features.add(mx)
        features.add(mn)
        features.add(energy)
    }
}
