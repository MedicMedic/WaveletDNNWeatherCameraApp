package com.example.noteanalyzer

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import androidx. exifinterface. media. ExifInterface
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.launch

class AnalysisActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_analysis)

        val imageUriString = intent.getStringExtra("image_uri")
        val selectedClef = intent.getStringExtra("selected_clef")
        val selectedKey = intent.getStringExtra("selected_key")

        val imageUri = Uri.parse(imageUriString)

        val imageView: ImageView = findViewById(R.id.capturedImage)
        // Run image loading and rotation in a coroutine
        GlobalScope.launch(Dispatchers.Main) {
            val bitmap = loadBitmapFromUri(imageUri)
            imageView.setImageBitmap(bitmap) // Set the rotated bitmap after processing
        }

        val clefTextView: TextView = findViewById(R.id.selected_clef)
        val keyTextView: TextView = findViewById(R.id.selected_key)
        clefTextView.text = selectedClef
        keyTextView.text = selectedKey


    }

    private fun loadBitmapFromUri(uri: Uri): Bitmap? {
        return try {
            val inputStream = contentResolver.openInputStream(uri)
            val bitmap = BitmapFactory.decodeStream(inputStream)

            inputStream?.close()

            val exifInterface = ExifInterface(inputStream!!)
            val orientation = exifInterface.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)


            val rotatedBitmap = when (orientation) {
                ExifInterface.ORIENTATION_ROTATE_90 -> rotateBitmap(bitmap, 90f)
                ExifInterface.ORIENTATION_ROTATE_180 -> rotateBitmap(bitmap, 180f)
                ExifInterface.ORIENTATION_ROTATE_270 -> rotateBitmap(bitmap, 270f)
                else -> bitmap
            }
                rotatedBitmap
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
}