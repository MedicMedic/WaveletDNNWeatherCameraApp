package com.example.noteanalyzer

import android.content.ContentValues
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.exifinterface.media.ExifInterface
import android.graphics.Matrix
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.provider.MediaStore
import android.util.Log
import java.text.SimpleDateFormat
import java.util.Locale

class AnalysisActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_analysis)

        val imageUriString = intent.getStringExtra("image_uri")
        val selectedClef = intent.getStringExtra("selected_clef") ?: "Selected Clef: "
        val selectedKey = intent.getStringExtra("selected_key") ?: "Selected Key: "

        val imageUri = Uri.parse(imageUriString)

        val imageView: ImageView = findViewById(R.id.capturedImage)
        val bitmap = loadBitmapFromUri(imageUri)
        imageView.setImageBitmap(bitmap) // Set the rotated bitmap after processing


        val clefTextView: TextView = findViewById(R.id.selected_clef)
        val keyTextView: TextView = findViewById(R.id.selected_key)
        clefTextView.text = selectedClef
        keyTextView.text = selectedKey
        Log.d("AnalysisActivity", "Clef Text: ${clefTextView.text}")
        Log.d("AnalysisActivity", "Key Text: ${keyTextView.text}")


        val readNotesButton: Button = findViewById(R.id.read_notes_button)
        val progressLoader: ProgressBar = findViewById(R.id.progress_loader)

        readNotesButton.setOnClickListener {
            // Show progress loader and disable button
            progressLoader.visibility = ProgressBar.VISIBLE
            readNotesButton.isEnabled = false

            // Simulate processing with a delay
            Handler(Looper.getMainLooper()).postDelayed({
                // Hide progress loader
                progressLoader.visibility = ProgressBar.GONE

                // Change button to "Save and Return"
                readNotesButton.text = "Save and Return"
                readNotesButton.isEnabled = true

                // Update button action
                readNotesButton.setOnClickListener {
                    saveImageAndReturn(bitmap)
                }
            }, 1000) // Simulate 1 second of processing
        }
    }

    private fun loadBitmapFromUri(uri: Uri): Bitmap? {
        return try {
            val inputStream = contentResolver.openInputStream(uri)
            val bitmap = BitmapFactory.decodeStream(inputStream)
            inputStream?.close()

            // Check and correct image orientation
            val input = contentResolver.openInputStream(uri)
            val exif = ExifInterface(input!!)
            val orientation = exif.getAttributeInt(
                ExifInterface.TAG_ORIENTATION,
                ExifInterface.ORIENTATION_NORMAL
            )

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

    private fun saveImageAndReturn(bitmap: Bitmap?) {
        bitmap?.let {
            val name = "Processed" + SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US)
                .format(System.currentTimeMillis()) // Create a timestamped name
            val contentValues = ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, name) // File name
                put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg") // MIME type
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    put(
                        MediaStore.Images.Media.RELATIVE_PATH,
                        "Pictures/NoteAnalyzer/Processed Sheet Music"
                    ) // Folder path
                }
            }
            val resolver = contentResolver
            val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)
            uri?.let {
                try {
                    val outputStream = resolver.openOutputStream(it)
                    outputStream?.use { os ->
                        bitmap.compress(Bitmap.CompressFormat.JPEG, 100, os)
                        os.close()
                        Toast.makeText(this, "Image saved to gallery!", Toast.LENGTH_SHORT).show()
                    }
                }catch (e: Exception) {
                    e.printStackTrace()
                    Toast.makeText(this, "Failed to save image", Toast.LENGTH_SHORT).show()
                }
            }
            // Return to the main menu
            finish()
        }
    }
}