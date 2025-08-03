package com.example.spotxai.data

import android.content.Context
import android.graphics.Bitmap
import android.view.Surface
import com.example.spotxai.domain.Classification
import com.example.spotxai.domain.SpotXClassifier
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.core.vision.ImageProcessingOptions
import org.tensorflow.lite.task.vision.classifier.ImageClassifier
import org.tensorflow.lite.task.vision.classifier.Classifications

/**
 * TfLiteLandmarkClassifier:
 * - Uses TensorFlow Lite Task Vision ImageClassifier to classify images.
 * - Loads the TFLite model from assets.
 * - Returns top predictions with a confidence score.
 */
class TfLiteLandmarkClassifier(
    private val context: Context,
    private val threshold: Float = 0.4f,
    private val maxResults: Int = 3
) :SpotXClassifier{

    // ImageClassifier instance (from TFLite Task Vision library)
    private var classifier: ImageClassifier? = null

    /**
     * Sets up the TFLite classifier with options.
     * - Loads model from assets (landmarks.tflite)
     * - Applies threshold, max results and number of threads
     */
    private fun setupClassifier() {
        // Base options =  Ye model ke low-level settings define karta hai.
        // : e.g., threads, GPU delegate (here using 2 threads)
        val baseOptions = BaseOptions.builder()
            .setNumThreads(2)
            .build()
    /**
     * ImageClassifier-specific options
     * options ka kaam hai model se result kaise milega, wo control karna.
     *@return ( define options ki result kaise ayegi )
    */
        val options = ImageClassifier.ImageClassifierOptions.builder()
            .setBaseOptions(baseOptions)
            .setMaxResults(maxResults)        // top-N predictions
            .setScoreThreshold(threshold)     // minimum confidence filter
            .build()

        // Create classifier instance by loading model from assets
        classifier = ImageClassifier.createFromFileAndOptions(
            context,
            "landmarks.tflite", // model file name (inside assets folder)
            options
        )
    }

    /**
     * @return List of Classification results (name + score)
     */
    override fun classify(bitmap: Bitmap, rotation: Int): List<Classification> {
        // If classifier not initialized yet, load it
        if (classifier == null) {
            setupClassifier()
        }

        // Convert bitmap to TensorImage (model-compatible format)
        val tensorImage = ImageProcessor.Builder().build()
            .process(TensorImage.fromBitmap(bitmap))

        // Set correct orientation of the image based on camera rotation
        val imageProcessingOptions = ImageProcessingOptions.builder()
            .setOrientation(getOrientationFromRotation(rotation))
            .build()

        // Run model
        val results: List<Classifications>? = classifier?.classify(tensorImage, imageProcessingOptions)

        // Map results into custom Classification objects
        return results?.flatMap { classifications ->
            classifications.categories.map { category ->
                Classification(
                    name = category.displayName,
                    score = category.score
                )
            }
        }?.distinctBy { it.name } ?: emptyList() // remove duplicates and handle null
    }

    /**
     * Maps CameraX Surface.ROTATION_* values to TFLite Orientation enum
     * so that model receives image in the correct direction.
     */
    private fun getOrientationFromRotation(rotation: Int): ImageProcessingOptions.Orientation {
        return when (rotation) {
            Surface.ROTATION_270 -> ImageProcessingOptions.Orientation.BOTTOM_RIGHT
            Surface.ROTATION_90  -> ImageProcessingOptions.Orientation.TOP_LEFT
            Surface.ROTATION_180 -> ImageProcessingOptions.Orientation.RIGHT_BOTTOM
            else                  -> ImageProcessingOptions.Orientation.RIGHT_TOP
        }
    }
}
