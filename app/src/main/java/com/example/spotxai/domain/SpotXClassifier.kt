package com.example.spotxai.domain

import android.graphics.Bitmap

interface SpotXClassifier {
    fun classify(bitmap: Bitmap, rotation: Int): List<Classification>
}