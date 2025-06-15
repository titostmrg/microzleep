plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

val camerax_version = "1.3.3"

android {
    namespace = "com.example.microzleepz"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.example.microzleepz"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    aaptOptions {
        noCompress("tflite", "task")
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions {
        jvmTarget = "11"
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.androidx.activity)
    implementation(libs.androidx.constraintlayout)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)

    // CameraX
    implementation("androidx.camera:camera-core:$camerax_version")
    implementation("androidx.camera:camera-camera2:$camerax_version")
    implementation("androidx.camera:camera-lifecycle:$camerax_version")
    implementation("androidx.camera:camera-view:$camerax_version")
    implementation("androidx.camera:camera-extensions:$camerax_version")

    // MediaPipe Tasks Vision (Penting untuk FaceLandmarker)
    implementation("com.google.mediapipe:tasks-vision:0.10.11")

    // TensorFlow Lite (Penting untuk menjalankan model .tflite)
    implementation("org.tensorflow:tensorflow-lite:2.15.0") // Versi stabil terbaru
    implementation("org.tensorflow:tensorflow-lite-support:0.1.0") // Untuk utilitas TFLite

    // OkHttp untuk HTTP requests
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    // GSON untuk parsing JSON
    implementation("com.google.code.gson:gson:2.10.1")
}
