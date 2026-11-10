plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}
val ortVersion = "1.31.1"
android {
    namespace = "com.tronx.voice"
    compileSdk = 34
    defaultConfig {
        applicationId = "com.tronx.voice"
        minSdk = 29
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
        ndk { abiFilters += listOf("arm64-v8a") }
        buildConfigField("String", "ORT_VERSION", "\"$ortVersion\"")
    }
    buildFeatures { buildConfig = true }
    compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
    kotlinOptions { jvmTarget = "17" }
    androidResources { noCompress += listOf("json", "onnx", "onnx_data", "txt") }
}
dependencies { implementation("com.microsoft.onnxruntime:onnxruntime-android:$ortVersion") }
