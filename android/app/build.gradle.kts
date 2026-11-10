plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}
val ortVersion = "1.31.1"
android {
    namespace = "com.tronx.bench"
    compileSdk = 34
    defaultConfig {
        applicationId = "com.tronx.bench"
        minSdk = 29
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
        ndk { abiFilters += listOf("arm64-v8a") }
        buildConfigField("String", "ORT_VERSION", "\"$ortVersion\"")
    }
    buildFeatures { buildConfig = true }
    // Models ship INSIDE the APK (copied from src/<flavor>/assets/models/ at build time; see README "Android app").
    flavorDimensions += "models"
    productFlavors {
        create("quant") { dimension = "models"; applicationIdSuffix = ".quant"; resValue("string", "app_name", "TronX Bench B (INT8+INT4)") }
        create("int8") { dimension = "models"; applicationIdSuffix = ".int8"; resValue("string", "app_name", "TronX Bench (INT8)") }
    }
    buildTypes {
        release { isMinifyEnabled = false; signingConfig = signingConfigs.getByName("debug") }
    }
    compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
    kotlinOptions { jvmTarget = "17" }
    androidResources { noCompress += listOf("json", "onnx", "onnx_data") }
}
dependencies {
    implementation("com.microsoft.onnxruntime:onnxruntime-android:$ortVersion")
}
