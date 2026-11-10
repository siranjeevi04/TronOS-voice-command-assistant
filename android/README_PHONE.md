# TronX Bench: running the on-device benchmark (Moto G34 5G)

## What you get (models are INSIDE the APKs - nothing to copy)
- `TronxBench_A_fp16.apk` (1.4 GB): `1_base_fp16` + `2_ft_fp16`   -> app "TronX Bench A (FP16)"
- `TronxBench_B_int8_int4.apk` (0.7 GB): `3_ft_int8` + `4_ft_int4` -> app "TronX Bench B (INT8+INT4)"
On first launch each app unpacks its models (~30-60 s, shown on screen). Both apps can be installed side by side.

## 1. Prepare the phone (same conditions for every run)
1. Charge to ~100 %, then UNPLUG (battery drain can only be measured on battery).
2. Settings > Battery > Battery saver: OFF. Close all other apps. Turn on airplane mode (Wi-Fi off) if possible.
3. Let the phone cool for ~10 min, lying flat on a table, out of direct sunlight. Note the room temperature.
4. Enable USB debugging (Settings > About phone > tap "Build number" 7 times > Developer options > USB debugging).

## 2. Install both APKs
Copy the two APK files to the phone (Google Drive / USB file transfer / `adb install -r <apk>`), tap each to install
(allow "install unknown apps"). Uninstall the older "TronX Bench" first if it is still on the phone.

## 3. Run (keep the app in the foreground, screen on; the app keeps the screen awake)
1. Type the room temperature in "Ambient temperature".
2. Leave defaults: threads 4, repeated runs 3, prompts 10, thermal 10 min, cool-down 60 s.
3. Press **1) Device info**.
4. Press **2) Run ALL variants** in app A, then (after it finishes and the phone cools ~5 min) in app B.
   Each runs sections A-F for its 2 models with a 60 s cool-down in between.
5. In app B select `4_ft_int4` and press **4) Thermal test** (10 min of sustained inference).
6. Press **Copy everything** and paste it to me (or **Share report**). Screenshots also work.

Reports are also saved on the phone and can be pulled with:
```
adb pull /sdcard/Android/data/com.tronx.bench.fp16/files/reports/ A_reports
adb pull /sdcard/Android/data/com.tronx.bench.quant/files/reports/ B_reports
```

## What each section on screen means (assignment Section 9)
- DEVICE: manufacturer/model, SoC, CPU, GPU, NPU (not used), RAM, Android version, runtime + version, power mode, ambient temperature (typed), battery at start.
- A: model file size.
- B: memory before/after loading, loading peak, loading time, memory released after loading.
- C: baseline / peak / after memory during inference and the increase caused by inference (VmRSS sampled every 15 ms).
- D: KV cache at 512 / 1,024 / 2,048 tokens with peak RAM, TTFT and decode tok/s (exact-length prompts, 32 output tokens).
- E: prefill and decode tokens/s, average and spread over repeated runs after 2 warm-up runs.
- F: time to first token (model loading and tokenisation excluded; prompts are pre-tokenised).
- G: temperatures, thermal status/headroom, CPU MHz, tok/s every 30 s, degradation, battery used.
