# TronX Bench (INT8): on-device Section 9 benchmark (Moto G34 5G)

## APK
- `TronxBench_INT8.apk` (463 MB): fine-tuned Granite-4.0-H-350M + LoRA D, ONNX INT8 (`3_ft_int8`, 424 MB, same file as the Mac run and the voice app).
  App name on the phone: "TronX Bench (INT8)". The model is inside the APK; first launch unpacks it (~30 s, shown on screen).

## 1. Prepare the phone (same conditions for every run)
1. Charge to ~100 %, then UNPLUG (battery drain can only be measured on battery).
2. Battery saver OFF. Close all other apps (also TronX Voice). Airplane mode on if possible.
3. Let the phone cool ~10 min lying flat, out of sunlight. Note the room temperature.

## 2. Run (keep the app in the foreground; it keeps the screen awake)
1. Wait for "Models ready: 3_ft_int8".
2. Open "Section 9 benchmark (KV cache, repeated runs, thermal)".
3. Type the room temperature in "Ambient temperature". Leave the other defaults (threads 4, repeated runs 3, prompts 10, thermal 10 min).
4. Press "Device info".
5. Press "Run benchmark on SELECTED model (A-F)". About 10-15 min. Covers file size, loading memory, peak inference RAM, KV cache at 512 / 1,024 / 2,048 tokens, tokens/s, TTFT.
6. Let the phone cool ~5 min. Press "Thermal test on SELECTED model (G)". 10 min of sustained inference, one row every 30 s.
7. Send the results: "Share" or "Copy" at the top (preferred, the output is long), or screenshots.

## What each section means (assignment Section 9)
- DEVICE: model, SoC, CPU, RAM, Android version, runtime + version, power mode, ambient temperature, battery at start.
- A: model file size.
- B: memory before/after loading, loading peak, loading time, memory released after loading.
- C: baseline / peak / after memory during inference; increase caused by inference (VmRSS sampled every 15 ms).
- D: KV cache at 512 / 1,024 / 2,048 tokens with peak RAM, TTFT and decode tok/s (exact-length prompts, 32 output tokens).
- E: prefill and decode tokens/s, average and spread over 3 repeated runs after 2 warm-up runs.
- F: time to first token (model loading and tokenisation excluded; prompts pre-tokenised).
- G: battery temperature, thermal status/headroom, CPU MHz, tok/s every 30 s, performance change, battery used.
