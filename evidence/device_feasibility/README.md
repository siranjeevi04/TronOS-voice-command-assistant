# Why 1B-class models (even INT4) are not efficient on Moto G34 5G

## Cited facts
| Fact | Value | Source / file |
|---|---|---|
| Moto G34 5G SoC | Snapdragon 695: 2x Cortex-A78 2.2 GHz + 6x Cortex-A55, Adreno 619, 4 or 8 GB LPDDR4X | LineageOS wiki (fogos), FoneArena |
| SD695 memory | "2x16 LP4x @ 2133 MHz" = **17.06 GB/s** | Qualcomm product brief -> `01_qualcomm_snapdragon-695_product_brief.pdf` |
| SD 8 Gen 1 memory | 4x16-bit LPDDR5 @ 3200 MHz = **51.2 GB/s** (3.0x SD695) | AnandTech launch article |
| MEASURED: Llama-3.2-1B Q4_0, llama.cpp CPU, SD 8 Gen 1 (OnePlus 10 Pro) | **12.0 tok/s decode, ~80 tok/s prefill, ~600 ms TTFT** | avisre/snapdragon-npu-llm BENCHMARKS.md -> `02_measured_llama-3.2-1b_q4_0_snapdragon-8gen1.png` |
| Official Q4_K_M sizes | LFM3-1.7B 1100 MB, Granite-4.0-1B 1024 MB, LFM2.5-350M 229 MB, Granite-4.0-H-350M 223 MB | HF API file listing |

## Key physics
LLM decoding reads every weight once per generated token, so tok/s <= memory bandwidth / model bytes.
INT4 shrinks bytes but the G34 has only 17 GB/s.
- Even at an impossible 100% bandwidth, a 1B Q4 model is capped at 15.5-16.7 tok/s on the G34.
- llama.cpp actually reached 18% of the theoretical bandwidth limit in the measured 8 Gen 1 run.

## Derived estimates (`estimate_moto_g34.py`, output in `estimate_output.txt`)
One command = ~90 prompt tokens + ~25 JSON output tokens.

| Model (Q4_K_M) | Decode tok/s | Prompt time (TTFT) | Time to write JSON | Total per command |
|---|---|---|---|---|
| LFM3-1.7B | 2.8 | 3.1 s | 8.9 s | **~12 s** |
| Granite-4.0-1B | 3.0 | 2.9 s | 8.3 s | **~11 s** |
| LFM2.5-350M | 13.4 | 0.6 s | 1.9 s | **~2.5 s** |
| Granite-4.0-H-350M | 13.8 | 0.6 s | 1.8 s | **~2.4 s** |

## Caveats (stated honestly)
- These are ESTIMATES scaled from one measured device; not measured on a G34.
- CPU compute ratio 2.0 (8 Gen 1 vs 695) is an assumption.
- Hybrid architectures (LFM conv, Granite Mamba2) may run somewhat faster than Llama-style scaling predicts.
- RAM: on the 4 GB variant, a ~1.1 GB model + KV cache + STT competes with Android itself; must be checked on device (MemAvailable in /proc/meminfo).
- Final proof = Section 9 on-device measurement.
