"""Estimate per-command LLM latency on Moto G34 5G (Snapdragon 695) from published inputs.

Every input below is a cited fact; every output is a DERIVED ESTIMATE, not a measurement.
Real numbers come from the on-device benchmark (assignment Section 9).

Method: scale a *measured* llama.cpp CPU result on Snapdragon 8 Gen 1 to Snapdragon 695.
  - decode (token generation) is memory-bandwidth bound -> scale by bandwidth ratio and bytes read per token
  - prefill (prompt processing) is compute bound        -> scale by parameter count and CPU compute ratio
"""

# ---- cited inputs -------------------------------------------------------------------------
SD695_BW = 2 * 16 / 8 * 2133e6 * 2 / 1e9   # Qualcomm brief: "2x16 LP4x @ 2133 MHz" (DDR -> x2) = 17.06 GB/s
SD8G1_BW = 51.2                            # AnandTech: 4x16-bit LPDDR5 @ 3200 MHz = 51.2 GB/s
# Measured, OnePlus 10 Pro (SD 8 Gen 1), llama.cpp CPU, Llama-3.2-1B Q4_0 (avisre/snapdragon-npu-llm BENCHMARKS.md)
REF_DECODE_TPS = 12.0
REF_PREFILL_TPS = 80.0
REF_PARAMS = 1.24e9          # Llama-3.2-1B parameter count
REF_FILE_GB = 0.77           # Llama-3.2-1B Q4_0 GGUF ~0.77 GB (approx.)
# CPU compute ratio SD8G1 / SD695: 1x X2 @3.0 + 3x A710 @2.5 (4 big cores) vs 2x A78 @2.2 (2 big cores).
CPU_RATIO = 2.0              # ASSUMPTION (conservative); not a measured figure

# Our task (measured on our own prompt format): compact system prompt + command ~ 90 tokens, JSON answer ~ 25 tokens
PROMPT_TOK, OUTPUT_TOK = 90, 25

# Official Q4_K_M GGUF file sizes (HF API, 2026-11-10) and parameter counts (model cards)
MODELS = {
    "LFM3-1.7B (Q4_K_M)":          dict(file_gb=1.100, params=1.73e9),
    "Granite-4.0-1B (Q4_K_M)":     dict(file_gb=1.024, params=1.60e9),
    "LFM2.5-350M (Q4_K_M)":        dict(file_gb=0.229, params=0.35e9),
    "Granite-4.0-H-350M (Q4_K_M)": dict(file_gb=0.223, params=0.34e9),
}

# ---- derived --------------------------------------------------------------------------------
bw_ratio = SD8G1_BW / SD695_BW
ref_ceiling = SD8G1_BW / REF_FILE_GB
efficiency = REF_DECODE_TPS / ref_ceiling   # fraction of theoretical bandwidth llama.cpp actually achieved

print(f"Snapdragon 695 bandwidth  : {SD695_BW:.2f} GB/s   (8 Gen 1: {SD8G1_BW} GB/s, {bw_ratio:.1f}x more)")
print(f"Reference achieved {REF_DECODE_TPS} tok/s vs theoretical {ref_ceiling:.0f} tok/s -> efficiency {efficiency:.0%}\n")
print(f"{'model':30s} {'ceiling':>8s} {'decode':>8s} {'prefill':>8s} {'TTFT':>7s} {'decode25':>9s} {'total':>7s}")
for name, m in MODELS.items():
    ceiling = SD695_BW / m["file_gb"]                       # absolute max tok/s if 100% bandwidth used
    decode = ceiling * efficiency                           # same efficiency as measured reference
    prefill = REF_PREFILL_TPS * (REF_PARAMS / m["params"]) / CPU_RATIO
    ttft = PROMPT_TOK / prefill
    gen = OUTPUT_TOK / decode
    print(f"{name:30s} {ceiling:7.1f}  {decode:7.1f}  {prefill:7.0f}  {ttft:5.2f}s  {gen:7.2f}s  {ttft+gen:5.1f}s")
