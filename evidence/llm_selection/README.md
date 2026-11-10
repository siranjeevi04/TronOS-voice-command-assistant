# Evidence for LLM shortlist (captured 2026-11-10, live fetch via Firecrawl, full-page screenshots)

| File | Source URL | What it proves |
|---|---|---|
| 07_liquid_benchmark_chart_official.png | https://aypchzzf9pftwuto.public.blob.vercel-storage.com/lfm2_5_230m_benchmarks-EcJisF1zm7I7KiST75d7XsE2LwE0lD.png (chart embedded in Liquid blog) | Same-harness comparison of 6 small models: IFEval LFM2.5-350M 76.96 / Granite-4.0-H-350M 61.27 / Qwen3.5-0.8B 59.94 / Gemma 3 1B 63.49; BFCLv4 21.86 / 13.28 / 18.70 / 7.17 |
| 01_liquid_lfm2.5-230m_blog_benchmark_tables.png | https://www.liquid.ai/blog/lfm2-5-230m (published 2026-06-25, modified 2026-10-27) | Full tables incl. BFCLv3: LFM2.5-350M 44.11, Granite-4.0-H-350M 43.07, Granite-4.0-350M 39.58, Qwen3.5-0.8B 35.08, Gemma 3 1B 16.61. VENDOR-RUN (Liquid AI) |
| 02_llm-stats_bfcl-v4_leaderboard.png | https://llm-stats.com/benchmarks/bfcl-v4 (modified 2026-11-09) | Third-party aggregated BFCL-V4: <=2B models LFM3-1.7B 0.714, Qwen3.5-2B 0.436, K2 Horizon 0.9B 0.280, Qwen3.5-0.8B 0.253 |
| 03_ibm_granite-4.0-1b_card_nano_benchmarks.png | https://huggingface.co/ibm-granite/granite-4.0-1b | IBM's own numbers: BFCL v3 350M 39.32 / H-350M 43.32 / 1B 54.82 / H-1B 50.21; IFEval avg 55.4 / 61.63 / 77.38 / 78.53; Apache-2.0 |
| 04_qwen3.5-0.8b_card_benchmarks.png | https://huggingface.co/Qwen/Qwen3.5-0.8B | Qwen's own numbers: IFEval 52.1 (non-thinking), BFCL-V4 25.3, TAU2 11.6 (thinking); hybrid Gated DeltaNet arch |
| 05_lfm2.5-350m_card.png | https://huggingface.co/LiquidAI/LFM2.5-350M | 350M, 10 conv + 6 GQA layers, 32K ctx, GGUF / llama.cpp, tool calling + JSON |
| 06_liquid_lfm-license.png | https://www.liquid.ai/lfm-license | LFM Open License v1.0: commercial use only if annual revenue <= USD 10M |

Caveats: no public benchmark measures our exact task (single command -> intent+slot JSON). BFCL (function calling) and IFEval (format following) are proxies. Final proof = our own bake-off on the validation split.
