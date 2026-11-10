# TronOS Voice Command Parser: On-Device LLM Technical Report

**Model:** IBM Granite-4.0-H-350M + LoRA (attention + MLP), merged, ONNX INT8
**Speech-to-text:** Moonshine v2 streaming small (ONNX q8), used as-is
**Target device:** Motorola Moto G34 5G (Snapdragon 695, 8 GB RAM), ONNX Runtime Android 1.31.1
**Date:** 10 November 2026

---

## 1. Executive summary

TronOS needs a small model that runs fully on the phone and turns a spoken command into JSON that Android services can execute. We built the full pipeline: microphone, voice activity detection, Moonshine speech-to-text, then a fine-tuned Granite-4.0-H-350M that outputs the JSON. It runs on a mid-range Moto G34 5G with no network permission.

Main results (held-out test set, 613 commands, never used for training or model selection; precision stated per column):

| | Base FP32 | Fine-tuned FP32 (LoRA D, merged) | Fine-tuned INT8 (deployed) |
|---|---:|---:|---:|
| Exact JSON match | 32.5% | **89.1%** | 86.3% |
| Intent accuracy | 74.2% | 94.9% | 94.3% |
| Field accuracy | 59.3% | 88.8% | 86.7% |
| JSON validity | 98.9% | 100% | 100% |
| Unsupported-request recall | 33.0% | 99.1% | 99.1% |

FP32 = full precision (base: PyTorch; fine-tuned: ONNX Runtime, identical to PyTorch). INT8 = ONNX Runtime CPU (Section 8).

- LoRA fine-tuning raised exact JSON match from 32.5% to 89.1% (+56.6 points) by training only 1.25% of the parameters (4.26 M).
- The adapter was merged into the base weights and quantized. INT8 keeps 94% of outputs byte-identical to full precision and loses 2.8 points of exact match. The file shrinks from 1.3 GB (FP32) to 404 MB.
- On the Moto G34 the INT8 model loads in 4.1 s and uses about 0.9 GB of process RAM. Time to first token is 5.9 s and decode speed is 9.8 tokens/s. A 10-minute sustained test caused mild throttling (−8% decode speed).
- **Recommended configuration for TronOS: fine-tuned INT8** (Section 17). INT4 is 37% smaller but loses 11 points and starts acting on out-of-scope requests.
- Main limitations (Section 16): phone INT8 output differs from Mac INT8 for the same file, the long system prompt makes time to first token slow, and some phrasing patterns are still unlearned.

---

## 2. Use case and requirements

The user speaks a command. The phone must return one JSON object with an `intent` and only the fields that were actually said:

| Intent | Fields |
|---|---|
| `set_alarm` | `time` (HH:MM, 24 h), `date`, `label`, `in` (relative, HH:MM from now) |
| `create_reminder` | `task`, `time`, `date`, `in` |
| `open_app` | `app_name` |
| `make_call` | `contact` |
| `send_message` | `contact`, `message`, `app` |
| `change_setting` | `setting`, `action` (on/off/set/increase/decrease), `value` |
| `unsupported` | none |

Examples:
```
"Wake me up at 6:30 tomorrow"          -> {"intent": "set_alarm", "time": "06:30", "date": "tomorrow"}
"Text Mom that I'm on my way"          -> {"intent": "send_message", "contact": "Mom", "message": "I'm on my way"}
"Send 2000 rupees to my mom's account" -> {"intent": "unsupported"}
```

Requirements we designed for:
1. **Valid, exact JSON.** Android code consumes the output directly, so the headline metric is exact match: correct intent, every field correct, no extra fields.
2. **Never invent fields.** A guessed time or contact creates a wrong alarm or calls the wrong person.
3. **Refuse what is not supported** (payments, bookings, questions). Acting on an unsupported request is worse than doing nothing.
4. **Run fully on device** on a mid-range phone: limited RAM, memory bandwidth (17 GB/s on the Snapdragon 695), CPU and battery.

Project decisions beyond the problem statement:
- **Language: English only**, matching the English-only Moonshine speech model.
- **Time without am/pm is kept literal** ("alarm for 7" gives `07:00`); Android can confirm.
- **Multi-action commands: the first action decides.** If the first action is unsupported, the whole command is unsupported, so the assistant never half-executes something like a payment.
- **Added an optional `in` field** for relative times ("remind me in 20 minutes" gives `"in": "00:20"`). Without it the base model invented clock times.

---

## 3. Base model selection

**Selection rules:**
1. At most about 400 M parameters. A 1 B-class model was estimated at 11–12 s per command on the Moto G34, because decode speed is limited by memory bandwidth (17 GB/s).
2. Strongest published instruction-following (IFEval) and function-calling (BFCL) scores.
3. A licence that allows commercial use.
4. Mature tooling for export to a mobile runtime.

We surveyed every published model at or below 400 M (evidence screenshots are in `evidence/llm_selection/`):

| Model | Params | IFEval | BFCL v3 | Licence | Verdict |
|---|---:|---:|---:|---|---|
| LiquidAI LFM2.5-350M | 350 M | 76.96 | 44.11 | LFM Open License (commercial only if revenue ≤ $10 M) | Finalist |
| **IBM Granite-4.0-H-350M** | 340 M | 61.3 | 43.1 | **Apache-2.0** | **Finalist → selected** |
| LFM2.5-230M | 230 M | 71.71 | 43.26 | LFM Open License | Dropped (same family, weaker) |
| Granite-4.0-350M (dense) | 350 M | 53.48 | 39.58 | Apache-2.0 | Reserve |
| Gemma-3-270M-it | 270 M | 51.2 | n/a | Gemma terms, gated | Dropped |
| SmolLM2-360M-Instruct | 360 M | 41.0 | n/a | Apache-2.0 | Dropped |
| MobileLLM-R1-360M | 360 M | n/a | n/a | Non-commercial | Dropped |
| Qwen3.5-0.8B | 800 M | 59.94 | 35.08 | Apache-2.0 | Too large, lower BFCL |

**Bake-off of the two finalists:** same system prompt, zero-shot, 395 validation commands.

| | Granite-4.0-H-350M | LFM2.5-350M |
|---|---:|---:|
| Exact JSON | **41.8%** | 29.6% |
| Intent | **82.3%** | 78.7% |
| Fields | **74.4%** | 49.7% |
| Valid JSON | **98.5%** | 96.5% |
| Target loss | **0.167** | 0.326 |

**Granite-4.0-H-350M won on every metric** and has the permissive Apache-2.0 licence, so it was selected.

Granite-4.0-H-350M architecture (verified in `results/model_inspection/`):

| Property | Value |
|---|---|
| Parameters | 340,332,224 |
| Layers | 32 hybrid: **28 Mamba2** (state-space) + **4 attention** (layers 10, 13, 17, 27) |
| Hidden size / heads / KV heads | 768 / 12 / 4 (grouped-query attention) |
| MLP in every layer | `shared_mlp.input_linear` 768→4096 (fused gate + up), `output_linear` 2048→768, SwiGLU |
| Vocabulary | 100,352 (tied embeddings) |
| Why it suits the phone | Only 4 layers keep a KV cache that grows with context; the 28 Mamba layers keep a fixed-size state |

---

## 4. Dataset preparation and statistics

### 4.1 Source and generation
The dataset is a combination: **96.6% synthetic, template-generated, plus 120 hand-written commands** (test set only). There is no public dataset with this exact schema.

The generator `scripts/generate_dataset_v12.py` (seed 13) was designed from a failure analysis of the base model (`docs/granite_failure_analysis.md`). Each base-model failure type became a data rule:

| Base-model failure | Data rule |
|---|---|
| Invents optional fields (adds a time, date or message) | "Say only what was said" minimal pairs; ≥35% of samples have no optional field |
| Confuses look-alike intents (cancel alarm vs set alarm, play music vs open app) | Hard-negative pairs at every intent boundary |
| Drops stated fields | Full-slot examples, varied slot positions |
| 12 h → 24 h time errors | Time-expression coverage table (half past, quarter to, noon, 7.30, 7:30 pm …) |
| Keys outside the schema, invented intents | Closed key set; distractor words ignored |
| Copies values from the prompt's examples | Value pools decorrelated from the 7 prompt examples |
| Unsupported requests answered with an action | 75 unsupported templates across 15+ categories (payments, bookings, questions, cancellations, multi-step) |

**Speech-to-text realism:** 70% of commands mimic Moonshine's output style (digits like "6:30 pm", casing, punctuation). 30% are lowercase, unpunctuated, or have spelled-out numbers.

### 4.2 Cleaning, normalisation, duplicates
- **Canonical values:** a dictionary for apps and settings ("file manager" → `Files`, "Wi-Fi" → `wifi`). Task and message text is kept verbatim. Times are always HH:MM in 24 h.
- **Duplicates removed:** 818 exact, 314 after normalisation (case, punctuation, spaces), and 6 near-duplicates between evaluation splits. **0 duplicates remain.**

### 4.3 Splits and leakage prevention
- **Split by template, not by sentence:** a template appears in only one split, so test sentences are new wordings, not paraphrases of training rows. The split is stratified by intent and field set.
- **Test-only value pools:** 129 test commands use names, apps and times never seen in training.
- **Hand-written commands:** 120 written by hand, test only.
- **Checks** (`data/v1.2/manifest.json`): template overlap train/val = 0 and train/test = 0; text overlap = 0.
- **Use of each split:** model and hyper-parameter choices were made on validation only. The test set was used only for the final numbers.

### 4.4 Statistics

| Item | Value |
|---|---|
| Total samples | 3,477 |
| Training | 2,510 |
| Validation | 354 |
| Test | 613 (493 template + 120 hand-written) |
| Intents | 7 |
| Language | English |
| Synthetic share | 96.6% |
| Duplicate samples remaining | 0 |
| Templates | 260 (train 162 / val 43 / test 55) |

| Intent | Train | Val | Test | Total |
|---|---:|---:|---:|---:|
| set_alarm | 360 | 60 | 96 | 516 |
| create_reminder | 360 | 59 | 94 | 513 |
| open_app | 359 | 36 | 76 | 471 |
| make_call | 359 | 36 | 74 | 469 |
| send_message | 359 | 48 | 72 | 479 |
| change_setting | 355 | 57 | 95 | 507 |
| unsupported | 358 | 58 | 106 | 522 |

### 4.5 Representative examples (from the test set)
```
"Get me up at 10:30 p.m.."                          -> {"intent": "set_alarm", "time": "22:30"}
"don't let me forget my umbrella"                   -> {"intent": "create_reminder", "task": "my umbrella"}
"go to settings"                                    -> {"intent": "open_app", "app_name": "Settings"}
"call 100"                                          -> {"intent": "make_call", "contact": "100"}
"Send Amma a message via Telegram: dinner is ready." -> {"intent": "send_message", "contact": "Amma", "message": "dinner is ready", "app": "Telegram"}
"Turn off Wi-Fi and open WhatsApp."                 -> {"intent": "change_setting", "setting": "wifi", "action": "off"}   (first action decides)
"send money to priya and text her"                  -> {"intent": "unsupported"}                                         (first action unsupported)
"Show missed calls from Amma."                      -> {"intent": "unsupported"}
```

### 4.6 Training format
Each sample is a chat: the **system prompt** (schema + 7 examples, 490 tokens), the **user command**, and the **assistant JSON**. The loss is computed only on the assistant JSON tokens, with the prompt masked. The same prompt is used in training, evaluation and on the phone.

### 4.7 Why the dataset is suitable
- Every rule targets a measured failure of the base model.
- Every intent and field combination is covered with balanced classes.
- The input style matches what the speech model actually produces.
- The test set has unseen templates, unseen values and real hand-written commands, so the score measures generalisation, not memorisation.

### 4.8 Dataset limitations
- Mostly synthetic: real speech has more disfluency, code-mixing and STT errors than templates produce.
- English only. Indian-English names are covered in the value pools, but no Hinglish.
- A few template artefacts exist (for example "remind me for me").
- Some gold labels are debatable: "pay the credit card bill on the 5th" is labelled as one task, not task + date.

---

## 5. Fine-tuning methodology

| Setting | Value | Why |
|---|---|---|
| Method | LoRA (PEFT 0.21), base frozen | Required; small adapter, fast, no catastrophic forgetting |
| Rank r / alpha | 16 / 32 (scale α/r = 2) | Common setting for small models; enough capacity (4.26 M params for D) with a 17 MB adapter |
| Dropout | 0.05 | Light regularisation for a 2.5 k-sample dataset |
| Learning rate | 2e-4, cosine, 5% warm-up | Standard LoRA learning rate; validation loss fell smoothly without spikes |
| Epochs | 2 (314 steps) | Validation loss for D: 0.326 → 0.031 (epoch 1) → 0.0215 (epoch 2); the gain was flattening |
| Effective batch | 16 (micro-batch 1 × gradient accumulation 16) | T4 16 GB memory: Mamba2 runs a PyTorch fallback with large activations |
| Max length | 640 tokens | Longest sample is 553 tokens (system prompt 490) |
| Precision | FP32 | T4 has no BF16; avoids FP16 overflow in the Mamba fallback |
| Gradient clipping | max norm 1.0 | Stability |
| Loss | Cross-entropy on JSON tokens only | Teaches the output, not the prompt |
| Seed | 42 | Reproducibility |
| Hardware | Kaggle 2× NVIDIA T4 (one run per GPU) | Mac MPS was ~6–9 h per run |
| Time per experiment | A 51 min, B 52 min, C 61 min, D 61 min | |

**Script:** `scripts/train_lora.py`. The configuration is saved in each `runs/granite/<exp>/train_log.json`.

**Learning curves** (`results/plots/lora_loss_curves.png`):
- All four runs converge with no divergence.
- D's training loss ends at 0.0003 and its validation loss at 0.0215.
- Validation loss keeps decreasing through epoch 2, so the low training loss is fitting, not harmful overfitting.
- Test exact match (89.1%) is close to validation exact match (91.2%), which confirms the model generalises.

![LoRA learning curves](../results/plots/lora_loss_curves.png)

---

## 6. LoRA layer selection

Four experiments on the same data, hyper-parameters and seed. Only the target modules change:

| Exp | Target modules | Layers touched | Trainable params | % of base | Adapter size |
|---|---|---|---:|---:|---:|
| A | attention `q_proj`, `v_proj` | 4 | 163,840 | 0.05% | 0.6 MB |
| B | attention `q,k,v,o_proj` | 4 | 327,680 | 0.10% | 1.3 MB |
| C | MLP `input_linear`, `output_linear` | 32 | 3,932,160 | 1.16% | 15.0 MB |
| **D** | **attention q,k,v,o + MLP** | **32** | **4,259,840** | **1.25%** | **16.3 MB** |

All rows FP32, PyTorch, unmerged adapter (identical to merged ONNX FP32).

| Exp | Val exact | Val loss | **Test exact** | Test intent | Test fields | Valid JSON | Unsupported recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base (FP32) | 30.2% | 0.326 | 32.5% | 74.2% | 59.3% | 98.9% | 33.0% |
| A | 81.6% | 0.0446 | 77.5% | 89.6% | 81.1% | 100% | 81.1% |
| B | 85.0% | 0.0429 | 81.4% | 91.0% | 83.4% | 100% | 89.6% |
| C | 91.2% | 0.0229 | 85.5% | 93.6% | 88.7% | 100% | 91.5% |
| **D** | **91.2%** | **0.0215** | **89.1%** | **94.9%** | **88.8%** | **100%** | **99.1%** |

![Quality by experiment](../results/plots/lora_quality_test.png)

**Why D was selected:**
- It was chosen on the **validation** set. It tied C on exact match (91.2%) and was better on validation loss, intent accuracy and field accuracy.
- The test set confirmed the choice: D is the best on every metric, and unsupported recall rose from 91.5% (C) to 99.1%.

**What the experiments show about this architecture:**
- **Attention-only adapters (A, B) are limited** because Granite-H has only 4 attention layers out of 32. Even so, they lift exact match from 32.5% to 77–81%. Attention controls which input words are copied into the output fields (names, message text).
- **MLP adapters (C) touch all 32 layers** and give the largest single gain: +8 points over B. The MLP is where each token's features are transformed into the output format and intent decision.
- **Combining both (D)** adds the attention copy behaviour on top of the MLP capacity. Gains: +3.6 points exact match and +7.6 points unsupported recall over C, for only 8% more parameters.
- **The Mamba2 mixer** (`in_proj`, `out_proj`, convolution, state-space scan) was left frozen in all experiments, keeping its pre-trained long-range behaviour. Targeting it is listed as future work.

---

## 7. LoRA execution explanation

### 7.1 One adapted linear layer
A frozen weight matrix `W` (d_out × d_in) gets two small trainable matrices: `A` (r × d_in) and `B` (d_out × r), with r = 16. For an input vector `x`:

```
h = W·x  +  (α/r) · B·(A·x)          α/r = 32/16 = 2
```

- `A·x` compresses the input to 16 numbers, and `B` expands them back to d_out.
- `B` starts at zero, so the model starts exactly as the base model and learns only a low-rank correction.
- During training, gradients flow only into A and B. W gets no gradient and no optimizer state, which is why training needs little memory.

Example: `q_proj` (768 → 768). A full update would be 589,824 values. LoRA uses 16 × (768 + 768) = 24,576 values, 4% of that.

### 7.2 Where the adapter runs inside one Granite-H layer (experiment D)
Each of the 32 layers is: `x → RMSNorm → mixer → +residual → RMSNorm → MLP → +residual`.

```
                     ┌──────────── layers 10, 13, 17, 27 (attention) ────────────┐
x ─► RMSNorm ─► mixer│ q = (Wq + 2·Bq·Aq)·x   k = (Wk + 2·Bk·Ak)·x                   │─► + ─► RMSNorm ─► MLP ─► + ─► next layer
                     │ v = (Wv + 2·Bv·Av)·x   softmax(qkᵀ/√d)·v → (Wo + 2·Bo·Ao)     │
                     └──────────── other 28 layers: Mamba2 mixer, FROZEN ────────────┘

MLP (all 32 layers):   [gate | up] = (W_in + 2·B_in·A_in)·x          768 → 4096, split 2048 + 2048
                       y = (W_out + 2·B_out·A_out)·( SiLU(gate) ⊙ up ) 2048 → 768
```

**Per token, LoRA D changes:**
- In all 32 layers: the MLP's input projection (to the gate and up halves) and its output projection.
- In the 4 attention layers: the query, key and value projections, which change which earlier tokens the model attends to (for example, copying "Mom" into `contact`), and the output projection.
- The Mamba2 token mixing in 28 layers is unchanged.

### 7.3 Inference: merge, then deploy
For deployment the adapter is **merged** into the base weights once (`scripts/export_onnx.py merge`):

```
W' = W + (α/r) · B·A      for every adapted matrix
```

The merged model has **exactly the same shapes and parameter count as the base** (340 M). Consequences:
- **No extra latency or memory at inference.** The phone runs one plain matrix multiply per layer, not W·x plus B·A·x.
- **No PEFT or LoRA code on the phone.** Any runtime that runs the base model runs the fine-tuned one.
- **Quantization applies to the merged weights.** INT8 and INT4 must keep the small LoRA delta (Section 8.3 shows INT4 partly loses it).

The unmerged adapter (16.3 MB) is still useful: several task adapters could be swapped at run time on one base model. That is listed as future work.

---

## 8. Quantization methodology

### 8.1 Conversion pipeline
```
LoRA D adapter ─merge─► HF FP32 checkpoint ─► ONNX FP32 ─┬─► FP16  (weights cast, FP16 graph)
                                                          ├─► INT8  (dynamic, per-channel)
                                                          └─► INT4  (block-wise weight-only)
```

- **ONNX export:** our merged weights are written into the public `onnx-community/granite-4.0-h-350m-ONNX` graph structure (Mamba2 loop as an If-subgraph, GroupQueryAttention). The ONNX graph includes the KV cache and Mamba state as inputs and outputs.
- **Verified:** ONNX FP32 gives **identical test metrics** to PyTorch (89.1% exact, target loss 0.0622).
- **FP16:** the standard float16 converter broke the Mamba subgraphs, so we wrote our weights into the reference FP16 graph instead (0 difference against the reference).
- **INT8:** `onnxruntime.quantization.quantize_dynamic`
  - Weights are INT8, symmetric, **per output channel**, applied to all MatMuls including inside the Mamba subgraph (`EnableSubgraph`).
  - Activations are quantized **dynamically at run time** (scale from each tensor's range).
  - The embedding table is stored as INT8 with a per-row scale; the output head (`lm_head`) was untied from it and quantized separately.
- **INT4:** `MatMulNBitsQuantizer`
  - Weight-only, blocks of 32 values with one FP scale per block, symmetric.
  - The embedding uses `GatherBlockQuantized`.
  - `accuracy_level=4`, so compute uses INT8 activations.
- **Cleanup:** unused FP32 initialisers were stripped. Before that, INT8 was 868 MB and INT4 477 MB.

**Script:** `scripts/export_onnx.py` (subcommands `merge`, `onnx`, `onnx16`, `verify`, `quant`). Runtime: ONNX Runtime 1.31.1.

### 8.2 Sizes

| Variant | Size (MiB) | vs FP32 |
|---|---:|---:|
| FP32 | 1,301 | 1.00× |
| FP16 | 651.5 | 0.50× |
| **INT8** | **404** | **0.31×** |
| INT4 | 252.7 | 0.19× |
| LoRA adapter alone (FP32) | 16.3 | n/a |

### 8.3 Quality impact (held-out test, 613 commands, same ONNX Runtime on the Mac)

| Variant | Exact | Intent | Fields | Valid JSON | Unsupported recall | Target loss |
|---|---:|---:|---:|---:|---:|---:|
| FP32 | 89.1% | 94.9% | 88.8% | 100% | 99.1% | 0.062 |
| FP16 | ⟨pending⟩ | | | | | |
| INT8 | 86.3% | 94.3% | 86.7% | 100% | 99.1% | 0.064 |
| INT4 | 77.8% | 89.9% | 82.4% | 99.8% | 78.3% | 0.082 |

Error breakdown against FP32 (`results/quant/quant_error_breakdown.json`):

| | INT8 | INT4 |
|---|---:|---:|
| Output byte-identical to FP32 | 574/613 (94%) | 501/613 (82%) |
| Correct in FP32 → wrong | 18 | 72 |
| Wrong in FP32 → correct | 1 | 3 |
| Main new error | dropped optional field (7), small value changes | **unsupported request → real action (22)**; time value changes (12, e.g. 21:00 → 09:00) |

![Quantization quality](../results/plots/quant_quality.png)

**Interpretation:**
- 8-bit per-channel weights keep the small LoRA correction almost intact.
- 4-bit blocks (16 levels per 32 weights) round part of it away, so behaviour drifts back toward the base model. The clearest sign: the base model's habit of acting on unsupported requests returns.

---

## 9. Model quality comparison

Same prompt, same greedy decoding (max 64 new tokens), same scoring code (`scripts/common.py`) for every variant.

### 9.1 Before and after fine-tuning and quantization (test set, 613)

| Metric | Base FP32 | Base FP16 | Fine-tuned FP32 | Fine-tuned FP16 | Fine-tuned INT8 | Fine-tuned INT4 |
|---|---:|---:|---:|---:|---:|---:|
| Intent accuracy | 74.2% | ⟨pending⟩ | 94.9% | ⟨pending⟩ | 94.3% | 89.9% |
| Field accuracy | 59.3% | ⟨pending⟩ | 88.8% | ⟨pending⟩ | 86.7% | 82.4% |
| Exact JSON match | 32.5% | ⟨pending⟩ | 89.1% | ⟨pending⟩ | 86.3% | 77.8% |
| JSON validity | 98.9% | ⟨pending⟩ | 100% | ⟨pending⟩ | 100% | 99.8% |
| Unsupported recall | 33.0% | ⟨pending⟩ | 99.1% | ⟨pending⟩ | 99.1% | 78.3% |

On the 120 hand-written test commands, exact match went from **44.2% (base FP32) to 92.5% (fine-tuned FP32)**. On the template commands, it went from 29.6% to 88.2%.

### 9.2 What changed (real test outputs)

| Command | Base FP32 | Fine-tuned FP32 (D) |
|---|---|---|
| enable airplane mode | `{"intent": "set_alarm", "time": "08:00", "date": "today", "in": "relative time"}` | `{"intent": "change_setting", "setting": "airplane_mode", "action": "on"}` |
| send 2000 to my mom's account | `{"intent": "send_message", "contact": "Mom", "message": "2000"}` | `{"intent": "unsupported"}` |
| what's 25 times 4 | `{"intent": "multiply", "number1": 25, "number2": 4}` | `{"intent": "unsupported"}` |
| set an alarm for 11:59 pm | `{"intent": "set_alarm", "time": "11:59", "date": "tomorrow", "in": "in 59 minutes"}` | `{"intent": "set_alarm", "time": "23:59"}` |
| remind me to switch off the heater | `{..., "task": "switch off the heater", "time": "17:00"}` (invented time) | `{"intent": "create_reminder", "task": "switch off the heater"}` |
| launch the camera i want to take a picture | `{"intent": "create_reminder", "task": "take a picture", ...}` | `{"intent": "open_app", "app_name": "Camera"}` |

**Pattern:** the base model knows the format but invents fields, invents intents, converts times wrongly and acts on unsafe requests. Fine-tuning fixed these: 350 test commands went from wrong to right, and 3 went from right to wrong.

### 9.3 Hard prompts (new phrasing patterns, outside the dataset)
On 10 new, harder patterns written after training:
- Base FP32: 3/10
- Fine-tuned INT8 (Mac): 5/10
- Fine-tuned FP32: 7/10

The remaining failures are listed in Section 16.

---

## 10. Android deployment architecture

Two Android apps were built: Kotlin, ONNX Runtime Android 1.31.1, arm64-v8a, models bundled in the APK.

**TronX Voice** (`android_voice/`): the product pipeline.
```
 Mic (AudioRecord 16 kHz mono)
   │  512-sample windows (32 ms)
   ▼
 Silero VAD v5 (ONNX, 2 MB) ── speech prob ≥ 0.5 → recording; 2.0 s silence → end of command
   │  utterance audio (8 s no-speech timeout, 15 s max)
   ▼
 Moonshine v2 streaming small (ONNX q8, encoder + merged decoder, greedy, KV cache)
   │  transcript  ── no letters? → discard (no LLM call)
   ▼
 Granite byte-level BPE tokenizer (pure Kotlin, verified identical to HF on 3,484 texts)
   │  ids = cached system-prompt prefix (493) + command + chat suffix (5)
   ▼
 Granite-4.0-H-350M + LoRA D, merged, ONNX INT8 (ONNX Runtime CPU EP, 4 threads, graph opt ALL)
   │  greedy decode, max 64 tokens, stop at <|end_of_text|>; KV cache (4 layers) + Mamba state (28 layers) as I/O tensors
   ▼
 JSON parse → intent + fields   ── (TronOS dispatch to AlarmClock / SmsManager / Settings intents)
```

- **Inputs:** speak (tap to talk), type, or pick one of 10 preset hard prompts.
- **Shown with every reply:** the raw JSON, valid-JSON flag, RAM (now and peak), speech-to-JSON time, STT time, LLM time, time to first token, decode tokens/s, tokens in and out.

**TronX Bench (INT8)** (`android/`): runs the assignment Section 9 measurements (A–G) and shows every number on screen; results leave the phone only through Share or Copy.

**Deployment details:**
- **Packaging:** models are stored uncompressed in the APK assets and copied once to app-private storage on first launch (ONNX Runtime needs a file path for external weights). Uninstalling the app removes them.
- **Privacy:** no network permission. The ONNX Runtime library's `INTERNET`, `ACCESS_NETWORK_STATE` and telemetry initializer were removed from the merged manifest. The only permission is `RECORD_AUDIO`, in the voice app.
- **Identical input:** the on-device tokenizer runs a start-up self-check against Mac token ids (20/20 identical). Prompt token counts matched the Mac on every phone command.
- **Memory budget (voice app):** LLM about 0.9 GB process peak, STT q8 158 MB on disk, VAD 2 MB. The estimated total is about 1.1–1.2 GB on an 8 GB phone.

---

## 11. Model file size and memory measurements

**Methodology:**
- Process resident memory (VmRSS from `/proc/self/status`) sampled every 15 ms.
- ONNX Runtime CPU EP with 4 threads.
- Phone: Moto G34 5G, Android 15, battery saver off, on battery, ambient 32 °C.
- The Mac numbers use the same app logic in Python (`scripts/bench_mac.py`) on an Apple M5 (macOS 26.6).

### 11.1 Device
| | |
|---|---|
| Phone | Motorola moto g34 5G, Qualcomm SM6375 (Snapdragon 695 5G) |
| CPU | 2× Cortex-A78 2.2 GHz + 6× Cortex-A55 1.8 GHz |
| GPU / NPU | Adreno 619 / not used (CPU execution provider) |
| RAM | 7,492 MB total, 2,038 MB available at start |
| Android | 15 (API 35) |
| Runtime | ONNX Runtime Android 1.31.1, CPU EP, 4 intra-op threads, graph optimisation ALL |
| Power mode / ambient / battery | Normal (saver off) / 32 °C / 66%, 34.8 °C |

### 11.2 Model files

| File | Size |
|---|---:|
| Base FP16 | 651.5 MiB |
| Fine-tuned adapter (LoRA D, FP32 safetensors) | 16.3 MiB |
| Fine-tuned merged FP16 | 651.5 MiB |
| Fine-tuned INT8 | 404.4 MiB (423,992,240 bytes) |
| Fine-tuned INT4 | 252.7 MiB |

### 11.3 Loading memory and peak inference RAM: fine-tuned INT8 on the phone

| Measurement | Value |
|---|---:|
| Memory before loading | 200.2 MB |
| Memory after loading | 527.3 MB (+327.1 MB) |
| Loading peak | 663.0 MB (+462.9 MB) |
| Loading time | 4.09 s |
| Released after loading | 138.3 MB |
| Baseline (model loaded, idle, after warm-up) | 825.9 MB |
| Peak during inference (30 runs) | 895.9 MB |
| After inference | 875.5 MB |
| Increase attributable to inference | 70.0 MB |

Baseline rises from 527 MB to 826 MB after the first generations, mainly because ONNX Runtime allocates its working buffers (memory arena) on first use and keeps them for later calls.

### 11.4 Other variants (Mac, same method)
⟨pending: Mac bench table for Base FP16 / FT FP16 / FT INT8 / FT INT4⟩

---

## 12. KV cache measurement

**Method:**
- The prompt is filled to exactly 512, 1,024 or 2,048 tokens, then exactly 32 tokens are decoded with end-of-sequence ignored. 2 runs each, averaged.
- KV size is computed from the actual cache tensors after generation.
- KV cache data type: FP32 in all variants, also for FP16 (state tensors). KV cache quantization: no.

**Granite-H specifics:**
- Only the 4 attention layers keep a KV cache: 4 layers × (K + V) × 4 KV heads × 64 dims × 4 bytes = **8 KB per token**.
- The 28 Mamba2 layers keep a **fixed 21.8 MB state** regardless of context length.

**Fine-tuned INT8 on the phone:**

| Context (prompt + output) | KV cache | Mamba state | Peak process RAM | Time to first token | Decode tok/s |
|---|---:|---:|---:|---:|---:|
| 512 + 32 | 4.24 MB | 21.77 MB | 891 MB | 6.26 s | 8.64 |
| 1,024 + 32 | 8.24 MB | 21.77 MB | 1,084 MB | 11.67 s | 9.55 |
| 2,048 + 32 | 16.24 MB | 21.77 MB | 1,404 MB | 23.32 s | 9.84 |

![KV cache vs context](../results/plots/phone_kv_cache.png)

**Observations:**
- The KV cache grows only 8 KB per token, 1/8 of what a pure transformer with the same attention width would need (32 attention layers = 64 KB/token).
- Decode speed does not drop with context length, because the Mamba layers do constant work per token.
- What grows is prefill time and temporary activation memory during prefill: peak RAM rises by about 0.5 GB from 512 to 2,048 tokens. That comes from the prefill activation buffers, not the KV cache.

⟨pending: KV table for other variants (Mac)⟩

---

## 13. TTFT and tokens/sec measurements

**Method:**
- 10 real command prompts (501–508 tokens, including the 490-token system prompt) × 3 repeated runs.
- 2 un-timed warm-up runs first.
- Greedy decoding, max 48 new tokens, stop at end of sequence.
- **Time to first token** is measured from the first `session.run` call to the first token id. Model loading is excluded. Tokenization is excluded because prompts are pre-tokenized.

**Fine-tuned INT8 on the phone (n = 30):**

| Metric | Average | Std | Min | Max |
|---|---:|---:|---:|---:|
| Time to first token | 5,921 ms | 300 | 5,481 | 7,240 |
| Prompt processing (prefill) | 85.4 tok/s | 3.9 | 69.5 | 92.3 |
| Decode | 9.76 tok/s | 1.15 | 7.30 | 11.30 |
| Generated tokens per answer | 14.5 | 6.3 | 6 | 22 |

A typical command (about 505 prompt tokens and 15 output tokens) takes about 7.5 s end to end on the phone. Most of that is prefill of the system prompt, which is identical for every command (Section 18).

**Same INT8 file, same 21 prompts, Mac (Apple M5) vs phone:**

| | Mac M5 | Moto G34 |
|---|---:|---:|
| Time to first token | 0.77 s | 6.2 s |
| Total per command | 1.0 s | 7.9 s |
| Decode | 74 tok/s | 10.2 tok/s |

⟨pending: TTFT/decode for other variants (Mac)⟩

---

## 14. Thermal testing

**Method:**
- 10 minutes of continuous generation with the fine-tuned INT8 model (76 generations).
- On battery, ambient 32 °C.
- Sampled every 30 s: battery temperature, Android thermal status, CPU frequency per core, battery %, current, decode tok/s and time to first token.

| t (s) | Decode tok/s | Time to first token (ms) | Battery °C | Thermal status | CPU MHz (6 little / 2 big) |
|---:|---:|---:|---:|---|---|
| 36 | 9.69 | 5,785 | 35.5 | NONE | 1516 / 2054 |
| 131 | 10.20 | 6,117 | 35.7 | NONE | 1708 / 2054 |
| 247 | 9.11 | 5,742 | 35.9 | NONE | **940** / 2054 |
| 371 | 9.52 | 5,927 | 36.1 | **SEVERE (3)** | 940 / 2054 |
| 401 | 9.63 | 5,889 | 36.1 | **SEVERE (3)** | 940 / 2054 |
| 469 | **5.76** | **9,052** | 36.1 | NONE | 1708 / **940** |
| 530 | 9.23 | 5,819 | 36.2 | NONE | 1708 / 2054 |
| 608 | 8.92 | 5,882 | 36.3 | NONE | 940 / 2054 |

![Thermal test](../results/plots/phone_thermal.png)

| Summary | Value |
|---|---|
| Initial battery temperature | 35.4 °C |
| Maximum during inference | 36.3 °C (CPU sensors up to about 62 °C) |
| After sustained inference | 36.3 °C |
| Decode tok/s first 30 s → last 30 s | 9.69 → 8.92 (**−8.0%**) |
| Throttling indicators | Thermal status SEVERE at 371–401 s; little cores capped 1708 → 940 MHz; one window with the big cores capped to 940 MHz (decode −40% for 30 s) |
| Battery consumption | 64% → 61% (**3% in 10 min**), average current about 0.78 A |

**Verdict:**
- Throttling is real but mild for sustained use. The governor caps clocks intermittently, but decode speed recovers within 30 s and the 10-minute average loss is 8%.
- Real voice use is bursty (one command every few seconds or minutes), so throttling is unlikely in normal use.
- The Android thermal-headroom API returned NaN on this device, so the verdict uses thermal status plus CPU frequency plus tokens/s.

---

## 15. Results and observations

1. **Fine-tuning is the biggest lever.** +56.6 points exact match with 1.25% of parameters trained. The prompt alone gives the format; LoRA teaches the schema rules: no invented fields, 24 h times, refusing unsupported requests.
2. **Target-module choice matters on a hybrid model.** With only 4 attention layers, attention-only LoRA is capacity-limited. The MLP adapter reaches all 32 layers. Attention + MLP (D) is best, especially for refusing unsupported requests (99.1%).
3. **INT8 is the sweet spot.** −2.8 points for a 3.2× smaller file than FP32 (1.6× smaller than FP16). INT4 saves another 151 MiB but loses 8.5 more points, and its errors are the dangerous kind: acting on unsupported requests.
4. **The hybrid architecture suits phones.** The KV cache is only 8 KB/token and decode speed stays constant from 512 to 2,048 tokens of context.
5. **Latency is dominated by the long system prompt.** Prefill takes about 85 tok/s on the phone, so 490 prompt tokens take about 5.7 s. Decode of a 15-token answer takes about 1.5 s.
6. **The same INT8 file behaves differently on the phone.** Over 32 commands where we compared outputs, exact match was 71.9% on the Mac and 50.0% on the phone; only 14 of 32 outputs were identical. Inputs were verified identical and the Mac is deterministic. The cause is dynamic INT8 activation quantization: activations are rounded at run time, and different ARM kernels (Snapdragon 695 vs Apple M5) produce slightly different values that flip close greedy choices. Accuracy measured off-device over-estimates phone accuracy for dynamic INT8.
7. **Thermals are acceptable:** −8% decode speed after 10 minutes of continuous load, 3% battery.

---

## 16. Limitations and failure cases

**Model failures (fine-tuned D, FP32, hand-written test commands):**

| Command | Expected | Model output | Failure |
|---|---|---|---|
| book an uber to the station | unsupported | `make_call`, contact "Uber" | Unsupported service mapped to a call |
| reply to meera saying sure | send_message | unsupported | "reply" phrasing not learned |
| whatsapp karthik ok done | send_message via WhatsApp | open_app WhatsApp | App name used as a verb |
| let joseph know the meeting moved to 4 | send_message | create_reminder 04:00 | Message content containing a time |
| tell dad i need the car tonight | send_message | create_reminder, invented label | Message content resembles a task |
| call uncle raju | contact "Uncle Raju" | "Raju" | Relationship prefix dropped |
| call dad on speaker phone | `{"contact": "Dad"}` | extra key `phone_type` | Out-of-schema key |

**Patterns still unlearned** (hard prompts):
- Negation ("don't set an alarm, remind me instead").
- A message whose content is itself a command.
- "Tonight" with no am/pm.
- "Turn it down to N" read as `set`.
- Disguised alarms ("make sure I'm awake by 5:15").

**System limitations:**
- **Phone vs Mac INT8 drift** (Section 15.6): the phone produced 3 non-JSON outputs in 32 commands, for example "I'm sorry, I can't assist with that." Phone-side accuracy is lower than the Mac-measured 86.3%.
- **Latency:** about 6 s to first token on the Moto G34 because the 490-token system prompt is processed for every command.
- **Speech-to-text errors propagate:** names (Ravi → "Navi"), app names (WhatsApp → "watcher") and spoken numbers ("six thirty" → "30 pm"). The LLM cannot fix a wrong transcript. Moonshine small WER on our 150 test clips is 9.07% (q8).
- **English only.** No Hindi, Tamil or code-mixed commands.
- **Synthetic data:** real-user phrasing will be more diverse than the templates.
- **Measurement gaps:**
  - Gradient norm and fine-grained validation curves were not logged in the reported training runs. Validation loss was measured per epoch; the script now logs these for future runs.
  - Phone RAM after unloading the model stays at about 1.39 GB because the native allocator keeps freed pages.
  - The thermal headroom API is not supported on this phone.

---

## 17. Recommended model configuration for TronOS

**Granite-4.0-H-350M + LoRA D (attention + MLP, r = 16), merged, ONNX INT8 (dynamic, per-channel), ONNX Runtime CPU, 4 threads.**

| Criterion | Why INT8 |
|---|---|
| Accuracy | 86.3% exact, 94.3% intent, 100% valid JSON, 99.1% unsupported recall: within 2.8 points of full precision |
| Safety | Keeps unsupported recall at 99.1% (INT4 drops to 78.3% and starts acting on payment and booking requests) |
| Size | 404 MiB: fits in the APK and leaves RAM for STT + app on 4–8 GB phones |
| Memory | About 0.9 GB process peak; about 1.2 GB with speech-to-text |
| Speed | 9.8 tok/s decode, about 7.5 s per command on a low-cost Snapdragon 695 (about 1 s on an Apple M5) |
| Thermal | −8% after 10 min continuous; normal bursty use is unaffected |

**Rejected alternatives:**
- **FP16:** 1.6× larger than INT8 for ⟨pending: FP16 accuracy gain⟩. CPU FP16 kernels on mid-range ARM are slow.
- **INT4:** saves 151 MiB but loses 8.5 points and is unsafe on unsupported requests. Use it only if storage is critical, and then only with quantization-aware fine-tuning.

**Two changes recommended before shipping** (Section 18): JSON-constrained decoding and system-prompt caching.

---

## 18. Future optimization opportunities

1. **Cache the system prompt (prefix caching).** The 490-token prompt is identical for every command. Run it once at start-up and keep the KV cache + Mamba state; each command then only processes about 15 tokens. Expected time to first token on the Moto G34: about 6 s → about 0.3 s. Granite-H makes this cheap: the saved state is about 26 MB.
2. **JSON-constrained decoding.** Allow only tokens that keep the output a valid JSON object with schema keys. This removes non-JSON replies (3 in 32 phone commands) and out-of-schema keys at almost no cost.
3. **Weight-only or static INT8.** Keep weights INT8 but avoid run-time activation rounding, using MatMulNBits 8-bit or calibrated static INT8. This removes the phone-vs-Mac drift at the same file size.
4. **Shorter prompt through fine-tuning.** The adapter has learned the schema, so the 7 in-prompt examples could be dropped and the model retrained with a compact prompt (about 75 tokens). That cuts prefill by about 6× on top of item 1.
5. **Data v1.3** for the unlearned patterns (negation, "reply to", app-as-verb, message containing a time or command, "tonight", relationship names), plus real recorded speech.
6. **Use the NPU/GPU** (QNN or NNAPI execution provider) for prefill on Snapdragon devices.
7. **Quantization-aware LoRA** (QLoRA-style or QAT) if INT4 is needed for low-storage devices.
8. **Multilingual support** (Hinglish, Tamil) with a multilingual STT and a matching dataset.
9. **Adapter hot-swap:** keep one INT8 base and load small task adapters at run time.

---

## Attachments

| Required | Location |
|---|---|
| Dataset + manifest | `data/v1.2/{train,val,test}.jsonl`, `data/v1.2/manifest.json` (sha256 per file) |
| Dataset scripts | `scripts/generate_dataset_v12.py`, `docs/dataset_design_v1.2.md`, `docs/granite_failure_analysis.md` |
| Training scripts | `scripts/train_lora.py`, `kaggle_job/` (Kaggle runner) |
| LoRA configuration | `runs/granite/<A_qv,B_attn,C_mlp,D_attn_mlp>/train_log.json` and `adapter/adapter_config.json` |
| Quantization scripts/config | `scripts/export_onnx.py` |
| Evaluation scripts | `scripts/eval_hf.py`, `scripts/eval_onnx.py`, `scripts/common.py`, `scripts/error_analysis.py` |
| Android implementation | `android_voice/` (TronX Voice), `android/` (TronX Bench), APKs in `deliverables/phone/` |
| Raw benchmark results | `results/eval/*/metrics.json` + `predictions.jsonl`, `results/phone/*.json`, `results/bench_mac/*.json`, `results/quant/` |
| Screenshots / logs | `deliverables/screenshots/`, phone Share logs in `results/phone/` |
