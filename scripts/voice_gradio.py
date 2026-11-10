"""Mac prototype of the full TronOS voice pipeline (same ONNX models the phone will run):

  mic -> Silero VAD -> utterance ends after 2.0 s of silence -> Moonshine v2 small (ONNX q8) -> text
      -> empty/no letters? discard : Granite-4.0-H-350M + LoRA D (ONNX INT8) -> JSON

  .venv/bin/python scripts/voice_gradio.py      # http://127.0.0.1:7861
"""
import json
import os
import re
import sys
import time
from pathlib import Path

import gradio as gr
import numpy as np
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).parent))
from common import build_messages, parse_output  # noqa: E402
from eval_onnx import OnnxLM  # noqa: E402
from moonshine_onnx import MoonshineONNX, SileroVAD  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SR = 16000
SILENCE_S = 2.0          # end of command after this much silence
SPEECH_TH = 0.5          # Silero speech probability threshold
MIN_SPEECH_S = 0.25      # ignore blips shorter than this
LOG = ROOT / "results/voice/voice_tests.jsonl"
LOG.parent.mkdir(parents=True, exist_ok=True)


def rss_mb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)  # macOS: bytes -> MB (peak)


print("loading models ...", flush=True)
stt = MoonshineONNX("quantized", threads=4)
vad = SileroVAD()
tok = AutoTokenizer.from_pretrained(ROOT / "models/granite-4.0-h-350m")
LLMS = {"FT INT8 (phone model)": "models/onnx/ft_int8", "FT FP32": "models/onnx/ft_fp32",
        "FT INT4": "models/onnx/ft_int4", "Base FP32 (untrained)": "models/onnx/base_fp32"}
loaded = {}


def llm(name):
    if name not in loaded:
        loaded.clear()
        loaded[name] = OnnxLM(ROOT / LLMS[name], threads=4)
    return loaded[name]


llm("FT INT8 (phone model)")
print("ready", flush=True)


def new_state():
    vad.reset()
    return {"buf": np.zeros(0, np.float32), "pending": np.zeros(0, np.float32), "speech": False,
            "speech_samples": 0, "silence_samples": 0, "history": []}


def run_pipeline(audio, llm_name, st):
    t0 = time.perf_counter()
    text = stt.transcribe(audio)
    t_stt = (time.perf_counter() - t0) * 1000
    rec = {"time": time.strftime("%H:%M:%S"), "audio_s": round(len(audio) / SR, 2), "transcript": text, "stt_ms": round(t_stt),
           "llm": llm_name}
    if not re.search(r"[A-Za-z]", text):
        rec["result"] = "DISCARDED (empty transcript)"
        return rec
    ids = tok(tok.apply_chat_template(build_messages(text, "schema"), add_generation_prompt=True, tokenize=False),
              add_special_tokens=False)["input_ids"]
    m = llm(llm_name)
    t1 = time.perf_counter()
    out = m.generate(ids, 64, tok.eos_token_id)
    t_llm = (time.perf_counter() - t1) * 1000
    raw = tok.decode([t for t in out if t != tok.eos_token_id], skip_special_tokens=True).strip()
    pred, valid = parse_output(raw)
    rec.update({"json": raw, "valid_json": valid, "prompt_tokens": len(ids), "out_tokens": len(out),
                "llm_ms": round(t_llm), "total_ms": round(t_stt + t_llm), "peak_rss_mb": round(rss_mb())})
    return rec


def fmt(rec):
    if "json" not in rec:
        return f"[{rec['time']}] heard: '{rec['transcript']}' -> {rec['result']} (STT {rec['stt_ms']} ms)"
    return (f"[{rec['time']}] heard ({rec['audio_s']} s audio): {rec['transcript']}\n"
            f"   -> {rec['json']}\n"
            f"   STT {rec['stt_ms']} ms | LLM {rec['llm_ms']} ms ({rec['prompt_tokens']} prompt tok, {rec['out_tokens']} out) | "
            f"speech-end -> JSON {rec['total_ms']} ms | peak RAM {rec['peak_rss_mb']} MB")


def stream(chunk, llm_name, st):
    if st is None:
        st = new_state()
    if chunk is None:
        return st, gr.update(), gr.update(), gr.update()
    sr, data = chunk
    x = np.asarray(data)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / np.iinfo(x.dtype).max
    x = x.astype(np.float32)
    if sr != SR:
        g = np.gcd(int(sr), SR); x = resample_poly(x, SR // g, int(sr) // g).astype(np.float32)
    st["pending"] = np.concatenate([st["pending"], x])
    status = None
    while len(st["pending"]) >= 512:
        win, st["pending"] = st["pending"][:512], st["pending"][512:]
        p = vad.prob(win)
        if p >= SPEECH_TH:
            st["speech"] = True; st["speech_samples"] += 512; st["silence_samples"] = 0
            st["buf"] = np.concatenate([st["buf"], win])
        elif st["speech"]:
            st["silence_samples"] += 512
            st["buf"] = np.concatenate([st["buf"], win])
            if st["silence_samples"] >= SILENCE_S * SR:
                audio = st["buf"][: len(st["buf"]) - int((SILENCE_S - 0.3) * SR)]  # keep 0.3 s tail
                long_enough = st["speech_samples"] >= MIN_SPEECH_S * SR
                st.update(buf=np.zeros(0, np.float32), speech=False, speech_samples=0, silence_samples=0)
                if long_enough:
                    rec = run_pipeline(audio, llm_name, st)
                    st["history"].insert(0, fmt(rec))
                    with LOG.open("a") as f:
                        f.write(json.dumps(rec) + "\n")
                    status = "processed: " + rec["transcript"]
    listening = "🎙️ speaking..." if st["speech"] else "👂 listening (stop talking for 2 s to send)"
    last = st["history"][0] if st["history"] else "(nothing yet)"
    return st, status or listening, last, "\n\n".join(st["history"][:15])


with gr.Blocks(title="TronOS voice -> JSON (Moonshine + Granite, ONNX)") as demo:
    gr.Markdown("## TronOS voice pipeline on the Mac (same ONNX models as the phone)\n"
                "Mic → Silero VAD → **2 s silence ends the command** → Moonshine v2 small (ONNX q8) → "
                "Granite-4.0-H-350M + LoRA D (ONNX) → JSON. Empty transcripts are discarded.\n"
                "Click **Record**, speak a command, pause 2 s; speak the next one, pause again ...")
    st = gr.State(None)
    with gr.Row():
        mic = gr.Audio(sources=["microphone"], streaming=True, type="numpy", label="Microphone (keep recording)")
        llm_pick = gr.Dropdown(list(LLMS), value="FT INT8 (phone model)", label="LLM variant")
    status = gr.Textbox(label="Status")
    last = gr.Textbox(label="Last command", lines=4)
    hist = gr.Textbox(label="History (newest first)", lines=14)
    mic.stream(stream, [mic, llm_pick, st], [st, status, last, hist], stream_every=0.25)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861)
