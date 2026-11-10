"""Gradio test bench for the STT stage: Moonshine v2 streaming small (moonshine-ai/moonshine-streaming-small).

Speak (or upload) a TronOS command, optionally pick/type what you said, and the app shows:
transcript, processing latency, audio length, real-time factor and word error rate (WER).
Every test is logged to results/stt/gradio_tests.jsonl (+ the audio in results/stt/audio/) so
accuracy can be summarised across many utterances.

  .venv/bin/python scripts/stt_gradio.py            # opens http://127.0.0.1:7860
"""
from __future__ import annotations

import datetime
import json
import re
import time
from pathlib import Path

import gradio as gr
import jiwer
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import AutoProcessor, MoonshineStreamingForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models/stt/moonshine-streaming-small"
LOG = ROOT / "results/stt/gradio_tests.jsonl"
AUDIO_DIR = ROOT / "results/stt/audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cpu"  # phone runs on CPU too; keep the test honest and simple
processor = AutoProcessor.from_pretrained(MODEL_DIR)
model = MoonshineStreamingForConditionalGeneration.from_pretrained(MODEL_DIR).to(DEVICE).eval()
SR = processor.feature_extractor.sampling_rate
N_PARAMS = sum(p.numel() for p in model.parameters())

# Commands to read aloud: one or more per intent, incl. the hard cases found in the LLM error analysis.
SUGGESTED = [
    "Set an alarm for 6:30 AM tomorrow.",
    "Open YouTube and play latest song.",
    "Transfer money to Ravi.",
    "Wake me up at quarter to seven in the morning.",
    "Set an alarm for half past seven in the evening on Monday for school run.",
    "Remind me to review the slides tomorrow at 10 AM.",
    "Don't let me forget to pay the electricity bill on Friday.",
    "Call Keerthi on speaker.",
    "Video call Siddharth.",
    "Send a message to Priya on WhatsApp saying I'll be there in five minutes.",
    "Text Anil that the meeting is cancelled.",
    "Turn on Bluetooth.",
    "Set the brightness to forty percent.",
    "Increase the volume.",
    "Deactivate night mode.",
    "Open file manager.",
    "Cancel my 5 PM alarm.",
    "What is the capital of France?",
    "Play some music.",
    "Remind me to call the plumber at two ten PM.",
]


def normalise(t: str) -> str:
    """Lowercase, drop punctuation (keep ':' in times), collapse spaces: so WER counts words, not commas."""
    t = t.lower().replace("’", "'")
    t = re.sub(r"[^\w\s:']", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def to_mono_16k(audio):
    sr, data = audio
    x = np.asarray(data)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / np.iinfo(x.dtype).max
    x = x.astype(np.float32)
    if sr != SR:
        g = np.gcd(int(sr), int(SR))
        x = resample_poly(x, SR // g, int(sr) // g).astype(np.float32)
    return x


@torch.no_grad()
def transcribe_array(x: np.ndarray) -> tuple[str, float]:
    inputs = processor(x, return_tensors="pt", sampling_rate=SR).to(DEVICE)
    # Model-card guard against hallucination loops: ~6.5 tokens per second of audio.
    max_len = max(10, int(inputs.attention_mask.sum(dim=-1).max().item() * 6.5 / SR))
    t0 = time.perf_counter()
    ids = model.generate(**inputs, max_length=max_len)
    dt = time.perf_counter() - t0
    return processor.decode(ids[0], skip_special_tokens=True).strip(), dt


# warm-up so the first measured latency is not inflated by lazy initialisation
transcribe_array(np.zeros(SR, dtype=np.float32))


def run(audio, reference, note):
    if audio is None:
        return "No audio.", "", ""
    x = to_mono_16k(audio)
    dur = len(x) / SR
    text, dt = transcribe_array(x)
    stats = f"audio {dur:.2f} s | processing {dt*1000:.0f} ms | real-time factor {dt/dur:.3f} (lower = faster)"
    wer_line = ""
    rec = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "model": "moonshine-streaming-small",
           "device": DEVICE, "audio_s": round(dur, 3), "latency_ms": round(dt * 1000, 1), "rtf": round(dt / dur, 4),
           "transcript": text, "reference": reference or "", "note": note or ""}
    if reference and reference.strip():
        ref_n, hyp_n = normalise(reference), normalise(text)
        out = jiwer.process_words(ref_n, hyp_n)
        rec.update({"ref_norm": ref_n, "hyp_norm": hyp_n, "wer": round(out.wer, 4), "ref_words": len(ref_n.split()),
                    "subs": out.substitutions, "dels": out.deletions, "ins": out.insertions, "exact": ref_n == hyp_n})
        wer_line = (f"WER {out.wer*100:.1f}%  (substitutions {out.substitutions}, deletions {out.deletions}, "
                    f"insertions {out.insertions})  |  exact sentence match: {'YES' if ref_n == hyp_n else 'NO'}\n"
                    f"reference : {ref_n}\nheard     : {hyp_n}")
    else:
        wer_line = ("NO REFERENCE -> accuracy not measured. Pick a suggested command or type exactly what you said "
                    "in 'Reference' BEFORE clicking Transcribe.")
    wav = AUDIO_DIR / f"{rec['time'].replace(':', '-')}.wav"
    sf.write(wav, x, SR)
    rec["audio_file"] = str(wav.relative_to(ROOT))
    with LOG.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return text, stats, wer_line


def summary():
    if not LOG.exists():
        return "No tests logged yet."
    rows = [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()]
    scored = [r for r in rows if "wer" in r]
    lines = [f"tests logged: {len(rows)}  |  with reference: {len(scored)}",
             f"mean latency {np.mean([r['latency_ms'] for r in rows]):.0f} ms  |  mean RTF {np.mean([r['rtf'] for r in rows]):.3f}"]
    if scored:
        words = sum(r["ref_words"] for r in scored)
        errs = sum(r["subs"] + r["dels"] + r["ins"] for r in scored)
        lines.append(f"corpus WER {100*errs/words:.1f}% over {words} words  |  exact sentences "
                     f"{sum(r['exact'] for r in scored)}/{len(scored)}")
        bad = [r for r in scored if not r["exact"]]
        if bad:
            lines.append("\nmisheard:")
            lines += [f"  said : {r['ref_norm']}\n  heard: {r['hyp_norm']}" for r in bad[-15:]]
    return "\n".join(lines)


with gr.Blocks(title="TronOS STT test - Moonshine v2 streaming small") as demo:
    gr.Markdown(f"## TronOS STT test: Moonshine v2 streaming small\n"
                f"`moonshine-ai/moonshine-streaming-small`, {N_PARAMS/1e6:.0f}M params, device: {DEVICE}, input {SR} Hz.\n"
                "1) pick or type the command you will say  2) record (or upload)  3) Transcribe.")
    with gr.Row():
        with gr.Column():
            pick = gr.Dropdown(SUGGESTED, label="Suggested command to read (optional)", allow_custom_value=True)
            ref = gr.Textbox(label="Reference: what you actually said (for WER)")
            pick.change(lambda s: s, pick, ref)
            audio = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Speak / upload")
            note = gr.Textbox(label="Note (optional: accent, noise, distance...)")
            btn = gr.Button("Transcribe", variant="primary")
        with gr.Column():
            out_text = gr.Textbox(label="Transcript")
            out_stats = gr.Textbox(label="Speed")
            out_wer = gr.Textbox(label="Accuracy vs reference", lines=4)
            sbtn = gr.Button("Show summary of all tests")
            out_sum = gr.Textbox(label="Summary", lines=14)
    btn.click(run, [audio, ref, note], [out_text, out_stats, out_wer])
    sbtn.click(summary, None, out_sum)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
