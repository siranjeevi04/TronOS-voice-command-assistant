"""Compare small open-source English STT models on TronOS voice commands.

Audio: macOS built-in TTS voices (3 Indian-English: Aman, Rishi, Tara; 1 US: Samantha; 1 UK: Daniel)
speaking 30 commands -> 150 clips at 16 kHz. SYNTHETIC speech: good for a controlled, identical
comparison between models, NOT a substitute for real human recordings.

Metrics per model:
  WER          word error rate after the standard Whisper English normaliser (Open ASR Leaderboard
               convention) + joining "6 30" -> "630" so digits and spoken numbers compare equally
  key-term acc share of critical slot words (times, numbers, names, app names) present in the transcript
  exact        normalised transcript == normalised reference
  latency      CPU wall time per clip (after warm-up), real-time factor

  .venv/bin/python scripts/stt_benchmark.py --models moonshine-streaming-small whisper-small.en ...
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

ROOT = Path(__file__).resolve().parents[1]
AUDIO = ROOT / "data/stt_bench/audio"
OUT = ROOT / "results/stt/benchmark"
VOICES = ["Aman", "Rishi", "Tara", "Samantha", "Daniel"]   # en_IN x3, en_US, en_GB

# (spoken command, critical key terms that must survive for the LLM to fill the slot)
COMMANDS = [
    ("Set an alarm for 6:30 AM tomorrow.", ["6:30", "tomorrow"]),
    ("Open YouTube and play latest song.", ["youtube"]),
    ("Transfer money to Ravi.", ["transfer", "ravi"]),
    ("Wake me up at quarter to seven in the morning.", ["quarter to 7", "morning"]),
    ("Set an alarm for half past seven in the evening on Monday for school run.", ["half past 7", "monday", "school run"]),
    ("Remind me to review the slides tomorrow at 10 AM.", ["review the slides", "10 am"]),
    ("Don't let me forget to pay the electricity bill on Friday.", ["electricity bill", "friday"]),
    ("Call Keerthi on speaker.", ["keerthi"]),
    ("Video call Siddharth.", ["video call", "siddharth"]),
    ("Send a message to Priya on WhatsApp saying I'll be there in five minutes.", ["priya", "whatsapp", "5 minutes"]),
    ("Text Anil that the meeting is cancelled.", ["anil", "meeting is cancelled"]),
    ("Turn on Bluetooth.", ["bluetooth"]),
    ("Set the brightness to forty percent.", ["brightness", "40%"]),
    ("Increase the volume.", ["increase", "volume"]),
    ("Deactivate night mode.", ["deactivate", "night mode"]),
    ("Open file manager.", ["file manager"]),
    ("Cancel my 5 PM alarm.", ["cancel", "5 pm"]),
    ("What is the capital of France?", ["capital", "france"]),
    ("Play some music.", ["play", "music"]),
    ("Remind me to call the plumber at 2:10 PM.", ["plumber", "2:10 pm"]),
    ("Set an alarm for 5:45 AM on Saturday.", ["5:45", "saturday"]),
    ("Call Amma.", ["amma"]),
    ("Message Karthik on Telegram saying lunch at one.", ["karthik", "telegram", "lunch at 1"]),
    ("Open PhonePe.", ["phonepe"]),
    ("Open Instagram.", ["instagram"]),
    ("Remind me to take my medicine at 9 PM tonight.", ["medicine", "9 pm"]),
    ("Turn off mobile data.", ["off", "mobile data"]),
    ("Set volume to 70 percent.", ["volume", "70%"]),
    ("Call Lakshmi.", ["lakshmi"]),
    ("Wake me up at 11:15 PM on Wednesday.", ["11:15", "wednesday"]),
]

MODELS = {
    "moonshine-streaming-tiny": "moonshine-ai/moonshine-streaming-tiny",
    "moonshine-streaming-small": "moonshine-ai/moonshine-streaming-small",
    "moonshine-streaming-medium": "moonshine-ai/moonshine-streaming-medium",
    "whisper-base.en": "openai/whisper-base.en",
    "distil-small.en": "distil-whisper/distil-small.en",
    "whisper-small.en": "openai/whisper-small.en",
    "whisper-large-v3-turbo": "openai/whisper-large-v3-turbo",
}

_norm = EnglishTextNormalizer(json.load(open(hf_hub_download("openai/whisper-small.en", "normalizer.json"))))


def norm(t: str) -> str:
    t = _norm(t)
    t = re.sub(r"\b(\d{1,2}) (\d{2})\b", r"\1\2", t)  # "6 30" -> "630" (matches spoken "six thirty" -> "630")
    return re.sub(r"\s+", " ", t).strip()


def make_audio():
    AUDIO.mkdir(parents=True, exist_ok=True)
    clips = []
    for v in VOICES:
        for i, (text, keys) in enumerate(COMMANDS):
            wav = AUDIO / f"{v}_{i:02d}.wav"
            if not wav.exists():
                aiff = wav.with_suffix(".aiff")
                subprocess.run(["say", "-v", v, "-o", str(aiff), text], check=True)
                subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", str(aiff), str(wav)], check=True)
                aiff.unlink()
            clips.append({"voice": v, "idx": i, "text": text, "keys": keys, "wav": wav})
    return clips


def load(name):
    rid = MODELS[name]
    proc = AutoProcessor.from_pretrained(rid)
    if name.startswith("moonshine"):
        from transformers import MoonshineStreamingForConditionalGeneration as M
    else:
        from transformers import WhisperForConditionalGeneration as M
    model = M.from_pretrained(rid).eval()
    return proc, model


@torch.no_grad()
def transcribe(name, proc, model, x):
    sr = proc.feature_extractor.sampling_rate
    if name.startswith("moonshine"):
        inp = proc(x, return_tensors="pt", sampling_rate=sr)
        max_len = max(10, int(inp.attention_mask.sum(-1).max().item() * 6.5 / sr))
        t0 = time.perf_counter()
        ids = model.generate(**inp, max_length=max_len)
    else:
        inp = proc(x, return_tensors="pt", sampling_rate=sr)
        kw = {"language": "en", "task": "transcribe"} if not name.endswith(".en") else {}
        t0 = time.perf_counter()
        ids = model.generate(inp.input_features, max_new_tokens=96, **kw)
    dt = time.perf_counter() - t0
    return proc.batch_decode(ids, skip_special_tokens=True)[0].strip(), dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    clips = make_audio()
    summary = {}
    for name in args.models:
        proc, model = load(name)
        n_params = sum(p.numel() for p in model.parameters())
        transcribe(name, proc, model, np.zeros(16000, dtype=np.float32))  # warm-up
        rows = []
        for c in clips:
            x, sr = sf.read(c["wav"], dtype="float32")
            hyp, dt = transcribe(name, proc, model, x)
            r, h = norm(c["text"]), norm(hyp)
            m = jiwer.process_words(r, h)
            keys_ok = [norm(k) in h for k in c["keys"]]
            rows.append({**{k: c[k] for k in ("voice", "idx", "text")}, "hyp": hyp, "ref_n": r, "hyp_n": h,
                         "errors": m.substitutions + m.deletions + m.insertions, "ref_words": len(r.split()),
                         "exact": r == h, "keys_ok": keys_ok, "latency_s": dt, "audio_s": len(x) / sr})
        (OUT / f"{name}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))

        def agg(sub):
            w = sum(r["ref_words"] for r in sub)
            return {"wer": round(100 * sum(r["errors"] for r in sub) / w, 2),
                    "key_acc": round(100 * np.mean([k for r in sub for k in r["keys_ok"]]), 1),
                    "exact": round(100 * np.mean([r["exact"] for r in sub]), 1)}
        s = {"params_M": round(n_params / 1e6, 1), **agg(rows),
             "wer_indian": agg([r for r in rows if r["voice"] in ("Aman", "Rishi", "Tara")])["wer"],
             "key_acc_indian": agg([r for r in rows if r["voice"] in ("Aman", "Rishi", "Tara")])["key_acc"],
             "latency_ms_mean": round(1000 * np.mean([r["latency_s"] for r in rows]), 1),
             "rtf_mean": round(float(np.mean([r["latency_s"] / r["audio_s"] for r in rows])), 4)}
        summary[name] = s
        print(name, json.dumps(s), flush=True)
        del model
    prev = json.loads((OUT / "summary.json").read_text()) if (OUT / "summary.json").exists() else {}
    prev.update(summary)
    (OUT / "summary.json").write_text(json.dumps(prev, indent=2))


if __name__ == "__main__":
    main()
