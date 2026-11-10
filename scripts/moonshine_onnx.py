"""Moonshine v2 streaming-small (Workmind ONNX export) + Silero VAD, with ONNX Runtime only.

The same two ONNX graphs + logic are what the Android app will run.
  encoder:  input_values [1, n_samples] (16 kHz, zero-padded to a multiple of 80) -> last_hidden_state [1, T, 620]
  decoder:  merged graph with use_cache_branch; 10 layers x {decoder,encoder} x {key,value} KV cache
"""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
STT_DIR = ROOT / "models/stt/moonshine-streaming-small-ONNX"
VAD_PATH = ROOT / "models/vad/silero_vad.onnx"


class MoonshineONNX:
    def __init__(self, variant="quantized", threads=4):
        so = ort.SessionOptions(); so.intra_op_num_threads = threads
        suf = "" if variant == "fp32" else f"_{variant}"
        self.enc = ort.InferenceSession(str(STT_DIR / f"onnx/encoder_model{suf}.onnx"), so, providers=["CPUExecutionProvider"])
        self.dec = ort.InferenceSession(str(STT_DIR / f"onnx/decoder_model_merged{suf}.onnx"), so, providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(str(STT_DIR / "tokenizer.json"))
        g = json.loads((STT_DIR / "generation_config.json").read_text())
        self.start = g.get("decoder_start_token_id", 1); self.eos = g.get("eos_token_id", 2)
        self.dec_in = {i.name: i for i in self.dec.get_inputs()}
        self.dec_out = [o.name for o in self.dec.get_outputs()]
        self.past = [n for n in self.dec_in if n.startswith("past_key_values")]

    def transcribe(self, audio: np.ndarray, sr=16000):
        x = audio.astype(np.float32)
        pad = (-len(x)) % 80                      # v2 frontend needs whole 5 ms frames
        if pad:
            x = np.concatenate([x, np.zeros(pad, np.float32)])
        h = self.enc.run(None, {"input_values": x[None, :]})[0]
        max_len = max(8, int(len(audio) / sr * 6.5) + 4)   # model-card guard against loops
        cache = {n: np.zeros([1, 8, 0, 64], np.float32) for n in self.past}
        ids = [self.start]; cur = np.array([[self.start]], np.int64)
        for step in range(max_len):
            feed = {"input_ids": cur, "encoder_hidden_states": h, **cache}
            if "use_cache_branch" in self.dec_in:
                feed["use_cache_branch"] = np.array([step > 0])
            out = dict(zip(self.dec_out, self.dec.run(None, feed)))
            nxt = int(out["logits"][0, -1].argmax())
            for n in self.past:
                pn = n.replace("past_key_values", "present")
                if step == 0 or ".decoder." in n:      # encoder K/V are computed once at step 0
                    cache[n] = out[pn]
            if nxt == self.eos:
                break
            ids.append(nxt); cur = np.array([[nxt]], np.int64)
        return self.tok.decode(ids[1:], skip_special_tokens=True).strip()


class SileroVAD:
    """Silero VAD v5: 512-sample windows at 16 kHz -> speech probability."""
    def __init__(self):
        self.s = ort.InferenceSession(str(VAD_PATH), providers=["CPUExecutionProvider"])
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), np.float32); self.ctx = np.zeros(64, np.float32)

    def prob(self, chunk512: np.ndarray) -> float:
        x = np.concatenate([self.ctx, chunk512.astype(np.float32)])[None, :]
        out, self.state = self.s.run(None, {"input": x, "state": self.state, "sr": np.array(16000, np.int64)})
        self.ctx = chunk512[-64:].astype(np.float32)
        return float(out[0, 0])
