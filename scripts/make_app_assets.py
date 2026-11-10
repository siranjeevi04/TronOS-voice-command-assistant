"""Build every tokenizer/prompt asset the two Android apps read.

  python scripts/make_app_assets.py [--android-dir DIR] [--voice-dir DIR]

Writes, for the benchmark app (android/) and the voice app (android_voice/):
  prompts.json  pre-tokenised benchmark prompts (identical ids for every model variant, Section 9E), exact-length
                context prompts (512/1K/2K), chat prefix/suffix ids around a command, special ids, preset prompts
  vocab.json    Granite token -> id map (on-device byte-level BPE encode/decode)
  merges.txt    Granite BPE merge ranks (on-device encode)
and for the voice app only:
  stt_vocab.json  Moonshine token pieces + special ids + start/eos (on-device STT decode)
Preset prompts: benchmark app = results/phone_test_prompts.json + results/hard_prompts_v1.json (20);
voice app = results/hard_prompts_v1.json (10).
"""
import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from common import SCHEMA_PROMPT, build_messages  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--android-dir", default=str(ROOT / "android/app/src/main/assets"))
ap.add_argument("--voice-dir", default=str(ROOT / "android_voice/app/src/main/assets"))
args = ap.parse_args()
OUT, VOICE = Path(args.android_dir), Path(args.voice_dir)
OUT.mkdir(parents=True, exist_ok=True); VOICE.mkdir(parents=True, exist_ok=True)
tok = AutoTokenizer.from_pretrained(ROOT / "models/granite-4.0-h-350m")

# 20 benchmark commands = the first 20 rows of the held-out v1.2 TEST split (fixed, documented)
test = [json.loads(l) for l in (ROOT / "data/v1.2/test.jsonl").read_text().splitlines() if l.strip()]
cmds = [r["text"] for r in test[:20]]


def ids_for(text):
    p = tok.apply_chat_template(build_messages(text, "schema"), add_generation_prompt=True, tokenize=False)
    return tok(p, add_special_tokens=False)["input_ids"]


bench = [{"text": c, "ids": ids_for(c)} for c in cmds]

# exact-length context prompts: system + user(filler ... command) + assistant header
prefix = tok.apply_chat_template(build_messages("<<FILL>>", "schema"), add_generation_prompt=True, tokenize=False)
a, b = prefix.split("<<FILL>>")
a_ids = tok(a, add_special_tokens=False)["input_ids"]
cmd_ids = tok(" Call Ravi.", add_special_tokens=False)["input_ids"]  # short so that 512 still fits the 493-token schema prompt
b_ids = tok(b, add_special_tokens=False)["input_ids"]
filler_text = " ".join(f"Context note {i}: the user earlier asked to open YouTube, call Ravi, and turn on Bluetooth." for i in range(400))
filler = tok(filler_text, add_special_tokens=False)["input_ids"]
ctx = {}
for n in (512, 1024, 2048):
    k = n - len(a_ids) - len(cmd_ids) - len(b_ids)
    assert k >= 0, (n, k)
    ctx[str(n)] = a_ids + filler[:k] + cmd_ids + b_ids
    assert len(ctx[str(n)]) == n, (n, len(ctx[str(n)]))

sys_only = tok.apply_chat_template([{"role": "system", "content": SCHEMA_PROMPT}], tokenize=False)
special = sorted(set(tok.all_special_ids) | {i for t, i in tok.get_vocab().items() if t.startswith("<|") and t.endswith("|>")})
demo = [x["text"] for x in json.loads((ROOT / "results/phone_test_prompts.json").read_text())]
hard = [x["text"] for x in json.loads((ROOT / "results/hard_prompts_v1.json").read_text())]
base = {"model": "ibm-granite/granite-4.0-h-350m (+ LoRA)", "eos": tok.eos_token_id,
        "schema_prompt_tokens": len(tok(sys_only, add_special_tokens=False)["input_ids"]),
        "bench_prompts": bench, "ctx_prompts": ctx, "special_ids": special,
        "source": "first 20 commands of data/v1.2/test.jsonl; chat template + schema system prompt (scripts/common.py)",
        "chat_prefix_ids": a_ids, "chat_suffix_ids": b_ids}
merges = [m if isinstance(m, str) else " ".join(m) for m in json.loads(tok.backend_tokenizer.to_str())["model"]["merges"]]
for d, presets in ((OUT, demo + hard), (VOICE, hard)):
    (d / "prompts.json").write_text(json.dumps({**base, "test_prompts": presets}))
    (d / "vocab.json").write_text(json.dumps(tok.get_vocab(), ensure_ascii=False))
    (d / "merges.txt").write_text("\n".join(merges), encoding="utf-8")

# Moonshine STT vocabulary for the voice app
stt = json.loads((ROOT / "models/stt/moonshine-streaming-small-ONNX/tokenizer.json").read_text())
gen = json.loads((ROOT / "models/stt/moonshine-streaming-small-ONNX/generation_config.json").read_text())
vocab = stt["model"]["vocab"]
added = stt.get("added_tokens", [])
pieces = [None] * (max(max(vocab.values()), max((a["id"] for a in added), default=0)) + 1)
for t, i in vocab.items():
    pieces[i] = t
for a in added:
    pieces[a["id"]] = a["content"]
(VOICE / "stt_vocab.json").write_text(json.dumps({"pieces": pieces, "special": sorted(a["id"] for a in stt.get("added_tokens", []) if a.get("special")),
                                                  "start": gen.get("decoder_start_token_id", 1), "eos": gen.get("eos_token_id", 2)}))
print("bench prompt lengths:", min(len(x["ids"]) for x in bench), "-", max(len(x["ids"]) for x in bench))
print("ctx:", {k: len(v) for k, v in ctx.items()}, "prefix", len(a_ids), "suffix", len(b_ids), "eos", tok.eos_token_id,
      "vocab", len(tok.get_vocab()), "merges", len(merges), "stt pieces", len(pieces))
