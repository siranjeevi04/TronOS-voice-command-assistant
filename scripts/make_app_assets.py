"""Build the Android benchmark assets: pre-tokenised prompts (identical for every model variant, Section 9E)
and the vocabulary for on-device decoding.

  python scripts/make_app_assets.py
Writes android/app/src/main/assets/{prompts.json,vocab.json}
"""
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from common import SCHEMA_PROMPT, build_messages  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "android/app/src/main/assets"
OUT.mkdir(parents=True, exist_ok=True)
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
(OUT / "prompts.json").write_text(json.dumps({
    "model": "ibm-granite/granite-4.0-h-350m (+ LoRA)", "eos": tok.eos_token_id,
    "schema_prompt_tokens": len(tok(sys_only, add_special_tokens=False)["input_ids"]),
    "bench_prompts": bench, "ctx_prompts": ctx, "special_ids": special,
    "source": "first 20 commands of data/v1.2/test.jsonl; chat template + schema system prompt (scripts/common.py)"}))
(OUT / "vocab.json").write_text(json.dumps(tok.get_vocab(), ensure_ascii=False))
print("bench prompt lengths:", min(len(x["ids"]) for x in bench), "-", max(len(x["ids"]) for x in bench))
print("ctx:", {k: len(v) for k, v in ctx.items()}, "eos", tok.eos_token_id, "vocab", len(tok.get_vocab()))
