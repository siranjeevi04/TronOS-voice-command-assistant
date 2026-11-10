"""Evaluate a HF model (optionally + LoRA adapter) on a split: greedy decoding, JSON
validity, intent / field / exact-match accuracy, and masked target loss.

  python scripts/eval_hf.py --model models/Qwen2.5-0.5B-Instruct --split test --prompt schema --name base_schema
  python scripts/eval_hf.py --model models/Qwen2.5-0.5B-Instruct --adapter runs/lora_D_attn_mlp/adapter --split test --name ft_D
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from common import aggregate, build_messages, parse_output, score_sample  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(p):
    return [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--merge", action="store_true", help="merge adapter before eval (timing comparison)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--prompt", default="compact", choices=["compact", "schema"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-loss", action="store_true")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"], help="weights + compute precision")
    ap.add_argument("--out-dir", default="results/eval")
    ap.add_argument("--loss-batch", type=int, default=4,
                    help="batch for target-loss pass; small because full-vocab fp32 logits are large "
                         "(16 x 470 tok x 100k vocab x 4 B = 3 GB caused swap thrash on a 24 GB Mac)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    mpath = ROOT / args.model if (ROOT / args.model).exists() else args.model  # local dir or HF repo id
    tok = AutoTokenizer.from_pretrained(mpath)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(mpath, dtype=torch.float16 if args.dtype == "fp16" else torch.float32).to(device)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ROOT / args.adapter).to(device)
        if args.merge:
            model = model.merge_and_unload()
    model.eval()
    # Stop on the chat end-of-turn token as well as the tokenizer's EOS.
    # Stop on the model's end-of-turn token(s); only add a candidate if it really exists in the vocab.
    stop_ids = {tok.eos_token_id}
    for t in ("<|im_end|>", "<|end_of_text|>", "<|eot_id|>"):
        tid = tok.convert_tokens_to_ids(t)
        if tid is not None and tid != tok.unk_token_id and tok.convert_ids_to_tokens(tid) == t:
            stop_ids.add(tid)
    stop_ids = sorted(stop_ids - {None})

    rows = load_jsonl(ROOT / args.data_dir / f"{args.split}.jsonl")
    if args.limit:
        rows = rows[: args.limit]
    prompts = [tok.apply_chat_template(build_messages(r["text"], args.prompt), add_generation_prompt=True,
                                       tokenize=False) for r in rows]
    outs = []
    gen_tokens = 0
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(rows), args.batch):
            enc = tok(prompts[i : i + args.batch], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            g = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False, temperature=None, top_p=None,
                               top_k=None, eos_token_id=stop_ids, pad_token_id=tok.pad_token_id)
            new = g[:, enc["input_ids"].shape[1]:]
            for seq in new:
                ids = seq.tolist()
                n = next((k for k, t in enumerate(ids) if t in stop_ids), len(ids))
                gen_tokens += n
                outs.append(tok.decode(ids[:n], skip_special_tokens=True))
            print(f"{min(i+args.batch, len(rows))}/{len(rows)}", flush=True)
    gen_time = time.time() - t0

    # masked loss on gold target tokens (same definition as training)
    loss = None
    if not args.no_loss:
        tot, ntok = 0.0, 0
        tok.padding_side = "right"
        with torch.no_grad():
            for i in range(0, len(rows), args.loss_batch):
                batch = rows[i : i + args.loss_batch]
                ids_l, lab_l = [], []
                for r, pr in zip(batch, prompts[i : i + args.loss_batch]):
                    full = tok.apply_chat_template(build_messages(r["text"], args.prompt) +
                                                   [{"role": "assistant", "content": r["target"]}],
                                                   tokenize=False).rstrip("\n")
                    p_ids = tok(pr, add_special_tokens=False)["input_ids"]
                    f_ids = tok(full, add_special_tokens=False)["input_ids"]
                    ids_l.append(f_ids)
                    lab_l.append([-100] * len(p_ids) + f_ids[len(p_ids):])
                L = max(map(len, ids_l))
                ids = torch.full((len(batch), L), tok.pad_token_id)
                lab = torch.full((len(batch), L), -100)
                att = torch.zeros((len(batch), L), dtype=torch.long)
                for j, (a, b) in enumerate(zip(ids_l, lab_l)):
                    ids[j, : len(a)] = torch.tensor(a)
                    lab[j, : len(b)] = torch.tensor(b)
                    att[j, : len(a)] = 1
                logits = model(input_ids=ids.to(device), attention_mask=att.to(device)).logits
                ll = lab[:, 1:].to(device)
                mask = ll != -100
                sel = logits[:, :-1][mask].float()  # only target positions -> small tensor
                tot += torch.nn.functional.cross_entropy(sel, ll[mask], reduction="sum").item()
                ntok += mask.sum().item()
                del logits
        loss = tot / ntok

    scored = []
    for r, raw in zip(rows, outs):
        pred, valid = parse_output(raw)
        s = score_sample(pred, r["output"])
        scored.append(dict(id=r["id"], text=r["text"], lang=r["lang"], source=r["source"],
                           uses_heldout_value=r.get("uses_heldout_value", False),
                           gold=r["output"], raw=raw, pred=pred, valid=valid, **s))
    metrics = aggregate(scored)
    metrics["by_source"] = {src: aggregate([x for x in scored if x["source"] == src])
                            for src in sorted({x["source"] for x in scored})}
    metrics["by_lang"] = {lg: {k: v for k, v in aggregate([x for x in scored if x["lang"] == lg]).items()
                               if k != "per_intent"} for lg in sorted({x["lang"] for x in scored})}
    metrics.update({"name": args.name, "model": args.model, "adapter": args.adapter, "merged": args.merge,
                    "split": args.split, "data_dir": args.data_dir, "prompt": args.prompt, "target_loss": loss,
                    "runtime": f"transformers {__import__('transformers').__version__} / torch {torch.__version__} / {device} {args.dtype}",
                    "gen_wall_s": gen_time, "gen_tokens": gen_tokens, "batch": args.batch})
    out = ROOT / args.out_dir / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with (out / "predictions.jsonl").open("w") as f:
        for x in scored:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(json.dumps({k: metrics[k] for k in ("n", "intent_accuracy", "field_accuracy", "exact_match",
                                              "json_validity", "unsupported_recall", "target_loss")}, indent=2))


if __name__ == "__main__":
    main()
