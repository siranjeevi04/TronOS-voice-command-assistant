"""Evaluate an ONNX variant (fp32/fp16/int8/int4) with ONNX Runtime CPU EP: same prompts, metrics and scoring as
scripts/eval_hf.py, so Stage 2-4 numbers are directly comparable. Also computes target-token loss (teacher forcing).

  python scripts/eval_onnx.py --model models/onnx/ft_int4 --split test --name ft_int4_test
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from common import aggregate, build_messages, parse_output, score_sample  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TOK = ROOT / "models/granite-4.0-h-350m"


class OnnxLM:
    def __init__(self, path, threads=8):
        so = ort.SessionOptions(); so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.s = ort.InferenceSession(str(Path(path) / "model.onnx"), so, providers=["CPUExecutionProvider"])
        self.inputs = {i.name: i for i in self.s.get_inputs()}
        self.states = [n for n in self.inputs if n.startswith("past_")]
        self.dtype = np.float16 if self.inputs[self.states[0]].type == "tensor(float16)" else np.float32
        self.out_names = [o.name for o in self.s.get_outputs()]

    def _zero(self, name):
        shp = [1 if (i == 0 and not isinstance(d, int)) else (d if isinstance(d, int) else 0)
               for i, d in enumerate(self.inputs[name].shape)]
        return np.zeros(shp, dtype=self.dtype)

    def generate(self, ids, max_new=64, eos=100257):
        cache = {n: self._zero(n) for n in self.states}
        x = np.array([ids], dtype=np.int64); total = len(ids); out = []
        keep = np.array(1, dtype=np.int64)
        for _ in range(max_new):
            feed = {"input_ids": x, "attention_mask": np.ones((1, total), dtype=np.int64), "num_logits_to_keep": keep, **cache}
            res = dict(zip(self.out_names, self.s.run(None, feed)))
            nxt = int(np.argmax(res["logits"][0, -1]))
            for n in self.states:
                cache[n] = res[n.replace("past_key_values", "present").replace("past_", "present_")]
            out.append(nxt)
            if nxt == eos:
                break
            x = np.array([[nxt]], dtype=np.int64); total += 1
        return out

    def target_nll(self, prompt_ids, target_ids):
        """Sum of -log p(target | prompt) via one teacher-forced pass, returning (nll_sum, n_tokens)."""
        ids = prompt_ids + target_ids
        k = len(target_ids)
        feed = {"input_ids": np.array([ids], dtype=np.int64), "attention_mask": np.ones((1, len(ids)), dtype=np.int64),
                "num_logits_to_keep": np.array(k + 1, dtype=np.int64), **{n: self._zero(n) for n in self.states}}
        logits = self.s.run(["logits"], feed)[0][0].astype(np.float64)   # positions len-k-1 .. len-1
        logits = logits[:-1]                                              # predict target tokens
        logits -= logits.max(-1, keepdims=True)
        logp = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
        return float(-logp[np.arange(k), target_ids].sum()), k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--split", default="test")
    ap.add_argument("--data-dir", default="data/v1.2"); ap.add_argument("--name", required=True)
    ap.add_argument("--limit", type=int); ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(TOK)
    lm = OnnxLM(ROOT / a.model, a.threads)
    rows = [json.loads(l) for l in (ROOT / a.data_dir / f"{a.split}.jsonl").read_text().splitlines() if l.strip()]
    if a.limit:
        rows = rows[: a.limit]
    scored, nll, ntok, gen_tok = [], 0.0, 0, 0
    t0 = time.time()
    for i, r in enumerate(rows):
        p = tok.apply_chat_template(build_messages(r["text"], "schema"), add_generation_prompt=True, tokenize=False)
        pid = tok(p, add_special_tokens=False)["input_ids"]
        full = tok.apply_chat_template(build_messages(r["text"], "schema") + [{"role": "assistant", "content": r["target"]}],
                                       tokenize=False).rstrip("\n")
        fid = tok(full, add_special_tokens=False)["input_ids"]
        out = lm.generate(pid, 64, tok.eos_token_id)
        gen_tok += len(out)
        raw = tok.decode([t for t in out if t != tok.eos_token_id], skip_special_tokens=True)
        l, k = lm.target_nll(pid, fid[len(pid):]); nll += l; ntok += k
        pred, valid = parse_output(raw)
        scored.append(dict(id=r["id"], text=r["text"], source=r["source"], gold=r["output"], raw=raw, pred=pred, valid=valid,
                           **score_sample(pred, r["output"])))
        if (i + 1) % 50 == 0:
            print(f"{i+1}/{len(rows)} {time.time()-t0:.0f}s", flush=True)
    m = aggregate(scored)
    m["by_source"] = {s: aggregate([x for x in scored if x["source"] == s]) for s in sorted({x["source"] for x in scored})}
    m.update({"name": a.name, "model": a.model, "split": a.split, "data_dir": a.data_dir, "prompt": "schema",
              "target_loss": nll / ntok, "runtime": f"onnxruntime {ort.__version__} CPUExecutionProvider ({a.threads} threads)",
              "state_dtype": lm.dtype.__name__, "gen_wall_s": time.time() - t0, "gen_tokens": gen_tok})
    out = ROOT / "results/eval" / a.name; out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(m, indent=2))
    (out / "predictions.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in scored))
    print(json.dumps({k: m[k] for k in ("n", "intent_accuracy", "field_accuracy", "exact_match", "json_validity",
                                        "unsupported_recall", "target_loss", "gen_wall_s")}, indent=1))


if __name__ == "__main__":
    main()
