"""Mac counterpart of the Android benchmark app (assignment Section 9 A-F), same methodology, same pre-tokenised
prompts (android/app/src/main/assets/prompts.json), same runtime (ONNX Runtime CPU EP, 4 threads, graph opt ALL).
Each variant runs in a fresh process so loading memory is clean.

  .venv/bin/python scripts/bench_mac.py                # all 4 Section 10 variants
  .venv/bin/python scripts/bench_mac.py --variant ft_int8
Output: results/bench_mac/<variant>.json
"""
import argparse
import json
import platform
import statistics as st
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "android/app/src/main/assets/prompts.json"
OUT = ROOT / "results/bench_mac"
VARIANTS = ["base_fp16", "ft_fp16", "ft_int8", "ft_int4"]


class Peak:
    """Samples process RSS every 15 ms (same as the Android app) and keeps the maximum."""
    def __init__(self, proc):
        self.p, self.peak, self.run = proc, 0.0, False

    def __enter__(self):
        self.run = True; self.peak = self.p.memory_info().rss / 2**20
        self.t = threading.Thread(target=self._loop, daemon=True); self.t.start(); return self

    def _loop(self):
        while self.run:
            self.peak = max(self.peak, self.p.memory_info().rss / 2**20); time.sleep(0.015)

    def __exit__(self, *a):
        self.run = False; self.t.join()


def one(variant, threads, reps, n_prompts):
    import gc
    import numpy as np
    import onnxruntime as ort
    import psutil
    proc = psutil.Process()
    rss = lambda: proc.memory_info().rss / 2**20
    P = json.loads(PROMPTS.read_text()); eos = P["eos"]
    d = ROOT / "models/onnx" / variant
    files = {f.name: f.stat().st_size for f in d.iterdir() if f.is_file() and f.name.startswith("model")}
    r = {"variant": variant, "files": files, "file_size_mb": sum(files.values()) / 2**20,
         "runtime": f"onnxruntime {ort.__version__} CPU EP, {threads} threads, graph opt ALL",
         "device": f"{platform.machine()} {subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True, text=True).stdout.strip()}"}
    # ---- B. loading memory
    gc.collect(); time.sleep(0.5); before = rss()
    with Peak(proc) as pk:
        t0 = time.perf_counter()
        so = ort.SessionOptions(); so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        s = ort.InferenceSession(str(d / "model.onnx"), so, providers=["CPUExecutionProvider"])
        load_ms = (time.perf_counter() - t0) * 1000
    after = rss(); gc.collect(); time.sleep(1.5); settled = rss()
    r["load"] = {"before_mb": before, "after_mb": after, "peak_mb": pk.peak, "load_ms": load_ms, "settled_mb": settled,
                 "released_mb": pk.peak - settled}
    inputs = {i.name: i for i in s.get_inputs()}; outs = [o.name for o in s.get_outputs()]
    states = [n for n in inputs if n.startswith("past_")]
    dt = np.float16 if inputs[states[0]].type == "tensor(float16)" else np.float32
    r["state_dtype"] = dt.__name__

    def zero(n):
        return np.zeros([1 if (i == 0 and not isinstance(x, int)) else (x if isinstance(x, int) else 0)
                         for i, x in enumerate(inputs[n].shape)], dtype=dt)

    def gen(ids, max_new, stop=True):
        cache = {n: zero(n) for n in states}; x = np.array([ids], np.int64); total = len(ids); toks = []
        keep = np.array(1, np.int64); t0 = time.perf_counter(); ttft = None; kv = ssm = 0
        for _ in range(max_new):
            res = dict(zip(outs, s.run(None, {"input_ids": x, "attention_mask": np.ones((1, total), np.int64),
                                              "num_logits_to_keep": keep, **cache})))
            nxt = int(np.argmax(res["logits"][0, -1]))
            if ttft is None:
                ttft = (time.perf_counter() - t0) * 1000; t1 = time.perf_counter()
            for n in states:
                cache[n] = res[n.replace("past_key_values", "present").replace("past_", "present_")]
            toks.append(nxt)
            if stop and nxt == eos:
                break
            x = np.array([[nxt]], np.int64); total += 1
        dec_s = time.perf_counter() - t1
        kv = sum(v.nbytes for k, v in cache.items() if "key_values" in k)
        ssm = sum(v.nbytes for k, v in cache.items() if "key_values" not in k)
        return dict(ttft_ms=ttft, prefill_tps=len(ids) / (ttft / 1000), n=len(toks),
                    decode_tps=(len(toks) - 1) / dec_s if len(toks) > 1 and dec_s > 0 else 0.0, kv_bytes=kv, ssm_bytes=ssm)

    prompts = [p["ids"] for p in P["bench_prompts"]][:n_prompts]
    for p in prompts[:2]:
        gen(p, 48)                                                     # 2 warm-up runs
    base = rss(); runs = []
    with Peak(proc) as pk:
        for _ in range(reps):
            for p in prompts:
                runs.append(gen(p, 48))
    after_inf = rss()
    f = lambda k: [x[k] for x in runs if x[k] > 0]
    stat = lambda v: {"avg": st.mean(v), "std": st.stdev(v) if len(v) > 1 else 0.0, "min": min(v), "max": max(v), "n": len(v)}
    r["prompts"] = {"n_prompts": len(prompts), "repeats": reps, "prompt_tokens": [len(p) for p in prompts],
                    "ttft_ms": stat(f("ttft_ms")), "prefill_tps": stat(f("prefill_tps")), "decode_tps": stat(f("decode_tps")),
                    "gen_tokens": stat([x["n"] for x in runs])}
    r["inference_ram"] = {"baseline_mb": base, "peak_mb": pk.peak, "after_mb": after_inf, "increase_mb": pk.peak - base}
    # ---- D. KV cache at exact context lengths, 32 tokens decoded (EOS ignored), 2 runs averaged
    r["kv_cache"] = {}
    for n, ids in sorted(P["ctx_prompts"].items(), key=lambda kv: int(kv[0])):
        with Peak(proc) as pk:
            g = [gen(ids, 32, stop=False) for _ in range(2)]
        r["kv_cache"][n] = {"prompt_tokens": len(ids), "output_tokens": 32, "kv_mb": g[0]["kv_bytes"] / 2**20,
                            "mamba_state_mb": g[0]["ssm_bytes"] / 2**20, "peak_mb": pk.peak,
                            "ttft_ms": st.mean(x["ttft_ms"] for x in g), "decode_tps": st.mean(x["decode_tps"] for x in g)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{variant}.json").write_text(json.dumps(r, indent=1))
    p = r["prompts"]; kv = r["kv_cache"]
    print(f"{variant}: file {r['file_size_mb']:.0f} MB | load {load_ms:.0f} ms +{after - before:.0f} MB (peak +{r['load']['peak_mb'] - before:.0f}) | "
          f"inf peak {r['inference_ram']['peak_mb']:.0f} MB | TTFT {p['ttft_ms']['avg']:.0f}±{p['ttft_ms']['std']:.0f} ms | "
          f"decode {p['decode_tps']['avg']:.1f} t/s | KV " + "/".join(f"{kv[k]['kv_mb']:.2f}" for k in kv) + " MB", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant"); ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=3); ap.add_argument("--prompts", type=int, default=10)
    a = ap.parse_args()
    if a.variant:
        one(a.variant, a.threads, a.repeats, a.prompts)
    else:
        for v in VARIANTS:                                             # fresh process per variant
            subprocess.run([sys.executable, __file__, "--variant", v, "--threads", str(a.threads),
                            "--repeats", str(a.repeats), "--prompts", str(a.prompts)], check=True)
            time.sleep(30)                                             # cool-down between variants
