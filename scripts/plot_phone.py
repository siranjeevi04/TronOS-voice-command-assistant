"""Report figures from the on-device (Moto G34, INT8) Section 9 run and the quantisation evals.

  .venv/bin/python scripts/plot_phone.py   -> results/plots/{phone_thermal,phone_kv_cache,quant_quality}.png
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/plots"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"          # categorical slots 1-3 (validated reference palette)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.grid.axis": "y", "axes.spines.top": False, "axes.spines.right": False})

d = json.loads((ROOT / "results/phone/int8_section9_20261110.json").read_text())


def mhz(s):
    """'1708x6,2054x2' / '1516x3,940x3,2054x2' -> (little avg MHz, big avg MHz)"""
    v = []
    for part in s.split(","):
        f, n = (part.split("x") + ["1"])[:2]
        v += [int(f)] * int(n)
    return sum(v[:6]) / 6, sum(v[6:]) / 2


# ---- 1. thermal: 3 small multiples sharing time (no dual axis)
rows = d["G_thermal"]["rows"]
t = [r[0] / 60 for r in rows]; tps = [r[1] for r in rows]; temp = [r[3] for r in rows]
lit = [mhz(r[7])[0] for r in rows]; big = [mhz(r[7])[1] for r in rows]
severe = [r[0] / 60 for r in rows if r[4] == "SEVERE"]
fig, ax = plt.subplots(3, 1, figsize=(9, 7.2), sharex=True, gridspec_kw={"height_ratios": [1.3, 1, 0.8]})
for a in ax:
    if severe:
        a.axvspan(min(severe) - 0.25, max(severe) + 0.25, color="#f3d9d0", lw=0, zorder=0)
ax[0].plot(t, tps, color=S1, lw=2, marker="o", ms=4)
ax[0].set_ylabel("decode tok/s"); ax[0].set_ylim(0, 12)
ax[0].set_title("Fine-tuned INT8 on Moto G34 5G: 10 min sustained inference (76 generations)", loc="left")
ax[0].annotate(f"start {tps[0]:.2f}", (t[0], tps[0]), xytext=(4, 8), textcoords="offset points", color=INK2, fontsize=9)
ax[0].annotate(f"end {tps[-1]:.2f} (-8%)", (t[-1], tps[-1]), xytext=(-80, 16), textcoords="offset points", color=INK2, fontsize=9)
i = tps.index(min(tps))
ax[0].annotate("big cores capped to 940 MHz", (t[i], tps[i]), xytext=(-150, -4), textcoords="offset points", color=INK2,
               fontsize=9, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax[0].text(min(severe), 1.0, "thermal status SEVERE", color=INK2, fontsize=9) if severe else None
ax[1].plot(t, big, color=S1, lw=2, label="2x Cortex-A78 (big)")
ax[1].plot(t, lit, color=S2, lw=2, label="6x Cortex-A55 (little)")
ax[1].set_ylabel("CPU MHz"); ax[1].set_ylim(0, 2400); ax[1].legend(loc="lower left", frameon=False, fontsize=9)
ax[2].plot(t, temp, color=S3, lw=2, marker="o", ms=4)
ax[2].set_ylabel("battery °C"); ax[2].set_ylim(34.5, 37); ax[2].set_xlabel("minutes")
fig.tight_layout(); fig.savefig(OUT / "phone_thermal.png", dpi=160); plt.close(fig)

# ---- 2. KV cache vs context: 3 panels
kv = d["D_kv"]; ctx = ["512", "1024", "2048"]; lab = ["512", "1,024", "2,048"]
fig, ax = plt.subplots(1, 3, figsize=(10, 3.4))
ax[0].bar(lab, [kv[c]["kv_mb"] for c in ctx], color=S1, width=0.55, label="KV cache (4 attention layers)")
ax[0].bar(lab, [kv[c]["mamba_mb"] for c in ctx], bottom=[kv[c]["kv_mb"] for c in ctx], color=S2, width=0.55,
          label="Mamba2 state (28 layers, fixed)", edgecolor=SURF, linewidth=2)
ax[0].set_title("State memory (MB)", loc="left"); ax[0].legend(frameon=False, fontsize=8, loc="upper left"); ax[0].set_ylim(0, 50)
for k, c in enumerate(ctx):
    ax[0].text(k, kv[c]["kv_mb"] + kv[c]["mamba_mb"] + 1, f"{kv[c]['kv_mb']:.1f} + {kv[c]['mamba_mb']:.1f}", ha="center", fontsize=8, color=INK2)
for a, key, title, f in [(ax[1], "peak_mb", "Peak process RAM (MB)", "{:.0f}"), (ax[2], "ttft_ms", "Time to first token (s)", "{:.1f}")]:
    v = [kv[c][key] / (1000 if key == "ttft_ms" else 1) for c in ctx]
    a.bar(lab, v, color=S1, width=0.55); a.set_title(title, loc="left")
    for k, x in enumerate(v):
        a.text(k, x * 1.02, f.format(x), ha="center", fontsize=8, color=INK2)
for a in ax:
    a.set_xlabel("context tokens (prompt, + 32 decoded)")
fig.suptitle("KV cache vs context, fine-tuned INT8 on Moto G34 5G (decode stays 8.6-9.8 tok/s)", x=0.01, ha="left",
             fontsize=11, fontweight="bold", color=INK)
fig.tight_layout(); fig.savefig(OUT / "phone_kv_cache.png", dpi=160); plt.close(fig)

# ---- 3. quantisation quality (test set): FP32 / (FP16 if available) / INT8 / INT4
var = [("FP32", "onnx_ft_fp32_test"), ("FP16", "onnx_ft_fp16_test"), ("INT8", "onnx_ft_int8_test"), ("INT4", "onnx_ft_int4_test")]
var = [(n, json.loads((ROOT / "results/eval" / k / "metrics.json").read_text())) for n, k in var
       if (ROOT / "results/eval" / k / "metrics.json").exists()]
mets = [("exact_match", "Exact JSON"), ("intent_accuracy", "Intent"), ("field_accuracy", "Fields"),
        ("json_validity", "Valid JSON"), ("unsupported_recall", "Unsupported recall")]
COL = {"FP32": S1, "FP16": S2, "INT8": S3, "INT4": "#eda100"}          # colour follows the variant, not its rank
w = 0.8 / len(var)
fig, ax = plt.subplots(figsize=(10, 3.8))
for j, (n, m) in enumerate(var):
    xs = [i + (j - (len(var) - 1) / 2) * w for i in range(len(mets))]
    ys = [m[k] * 100 for k, _ in mets]
    ax.bar(xs, ys, width=w, color=COL[n], label=n, edgecolor=SURF, linewidth=2)
    for x, y in zip(xs, ys):
        ax.text(x, y + 1, f"{y:.0f}", ha="center", fontsize=8, color=INK2)
ax.set_xticks(range(len(mets)), [l for _, l in mets]); ax.set_ylim(0, 110); ax.set_ylabel("% (held-out test, 613)")
ax.legend(frameon=False, ncol=len(var), loc="upper left", fontsize=9)
ax.set_title("Fine-tuned model after quantisation (ONNX Runtime)", loc="left")
fig.tight_layout(); fig.savefig(OUT / "quant_quality.png", dpi=160); plt.close(fig)
print("saved", [p.name for p in OUT.glob("*.png")])
