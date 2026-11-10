"""Headline figures for the README / report.

  .venv/bin/python scripts/plot_summary.py -> results/plots/{improvement_per_intent,mac_vs_phone_int8}.png
"""
import collections
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/plots"
S1, S2, S3, BASE = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29d"   # FP32 = slot 1, INT8 = slot 3 (as in quant_quality.png)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False})


def per_intent(name):
    c = collections.defaultdict(lambda: [0, 0])
    for r in map(json.loads, open(ROOT / "results/eval" / name / "predictions.jsonl")):
        g = r["gold"]["intent"]; c[g][0] += r["exact"]; c[g][1] += 1
    return {k: 100 * v[0] / v[1] for k, v in c.items()}, {k: v[1] for k, v in c.items()}


# ---- 1. exact JSON match per intent: base FP32 vs fine-tuned FP32 vs fine-tuned INT8
order = ["set_alarm", "create_reminder", "change_setting", "send_message", "make_call", "open_app", "unsupported"]
series = [("Base FP32", "granite_v12_base_test", BASE), ("Fine-tuned FP32 (LoRA D)", "granite_v12_loraD_attn_mlp_test", S1),
          ("Fine-tuned INT8 (deployed)", "onnx_ft_int8_test", S3)]
data = [(n, per_intent(k)[0], c) for n, k, c in series]
counts = per_intent("granite_v12_base_test")[1]
fig, ax = plt.subplots(figsize=(10, 5.2))
h = 0.26
for j, (n, v, c) in enumerate(data):
    ys = [i + (j - 1) * h for i in range(len(order))]
    ax.barh(ys, [v[k] for k in order], height=h, color=c, label=n, edgecolor=SURF, linewidth=1.5)
    for y, k in zip(ys, order):
        ax.text(v[k] + 0.8, y, f"{v[k]:.0f}", va="center", fontsize=8, color=INK2)
ax.set_yticks(range(len(order)), [f"{k}  (n={counts[k]})" for k in order]); ax.invert_yaxis()
ax.set_xlim(0, 108); ax.set_xlabel("exact JSON match, % (held-out test set, 613 commands)")
ax.grid(axis="y", visible=False)
ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=9)
ax.set_title("Before vs after fine-tuning, per intent", loc="left", pad=34)
fig.tight_layout(); fig.savefig(OUT / "improvement_per_intent.png", dpi=160); plt.close(fig)

# ---- 2. same INT8 file, same prompts: Mac vs phone (quality + speed as small multiples, no dual axis)
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.6))
lab = ["Mac M5", "Moto G34"]; col = [S1, S2]
q = {"Exact JSON": (50 / 62 * 100, 32 / 62 * 100), "Valid JSON": (100.0, 54 / 62 * 100)}
x = range(len(q)); w = 0.36
for j in range(2):
    xs = [i + (j - 0.5) * w for i in x]; ys = [v[j] for v in q.values()]
    ax[0].bar(xs, ys, width=w, color=col[j], label=lab[j], edgecolor=SURF, linewidth=1.5)
    for a, b in zip(xs, ys):
        ax[0].text(a, b + 1.5, f"{b:.0f}%", ha="center", fontsize=8, color=INK2)
ax[0].set_xticks(list(x), list(q)); ax[0].set_ylim(0, 112); ax[0].set_title("Quality, 62 commands (%)", loc="left")
ax[0].grid(axis="x", visible=False)
for a, title, vals, fmt in [(ax[1], "Time per command, 21 prompts (s)", {"First token": (0.766, 6.215), "Total": (1.017, 7.915)}, "{:.1f}"),
                            (ax[2], "Decode speed (tok/s)", {"Decode": (74.0, 10.2)}, "{:.0f}")]:
    xx = range(len(vals))
    for j in range(2):
        xs = [i + (j - 0.5) * w for i in xx]; ys = [v[j] for v in vals.values()]
        a.bar(xs, ys, width=w, color=col[j], edgecolor=SURF, linewidth=1.5)
        for p, b in zip(xs, ys):
            a.text(p, b * 1.02 + 0.05, fmt.format(b), ha="center", fontsize=8, color=INK2)
    a.set_xticks(list(xx), list(vals)); a.set_title(title, loc="left"); a.grid(axis="x", visible=False)
fig.suptitle("Same INT8 model file, same prompts: only the device differs (outputs identical on 28 of 62)",
             x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in col], labels=lab, frameon=False, ncol=2,
           loc="upper right", fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(OUT / "mac_vs_phone_int8.png", dpi=160); plt.close(fig)
print("saved")
