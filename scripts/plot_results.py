"""Plots for the report: LoRA loss curves (A-D) and quality comparisons.

  python scripts/plot_results.py --runs kaggle_job/out/tronx/runs/granite --eval results/eval
Writes PNGs to results/plots/.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/plots"
EXPS = ["A_qv", "B_attn", "C_mlp", "D_attn_mlp"]
LABEL = {"A_qv": "A: q,v", "B_attn": "B: q,k,v,o", "C_mlp": "C: MLP", "D_attn_mlp": "D: attn+MLP"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--eval", default=str(ROOT / "results/eval"))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    runs, ev = Path(a.runs), Path(a.eval)

    # 1) training + validation loss curves
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    for e in EXPS:
        p = runs / e / "train_log.json"
        if not p.exists():
            continue
        h = json.loads(p.read_text())["history"]
        tr = [(x["step"], x["train_loss"]) for x in h if "train_loss" in x]
        va = [(x["epoch"], x["val_loss"]) for x in h if "val_loss" in x]
        if tr:
            ax[0].plot(*zip(*tr), label=LABEL[e])
        lr = [(x["step"], x["lr"]) for x in h if "lr" in x]
        if lr:
            ax[2].plot(*zip(*lr), label=LABEL[e])
        vs = [(x["step"], x["val_loss_step"]) for x in h if "val_loss_step" in x]
        if vs:   # detailed validation curve (runs logged with val_every)
            ax[1].plot(*zip(*vs), marker=".", label=LABEL[e] + " (per step)")
        if va and not vs:
            ax[1].plot([0] + [v[0] for v in va], [json.loads(p.read_text())["val_loss_before"]] + [v[1] for v in va], marker="o", label=LABEL[e])
    ax[0].set(title="Training loss (JSON tokens only)", xlabel="optimizer step", ylabel="cross-entropy"); ax[0].legend()
    ax[1].set(title="Validation loss (measured per epoch)", xlabel="epoch", ylabel="cross-entropy"); ax[1].legend()
    ax[2].set(title="Learning rate (5% warm-up + cosine decay)", xlabel="optimizer step", ylabel="learning rate"); ax[2].legend()
    ax[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.tight_layout(); fig.savefig(OUT / "lora_training_curves.png", dpi=150); plt.close(fig)

    # 2) quality comparison: base vs A-D on test
    rows = [("Base (no FT)", ev / "granite_v12_base_test/metrics.json")] + \
           [(LABEL[e], ev / f"granite_v12_lora{e}_test/metrics.json") for e in EXPS]
    rows = [(n, json.loads(p.read_text())) for n, p in rows if p.exists()]
    if rows:
        metrics = [("exact_match", "Exact JSON"), ("intent_accuracy", "Intent"), ("field_accuracy", "Fields"),
                   ("json_validity", "Valid JSON"), ("unsupported_recall", "Unsupported recall")]
        fig, ax = plt.subplots(figsize=(11, 4.5))
        w = 0.8 / len(rows)
        for i, (n, m) in enumerate(rows):
            vals = [100 * (m.get(k) or 0) for k, _ in metrics]
            xs = [j + i * w for j in range(len(metrics))]
            b = ax.bar(xs, vals, w, label=n)
            ax.bar_label(b, fmt="%.0f", fontsize=7)
        ax.set_xticks([j + w * (len(rows) - 1) / 2 for j in range(len(metrics))], [t for _, t in metrics])
        ax.set(ylim=(0, 105), ylabel="% (held-out test set)", title="Granite-4.0-H-350M: base vs LoRA experiments A-D")
        ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "lora_quality_test.png", dpi=150); plt.close(fig)
    print("plots ->", OUT, sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
