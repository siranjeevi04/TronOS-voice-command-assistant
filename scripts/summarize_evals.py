"""Print a comparison table from results/eval/<name>/metrics.json.

  python scripts/summarize_evals.py zeroshot_val_          # prefix filter
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
prefix = sys.argv[1] if len(sys.argv) > 1 else ""


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}"


rows = []
for p in sorted((ROOT / "results/eval").glob(f"{prefix}*/metrics.json")):
    m = json.loads(p.read_text())
    rows.append([m["name"], m["n"], pct(m["intent_accuracy"]), pct(m["field_accuracy"]), pct(m["exact_match"]),
                 pct(m["json_validity"]), pct(m["unsupported_recall"]),
                 "-" if m.get("target_loss") is None else f"{m['target_loss']:.3f}"])
hdr = ["run", "n", "intent%", "field%", "exact%", "json_valid%", "unsupported_recall%", "target_loss"]
w = [max(len(str(r[i])) for r in rows + [hdr]) for i in range(len(hdr))]
print(" | ".join(h.ljust(w[i]) for i, h in enumerate(hdr)))
print("-+-".join("-" * x for x in w))
for r in rows:
    print(" | ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))
