"""Categorise every failure in an eval run (predictions.jsonl) into an error taxonomy.

  python scripts/error_analysis.py zeroshot_val_LFM2.5-350M_schema zeroshot_val_granite-4.0-h-350m_schema
Writes results/error_analysis/<run>.json and prints a summary with examples.
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import FIELD_ORDER, INTENTS, SETTINGS, ACTIONS, norm_value  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# values that appear in the schema prompt's few-shot examples (to detect copying)
FEWSHOT_VALUES = {"06:30", "tomorrow", "YouTube", "buy milk", "17:00", "Priya", "i am on my way",
                  "brightness", "40", "Mom"}
HHMM = re.compile(r"^\d{2}:\d{2}$")


def classify(r):
    """Return list of error tags for one prediction row."""
    tags = []
    gold, pred = r["gold"], r["pred"]
    if pred is None:
        return ["invalid_json"]
    if not r["valid"]:
        tags.append("extra_text_around_json")
    gi, pi = gold["intent"], pred.get("intent")
    if pi is None:
        keys = ",".join(sorted(pred))
        tags.append(f"no_intent_key[{keys[:40]}]")
        return tags
    if pi not in INTENTS:
        tags.append(f"invented_intent[{pi}]")
    if gi != pi:
        tags.append(f"wrong_intent[{gi}->{pi}]")
        return tags
    gf = {k: v for k, v in gold.items() if k != "intent"}
    pf = {k: v for k, v in pred.items() if k != "intent" and v not in (None, "")}
    allowed = set(FIELD_ORDER[gi])
    for k, v in gf.items():
        if k not in pf:
            tags.append(f"missing_field[{gi}.{k}]")
        elif norm_value(pf[k]) != norm_value(v):
            pv = pf[k]
            sub = "wrong_value"
            if k == "time" and isinstance(pv, str) and not HHMM.match(pv):
                sub = "time_not_HHMM"
            elif k == "time":
                sub = "time_wrong_HHMM"
            elif k == "date":
                sub = "date_wrong"
            elif k in ("task", "message") and norm_value(v) in norm_value(pv):
                sub = "text_slot_too_long"
            elif k in ("task", "message") and norm_value(pv) in norm_value(v):
                sub = "text_slot_truncated"
            elif k == "value" and isinstance(pv, str):
                sub = "value_not_integer"
            if str(pv) in FEWSHOT_VALUES and str(v) not in FEWSHOT_VALUES:
                sub += "+copied_from_example"
            tags.append(f"{sub}[{gi}.{k}]")
    for k in pf:
        if k not in gf:
            if k not in allowed:
                tags.append(f"spurious_unknown_key[{gi}.{k}]")
            else:
                tags.append(f"spurious_hallucinated_field[{gi}.{k}]")
    if gi == "change_setting":
        if "setting" in pf and pf["setting"] not in SETTINGS:
            tags.append("setting_not_in_enum")
        if "action" in pf and pf["action"] not in ACTIONS:
            tags.append("action_not_in_enum")
    return tags


def family(tag):
    return tag.split("[")[0]


def main():
    out_dir = ROOT / "results/error_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    for run in sys.argv[1:]:
        rows = [json.loads(line) for line in (ROOT / f"results/eval/{run}/predictions.jsonl").open()]
        fails = [r for r in rows if not r["exact"]]
        fam, full, ex = Counter(), Counter(), defaultdict(list)
        per_intent_fail = Counter(r["gold"]["intent"] for r in fails)
        per_src = Counter((r["source"], r["exact"]) for r in rows)
        for r in fails:
            tags = classify(r) or ["other"]
            r["error_tags"] = tags
            for t in set(tags):
                fam[family(t)] += 1
                full[t] += 1
                if len(ex[t]) < 3:
                    ex[t].append({"text": r["text"], "gold": r["gold"], "raw": r["raw"][:160]})
        res = {"run": run, "n": len(rows), "failures": len(fails),
               "fail_by_intent": dict(per_intent_fail),
               "by_source": {f"{s}|{'exact' if e else 'fail'}": c for (s, e), c in per_src.items()},
               "error_families": dict(fam.most_common()), "error_tags": dict(full.most_common()),
               "examples": ex}
        (out_dir / f"{run}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print(f"\n===== {run}: {len(fails)}/{len(rows)} not exact")
        print("fail by intent:", dict(per_intent_fail.most_common()))
        print("error families (samples affected):")
        for k, v in fam.most_common():
            print(f"  {v:4d}  {k}")
        print("top tags:")
        for k, v in full.most_common(25):
            e = ex[k][0]
            print(f"  {v:4d}  {k:55s} e.g. '{e['text'][:45]}' -> {e['raw'][:90]}")


if __name__ == "__main__":
    main()
