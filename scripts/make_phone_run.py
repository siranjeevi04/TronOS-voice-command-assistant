"""Pick 30 test commands for the on-phone accuracy run ("Run 30" button in the voice app).

Selection (deterministic, seed 7), from the 613-command held-out test set:
  * only commands the BASE model got wrong (medium-hard: fine-tuning was needed),
  * stratified over the 7 intents (5/5/4/5/3/3/5),
  * mixed outcomes on the Mac for the same INT8 file: 26 passed + 4 failed (~ its 86% test accuracy),
  * none of the 32 commands already run on the phone (results/phone/*), so phone total = 62 distinct commands,
  * about one third hand-written commands.

  python scripts/make_phone_run.py   -> android_voice/app/src/main/assets/phone_run.json
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "android_voice/app/src/main/assets/phone_run.json"


def load(name):
    return {r["id"]: r for r in map(json.loads, open(ROOT / "results/eval" / name / "predictions.jsonl"))}


base, int8, fp32 = load("granite_v12_base_test"), load("onnx_ft_int8_test"), load("onnx_ft_fp32_test")
seen = {x["text"].strip().lower() for x in json.loads((ROOT / "results/phone/mac_vs_phone_int8.json").read_text())["items"]}
seen |= {x["text"].strip().lower() for x in json.loads((ROOT / "results/phone/int8_bench_chat2_mac_vs_phone.json").read_text())}
rng = random.Random(7)
per_intent = {"set_alarm": 5, "create_reminder": 5, "change_setting": 4, "send_message": 5,
              "make_call": 3, "open_app": 3, "unsupported": 5}
fails = {"set_alarm": 1, "create_reminder": 1, "send_message": 1, "make_call": 1}   # Mac-INT8 failures per intent

pick = []
for intent, n in per_intent.items():
    pool = [i for i in int8 if int8[i]["gold"]["intent"] == intent and not base[i]["exact"]
            and 3 <= len(int8[i]["text"].split()) <= 16 and int8[i]["text"].strip().lower() not in seen]
    passed = sorted(i for i in pool if int8[i]["exact"])
    failed = sorted(i for i in pool if not int8[i]["exact"])
    k_fail = fails.get(intent, 0)
    chosen = rng.sample(failed, k_fail)
    hw = [i for i in passed if int8[i]["source"] == "handwritten"]
    syn = [i for i in passed if int8[i]["source"] != "handwritten"]
    k_hw = min(len(hw), (n - k_fail) // 3 + (1 if (n - k_fail) % 3 == 2 else 0))
    chosen += rng.sample(hw, k_hw) + rng.sample(syn, n - k_fail - k_hw)
    pick += chosen

rng.shuffle(pick)
items = [{"n": k + 1, "id": i, "text": int8[i]["text"], "source": int8[i]["source"], "gold": int8[i]["gold"],
          "mac_int8_raw": int8[i]["raw"].strip(), "mac_int8_exact": int8[i]["exact"],
          "mac_fp32_exact": fp32[i]["exact"], "base_fp32_exact": base[i]["exact"]} for k, i in enumerate(pick)]
OUT.write_text(json.dumps({"description": "30 held-out test commands (base wrong; Mac INT8 26 pass / 4 fail), scripts/make_phone_run.py",
                           "mac_int8_correct": sum(x["mac_int8_exact"] for x in items),
                           "mac_fp32_correct": sum(x["mac_fp32_exact"] for x in items),
                           "base_fp32_correct": sum(x["base_fp32_exact"] for x in items), "items": items}, ensure_ascii=False, indent=1))
print(f"{len(items)} items -> {OUT} | Mac INT8 {sum(x['mac_int8_exact'] for x in items)}/{len(items)}, "
      f"FP32 {sum(x['mac_fp32_exact'] for x in items)}/{len(items)}, base {sum(x['base_fp32_exact'] for x in items)}/{len(items)}, "
      f"hand-written {sum(x['source'] == 'handwritten' for x in items)}")
for x in items:
    print(("PASS " if x["mac_int8_exact"] else "FAIL ") + x["gold"]["intent"].ljust(16) + x["text"])
