"""Shared definitions: output schema, prompts, JSON canonicalisation and metrics.

Every script (dataset, training, evaluation, GGUF evaluation) imports from here so
that the prompt, the canonical JSON format and the scoring rules are identical at
every stage of the pipeline.
"""
from __future__ import annotations

import json
import re
from typing import Any

INTENTS = [
    "set_alarm",
    "create_reminder",
    "open_app",
    "make_call",
    "send_message",
    "change_setting",
    "unsupported",
]

# Field order used when serialising the target JSON (intent always first).
FIELD_ORDER = {
    "set_alarm": ["time", "date", "label", "in"],
    "create_reminder": ["task", "time", "date", "in"],
    "open_app": ["app_name"],
    "make_call": ["contact"],
    "send_message": ["contact", "message", "app"],
    "change_setting": ["setting", "action", "value"],
    "unsupported": [],
}

SETTINGS = [
    "wifi", "bluetooth", "mobile_data", "airplane_mode", "flashlight",
    "do_not_disturb", "brightness", "volume", "dark_mode", "location",
    "hotspot", "battery_saver", "auto_rotate",
]
ACTIONS = ["on", "off", "increase", "decrease", "set"]

# Compact prompt used for training and for every deployed (fine-tuned / quantised)
# model. Kept short on purpose: every system-prompt token costs prefill time on device.
SYSTEM_PROMPT = (
    "You are the TronOS voice command parser. Convert the user's command into one JSON object. "
    "Intents: set_alarm, create_reminder, open_app, make_call, send_message, change_setting, unsupported. "
    "Output only JSON."
)

# Long prompt (schema + few-shot) used only to give the *untrained* base model a fair
# chance. The fine-tuned model does not need it.
SCHEMA_PROMPT = """You are the TronOS voice command parser. Convert the user's command into exactly one JSON object and output nothing else.

Intents and fields (include a field only if the user said it):
- set_alarm: time ("HH:MM", 24-hour; if no am/pm is said, keep the hour as said), date ("today", "tomorrow", "day_after_tomorrow" or a weekday like "monday"), label, in (relative time as "HH:MM" from now, e.g. "in 20 minutes" -> "00:20")
- create_reminder: task (the user's own words), time ("HH:MM"), date, in
- open_app: app_name
- make_call: contact
- send_message: contact, message, app
- change_setting: setting (one of wifi, bluetooth, mobile_data, airplane_mode, flashlight, do_not_disturb, brightness, volume, dark_mode, location, hotspot, battery_saver, auto_rotate), action (on, off, increase, decrease, set), value (integer, only with set)
- unsupported: any other request (payments, shopping, weather, search, deleting things, smart home, ...). Output {"intent": "unsupported"}.
If the command contains several actions, use only the first one; if the first one is unsupported, output {"intent": "unsupported"}.

Examples:
User: Set an alarm for 6:30 AM tomorrow.
{"intent": "set_alarm", "time": "06:30", "date": "tomorrow"}
User: Open YouTube and play latest song.
{"intent": "open_app", "app_name": "YouTube"}
User: Transfer money to Ravi.
{"intent": "unsupported"}
User: remind me to buy milk at 5 pm
{"intent": "create_reminder", "task": "buy milk", "time": "17:00"}
User: text priya that i am on my way
{"intent": "send_message", "contact": "Priya", "message": "i am on my way"}
User: set brightness to 40 percent
{"intent": "change_setting", "setting": "brightness", "action": "set", "value": 40}
User: call mom
{"intent": "make_call", "contact": "Mom"}"""


def build_messages(text: str, prompt: str = "compact") -> list[dict[str, str]]:
    system = SYSTEM_PROMPT if prompt == "compact" else SCHEMA_PROMPT
    return [{"role": "system", "content": system}, {"role": "user", "content": text}]


def canonical_json(obj: dict[str, Any]) -> str:
    """Serialise a target dict with a deterministic key order."""
    intent = obj["intent"]
    out = {"intent": intent}
    for k in FIELD_ORDER.get(intent, []):
        if k in obj and obj[k] is not None:
            out[k] = obj[k]
    return json.dumps(out, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Parsing and scoring
# ---------------------------------------------------------------------------
_JSON_OBJ = re.compile(r"\{.*?\}", re.S)


def parse_output(raw: str) -> tuple[dict | None, bool]:
    """Return (parsed_dict, strictly_valid).

    strictly_valid: the whole generation (after stripping whitespace and an optional
    ```json fence) is a single JSON object. If it is not, we still try to recover the
    first {...} block so that intent/field accuracy can be scored, but the sample counts
    as *invalid JSON* for the JSON-validity metric.
    """
    s = raw.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, re.S)
    if fence:
        s = fence.group(1).strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj, True
    except Exception:
        pass
    m = _JSON_OBJ.search(raw)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj, False
        except Exception:
            pass
    return None, False


def norm_value(v: Any) -> str:
    """Normalise a slot value for comparison: case-, punctuation- and space-insensitive.
    Numbers and numeric strings compare equal ("40" == 40)."""
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, (int, float)):
        return str(int(v)) if float(v).is_integer() else str(v)
    s = str(v).lower().strip()
    s = re.sub(r"[^\w\s:]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def score_sample(pred: dict | None, gold: dict) -> dict[str, Any]:
    gold_fields = {k: v for k, v in gold.items() if k != "intent"}
    res: dict[str, Any] = {
        "intent_correct": False,
        "fields_total": len(gold_fields),
        "fields_correct": 0,
        "fields_spurious": 0,
        "fields_pred": 0,
        "exact": False,
    }
    if pred is None:
        return res
    res["intent_correct"] = pred.get("intent") == gold["intent"]
    pred_fields = {k: v for k, v in pred.items() if k != "intent" and v not in (None, "")}
    for k, v in gold_fields.items():
        if k in pred_fields and norm_value(pred_fields[k]) == norm_value(v):
            res["fields_correct"] += 1
    res["fields_spurious"] = len([k for k in pred_fields if k not in gold_fields])
    res["fields_pred"] = len(pred_fields)
    res["exact"] = (
        res["intent_correct"]
        and res["fields_correct"] == len(gold_fields)
        and res["fields_spurious"] == 0
    )
    return res


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """rows: dicts with keys gold, pred (parsed or None), valid (bool) + score fields."""
    n = len(rows)
    if n == 0:
        return {}
    intent_acc = sum(r["intent_correct"] for r in rows) / n
    ft = sum(r["fields_total"] for r in rows)
    fc = sum(r["fields_correct"] for r in rows)
    fs = sum(r["fields_spurious"] for r in rows)
    exact = sum(r["exact"] for r in rows) / n
    valid = sum(r["valid"] for r in rows) / n
    uns = [r for r in rows if r["gold"]["intent"] == "unsupported"]
    pred_uns = [r for r in rows if (r["pred"] or {}).get("intent") == "unsupported"]
    tp = sum(1 for r in uns if (r["pred"] or {}).get("intent") == "unsupported")
    uns_recall = tp / len(uns) if uns else None
    uns_precision = tp / len(pred_uns) if pred_uns else None
    per_intent = {}
    for it in INTENTS:
        sub = [r for r in rows if r["gold"]["intent"] == it]
        if sub:
            per_intent[it] = {
                "n": len(sub),
                "intent_acc": sum(r["intent_correct"] for r in sub) / len(sub),
                "exact": sum(r["exact"] for r in sub) / len(sub),
            }
    return {
        "n": n,
        "intent_accuracy": intent_acc,
        "field_accuracy": fc / ft if ft else None,  # recall over gold slots
        "field_precision": fc / fp if (fp := sum(r["fields_pred"] for r in rows)) else None,
        "spurious_fields": fs,
        "exact_match": exact,
        "json_validity": valid,
        "unsupported_recall": uns_recall,
        "unsupported_precision": uns_precision,
        "per_intent": per_intent,
    }
