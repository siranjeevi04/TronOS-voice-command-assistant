# Dataset design v1.2: derived from zero-shot failure analysis

Source of evidence: zero-shot runs of LFM2.5-350M and Granite-4.0-H-350M with the
schema + 7 examples system prompt on the validation split (395 commands), dataset v1.1.0.
Exact JSON match: LFM 29.6% (278 failures), Granite 41.8% (230 failures).
Raw data: `results/eval/zeroshot_val_*_schema/predictions.jsonl`,
`results/error_analysis/*.json` (script: `scripts/error_analysis.py`).
Gold labels of the failing samples were spot-checked: no label errors found.

## Failure scenarios (samples affected: LFM / Granite)

| # | Scenario | LFM | Granite | Real examples (command -> model output) |
|---|---|---|---|---|
| F1 | Hallucinated optional fields (adds what the user never said) | 92 | 77 | "can you increase volume?" -> `"action":"set","value":100`; "set an alarm for 2 am" -> adds `"date":"tomorrow"`; "Remind me to review the slides" -> adds `"time":"10:00"`; "Send a message to Sanjay" -> invents `"message":"hey","app":"WhatsApp"` |
| F2 | Wrong intent on look-alike commands | 84 | 70 | "cancel my 5 pm alarm" -> set_alarm; "what's on my calendar today" -> create_reminder; "play some music" -> open_app; "Don't let me forget to review the slides" -> send_message; "video call Manoj" -> open_app; "Launch Zomato" -> make_call; "deactivate night mode" -> unsupported |
| F3 | Missing stated fields | 93 | 36 | "...on monday for school run" -> label dropped; "message through whatsapp: thanks" -> app dropped; "remind me to submit the report at 05:15" -> task dropped |
| F4 | Time normalisation errors (12h -> 24h) | 18 | 21 | "half past seven in the evening" -> 17:30 (gold 19:30); "quarter to 2 pm" -> 14:45 (13:45); "two ten pm" -> 14:20 / 20:00 (14:10); "four ten am" -> 16:10 (04:10); "10.50am" -> "10.50" |
| F5 | Keys outside the schema / invented intents | 28 | 25 | unsupported -> `"action":"change_setting"`; "Call David on speaker" -> `"action":"on"`; "convert 100 dollars" -> `"intent":"convert_currency"` |
| F6 | Copying values from the prompt's examples | 31 | 2 | "...on monday" -> `"date":"tomorrow"` (the example's date); "recharge my phone with 299 plan" -> make_call `"contact":"Mom"` |
| F7 | Non-canonical values | 6 | 8+ | "open file manager" -> `"File Manager"` (canonical "Files"); task "review slides" vs spoken "review the slides" |

## Problems found in OUR dataset (v1.1.0)

| # | Problem | Evidence | Effect |
|---|---|---|---|
| D1 | Unsupported commands come from a few fixed sentences with noise variants | val: 56 unsupported from only 10 templates (one template 12x); test: 86 from 10 templates (16x each) | Metric dominated by a few phrasings; poor coverage of out-of-scope requests |
| D2 | Template-level split is not stratified by slot pattern | test has only 1 set_alarm with a label; train has 66 | Some fields barely tested; others barely trained |
| D3 | change_setting "value" rare | only 46 / 333 train samples carry a value | F1 (value hallucinated on increase/decrease) not counter-trained |
| D4 | Optional-slot absence not contrasted | few minimal pairs "with slot" vs "without slot" for the same phrasing | F1 / F3 |

## Dataset rules for v1.2 ("ideology")

1. **Say-only-what-was-said.** Every intent gets minimal-pair examples: the same phrasing
   with and without each optional field (date, label, time, app, message, value).
   Target: >= 35% of each intent's samples omit all optional fields. Fixes F1, F3, D4.
2. **Hard-negative boundaries.** For every supported intent, add look-alike UNSUPPORTED commands
   that reuse its words: cancel/delete/snooze/list alarms, delete/list reminders, read/reply-read
   messages, who called / call history, uninstall/update app, play music/video without an app,
   calendar queries, payments, currency, Q&A, smart-home. Plus cross-intent pairs:
   reminder vs message ("remind me to text Ravi" = reminder), reminder vs alarm,
   video call = make_call, "launch <app>" never make_call. Fixes F2.
3. **Unsupported must be generated, not listed.** Slot-filled templates in >= 15 categories,
   with a cap of 3 noise variants per base sentence. Fixes D1.
4. **Time-expression coverage table.** Every surface form x AM/PM balanced:
   h:mm am/pm, h.mm am, hmm-words ("two ten pm"), "half past", "quarter past/to",
   "in the morning/evening/at night", 24h, noon/midnight, 12 am / 12 pm edge cases.
   Each form >= 20 train samples. Fixes F4.
5. **Closed key set.** Only schema keys ever appear in targets; unsupported is always exactly
   `{"intent": "unsupported"}`; add commands with distractor words ("on speaker", "quickly",
   "right now") whose target ignores them. Fixes F5.
6. **Decorrelate from the prompt's examples.** Values used in the 7 prompt examples
   (06:30, tomorrow, YouTube, Priya, Mom, buy milk, 40) are capped at their natural frequency,
   and the example sentences re-appear in training with *different* values. Fixes F6.
7. **Canonical-value dictionary.** App aliases -> one canonical name; setting synonyms -> enum;
   task/message = the spoken words verbatim minus the leading "to"/"that". Fixes F7.
8. **Stratified split by slot signature.** Split templates per (intent, set of fields), so every
   field combination appears in train, val and test. Fixes D2.
9. **Keep what already works:** template-disjoint splits, test-only slot values, hand-written
   test set, dedup + near-dup filter, and the schema + examples system prompt in every
   training sample (same prompt as inference).

## How we will check the new dataset worked
- Re-run the same zero-shot baseline on v1.2 val/test (baseline numbers change because the split changes).
- After fine-tuning: per-scenario error counts F1-F7 must fall; headline metric = exact JSON match.
