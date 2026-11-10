# Granite-4.0-H-350M: where it fails, and the dataset that fixes it

Evidence: zero-shot (no fine-tuning), schema + 7 examples system prompt, validation split of
dataset v1.1.0 (395 commands). Raw outputs: `results/eval/zeroshot_val_granite-4.0-h-350m_schema/predictions.jsonl`.
Analysis output: `results/error_analysis/granite_deep_dive.txt` (script: `scripts/error_analysis.py` + deep-dive snippet).

## Headline
- Exact JSON: 165 / 395 = 41.8%. Failures: 230.
- **192 of 230 failures (83%) contain exactly ONE error** (36 have two, 2 have three).
  These are near misses: fixing one behaviour fixes the whole command.

| Intent | Exact | Failed / total |
|---|---|---|
| open_app | 75% | 14 / 57 |
| set_alarm | 47% | 30 / 57 |
| send_message | 44% | 32 / 57 |
| unsupported | 43% | 32 / 56 |
| make_call | 36% | 35 / 55 |
| create_reminder | 25% | 42 / 56 |
| change_setting | 21% | 45 / 57 |

## Failure scenarios (commands affected)

### G1. Adds fields the user never said: 77
| Sub-case | Count | Real example |
|---|---|---|
| `value` added to increase/decrease | 33 | "can you increase volume?" -> `"action":"increase","value":100` (also 10, 40, 0) |
| `time` invented for reminders | 20 | "Don't let me forget to review the slides" -> `"time":"10:00"` |
| `date` invented for alarms | 14 | "set an alarm for 2 am" -> `"date":"today"` |
| `message` invented | 9 | "Send a message to Sanjay" -> `"message":"I am not the TronOS voice..."` |

### G2. Filler words turned into keys: 21
"Call David on speaker" -> `"action":"on"`; "call deepa on speaker" -> `"speaker":"on"`.

### G3. Wrong intent: 70
| Gold -> predicted | Count | Real example |
|---|---|---|
| unsupported -> set_alarm | 21 | "cancel my 5 pm alarm" (12); "what is the capital of France?" (5); "recharge my phone with 299 plan" (3) -> set_alarm. **set_alarm is the model's default guess when unsure.** |
| make_call -> create_reminder | 12 | "Video call Papa" -> `{"intent":"create_reminder","task":"video call"}` |
| change_setting -> unsupported | 8 | "deactivate night mode" -> unsupported |
| create_reminder -> set_alarm | 7 | "don't let me forget to check my email" -> set_alarm 08:00 |
| open_app -> unsupported | 6 | "Launch Settings", "load telegram" -> unsupported |
| unsupported -> open_app | 4 | "play some music" -> open_app Spotify |
| change_setting -> set_alarm | 4 | "Deactivate silent mode" -> set_alarm |
| unsupported -> create_reminder | 3 | "what's on my calendar today" -> reminder "check calendar" |
| unsupported -> invented intent | 4 | "convert 100 dollars to rupees" -> `"intent":"convert_currency"` |

### G4. 12-hour -> 24-hour time conversion: 24
| Sub-case | Count | Real example (gold -> predicted) |
|---|---|---|
| AM/PM flipped (12 h off) | 12 | "four thirty in the morning" 04:30 -> 16:30; "two in the morning" 02:00 -> 14:00 |
| "half past" | 4 | "half past seven in the evening" 19:30 -> 17:30; "half past two am" 02:30 -> 06:30 (copied example time) |
| "quarter to" | 3 | "quarter to 4 pm" 15:45 -> 14:45 |
| minutes misread | 3 | "two ten pm" 14:10 -> 14:20 |
| not HH:MM | 1 | "10.50am" -> "10.50" |

### G5. Drops fields the user did say: 36
| Field | Count | Real example |
|---|---|---|
| send_message.app | 19 | "Send Papa a message through whatsapp: thanks..." -> no `app` |
| create_reminder.date | 11 | "Remind me to attend the yoga class on friday" -> `"time":"friday"` (weekday put in time) |
| set_alarm.label | 5 | "...on monday for school run" -> no `label` |

### G6. Value not copied faithfully: 25
| Sub-case | Count | Real example (gold -> predicted) |
|---|---|---|
| task paraphrased (drops "the", "my") | 12 | "review the slides" -> "review slides"; "check my email" -> "check email" |
| app name not canonical | 8 | "file manager" -> "File Manager" (Files); "phone pe" -> "Phone" (PhonePe) |
| message rewritten | 4 | "are you free tonight" -> "I am free tonight"; "lunch at one" -> "lunch" |
| contact changed | 2 | "Mummy" -> "Mom" (prompt example); "Keerthi" -> "Kerthi" |

### G7. Value format: 2
"this saturday" -> `"date":"this saturday"` (should be "saturday"); "10.50am" -> "10.50".

## Measurement problems in our v1.1 validation set
- make_call: 33 of 35 failures come from **two** templates ("on speaker" 20, "video call" 13); plain "call X" is fine.
- unsupported: 56 commands from only 10 templates.
- Per-intent scores are therefore driven by a few phrasings -> v1.2 must spread templates evenly.

## Dataset rules for Granite (v1.2)
| Rule | Fixes | What goes into the data |
|---|---|---|
| R1 Say only what was said | G1 | Minimal pairs with/without every optional field; >= 35% of each intent has no optional field; increase/decrease NEVER carry `value`, only "set ... to N" does; reminders without time; alarms without date; "send a message to X" without message |
| R2 Ignore filler words | G2 | Every intent gets distractors ("on speaker", "right now", "quickly", "please", "for me") with unchanged target; closed key set per intent |
| R3 Intent boundaries | G3 | Hard negatives for set_alarm look-alikes (cancel/delete/snooze/list alarms, "what time is my alarm"), general Q&A, payments/recharge, calendar queries, play music without app, currency; positives for "video call X" = make_call, "deactivate/disable/kill X" = change_setting off, "launch/load/fire up X" = open_app, "don't let me forget to X" = create_reminder. Unsupported generated from slot templates across >= 15 categories |
| R4 Time table | G4 | Every spoken time form x AM/PM balanced, >= 20 train samples each: "h:mm am/pm", "h.mm am", word times ("four thirty in the morning", "two ten pm"), "half past", "quarter past/to", "noon/midnight", 24h, "12 am/pm" |
| R5 Keep every stated field | G5 | Full-slot examples with varied prepositions ("on/via/through/using whatsapp"), weekday dates in reminders ("on friday"), alarm labels in several positions |
| R6 Verbatim copy | G6 | task/message = exact spoken words (only leading "to"/"that"/"saying" removed); canonical app dictionary with all aliases ("file manager"->Files, "phone pe"->PhonePe); contacts copied as spoken (Mummy stays Mummy) |
| R7 Normalised formats | G7 | "this/next/on saturday" -> "saturday"; all times "HH:MM" |
| R8 Even, stratified splits | measurement | Split templates by (intent, field set); cap each template's share of val/test; every field combination in train/val/test |
| R10 Match real STT output | STT tests | Moonshine small writes digits, casing and punctuation ("Set an alarm for 6:30 am tomorrow.", "Open YouTube and play the latest song"); spelled numbers are rare. Majority of commands in this style (+ a lowercase/no-punctuation share for robustness); a few realistic STT slips (inserted "the", dropped first word) with unchanged targets. Misheard names (Ravi->Navi) are copied as heard; correction happens on Android by fuzzy-matching contacts/apps |
| R9 Keep | - | Template-disjoint splits, test-only names/values, hand-written test set, dedup + near-dup filter, schema + 7 examples prompt in every sample |

## Success check
After fine-tuning on v1.2, re-run `scripts/error_analysis.py` on Granite: counts for G1-G7 must drop;
headline = exact JSON match on validation, final numbers on the held-out test set.
