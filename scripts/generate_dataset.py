"""Synthetic + hand-written dataset generator for the TronOS voice-command parser.

Dataset v1.1 generator. Superseded by generate_dataset_v12.py (dataset v1.2, used for all training and test results);
kept because its validation split (395 commands) was used for the zero-shot model bake-off (report Section 3.5).

Design (see report Section 4 for the full discussion):
  * Every utterance comes from a *template* (string with slots). Template IDs are
    partitioned per intent into disjoint train / val / test pools, so test phrasings
    are never seen during training or hyper-parameter selection.
  * Slot values (contacts, apps, reminder tasks, message bodies) are partitioned too:
    a block of values is reserved for test only.
  * ASR-style noise (lower-casing, punctuation loss, wake words, fillers, politeness)
    is applied after slot filling.
  * Exact and normalised duplicates are removed globally, then a near-duplicate
    filter (token Jaccard >= 0.9) drops any val/test sample too close to a train sample.
  * A small hand-written challenge set (data/raw/handwritten.jsonl) is added to test
    only; it contains phrasings written independently of the templates.

Usage:  python scripts/generate_dataset.py --seed 13
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import INTENTS, canonical_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GENERATOR_VERSION = "1.1.0"  # 1.1.0: English only (STT = Moonshine, English)

# ---------------------------------------------------------------------------
# Slot value pools. The last block of every list is reserved for TEST ONLY.
# ---------------------------------------------------------------------------
CONTACTS = [
    "Mom", "Dad", "Ravi", "Priya", "Arjun", "Ananya", "Rahul", "Sneha", "Karthik", "Divya",
    "Vikram", "Meera", "Suresh", "Lakshmi", "Aditya", "Pooja", "Ganesh", "Kavya", "Rohan", "Nisha",
    "John", "Sarah", "David", "Emily", "Michael", "Jessica", "Daniel", "Sophia", "Kevin", "Olivia",
    "Grandma", "Grandpa", "Anil", "Deepa", "Manoj", "Shalini", "Venkat", "Harini", "Sanjay", "Keerthi",
    "Amma", "Appa", "Akka", "Anna", "Thatha", "Paati", "Mummy", "Papa", "Bhaiya", "Didi",
    # --- test-only block (12) ---
    "Siddharth", "Revathi", "Imran", "Fatima", "Joseph", "Maria", "Balaji", "Gayathri",
    "Ashok", "Uncle Raju", "Aunty Geetha", "Nikhil",
]
CONTACTS_TEST_ONLY = 12

APPS = [
    ("YouTube", ["youtube", "you tube", "YouTube"]),
    ("WhatsApp", ["whatsapp", "whats app", "WhatsApp"]),
    ("Instagram", ["instagram", "insta", "Instagram"]),
    ("Spotify", ["spotify", "Spotify"]),
    ("Google Maps", ["google maps", "maps", "Google Maps"]),
    ("Gmail", ["gmail", "my email", "Gmail"]),
    ("Camera", ["camera", "the camera", "Camera"]),
    ("Settings", ["settings", "phone settings", "Settings"]),
    ("Chrome", ["chrome", "the browser", "Chrome"]),
    ("Calculator", ["calculator", "the calculator", "Calculator"]),
    ("Calendar", ["calendar", "my calendar", "Calendar"]),
    ("Gallery", ["gallery", "photos", "Gallery"]),
    ("Clock", ["clock", "the clock app", "Clock"]),
    ("Netflix", ["netflix", "Netflix"]),
    ("Facebook", ["facebook", "fb", "Facebook"]),
    ("Telegram", ["telegram", "Telegram"]),
    ("Amazon", ["amazon", "the amazon app", "Amazon"]),
    ("Flipkart", ["flipkart", "Flipkart"]),
    ("Play Store", ["play store", "the play store", "Play Store"]),
    ("Phonepe", ["phonepe", "phone pe", "PhonePe"]),
    ("Zomato", ["zomato", "Zomato"]),
    ("Swiggy", ["swiggy", "Swiggy"]),
    ("Uber", ["uber", "Uber"]),
    ("Files", ["files", "file manager", "Files"]),
    # --- test-only block (6) ---
    ("Hotstar", ["hotstar", "Hotstar"]),
    ("LinkedIn", ["linkedin", "linked in", "LinkedIn"]),
    ("Snapchat", ["snapchat", "Snapchat"]),
    ("Ola", ["ola", "Ola"]),
    ("Paytm", ["paytm", "Paytm"]),
    ("Notes", ["notes", "my notes", "Notes"]),
]
APPS_TEST_ONLY = 6

TASKS = [
    "buy milk", "call the plumber", "pay the electricity bill", "take my medicine", "drink water",
    "submit the report", "water the plants", "pick up the kids", "book train tickets", "go for a walk",
    "send the invoice", "check the oven", "feed the dog", "renew my insurance", "charge the car",
    "join the team meeting", "buy vegetables", "return the library books", "wish Ravi happy birthday",
    "pack my bag", "collect the laundry", "recharge my phone", "attend the yoga class", "lock the door",
    "take out the trash", "review the slides", "call the bank", "buy a gift for mom", "pay rent",
    "update my resume", "finish the assignment", "book a doctor appointment", "check my email",
    "clean the room", "order groceries", "transfer the files", "practice guitar", "read the newspaper",
    "visit grandma", "prepare for the interview", "back up my phone", "stretch", "start the washing machine",
    "call Priya", "message the landlord", "fill the water bottle", "turn off the geyser",
    # --- test-only block (12) ---
    "get the car serviced", "buy flowers", "pay the gas bill", "print the boarding pass",
    "defrost the chicken", "call the electrician", "check the mail box", "apply for leave",
    "buy batteries", "pick up the parcel", "take the cake out", "meditate",
]
TASKS_TEST_ONLY = 12

MESSAGES = [
    "i will be late", "i'm on my way", "call me back", "running ten minutes late", "happy birthday",
    "see you tomorrow", "i reached home", "what time is the meeting", "dinner is ready", "good morning",
    "i'm in a meeting", "can you pick me up", "thanks for the help", "let's meet at 5",
    "where are you", "don't forget the keys", "the train is delayed", "i'll call you later",
    "please send the document", "congratulations", "i am stuck in traffic", "lunch at one",
    "bring some snacks", "the parcel has arrived", "good night", "i miss you", "are you free tonight",
    "on the way to the office", "i forgot my wallet", "start without me", "meeting is cancelled",
    "the kids are asleep", "buy bread on your way", "my phone battery is low", "reached safely",
    "i'll be there in five minutes", "ok sounds good", "please call when free", "flight landed",
    "welcome home",
    # --- test-only block (10) ---
    "keep the door open", "the doctor said it's fine", "save me a seat", "i paid the bill",
    "check your email", "we won the match", "leaving the office now", "get well soon",
    "the movie starts at 7", "can we reschedule",
]
MESSAGES_TEST_ONLY = 10

MSG_APPS = [("WhatsApp", ["on whatsapp", "via whatsapp", "through whatsapp"]),
            ("Telegram", ["on telegram", "via telegram"]),
            ("SMS", ["by sms", "as an sms", "via text"])]

ALARM_LABELS = ["gym", "wake up", "meeting", "medicine", "school run", "flight", "workout", "cooking"]

HINGLISH_TASKS = ["dawai leni hai", "doodh lana hai", "bill bharna hai", "mummy ko call karna hai",
                  "paani peena hai", "report bhejni hai", "sabzi lani hai", "gas band karni hai"]
HINGLISH_MSGS = ["main late aaunga", "main raste mein hoon", "khana ready hai", "mujhe call karo",
                 "kal milte hain", "ghar pahunch gaya", "thoda late ho jayega", "meeting cancel hai"]
TANGLISH_MSGS = ["naan late aaven", "vandhutten", "saapadu ready", "call pannu", "naalaikku paakalaam",
                 "veetuku vandhutten"]


def split_pool(values, test_only: int):
    """Return (train_pool, test_pool). Test pool = all values (seen + reserved)."""
    return values[:-test_only], values


# ---------------------------------------------------------------------------
# Time / date surface forms -> canonical value
# ---------------------------------------------------------------------------
NUM_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
             "eleven", "twelve"]
MIN_WORDS = {0: "", 5: "oh five", 10: "ten", 15: "fifteen", 20: "twenty", 25: "twenty five",
             30: "thirty", 40: "forty", 45: "forty five", 50: "fifty"}


def _hm(h24: int, m: int) -> str:
    return f"{h24:02d}:{m:02d}"


def time_surface(rng: random.Random, lang: str = "en"):
    """Return (surface_text, 'HH:MM')."""
    if lang == "hi":
        part, lo, hi, off = rng.choice([("subah", 5, 11, 0), ("shaam", 5, 8, 12), ("raat", 8, 11, 12),
                                        ("dopahar", 1, 3, 12)])
        h = rng.randint(lo, hi)
        m = rng.choice([0, 0, 30, 15])
        mm = {0: "", 30: " saade", 15: " sava"}[m]
        if m == 30:
            s = f"{part} saade {h} baje"
        elif m == 15:
            s = f"{part} sava {h} baje"
        else:
            s = rng.choice([f"{part} {h} baje", f"{h} baje {part}"])
        del mm
        return s, _hm(h + off, m)
    if lang == "ta":
        part, lo, hi, off = rng.choice([("kaalai", 5, 11, 0), ("saayangalam", 4, 7, 12),
                                        ("raathiri", 8, 11, 12), ("madhiyam", 1, 3, 12)])
        h = rng.randint(lo, hi)
        m = rng.choice([0, 0, 30])
        s = f"{part} {h} manikku" if m == 0 else f"{part} {h} {m} ku"
        return s, _hm(h + off, m)

    kind = rng.choices(["hmm", "h", "words", "half", "quarter", "24h", "special"],
                       weights=[30, 20, 15, 8, 8, 10, 4])[0]
    h12 = rng.randint(1, 12)
    pm = rng.random() < 0.45
    ampm_form = rng.choice(["AM", "am", "a.m.", "in the morning"]) if not pm else \
        rng.choice(["PM", "pm", "p.m.", "in the evening" if 4 <= h12 <= 9 else "pm"])
    h24 = (h12 % 12) + (12 if pm else 0)
    if kind == "hmm":
        m = rng.choice([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
        sep = rng.choice([":", ":", "."])
        sp = rng.choice([" ", ""]) if ampm_form in ("AM", "am", "PM", "pm") else " "
        return f"{h12}{sep}{m:02d}{sp}{ampm_form}", _hm(h24, m)
    if kind == "h":
        sp = rng.choice([" ", ""]) if ampm_form in ("AM", "am", "PM", "pm") else " "
        oc = rng.choice(["", " o'clock"]) if "in the" in ampm_form else ""
        return f"{h12}{oc}{sp}{ampm_form}", _hm(h24, 0)
    if kind == "words":
        m = rng.choice(list(MIN_WORDS))
        mw = MIN_WORDS[m]
        word_ampm = ampm_form if ampm_form in ("am", "pm", "in the morning", "in the evening") else ampm_form.lower()
        return f"{NUM_WORDS[h12]}{' ' + mw if mw else ''} {word_ampm}", _hm(h24, m)
    if kind == "half":
        return f"half past {rng.choice([str(h12), NUM_WORDS[h12]])} {ampm_form}", _hm(h24, 30)
    if kind == "quarter":
        if rng.random() < 0.5:
            return f"quarter past {h12} {ampm_form}", _hm(h24, 15)
        # quarter to h12 -> (h12-1):45 in the same half of the day
        prev24 = (h24 - 1) % 24
        return f"quarter to {h12} {ampm_form}", _hm(prev24, 45)
    if kind == "24h":
        H = rng.randint(0, 23)
        m = rng.choice([0, 15, 30, 45, 10, 20, 40, 50])
        return f"{H:02d}:{m:02d}", _hm(H, m)
    return rng.choice([("noon", "12:00"), ("midnight", "00:00"), ("12 noon", "12:00")])


EN_DATES = [("tomorrow", "tomorrow"), ("today", "today"), ("tonight", "today"),
            ("day after tomorrow", "day_after_tomorrow")] + \
           [(f"on {d}", d) for d in ["monday", "tuesday", "wednesday", "thursday", "friday",
                                     "saturday", "sunday"]] + \
           [(f"this {d}", d) for d in ["friday", "saturday", "sunday", "monday"]]
HI_DATES = [("kal", "tomorrow"), ("aaj", "today"), ("parso", "day_after_tomorrow")]
TA_DATES = [("naalaikku", "tomorrow"), ("inniki", "today"), ("naalai", "tomorrow")]


def date_surface(rng, lang="en"):
    pool = {"en": EN_DATES, "hi": HI_DATES, "ta": TA_DATES}[lang]
    s, v = rng.choice(pool)
    return s, v


# ---------------------------------------------------------------------------
# Templates. Each template: (template_id, lang, pattern). Placeholders:
#   {time} {date} {label} {task} {app} {contact} {msg} {msgapp}
# Optional placeholders are expanded by separate template variants rather than
# randomly dropped, so the gold JSON is always derivable from the template.
# ---------------------------------------------------------------------------
T = defaultdict(list)


def add(intent, lang, *patterns):
    for p in patterns:
        T[intent].append((f"{intent}:{len(T[intent]):03d}", lang, p))


add("set_alarm", "en",
    "Set an alarm for {time} {date}.", "Set an alarm for {time}.", "Wake me up at {time} {date}.",
    "Wake me up at {time}.", "Alarm at {time} {date}.", "Can you set an alarm at {time}?",
    "I need an alarm for {time} {date}.", "Please set my alarm to {time}.",
    "Set a {label} alarm for {time}.", "Create an alarm for {time} {date} called {label}.",
    "{date} wake me at {time}.", "Put an alarm on for {time}.", "Alarm {time}.",
    "Set alarm {time} {date}.", "I want to wake up at {time} {date}.",
    "Schedule an alarm for {time} {date} labelled {label}.", "Get me up at {time}.",
    "Make an alarm for {time}.", "Ring an alarm at {time} {date}.", "Add an alarm at {time}.",
    "Could you wake me up {date} at {time}?", "Set my morning alarm for {time}.",
    "Alarm for {label} at {time} {date}.", "Set an alarm.", "I need an alarm.",
    "Set an alarm for {time} and call {contact}.", "Set an alarm for {time} {date} and turn on do not disturb.",
    "New alarm {time}.", "Wake me at {time} please.", "Set the alarm for {time} {date} for {label}.",
    )
add("set_alarm", "hi",
    "{date} {time} ka alarm laga do.", "{time} ka alarm set karo.", "Mujhe {date} {time} utha dena.",
    "{time} pe alarm lagao.", "Alarm laga do {date} {time}.", "Ek alarm set karo {time} ke liye.")
add("set_alarm", "ta",
    "{date} {time} alarm vai.", "{time} ku alarm set pannu.", "{date} {time} ennai ezhuppu.",
    "Alarm vai {time}.")

add("create_reminder", "en",
    "Remind me to {task}.", "Remind me to {task} at {time}.", "Remind me to {task} {date}.",
    "Remind me to {task} {date} at {time}.", "Set a reminder to {task}.",
    "Create a reminder to {task} at {time} {date}.", "Don't let me forget to {task}.",
    "Can you remind me at {time} to {task}?", "{date} at {time} remind me to {task}.",
    "Add a reminder: {task}.", "Reminder to {task} at {time}.", "Make a reminder to {task} {date}.",
    "Please remind me to {task} at {time}.", "I need a reminder to {task}.",
    "Set a reminder for {time} to {task}.", "Remind me {date} to {task}.",
    "Put a reminder to {task} for {time} {date}.", "Ping me at {time} to {task}.",
    "Note to self, {task} {date}.", "Remind me to {task} later at {time}.",
    "Create a reminder.", "Remind me.", "Set a reminder to {task} and open {app}.",
    "Help me remember to {task} at {time}.", "Schedule a reminder {date} at {time} to {task}.",
    "Remind me at {time} {date} that I have to {task}.",
    )
add("create_reminder", "hi",
    "Mujhe {time} yaad dilana ki {task}.", "{date} {time} reminder lagao ki {task}.",
    "Yaad dilana ki {task}.", "{date} yaad dila dena ki {task}.")
add("create_reminder", "ta",
    "{time} ku {task} nu remind pannu.", "{date} {task} nu nyabagam paduthu.")

add("open_app", "en",
    "Open {app}.", "Launch {app}.", "Start {app}.", "Open {app} please.", "Can you open {app}?",
    "Go to {app}.", "Bring up {app}.", "Open the {app} app.", "Fire up {app}.", "I want to use {app}.",
    "Show me {app}.", "Take me to {app}.", "Open {app} and play the latest song.",
    "Open {app} and search for cricket highlights.", "Launch {app} now.", "Run {app}.",
    "Switch to {app}.", "Open up {app}.", "Could you launch {app} for me?", "Pull up {app}.",
    "Open {app} and check my notifications.", "Get {app} open.", "Open the app.",
    "Open {app} and call {contact}.", "Load {app}.",
    )
add("open_app", "hi", "{app} kholo.", "{app} open karo.", "Zara {app} khol do.", "{app} chalu karo.")
add("open_app", "ta", "{app} open pannu.", "{app} thira.", "{app} open pannunga.")

add("make_call", "en",
    "Call {contact}.", "Phone {contact}.", "Ring {contact}.", "Dial {contact}.", "Give {contact} a call.",
    "Make a call to {contact}.", "Can you call {contact}?", "Call {contact} now.", "Please call {contact}.",
    "Place a call to {contact}.", "I want to talk to {contact}, call them.", "Get {contact} on the phone.",
    "Call {contact} on speaker.", "Start a call with {contact}.", "Ring up {contact}.",
    "Could you dial {contact} for me?", "Make a phone call.", "Call {contact} and set an alarm for {time}.",
    "Connect me to {contact}.", "Video call {contact}.", "Call {contact} please.", "Quickly call {contact}.",
    )
add("make_call", "hi", "{contact} ko call karo.", "{contact} ko phone lagao.", "{contact} ko call lagao.",
    "Zara {contact} ko phone karo.")
add("make_call", "ta", "{contact} ku call pannu.", "{contact} ku phone pannu.", "{contact} ah koopdu.")

add("send_message", "en",
    "Send a message to {contact} saying {msg}.", "Text {contact} that {msg}.", "Message {contact}: {msg}.",
    "Tell {contact} {msg}.", "Send {contact} a text saying {msg}.", "Send a message to {contact}.",
    "Text {contact} {msgapp} saying {msg}.", "Send {contact} a message {msgapp}: {msg}.",
    "Can you text {contact} {msg}?", "Drop {contact} a message that {msg}.",
    "Write to {contact} {msg}.", "Message {contact} {msgapp} that {msg}.",
    "Let {contact} know {msg}.", "Send a text to {contact}.", "Shoot a message to {contact} saying {msg}.",
    "Send a message.", "Reply to {contact} with {msg}.", "Send {contact} {msg} {msgapp}.",
    "Text {contact} and tell them {msg}.", "Send a message to {contact} saying {msg} and call {contact}.",
    "Please message {contact} that {msg}.", "Ping {contact} {msgapp} saying {msg}.",
    )
add("send_message", "hi", "{contact} ko message bhejo ki {msg}.", "{contact} ko bolo {msg}.",
    "{contact} ko {msgapp} pe message karo ki {msg}.", "{contact} ko text karo {msg}.")
add("send_message", "ta", "{contact} ku message anuppu {msg}.", "{contact} kitta sollu {msg}.")

# Settings are expressed as (template, setting, action, value) directly.
SETTING_NAMES = {
    "wifi": ["wifi", "wi-fi", "the wifi", "wireless internet"],
    "bluetooth": ["bluetooth", "the bluetooth", "BT"],
    "mobile_data": ["mobile data", "cellular data", "data"],
    "airplane_mode": ["airplane mode", "flight mode", "aeroplane mode"],
    "flashlight": ["flashlight", "the torch", "torch", "the flashlight"],
    "do_not_disturb": ["do not disturb", "DND", "silent mode"],
    "dark_mode": ["dark mode", "dark theme", "night mode"],
    "location": ["location", "GPS", "location services"],
    "hotspot": ["hotspot", "the hotspot", "mobile hotspot"],
    "battery_saver": ["battery saver", "power saving mode", "battery saving mode"],
    "auto_rotate": ["auto rotate", "screen rotation", "auto-rotate"],
}
LEVEL_NAMES = {"brightness": ["brightness", "the brightness", "screen brightness"],
               "volume": ["volume", "the volume", "media volume"]}
TOGGLE_ON = ["Turn on {s}.", "Switch on {s}.", "Enable {s}.", "Turn {s} on.", "{s} on.",
             "Can you turn on {s}?", "Please enable {s}.", "Activate {s}.", "Switch {s} on please.",
             "I need {s} on.", "Turn on {s} and open {app}."]
TOGGLE_OFF = ["Turn off {s}.", "Switch off {s}.", "Disable {s}.", "Turn {s} off.", "{s} off.",
              "Can you turn off {s}?", "Please disable {s}.", "Deactivate {s}.", "Kill {s}.",
              "Shut off {s}.", "Turn off {s} and call {contact}."]
LEVEL_UP = ["Increase {s}.", "Turn up {s}.", "Raise {s}.", "Increase {s} a bit.", "{s} up.",
            "Can you increase {s}?", "Bump up {s}."]
LEVEL_DOWN = ["Decrease {s}.", "Turn down {s}.", "Lower {s}.", "Reduce {s}.", "{s} down.",
              "Can you reduce {s} a little?", "Dim {s}."]
LEVEL_SET = ["Set {s} to {n} percent.", "Set {s} to {n}%.", "Change {s} to {n}.", "Put {s} at {n} percent.",
             "Make {s} {n} percent.", "{s} {n} percent."]
SPECIAL_SETTINGS = [  # (pattern, setting, action, value)
    ("Make the screen brighter.", "brightness", "increase", None),
    ("Make the screen darker.", "brightness", "decrease", None),
    ("It's too bright, dim the screen.", "brightness", "decrease", None),
    ("Make it louder.", "volume", "increase", None),
    ("I can't hear, volume up.", "volume", "increase", None),
    ("Make it quieter.", "volume", "decrease", None),
    ("Mute the phone.", "volume", "set", 0),
    ("Mute the volume.", "volume", "set", 0),
    ("Max volume.", "volume", "set", 100),
    ("Full brightness.", "brightness", "set", 100),
    ("Brightness to maximum.", "brightness", "set", 100),
    ("Turn it on.", None, "on", None),
    ("Change a setting.", None, None, None),
]
HI_TOGGLE = [("{s} on karo.", "on"), ("{s} chalu karo.", "on"), ("{s} band karo.", "off"), ("{s} off karo.", "off")]
HI_LEVEL = [("{s} badhao.", "increase"), ("{s} kam karo.", "decrease"), ("{s} thoda badha do.", "increase")]
TA_TOGGLE = [("{s} on pannu.", "on"), ("{s} off pannu.", "off"), ("{s} on pannunga.", "on"), ("{s} niruthu.", "off")]
TA_LEVEL = [("{s} adhigam pannu.", "increase"), ("{s} kammi pannu.", "decrease"), ("{s} koraiyi.", "decrease")]

SETTING_TEMPLATES = []  # (tid, lang, pattern, kind)
for p in TOGGLE_ON:
    SETTING_TEMPLATES.append(("en", p, "toggle", "on"))
for p in TOGGLE_OFF:
    SETTING_TEMPLATES.append(("en", p, "toggle", "off"))
for p in LEVEL_UP:
    SETTING_TEMPLATES.append(("en", p, "level", "increase"))
for p in LEVEL_DOWN:
    SETTING_TEMPLATES.append(("en", p, "level", "decrease"))
for p in LEVEL_SET:
    SETTING_TEMPLATES.append(("en", p, "level", "set"))
for p, s, a, v in SPECIAL_SETTINGS:
    SETTING_TEMPLATES.append(("en", p, "special", (s, a, v)))
for p, a in HI_TOGGLE:
    SETTING_TEMPLATES.append(("hi", p, "toggle", a))
for p, a in HI_LEVEL:
    SETTING_TEMPLATES.append(("hi", p, "level", a))
for p, a in TA_TOGGLE:
    SETTING_TEMPLATES.append(("ta", p, "toggle", a))
for p, a in TA_LEVEL:
    SETTING_TEMPLATES.append(("ta", p, "level", a))
for i, (lang, p, kind, a) in enumerate(SETTING_TEMPLATES):
    T["change_setting"].append((f"change_setting:{i:03d}", lang, (p, kind, a)))

# Unsupported: out-of-scope requests, hard negatives that look like supported intents,
# and garbage / incomplete ASR output.
add("unsupported", "en",
    "Transfer money to {contact}.", "Send {amount} rupees to {contact}.", "Pay {contact} {amount} on PhonePe.",
    "Order a pizza.", "Order biryani from Swiggy.", "Book a cab to the airport.", "Book a flight to Delhi.",
    "What's the weather like {date}?", "Will it rain {date}?", "Tell me a joke.", "Who won the match yesterday?",
    "What is the capital of France?", "Play some music.", "Play the latest song.",
    "Delete all my photos.", "Cancel my {time} alarm.", "Delete the reminder to {task}.",
    "What alarms do I have?", "Read my messages.", "Who called me?", "Uninstall {app}.", "Update {app}.",
    "Factory reset the phone.", "Turn on the living room lights.", "Set the AC to 24 degrees.",
    "Translate hello to Tamil.", "How far is Chennai from Bangalore?", "Navigate to the nearest petrol station.",
    "Search Google for cheap flights.", "What's my bank balance?", "Buy a phone case on Amazon.",
    "Take a selfie.", "Record a video.", "Scan this QR code.", "What's on my calendar today?",
    "Block {contact}.", "Delete {contact} from my contacts.", "Add {contact} to my contacts.",
    "How old is the universe?", "Lock my phone with a new password.", "Share my location with {contact}.",
    "Show my screen time.", "What's the news?", "Set a timer for 10 minutes.", "Start a stopwatch.",
    "What time is it?", "Convert 100 dollars to rupees.", "Book a table for two at 8.",
    "Recharge my phone with 299 plan.", "Check my train PNR status.", "Turn off the phone.",
    "Restart my phone.", "Um.", "Uh hello?", "Hmm.", "Okay.", "Never mind.", "Blah blah.",
    "asdf", "What.", "Sing a song.", "Who are you?", "Thank you.", "Snooze.",
    "Transfer money to {contact} and send a message to {contact}.", "Pay my electricity bill.",
    "Email the report to my boss.", "Post this on Instagram.", "Download this video.",
    )
add("unsupported", "hi", "{contact} ko paise bhejo.", "Aaj mausam kaisa hai?", "Ek pizza order karo.",
    "Mujhe ek joke sunao.", "{contact} ko {amount} rupaye transfer karo.", "Gaana bajao.")
add("unsupported", "ta", "{contact} ku panam anuppu.", "Inniki mazhai varuma?", "Oru joke sollu.",
    "Pizza order pannu.", "Paattu podu.")

AMOUNTS = ["500", "1000", "two thousand", "200", "fifty"]

# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
WAKE = ["hey tron", "ok tron", "tron", "hey tronos"]
POLITE_PRE = ["please", "can you", "could you please", "hey", "um", "uh", "quickly"]


def fill(rng, intent, lang, pattern, pools):
    """Fill a pattern; return (text, gold_dict)."""
    gold = {"intent": intent}
    if intent == "change_setting":
        p, kind, a = pattern
        if kind == "special":
            s, act, val = a
            text = p
            if s:
                gold["setting"] = s
            if act:
                gold["action"] = act
            if val is not None:
                gold["value"] = val
            return text, gold
        if kind == "toggle":
            s = rng.choice(list(SETTING_NAMES))
            surf = rng.choice(SETTING_NAMES[s])
        else:
            s = rng.choice(list(LEVEL_NAMES))
            surf = rng.choice(LEVEL_NAMES[s])
        n = rng.choice([10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 100])
        text = p.replace("{s}", surf).replace("{n}", str(n))
        if "{app}" in text:
            text = text.replace("{app}", rng.choice(rng.choice(pools["apps"])[1]))
        if "{contact}" in text:
            text = text.replace("{contact}", rng.choice(pools["contacts"]))
        gold.update({"setting": s, "action": a})
        if a == "set":
            gold["value"] = n
        return text, gold

    text = pattern
    vals = {}
    if "{time}" in text:
        surf, v = time_surface(rng, lang)
        text = text.replace("{time}", surf, 1)
        vals["time"] = v
        text = text.replace("{time}", surf)
    if "{date}" in text:
        surf, v = date_surface(rng, lang)
        text = text.replace("{date}", surf)
        vals["date"] = v
    if "tonight" in text and vals.get("time", "12:00") < "12:00":
        text = text.replace("tonight", "today")  # avoid "6 AM tonight"
    if "{label}" in text:
        lab = rng.choice(ALARM_LABELS)
        text = text.replace("{label}", lab)
        vals["label"] = lab
    if "{task}" in text:
        task = rng.choice(HINGLISH_TASKS) if lang == "hi" else rng.choice(pools["tasks"])
        text = text.replace("{task}", task)
        vals["task"] = task
    if "{app}" in text:
        canon, surfs = rng.choice(pools["apps"])
        text = text.replace("{app}", rng.choice(surfs))
        vals["app_name"] = canon
    if "{contact}" in text:
        c = rng.choice(pools["contacts"])
        text = text.replace("{contact}", c)
        vals["contact"] = c
    if "{msgapp}" in text:
        canon, surfs = rng.choice(MSG_APPS)
        surf = rng.choice(surfs)
        if lang == "hi":
            surf = canon.lower() if canon != "SMS" else "sms"
        text = text.replace("{msgapp}", surf)
        vals["app"] = canon
    if "{msg}" in text:
        m = {"hi": HINGLISH_MSGS, "ta": TANGLISH_MSGS}.get(lang)
        m = rng.choice(m) if m else rng.choice(pools["messages"])
        text = text.replace("{msg}", m)
        vals["message"] = m
    if "{amount}" in text:
        text = text.replace("{amount}", rng.choice(AMOUNTS))

    # Build gold. Only fields of the *first* (governing) intent are kept: secondary
    # clauses ("... and call Ravi") are ignored by design.
    if intent == "set_alarm":
        for k in ("time", "date", "label"):
            if k in vals:
                gold[k] = vals[k]
    elif intent == "create_reminder":
        for k in ("task", "time", "date"):
            if k in vals:
                gold[k] = vals[k]
    elif intent == "open_app":
        if "app_name" in vals:
            gold["app_name"] = vals["app_name"]
    elif intent == "make_call":
        if "contact" in vals:
            gold["contact"] = vals["contact"]
    elif intent == "send_message":
        for k in ("contact", "message", "app"):
            if k in vals:
                gold[k] = vals[k]
    # tonight -> evening times only make sense as pm; we leave the time as rendered.
    return text, gold


def asr_noise(rng, text):
    t = text
    if rng.random() < 0.15:
        t = rng.choice(POLITE_PRE) + " " + t[0].lower() + t[1:]
    if rng.random() < 0.12:
        t = rng.choice(WAKE) + " " + t[0].lower() + t[1:]
    if rng.random() < 0.55:  # ASR output usually has no punctuation
        t = re.sub(r"(?<!\d)\.|\.(?!\d)|[,!?;]", "", t)  # keep 8.30 intact
        t = t.replace(": ", " ")
    if rng.random() < 0.5:
        t = t.lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t


def norm_text(t):
    t = t.lower()
    t = re.sub(r"[^\w\s:]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def jaccard(a, b):
    sa, sb = set(a.split()), set(b.split())
    return len(sa & sb) / max(1, len(sa | sb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--per-intent", type=int, default=480, help="target generated samples per intent")
    ap.add_argument("--langs", default="en", help="comma list of template languages to keep (en, hi, ta)")
    args = ap.parse_args()
    keep = set(args.langs.split(","))
    for intent in list(T):
        T[intent] = [t for t in T[intent] if t[1] in keep]
    rng = random.Random(args.seed)

    train_contacts, all_contacts = split_pool(CONTACTS, CONTACTS_TEST_ONLY)
    train_apps, all_apps = split_pool(APPS, APPS_TEST_ONLY)
    train_tasks, all_tasks = split_pool(TASKS, TASKS_TEST_ONLY)
    train_msgs, all_msgs = split_pool(MESSAGES, MESSAGES_TEST_ONLY)
    pools_train = dict(contacts=train_contacts, apps=train_apps, tasks=train_tasks, messages=train_msgs)
    pools_test = dict(contacts=CONTACTS[-CONTACTS_TEST_ONLY:] * 2 + train_contacts,
                      apps=APPS[-APPS_TEST_ONLY:] * 2 + train_apps,
                      tasks=TASKS[-TASKS_TEST_ONLY:] * 2 + train_tasks,
                      messages=MESSAGES[-MESSAGES_TEST_ONLY:] * 2 + train_msgs)
    held_values = set(CONTACTS[-CONTACTS_TEST_ONLY:]) | {a for a, _ in APPS[-APPS_TEST_ONLY:]} | \
        set(TASKS[-TASKS_TEST_ONLY:]) | set(MESSAGES[-MESSAGES_TEST_ONLY:])

    # --- partition templates per intent (and per language so every split covers all langs)
    template_split = {}
    for intent in INTENTS:
        by_lang = defaultdict(list)
        for tid, lang, pat in T[intent]:
            by_lang[lang].append((tid, lang, pat))
        for lang, lst in by_lang.items():
            lst = lst[:]
            rng.shuffle(lst)
            n = len(lst)
            n_test = max(1, round(n * 0.15))
            n_val = max(1, round(n * 0.15)) if n >= 4 else 0
            for i, (tid, _, _) in enumerate(lst):
                template_split[tid] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")

    split_target = {"train": 0.70, "val": 0.12, "test": 0.18}
    samples = []
    seen_norm = {}
    dup_exact = dup_norm = 0
    seen_exact = set()
    for intent in INTENTS:
        tpls = T[intent]
        by_split = defaultdict(list)
        for tpl in tpls:
            by_split[template_split[tpl[0]]].append(tpl)
        for split, tlist in by_split.items():
            target = int(args.per_intent * split_target[split])
            pools = pools_train if split != "test" else pools_test
            attempts = 0
            made = 0
            while made < target and attempts < target * 30:
                attempts += 1
                tid, lang, pat = tlist[attempts % len(tlist)]
                text, gold = fill(rng, intent, lang, pat, pools)
                text = asr_noise(rng, text)
                if text in seen_exact:
                    dup_exact += 1
                    continue
                seen_exact.add(text)
                nt = norm_text(text)
                if nt in seen_norm:
                    dup_norm += 1
                    continue
                seen_norm[nt] = split
                samples.append(dict(text=text, output=gold, intent=intent, lang=lang, template_id=tid,
                                    split=split, source="synthetic_template"))
                made += 1

    # --- hand-written challenge set (test only)
    hw_path = ROOT / "data/raw/handwritten.jsonl"
    hand = [json.loads(line) for line in hw_path.read_text().splitlines() if line.strip()]
    hand = [h for h in hand if h.get("lang", "en") in keep]
    dup_hand = 0
    for h in hand:
        nt = norm_text(h["text"])
        if nt in seen_norm:
            dup_hand += 1
            # remove the synthetic copy so the hand-written gold wins and no leakage occurs
            samples = [s for s in samples if norm_text(s["text"]) != nt]
        seen_norm[nt] = "test"
        samples.append(dict(text=h["text"], output=h["output"], intent=h["output"]["intent"],
                            lang=h.get("lang", "en"), template_id="handwritten", split="test",
                            source="handwritten"))

    # --- near-duplicate leakage filter: drop val/test samples too close to any train sample
    train_norm = [norm_text(s["text"]) for s in samples if s["split"] == "train"]
    train_by_tok = defaultdict(list)
    for t in train_norm:
        for tok in set(t.split()):
            train_by_tok[tok].append(t)
    removed_near = []
    kept = []
    for s in samples:
        if s["split"] == "train":
            kept.append(s)
            continue
        nt = norm_text(s["text"])
        cands = set()
        for tok in set(nt.split()):
            if len(train_by_tok[tok]) < 400:
                cands.update(train_by_tok[tok])
        if any(jaccard(nt, c) >= 0.9 for c in cands):
            removed_near.append(s)
            if s["source"] == "handwritten":
                kept.append(s)  # never silently drop hand-written items; flag them instead
                s["near_duplicate_of_train"] = True
            continue
        kept.append(s)
    samples = kept

    for s in samples:
        s["target"] = canonical_json(s["output"])
        s["uses_heldout_value"] = any(str(v) in held_values for v in s["output"].values())
    # stable ids
    for s in samples:
        s["id"] = hashlib.sha1((s["split"] + "|" + s["text"]).encode()).hexdigest()[:12]

    out_dir = ROOT / "data/processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for split in ("train", "val", "test"):
        rows = [s for s in samples if s["split"] == split]
        rng.shuffle(rows)
        p = out_dir / f"{split}.jsonl"
        with p.open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        files[split] = p

    # --- statistics + manifest
    def stats(rows):
        return {
            "n": len(rows),
            "per_intent": dict(Counter(r["intent"] for r in rows)),
            "per_lang": dict(Counter(r["lang"] for r in rows)),
            "per_source": dict(Counter(r["source"] for r in rows)),
            "templates": len({r["template_id"] for r in rows}),
            "uses_heldout_value": sum(r["uses_heldout_value"] for r in rows),
            "missing_slot_examples": sum(1 for r in rows if r["intent"] != "unsupported"
                                         and len(r["output"]) == 1),
        }
    all_rows = samples
    tpl_overlap = {
        "train_val": len({r["template_id"] for r in all_rows if r["split"] == "train"} &
                         {r["template_id"] for r in all_rows if r["split"] == "val"}),
        "train_test": len({r["template_id"] for r in all_rows if r["split"] == "train"} &
                          {r["template_id"] for r in all_rows if r["split"] == "test" and r["template_id"] != "handwritten"}),
    }
    text_overlap = {
        "train_val": len({norm_text(r["text"]) for r in all_rows if r["split"] == "train"} &
                         {norm_text(r["text"]) for r in all_rows if r["split"] == "val"}),
        "train_test": len({norm_text(r["text"]) for r in all_rows if r["split"] == "train"} &
                          {norm_text(r["text"]) for r in all_rows if r["split"] == "test"}),
    }
    manifest = {
        "dataset": "tronos-voice-commands",
        "version": GENERATOR_VERSION,
        "seed": args.seed,
        "languages": sorted(keep),
        "generator": "scripts/generate_dataset.py",
        "created_from": {"templates": {k: len(v) for k, v in T.items()},
                         "handwritten_file": "data/raw/handwritten.jsonl", "handwritten_items": len(hand)},
        "total": stats(all_rows),
        "splits": {s: stats([r for r in all_rows if r["split"] == s]) for s in ("train", "val", "test")},
        "synthetic_percentage": round(100 * sum(r["source"] == "synthetic_template" for r in all_rows) / len(all_rows), 2),
        "deduplication": {"exact_duplicates_removed": dup_exact,
                          "normalised_duplicates_removed": dup_norm,
                          "handwritten_collisions_replaced": dup_hand,
                          "near_duplicates_removed_from_val_test": len([r for r in removed_near if r["source"] != "handwritten"]),
                          "handwritten_flagged_near_duplicate": len([r for r in removed_near if r["source"] == "handwritten"]),
                          "remaining_duplicates": len(all_rows) - len({norm_text(r["text"]) for r in all_rows})},
        "leakage_checks": {"template_overlap": tpl_overlap, "normalised_text_overlap": text_overlap},
        "files": {k: {"path": str(v.relative_to(ROOT)), "sha256": hashlib.sha256(v.read_bytes()).hexdigest(),
                      "rows": sum(1 for _ in v.open())} for k, v in files.items()},
    }
    (ROOT / "data/manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps({k: manifest[k] for k in ("total", "synthetic_percentage", "deduplication", "leakage_checks")}, indent=2))
    for s in ("train", "val", "test"):
        print(s, manifest["splits"][s]["n"], manifest["splits"][s]["per_intent"])


if __name__ == "__main__":
    main()
