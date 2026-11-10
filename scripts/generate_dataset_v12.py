"""Dataset v1.2 generator for the TronOS command parser (target model: Granite-4.0-H-350M).

Every design rule comes from the zero-shot failure analysis (docs/granite_failure_analysis.md):
  R1  say only what was said: each intent has templates WITH and WITHOUT every optional field
  R2  filler/distractor words ("on speaker", "quickly", "please") never change the target
  R3  intent boundaries: generated hard negatives (19 unsupported categories) + boundary positives
  R4  time-expression table: every spoken form, AM/PM balanced, sampled evenly
  R5  every stated field kept, with varied prepositions/positions
  R6  verbatim task/message, canonical app/setting names
  R7  normalised formats: "HH:MM", weekday names, "in" = "HH:MM" from now
  R8  split by template, stratified by (intent, field signature); val/test templates capped
  R9  template-disjoint splits, test-only slot values, hand-written test set, dedup, near-dup filter
  R10 ~70% Moonshine-style text (digits, casing, punctuation), ~30% lowercase/no punctuation/spelled numbers
User decisions (sheet "Decisions"): literal hour when no am/pm; optional "in" field for relative time;
multi-action -> FIRST action decides (unsupported first => unsupported).

  python scripts/generate_dataset_v12.py --seed 13 --per-intent 500
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
VERSION = "1.2.0"

# ------------------------------------------------------------------ slot pools (last block = TEST ONLY)
CONTACTS = ["Mom", "Dad", "Ravi", "Priya", "Arjun", "Ananya", "Rahul", "Sneha", "Karthik", "Divya", "Vikram",
            "Meera", "Suresh", "Lakshmi", "Aditya", "Pooja", "Ganesh", "Kavya", "Rohan", "Nisha", "John", "Sarah",
            "David", "Emily", "Michael", "Jessica", "Daniel", "Sophia", "Kevin", "Olivia", "Grandma", "Grandpa",
            "Anil", "Deepa", "Manoj", "Shalini", "Venkat", "Harini", "Sanjay", "Keerthi", "Amma", "Appa", "Akka",
            "Anna", "Mummy", "Papa", "Bhaiya", "Didi", "Uncle Mohan", "Aunty Rekha"]
CONTACTS_TEST = ["Siddharth", "Revathi", "Imran", "Fatima", "Joseph", "Maria", "Balaji", "Gayathri", "Ashok",
                 "Uncle Raju", "Aunty Geetha", "Nikhil"]
APPS = {  # canonical -> spoken aliases
    "YouTube": ["YouTube"], "WhatsApp": ["WhatsApp"], "Instagram": ["Instagram", "Insta"], "Spotify": ["Spotify"],
    "Google Maps": ["Google Maps", "Maps"], "Gmail": ["Gmail", "my email"], "Camera": ["Camera", "the camera"],
    "Settings": ["Settings", "phone settings"], "Chrome": ["Chrome", "the browser"],
    "Calculator": ["Calculator", "the calculator"], "Calendar": ["Calendar", "my calendar"],
    "Gallery": ["Gallery", "my photos"], "Clock": ["Clock", "the clock app"], "Netflix": ["Netflix"],
    "Facebook": ["Facebook", "FB"], "Telegram": ["Telegram"], "Amazon": ["Amazon"], "Flipkart": ["Flipkart"],
    "Play Store": ["Play Store", "the Play Store"], "PhonePe": ["PhonePe", "Phone Pe"], "Zomato": ["Zomato"],
    "Swiggy": ["Swiggy"], "Uber": ["Uber"], "Files": ["Files", "file manager", "my files"],
}
APPS_TEST = {"Hotstar": ["Hotstar"], "LinkedIn": ["LinkedIn"], "Snapchat": ["Snapchat"], "Ola": ["Ola"],
             "Paytm": ["Paytm"], "Notes": ["Notes", "my notes"]}
TASKS = ["buy milk", "call the plumber", "pay the electricity bill", "take my medicine", "drink water",
         "submit the report", "water the plants", "pick up the kids", "book train tickets", "go for a walk",
         "send the invoice", "check the oven", "feed the dog", "renew my insurance", "charge the car",
         "join the team meeting", "buy vegetables", "return the library books", "pack my bag",
         "collect the laundry", "recharge my phone", "attend the yoga class", "lock the door",
         "take out the trash", "review the slides", "call the bank", "buy a gift for Mom", "pay rent",
         "update my resume", "finish the assignment", "book a doctor appointment", "check my email",
         "clean the room", "order groceries", "practice guitar", "read the newspaper", "visit Grandma",
         "prepare for the interview", "back up my phone", "stretch", "start the washing machine", "call Priya",
         "text Ravi", "message the landlord", "fill the water bottle", "turn off the geyser"]
TASKS_TEST = ["get the car serviced", "buy flowers", "pay the gas bill", "print the boarding pass",
              "defrost the chicken", "call the electrician", "check the mailbox", "apply for leave",
              "buy batteries", "pick up the parcel", "take the cake out", "meditate"]
THINGS = ["my umbrella", "the electricity bill", "my keys", "the meeting", "Mom's birthday", "my passport",
          "the gym bag", "the rent"]                    # "remind me about X" / "don't let me forget X"
THINGS_TEST = ["my charger", "the parking ticket"]
MESSAGES = ["I will be late", "I'm on my way", "call me back", "running ten minutes late", "happy birthday",
            "see you tomorrow", "I reached home", "what time is the meeting", "dinner is ready", "good morning",
            "I'm in a meeting", "can you pick me up", "thanks for the help", "let's meet at 5", "where are you",
            "don't forget the keys", "the train is delayed", "I'll call you later", "please send the document",
            "congratulations", "I am stuck in traffic", "lunch at one", "bring some snacks",
            "the parcel has arrived", "good night", "are you free tonight", "on the way to the office",
            "I forgot my wallet", "start without me", "meeting is cancelled", "buy bread on your way",
            "my phone battery is low", "reached safely", "I'll be there in five minutes", "ok sounds good",
            "flight landed", "welcome home", "ok done"]
MESSAGES_TEST = ["keep the door open", "the doctor said it's fine", "save me a seat", "I paid the bill",
                 "check your email", "we won the match", "leaving the office now", "get well soon",
                 "the movie starts at 7", "can we reschedule"]
LABELS = ["gym", "wake up", "meeting", "medicine", "school run", "flight", "workout", "cooking", "yoga", "office"]
LABELS_TEST = ["prayer", "jogging"]
MSG_APPS = {"WhatsApp": ["on WhatsApp", "via WhatsApp", "through WhatsApp", "using WhatsApp", "over WhatsApp"],
            "Telegram": ["on Telegram", "via Telegram", "through Telegram"],
            "SMS": ["by SMS", "as an SMS", "via SMS"]}
SETTINGS_TOGGLE = {
    "wifi": ["Wi-Fi", "the Wi-Fi", "wireless internet"], "bluetooth": ["Bluetooth", "the Bluetooth"],
    "mobile_data": ["mobile data", "cellular data"], "airplane_mode": ["airplane mode", "flight mode"],
    "flashlight": ["the flashlight", "the torch", "torch"], "do_not_disturb": ["do not disturb", "DND", "silent mode"],
    "dark_mode": ["dark mode", "dark theme", "night mode"], "location": ["location", "GPS", "location services"],
    "hotspot": ["the hotspot", "mobile hotspot"], "battery_saver": ["battery saver", "power saving mode"],
    "auto_rotate": ["auto rotate", "screen rotation"]}
SETTINGS_LEVEL = {"brightness": ["brightness", "the brightness", "screen brightness"],
                  "volume": ["volume", "the volume", "media volume"]}
NUM_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
             10: "ten", 11: "eleven", 12: "twelve", 15: "fifteen", 20: "twenty", 25: "twenty five", 30: "thirty",
             40: "forty", 45: "forty five", 50: "fifty", 60: "sixty", 70: "seventy", 75: "seventy five",
             80: "eighty", 90: "ninety", 100: "hundred"}
MIN_WORDS = {5: "oh five", 10: "ten", 15: "fifteen", 20: "twenty", 25: "twenty five", 30: "thirty", 35: "thirty five",
             40: "forty", 45: "forty five", 50: "fifty", 55: "fifty five"}
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
AMOUNTS = ["500 rupees", "1000 rupees", "2,000 rupees", "200 rupees", "50 dollars"]
FOODS = ["a pizza", "biryani", "a coffee", "dosa", "burgers"]
PLACES = ["the airport", "the railway station", "Chennai", "the office", "the nearest petrol pump", "Bangalore"]
ROOMS = ["living room", "bedroom", "kitchen"]
SONGS = ["Despacito", "the latest Tamil songs", "some music", "my playlist", "Believer"]

# ------------------------------------------------------------------ time / duration / date surface forms (R4, R7)
TIME_FORMS = ["hmm_ampm", "h_ampm", "h_part", "hmm_part", "half_past", "quarter_past", "quarter_to", "24h",
              "special", "no_ampm", "oclock"]
PARTS = {"morning": ("in the morning", range(5, 12), 0), "afternoon": ("in the afternoon", range(12, 17), 12),
         "evening": ("in the evening", range(17, 22), 12), "night": ("at night", range(20, 24), 12)}


def _ampm(rng, pm):
    return rng.choice(["pm", "PM", "p.m."] if pm else ["am", "AM", "a.m."])


def time_expr(rng, words: bool, form: str | None = None):
    """Return (surface, 'HH:MM', form). words=True -> spelled numbers (30% robustness style)."""
    form = form or rng.choice(TIME_FORMS)
    h12 = rng.randint(1, 12)
    pm = rng.random() < 0.5
    h24 = (h12 % 12) + (12 if pm else 0)
    W = lambda n: NUM_WORDS.get(n, str(n)) if words else str(n)  # noqa: E731
    if form == "hmm_ampm":
        m = rng.choice([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
        s = f"{NUM_WORDS[h12]} {MIN_WORDS[m]} {_ampm(rng, pm)}" if words else f"{h12}:{m:02d} {_ampm(rng, pm)}"
        return s, f"{h24:02d}:{m:02d}", form
    if form == "h_ampm":
        return f"{W(h12)} {_ampm(rng, pm)}", f"{h24:02d}:00", form
    if form in ("h_part", "hmm_part"):
        part = rng.choice(list(PARTS))
        phrase, hrs, _ = PARTS[part]
        H = rng.choice(list(hrs))
        h = H % 12 or 12
        m = 0 if form == "h_part" else rng.choice([10, 15, 20, 30, 40, 45])
        if form == "h_part":
            s = f"{W(h)} {phrase}" if rng.random() < 0.7 else f"{W(h)} o'clock {phrase}"
        else:
            s = f"{NUM_WORDS[h]} {MIN_WORDS[m]} {phrase}" if words else f"{h}:{m:02d} {phrase}"
        return s, f"{H:02d}:{m:02d}", form
    if form in ("half_past", "quarter_past", "quarter_to"):
        tail = rng.choice([_ampm(rng, pm), "in the morning" if not pm else ("in the evening" if 5 <= h12 <= 9 else "pm")])
        if not pm and tail == "in the morning" and not 5 <= h12 <= 11:
            tail = _ampm(rng, pm)
        if form == "half_past":
            return f"half past {W(h12)} {tail}", f"{h24:02d}:30", form
        if form == "quarter_past":
            return f"quarter past {W(h12)} {tail}", f"{h24:02d}:15", form
        prev = (h24 - 1) % 24
        return f"quarter to {W(h12)} {tail}", f"{prev:02d}:45", form
    if form == "24h":
        H, m = rng.randint(0, 23), rng.choice([0, 10, 15, 20, 30, 40, 45, 50])
        return f"{H:02d}:{m:02d}", f"{H:02d}:{m:02d}", form
    if form == "special":
        return rng.choice([("noon", "12:00"), ("midnight", "00:00"), ("12 noon", "12:00"), ("12 pm", "12:00"),
                           ("12 am", "00:00"), ("12:30 pm", "12:30"), ("12:15 am", "00:15")]) + (form,)
    if form == "no_ampm":  # user decision: literal hour
        m = rng.choice([0, 0, 15, 30, 45])
        s = (f"{W(h12)}" if m == 0 else (f"{NUM_WORDS[h12]} {MIN_WORDS[m]}" if words else f"{h12}:{m:02d}"))
        return s, f"{h12:02d}:{m:02d}", form
    # oclock without am/pm -> literal hour
    return f"{W(h12)} o'clock", f"{h12:02d}:00", form


def duration_expr(rng, words: bool):
    """Relative time -> 'HH:MM' from now (user decision: optional field "in")."""
    c = rng.choice(["min", "min", "hour", "an_hour", "half", "hm", "mins90"])
    if c == "min":
        n = rng.choice([2, 5, 10, 15, 20, 25, 30, 40, 45, 50])
        return f"in {NUM_WORDS.get(n, str(n)) if words else n} {rng.choice(['minutes', 'mins'])}", f"00:{n:02d}"
    if c == "hour":
        n = rng.choice([2, 3, 4, 5, 6, 8])
        return f"in {NUM_WORDS[n] if words else n} hours", f"{n:02d}:00"
    if c == "an_hour":
        return rng.choice(["in an hour", "in one hour", "in 1 hour"]), "01:00"
    if c == "half":
        return rng.choice(["in half an hour", "in thirty minutes"]), "00:30"
    if c == "hm":
        h, m = rng.choice([1, 2]), rng.choice([15, 30, 45])
        return f"in {h} hour{'s' if h > 1 else ''} and {m} minutes", f"{h:02d}:{m:02d}"
    return "in 90 minutes", "01:30"


def date_expr(rng, words: bool):
    c = rng.random()
    if c < 0.25:
        return "tomorrow", "tomorrow"
    if c < 0.35:
        return "today", "today"
    if c < 0.42:
        return "day after tomorrow", "day_after_tomorrow"
    d = rng.choice(WEEKDAYS)
    return f"{rng.choice(['on', 'this', 'next', 'on', 'coming'])} {d.capitalize()}", d


# ------------------------------------------------------------------ templates
# Each template: (intent, pattern). Slots: {time} {date} {label} {in} {task} {thing} {app} {contact} {msg}
# {msgapp} {setting} {n} {amount} {food} {place} {room} {song} {other}  (other = second action, ignored)
T = defaultdict(list)


def add(intent, *pats):
    for p in pats:
        T[intent].append(p)


add("set_alarm",
    # time only
    "Set an alarm for {time}.", "Wake me up at {time}.", "Alarm at {time}.", "Can you set an alarm at {time}?",
    "Please set my alarm to {time}.", "Put an alarm on for {time}.", "Get me up at {time}.",
    "I need an alarm for {time}.", "Make an alarm for {time}.", "Set alarm {time}.",
    # time + date
    "Set an alarm for {time} {date}.", "Wake me up {date} at {time}.", "{date} wake me at {time}.",
    "I want to wake up at {time} {date}.", "Alarm for {time} {date}.", "Could you wake me up {date} at {time}?",
    # time + label (+ date)
    "Set a {label} alarm for {time}.", "Set an alarm for {time} for {label}.",
    "Set an alarm for {time} {date} called {label}.", "Alarm for {label} at {time} {date}.",
    "Create an alarm called {label} at {time}.", "Set the alarm for {time} {date} for {label}.",
    # relative ("in")
    "Wake me up {in}.", "Set an alarm {in}.", "Alarm {in}.", "Can you wake me {in}?",
    "Set a {label} alarm {in}.", "Ring an alarm {in}.",
    # missing slot (R1)
    "Set an alarm.", "I need an alarm.", "Can you set an alarm for me?", "Create a new alarm.",
    # multi-action, first supported (R3)
    "Set an alarm for {time} and {other}.", "Wake me up at {time} {date} and {other}.")
add("create_reminder",
    # task only
    "Remind me to {task}.", "Set a reminder to {task}.", "Don't let me forget to {task}.",
    "I need a reminder to {task}.", "Add a reminder: {task}.", "Create a reminder to {task}.",
    "Remind me about {thing}.", "Don't let me forget {thing}.", "Help me remember to {task}.",
    # task + time
    "Remind me to {task} at {time}.", "Can you remind me at {time} to {task}?", "Reminder to {task} at {time}.",
    "Set a reminder for {time} to {task}.", "Ping me at {time} to {task}.", "At {time} remind me to {task}.",
    # task + date
    "Remind me to {task} {date}.", "Remind me {date} to {task}.", "Remind me about {thing} {date}.",
    "Don't let me forget to {task} {date}.", "Make a reminder to {task} {date}.",
    # task + time + date
    "Remind me to {task} {date} at {time}.", "{date} at {time} remind me to {task}.",
    "Create a reminder to {task} at {time} {date}.", "Schedule a reminder {date} at {time} to {task}.",
    # relative
    "Remind me to {task} {in}.", "Remind me {in} to {task}.", "{in}, remind me to {task}.",
    "Set a reminder {in} to {task}.",
    # missing slot
    "Remind me.", "Create a reminder.", "Set a reminder.", "I need a reminder.",
    # multi-action
    "Remind me to {task} and {other}.", "Set a reminder to {task} at {time} and {other}.")
add("open_app",
    "Open {app}.", "Launch {app}.", "Start {app}.", "Can you open {app}?", "Go to {app}.", "Bring up {app}.",
    "Open the {app} app.", "Fire up {app}.", "Load {app}.", "Switch to {app}.", "Pull up {app}.", "Run {app}.",
    "Take me to {app}.", "Open up {app}.", "Could you launch {app} for me?",
    "Open {app} and play the latest song.", "Open {app} and search for cricket highlights.",
    "Open {app} and {other}.", "Launch {app} and check my notifications.",
    "Open the app.", "Open an app.")
add("make_call",
    "Call {contact}.", "Phone {contact}.", "Ring {contact}.", "Dial {contact}.", "Give {contact} a call.",
    "Make a call to {contact}.", "Can you call {contact}?", "Place a call to {contact}.",
    "Get {contact} on the phone.", "Start a call with {contact}.", "Connect me to {contact}.",
    "Give {contact} a ring.", "Call {contact} on speaker.", "Call {contact} on speakerphone.",
    "Video call {contact}.", "Start a video call with {contact}.", "Make a video call to {contact}.",
    "Call {contact} and {other}.", "Make a phone call.", "Call someone.")
add("send_message",
    # contact + msg
    "Send a message to {contact} saying {msg}.", "Text {contact} that {msg}.", "Message {contact}: {msg}.",
    "Tell {contact} {msg}.", "Send {contact} a text saying {msg}.", "Let {contact} know {msg}.",
    "Reply to {contact} with {msg}.", "Drop {contact} a message that {msg}.",
    # + app
    "Send a message to {contact} {msgapp} saying {msg}.", "Text {contact} {msgapp} that {msg}.",
    "Send {contact} a message {msgapp}: {msg}.", "Message {contact} {msgapp} saying {msg}.",
    "WhatsApp {contact} {msg}.", "Send a WhatsApp to {contact} saying {msg}.",
    # contact only (+ app)
    "Send a message to {contact}.", "Text {contact}.", "Message {contact} {msgapp}.", "Send {contact} a text.",
    # missing slot
    "Send a message.", "I want to send a text.",
    # multi-action
    "Text {contact} that {msg} and {other}.")
add("unsupported",
    # 1 alarm management (look-alikes of set_alarm)
    "Cancel my {time} alarm.", "Delete the alarm for {time}.", "Turn off my {time} alarm.", "Snooze the alarm.",
    "Stop the alarm.", "Cancel all my alarms.",
    # 2 alarm queries
    "What alarms do I have?", "What time is my alarm set for?", "Is my alarm on for {date}?",
    # 3 reminder management
    "Delete the reminder to {task}.", "Cancel my reminder for {date}.", "What are my reminders?",
    # 4 reading messages
    "Read my messages.", "Read my last message from {contact}.", "Did {contact} text me?",
    # 5 call history
    "Who called me?", "Show missed calls from {contact}.", "Block calls from {contact}.",
    # 6 app management
    "Uninstall {app}.", "Update {app}.", "Clear the cache of {app}.", "Install {app}.",
    # 7 media without app
    "Play {song}.", "Play the latest song.", "Next song.", "Pause the music.",
    # 8 calendar
    "What's on my calendar {date}?", "Add a meeting with {contact} to my calendar.", "Do I have any meetings {date}?",
    # 9 payments
    "Transfer {amount} to {contact}.", "Pay {contact} {amount}.", "Send {amount} to {contact}.",
    "Recharge my phone with the 299 plan.", "What's my bank balance?", "Pay my electricity bill.",
    # 10 shopping / food / cab
    "Order {food}.", "Book a cab to {place}.", "Buy a phone case on Amazon.", "Book a table for two at {time}.",
    # 11 weather / news / sports
    "What's the weather like {date}?", "Will it rain {date}?", "What's the news today?", "Who won the match yesterday?",
    # 12 Q&A / maths / translation / conversion
    "What is the capital of France?", "What's 25 times 4?", "Translate hello to Hindi.",
    "Convert 100 dollars to rupees.", "How far is Chennai from Bangalore?", "Who is the prime minister of India?",
    # 13 smart home (look-alikes of change_setting)
    "Turn on the {room} lights.", "Switch off the {room} fan.", "Set the AC to 24 degrees.",
    # 14 out-of-scope device actions
    "Take a selfie.", "Record a video.", "Take a screenshot.", "Restart my phone.", "Factory reset the phone.",
    # 15 navigation / location
    "Navigate to {place}.", "Where am I?", "Share my location with {contact}.",
    # 16 contacts management
    "Add {contact} to my contacts.", "Delete {contact} from my contacts.", "Block {contact}.",
    # 17 timers (look-alikes of "in")
    "Set a timer for 10 minutes.", "Start a stopwatch.", "Set a timer {in}.",
    # 18 chit-chat / noise
    "Hello.", "Thank you.", "Tell me a joke.", "Who are you?", "Never mind.", "Um.",
    # 19 multi-action with unsupported FIRST (user decision: first action decides)
    "Transfer {amount} to {contact} and text them.", "Order {food} and set an alarm for {time}.",
    "Play some music and open {app}.")

# change_setting templates: (pattern, kind, action)  kind: toggle | level | set | special
CS = []
for p in ["Turn on {s}.", "Switch on {s}.", "Enable {s}.", "Turn {s} on.", "{S} on.", "Activate {s}.",
          "Can you turn on {s}?", "Please enable {s}.", "I need {s} on."]:
    CS.append((p, "toggle", "on"))
for p in ["Turn off {s}.", "Switch off {s}.", "Disable {s}.", "Turn {s} off.", "{S} off.", "Deactivate {s}.",
          "Kill {s}.", "Shut off {s}.", "Can you turn off {s}?", "Stop {s}."]:
    CS.append((p, "toggle", "off"))
for p in ["Increase {s}.", "Turn up {s}.", "Raise {s}.", "Increase {s} a bit.", "Can you increase {s}?",
          "Bump up {s}.", "Make {s} higher."]:
    CS.append((p, "level", "increase"))
for p in ["Decrease {s}.", "Turn down {s}.", "Lower {s}.", "Reduce {s}.", "Can you reduce {s} a little?",
          "Dim {s}.", "Make {s} lower."]:
    CS.append((p, "level", "decrease"))
for p in ["Set {s} to {n}.", "Change {s} to {n}.", "Put {s} at {n}.", "Make {s} {n}.", "{S} {n}.",
          "Increase {s} to {n}.", "Reduce {s} to {n}.", "Can you set {s} to {n}?"]:
    CS.append((p, "set", "set"))
for p, s, a, v in [("Make the screen brighter.", "brightness", "increase", None),
                   ("Make the screen darker.", "brightness", "decrease", None),
                   ("Make it louder.", "volume", "increase", None), ("Make it quieter.", "volume", "decrease", None),
                   ("Mute the phone.", "volume", "set", 0), ("Turn the volume all the way down.", "volume", "set", 0),
                   ("Max volume.", "volume", "set", 100), ("Full brightness.", "brightness", "set", 100),
                   ("Put the phone on silent.", "do_not_disturb", "on", None),
                   ("Turn it on.", None, "on", None), ("Change a setting.", None, None, None),
                   ("Turn on Bluetooth and {other}.", "bluetooth", "on", None),
                   ("Turn off Wi-Fi and {other}.", "wifi", "off", None)]:
    CS.append((p, "special", (s, a, v)))
for p in CS:
    T["change_setting"].append(p)

# Second actions for multi-action templates (their content is ignored by the target)
OTHERS = ["call {c2}", "open WhatsApp", "turn on Bluetooth", "send a message to {c2}", "set an alarm for 7 am",
          "play some music", "check the weather"]
FILLERS_PRE = ["please", "hey Tron,", "OK Tron,", "um", "quickly", "can you", "could you"]
FILLERS_POST = ["please", "right now", "for me", "quickly"]


OPTIONAL = {"set_alarm": {"date", "label", "in"}, "create_reminder": {"time", "date", "in"},
            "send_message": {"message", "app"}, "change_setting": {"value"}}

# ------------------------------------------------------------------ rendering
def signature(intent, pat):
    if intent == "change_setting":
        p, kind, a = pat
        if kind == "special":
            s, act, v = a
            return "cs:" + ",".join(k for k, x in (("setting", s), ("action", act), ("value", v)) if x is not None)
        return f"cs:setting,action{',value' if kind == 'set' else ''}"
    slots = re.findall(r"\{(\w+)\}", pat)
    fields = {"time": "time", "date": "date", "label": "label", "in": "in", "task": "task", "thing": "task",
              "app": "app_name", "contact": "contact", "msg": "message", "msgapp": "app"}
    if intent == "unsupported":
        return "unsupported"
    return intent + ":" + ",".join(sorted({fields[s] for s in slots if s in fields}))


def fill(rng, intent, pat, pools, words):
    gold = {"intent": intent}
    if intent == "change_setting":
        p, kind, a = pat
        if kind == "special":
            s, act, v = a
            text = p.replace("{other}", rng.choice(OTHERS).replace("{c2}", rng.choice(pools["contacts"])))
            for k, x in (("setting", s), ("action", act), ("value", v)):
                if x is not None:
                    gold[k] = x
            return text, gold
        if kind == "toggle":
            s = rng.choice(list(SETTINGS_TOGGLE)); surf = rng.choice(SETTINGS_TOGGLE[s])
        else:
            s = rng.choice(list(SETTINGS_LEVEL)); surf = rng.choice(SETTINGS_LEVEL[s])
        n = rng.choice([10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 100])
        nsurf = (f"{NUM_WORDS[n]} percent" if words else rng.choice([f"{n}%", f"{n} percent", f"{n} percent"]))
        text = p.replace("{s}", surf).replace("{S}", surf[0].upper() + surf[1:]).replace("{n}", nsurf)
        gold.update({"setting": s, "action": a})
        if kind == "set":
            gold["value"] = n
        return text, gold

    text, vals = pat, {}
    if "{other}" in text:
        text = text.replace("{other}", rng.choice(OTHERS).replace("{c2}", rng.choice(pools["contacts"])))
    if "{time}" in text:
        s, v, _ = time_expr(rng, words); text = text.replace("{time}", s); vals["time"] = v
    if "{in}" in text:
        s, v = duration_expr(rng, words); text = text.replace("{in}", s); vals["in"] = v
    if "{date}" in text:
        s, v = date_expr(rng, words)
        if "time" in vals and s == "today" and vals["time"] >= "17:00" and rng.random() < 0.4:
            s = "tonight"
        text = text.replace("{date}", s); vals["date"] = v
    if "{label}" in text:
        x = rng.choice(pools["labels"]); text = text.replace("{label}", x); vals["label"] = x
    if "{task}" in text:
        x = rng.choice(pools["tasks"]); text = text.replace("{task}", x); vals["task"] = x
    if "{thing}" in text:
        x = rng.choice(pools["things"]); text = text.replace("{thing}", x); vals["task"] = x
    if "{app}" in text:
        canon = rng.choice(pools.get("app_keys") or list(pools["apps"]))
        text = text.replace("{app}", rng.choice(pools["apps"][canon]))
        vals["app_name"] = canon
    if "{contact}" in text:
        x = rng.choice(pools["contacts"]); text = text.replace("{contact}", x); vals["contact"] = x
    if "{msgapp}" in text:
        canon = rng.choice(list(MSG_APPS)); text = text.replace("{msgapp}", rng.choice(MSG_APPS[canon]))
        vals["app"] = canon
    if intent == "send_message" and text.startswith(("WhatsApp ", "Send a WhatsApp")):
        vals["app"] = "WhatsApp"
    if "{msg}" in text:
        x = rng.choice(pools["messages"]); text = text.replace("{msg}", x); vals["message"] = x
    for k, pool in (("{amount}", AMOUNTS), ("{food}", FOODS), ("{place}", PLACES), ("{room}", ROOMS), ("{song}", SONGS)):
        text = text.replace(k, rng.choice(pool))
    keep = {"set_alarm": ("time", "date", "label", "in"), "create_reminder": ("task", "time", "date", "in"),
            "open_app": ("app_name",), "make_call": ("contact",), "send_message": ("contact", "message", "app"),
            "unsupported": ()}[intent]
    for k in keep:
        if k in vals:
            gold[k] = vals[k]
    return text, gold


def style(rng, text, moonshine: bool):
    """R10: Moonshine style keeps casing/punctuation/digits; robust style lowercases and drops punctuation.
    Fillers (R2) are added in both styles and never change the target."""
    t = text[0].upper() + text[1:]
    if rng.random() < 0.18:
        f = rng.choice(FILLERS_PRE)
        t = f[0].upper() + f[1:] + " " + t[0].lower() + t[1:]
    if rng.random() < 0.10:
        t = re.sub(r"([.?])$", "", t) + " " + rng.choice(FILLERS_POST) + t[-1] if t[-1] in ".?" else t
    if not moonshine:
        t = t.lower()
        t = re.sub(r"(?<!\d)[.,!?;:](?!\d)", "", t)  # keep 6:30 / 8.50 intact
    return re.sub(r"\s+", " ", t).strip()


def norm_text(t):
    t = t.lower()
    t = re.sub(r"[^\w\s:]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def jacc(a, b):
    sa, sb = set(a.split()), set(b.split())
    return len(sa & sb) / max(1, len(sa | sb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--per-intent", type=int, default=500, help="target samples per intent across all splits")
    ap.add_argument("--split", default="0.72,0.12,0.16")
    ap.add_argument("--max-per-template-eval", type=int, default=12, help="cap per template in val/test (R8)")
    ap.add_argument("--moonshine-share", type=float, default=0.7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    fr = dict(zip(("train", "val", "test"), map(float, args.split.split(","))))

    pools_train = dict(contacts=CONTACTS, apps=APPS, tasks=TASKS, things=THINGS, messages=MESSAGES, labels=LABELS)
    # test split: test-only values weighted so roughly 40-50% of slot values are unseen in training (R9)
    pools_eval = dict(contacts=CONTACTS + CONTACTS_TEST * 4, apps={**APPS, **APPS_TEST},
                      tasks=TASKS + TASKS_TEST * 4, things=THINGS + THINGS_TEST * 4,
                      messages=MESSAGES + MESSAGES_TEST * 4, labels=LABELS + LABELS_TEST * 5,
                      app_keys=list(APPS) + list(APPS_TEST) * 4)
    held = set(CONTACTS_TEST) | set(APPS_TEST) | set(TASKS_TEST) | set(THINGS_TEST) | set(MESSAGES_TEST) | set(LABELS_TEST)

    # ---- R8: split templates per (intent, signature)
    tmpl = []  # (tid, intent, pat, sig)
    for intent in INTENTS:
        for i, pat in enumerate(T[intent]):
            tmpl.append((f"{intent}:{i:03d}", intent, pat, signature(intent, pat)))
    groups = defaultdict(list)
    for t in tmpl:
        groups[(t[1], t[3])].append(t)
    split_of = {}
    for key, lst in sorted(groups.items()):
        lst = lst[:]
        rng.shuffle(lst)
        n = len(lst)
        if n >= 3:
            n_test = max(1, round(n * 0.2)); n_val = max(1, round(n * 0.15))
        elif n == 2:
            n_test, n_val = 1, 0
        else:
            n_test, n_val = 0, 0
        for i, t in enumerate(lst):
            split_of[t[0]] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
    # signatures with a single template appear in train only; their *values* still vary.

    # ---- generate
    samples, seen = [], set()
    dup_exact = dup_norm = 0
    base_count = Counter()
    for intent in INTENTS:
        by_split = defaultdict(list)
        for t in tmpl:
            if t[1] == intent:
                by_split[split_of[t[0]]].append(t)
        for split, tl in by_split.items():
            target = int(args.per_intent * fr[split])
            cap = None if split == "train" else args.max_per_template_eval
            per_t = Counter()
            pools = pools_train if split != "test" else pools_eval
            made = attempts = 0
            while made < target and attempts < target * 40:
                attempts += 1
                tid, _, pat, sig = tl[attempts % len(tl)]
                if cap and per_t[tid] >= cap:
                    if all(per_t[x[0]] >= cap for x in tl):
                        break
                    continue
                moon = rng.random() < args.moonshine_share
                raw, gold = fill(rng, intent, pat, pools, words=not moon and rng.random() < 0.5)
                base = norm_text(raw)
                if intent == "unsupported" and base_count[base] >= 3:  # R3: max 3 variants per base sentence
                    continue
                text = style(rng, raw, moon)
                nt = norm_text(text)
                if text in seen:
                    dup_exact += 1; continue
                if nt in seen:
                    dup_norm += 1; continue
                seen.add(text); seen.add(nt)
                base_count[base] += 1
                per_t[tid] += 1
                samples.append(dict(text=text, output=gold, intent=intent, lang="en", template_id=tid,
                                    signature=sig, split=split, source="synthetic_template",
                                    style="moonshine" if moon else "robust"))
                made += 1

    # ---- hand-written test set (relabelled to v1.2 policy where needed)
    hand = [json.loads(l) for l in (ROOT / "data/raw/handwritten.jsonl").read_text().splitlines() if l.strip()]
    hand = [h for h in hand if h.get("lang", "en") == "en"]
    hw_n = {norm_text(h["text"]) for h in hand}
    samples = [s for s in samples if norm_text(s["text"]) not in hw_n]
    for h in hand:
        samples.append(dict(text=h["text"], output=h["output"], intent=h["output"]["intent"], lang="en",
                            template_id="handwritten", signature="handwritten", split="test", source="handwritten",
                            style="handwritten"))

    # ---- near-duplicate filter (Jaccard >= 0.9 vs any train sample) for val/test
    train_n = [norm_text(s["text"]) for s in samples if s["split"] == "train"]
    idx = defaultdict(list)
    for t in train_n:
        for w in set(t.split()):
            idx[w].append(t)
    kept, near = [], 0
    for s in samples:
        if s["split"] == "train" or s["source"] == "handwritten":
            kept.append(s); continue
        nt = norm_text(s["text"])
        cands = set()
        for w in set(nt.split()):
            if len(idx[w]) < 500:
                cands.update(idx[w])
        if any(jacc(nt, c) >= 0.9 for c in cands):
            near += 1; continue
        kept.append(s)
    samples = kept
    for s in samples:
        s["target"] = canonical_json(s["output"])
        s["uses_heldout_value"] = any(str(v) in held for v in s["output"].values())
        s["id"] = hashlib.sha1((s["split"] + "|" + s["text"]).encode()).hexdigest()[:12]

    out = ROOT / "data/v1.2"
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for split in ("train", "val", "test"):
        rows = [s for s in samples if s["split"] == split]
        rng.shuffle(rows)
        p = out / f"{split}.jsonl"
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        files[split] = p

    def stats(rows):
        return {"n": len(rows), "per_intent": dict(Counter(r["intent"] for r in rows)),
                "per_source": dict(Counter(r["source"] for r in rows)), "per_style": dict(Counter(r["style"] for r in rows)),
                "templates": len({r["template_id"] for r in rows}), "signatures": len({r["signature"] for r in rows}),
                "uses_heldout_value": sum(r["uses_heldout_value"] for r in rows),
                "with_in_field": sum("in" in r["output"] for r in rows),
                "no_optional_field": sum(1 for r in rows if not (set(r["output"]) & OPTIONAL.get(r["intent"], set())))}
    S = {k: [s for s in samples if s["split"] == k] for k in ("train", "val", "test")}
    tr_t = {r["template_id"] for r in S["train"]}
    manifest = {
        "dataset": "tronos-voice-commands", "version": VERSION, "seed": args.seed, "languages": ["en"],
        "generator": "scripts/generate_dataset_v12.py", "design": "docs/granite_failure_analysis.md (R1-R10)",
        "args": vars(args), "templates_per_intent": {k: len(v) for k, v in T.items()},
        "total": stats(samples), "splits": {k: stats(v) for k, v in S.items()},
        "synthetic_percentage": round(100 * sum(s["source"] == "synthetic_template" for s in samples) / len(samples), 2),
        "deduplication": {"exact_duplicates_removed": dup_exact, "normalised_duplicates_removed": dup_norm,
                          "near_duplicates_removed_from_val_test": near,
                          "remaining_duplicates": len(samples) - len({norm_text(s["text"]) for s in samples})},
        "leakage_checks": {
            "template_overlap_train_val": len(tr_t & {r["template_id"] for r in S["val"]}),
            "template_overlap_train_test": len(tr_t & {r["template_id"] for r in S["test"] if r["template_id"] != "handwritten"}),
            "text_overlap_train_val": len({norm_text(r["text"]) for r in S["train"]} & {norm_text(r["text"]) for r in S["val"]}),
            "text_overlap_train_test": len({norm_text(r["text"]) for r in S["train"]} & {norm_text(r["text"]) for r in S["test"]})},
        "files": {k: {"path": str(v.relative_to(ROOT)), "rows": len(S[k]),
                      "sha256": hashlib.sha256(v.read_bytes()).hexdigest()} for k, v in files.items()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps({k: manifest[k] for k in ("total", "synthetic_percentage", "deduplication", "leakage_checks")}, indent=1))
    for k in ("train", "val", "test"):
        print(k, manifest["splits"][k]["n"], manifest["splits"][k]["per_intent"])


if __name__ == "__main__":
    main()
