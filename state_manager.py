from __future__ import annotations
import json, os, datetime
from typing import Any, Dict

STATE_DIR = "state"
PREFS_PATH = os.path.join(STATE_DIR, "user_prefs.json")
EVENTS_LOG = os.path.join(STATE_DIR, "events.jsonl")

def ensure_state_dir():
    os.makedirs(STATE_DIR, exist_ok=True)

def load_prefs() -> Dict[str, Any]:
    ensure_state_dir()
    if os.path.exists(PREFS_PATH):
        with open(PREFS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "interests": [],
        "minutes": 20,
        "language_pref": "en",
        "difficulty_pref": None,  # "easier" | "harder" | None
        "history": {"done_ids": [], "skipped_ids": [], "saved_ids": []},
        "streak": 0,
        "last_done_date": None
    }

def save_prefs(p: Dict[str, Any]):
    ensure_state_dir()
    with open(PREFS_PATH, "w", encoding="utf-8") as f:
        json.dump(p, f, indent=2)

def append_event(user_event: Dict[str, Any]):
    ensure_state_dir()
    user_event["ts"] = datetime.datetime.utcnow().isoformat() + "Z"
    with open(EVENTS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(user_event) + "\n")

def update_streak_on_done(prefs: Dict[str, Any]):
    today = datetime.date.today().isoformat()
    last = prefs.get("last_done_date")
    if last is None:
        prefs["streak"] = 1
    else:
        last_date = datetime.date.fromisoformat(last)
        if last_date == datetime.date.today():
            pass
        elif last_date == datetime.date.today() - datetime.timedelta(days=1):
            prefs["streak"] = int(prefs.get("streak", 0)) + 1
        else:
            prefs["streak"] = 1
    prefs["last_done_date"] = today
