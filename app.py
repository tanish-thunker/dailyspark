import os 
import json
import datetime as dt
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from recommender import build_plan_for_today
from state_manager import load_prefs, save_prefs, append_event, update_streak_on_done

st.set_page_config(page_title="DailySpark ⚡", page_icon="⚡", layout="centered")

DATA_CSV = "data/capsules.csv"
EVENTS_LOG = os.path.join("state", "events.jsonl")

# ---------- Master interest catalog (100+ items; expandable) ----------
INTEREST_CATALOG = [
    # Programming & Tech (30)
    "programming", "python", "javascript", "typescript", "java", "c++", "c", "csharp",
    "go", "rust", "kotlin", "swift", "dart", "php", "ruby", "sql", "mongodb",
    "devops", "docker", "kubernetes", "linux", "git", "testing", "algorithms",
    "data-structures", "web-development", "backend", "frontend", "fullstack", "cloud",

    # Data / AI / ML (18)
    "data-science", "machine-learning", "deep-learning", "pandas", "numpy", "matplotlib",
    "scikit-learn", "tensorflow", "pytorch", "nlp", "computer-vision", "mlops",
    "statistics", "probability", "time-series", "data-engineering", "feature-engineering",
    "prompt-engineering",

    # Career & Productivity (14)
    "career", "interview-prep", "resume", "linkedin", "freelancing", "side-hustles",
    "public-speaking", "writing", "note-taking", "productivity", "time-management",
    "mindset", "entrepreneurship", "startups",

    # Business / Marketing (12)
    "marketing", "digital-marketing", "seo", "content-marketing", "copywriting",
    "social-media", "instagram-growth", "youtube-growth", "email-marketing",
    "brand-building", "sales", "analytics",

    # Finance & Money (10)
    "personal-finance", "investing", "stock-market", "options-basics", "crypto",
    "budgeting", "taxes", "financial-planning", "earn-online", "ecommerce",

    # Creative (10)
    "design", "ui-ux", "figma", "graphic-design", "illustration", "video-editing",
    "photography", "music-production", "storytelling", "blogging",

    # Languages & Communication (6)
    "english", "german", "spanish", "french", "japanese", "communication",

    # Health / Fitness / Mind (12)
    "fitness", "strength-training", "yoga", "mobility", "nutrition", "meditation",
    "mental-health", "sleep", "habit-building", "breathwork", "posture", "running",

    # Games / Strategy / Logic (8)
    "chess", "poker-basics", "sudoku", "rubiks-cube", "game-theory",
    "strategic-thinking", "critical-thinking", "logic-puzzles"
]

# ---------- Motive quotes ----------
QUOTES = [
    "Small steps. Big wins.",
    "Five focused minutes beat an hour of scrolling.",
    "Consistency compounds faster than motivation.",
    "Done is better than perfect.",
    "Tiny daily habits → massive annual gains.",
    "Win the day. Tomorrow thanks you.",
    "Show up. Spark. Repeat.",
    "One more minute than yesterday.",
    "Progress loves patience.",
    "You’re building future you.",
    "Little sparks light big fires.",
    "Invest minutes, harvest mastery."
]
os.makedirs("data", exist_ok=True)
os.makedirs("state", exist_ok=True)

@st.cache_data
def load_capsules():
    if not os.path.exists(DATA_CSV):
        st.error("Missing data/capsules.csv")
        return pd.DataFrame()
    df = pd.read_csv(DATA_CSV)
    mandatory = {"id","interest","format","title","url","duration_min","level","tags","language","source","quality_score"}
    missing = mandatory - set(df.columns)
    if missing:
        st.error(f"capsules.csv missing columns: {missing}")
        return pd.DataFrame()
    df["id"] = df["id"].astype(str)
    # normalize interest to lowercase to match catalog/customs
    df["interest"] = df["interest"].astype(str).str.lower()
    # ensure duration is numeric
    df["duration_min"] = pd.to_numeric(df["duration_min"], errors="coerce").fillna(0).astype(int)
    return df

def read_events():
    """Read events.jsonl into a list."""
    if not os.path.exists(EVENTS_LOG):
        return []
    out = []
    with open(EVENTS_LOG, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out

def events_to_df(events):
    if not events:
        return pd.DataFrame(columns=["ts","event","item_id"])
    df = pd.DataFrame(events)
    # ts is ISO; parse to datetime date
    try:
        df["ts_dt"] = pd.to_datetime(df["ts"], errors="coerce")
    except Exception:
        df["ts_dt"] = pd.NaT
    df["date"] = df["ts_dt"].dt.date
    return df

def onboarding(df: pd.DataFrame):
    st.subheader("Tell me your interests & daily time")

    # Combine catalog + any interests already present in CSV
    csv_interests = sorted(df["interest"].dropna().unique().tolist())
    all_interests = sorted(set(INTEREST_CATALOG) | set(csv_interests))

    default = [x for x in ["programming","python","chess","earn-online","earning-online"] if x in all_interests]
    chosen = st.multiselect("Pick interests (you can add more later)", all_interests, default=default or all_interests[:3])

    # Allow custom interest input (free text)
    with st.expander("Add a custom interest"):
        custom = st.text_input("Type a new interest and press Enter", "")
        if custom:
            custom_norm = custom.strip().lower()
            if custom_norm and custom_norm not in all_interests:
                all_interests.append(custom_norm)
                st.success(f"Added “{custom_norm}” to your available list (remember to add content later).")

    minutes = st.slider("Daily minutes", 5, 180, 20, 5)
    st.caption("Tip: You can change this later in Settings.")

    if st.button("Start"):
        # Include custom if provided and not already chosen
        if custom and custom.strip():
            cn = custom.strip().lower()
            if cn not in chosen:
                chosen.append(cn)
        if not chosen:
            st.warning("Pick at least one interest to get started.")
            return
        prefs = load_prefs()
        prefs["interests"] = [{"name": n, "weight": 1} for n in chosen]
        prefs["minutes"] = int(minutes)
        save_prefs(prefs)
        st.rerun()

def settings_panel(df: pd.DataFrame, prefs: dict):
    with st.expander("⚙️ Settings"):
        new_minutes = st.slider("Daily minutes", 5, 180, int(prefs["minutes"]), 5)
        if new_minutes != prefs["minutes"]:
            prefs["minutes"] = int(new_minutes)
            save_prefs(prefs)

        st.markdown("**Interests & Weights**")
        new_interests = []
        for it in prefs["interests"]:
            c1, c2, c3 = st.columns([3,1,1])
            with c1:
                st.write(f"• {it['name']}")
            with c2:
                w = st.number_input(f"Wt:{it['name']}", min_value=0.0, max_value=5.0,
                                    value=float(it.get("weight",1)), step=0.5, key=f"wt_{it['name']}")
            with c3:
                if st.button("Remove", key=f"rm_{it['name']}"):
                    append_event({"event":"remove_interest","interest":it["name"]})
                    continue
            it["weight"] = float(w)
            new_interests.append(it)
        prefs["interests"] = new_interests

        # Build full selectable list from catalog ∪ CSV ∪ current choices
        csv_interests = sorted(df["interest"].dropna().unique().tolist())
        master_interests = sorted(set(INTEREST_CATALOG) | set(csv_interests) | set([x["name"] for x in prefs["interests"]]))

        add_it = st.selectbox("Add interest", ["--"] + [i for i in master_interests
                              if i not in [x["name"] for x in prefs["interests"]]])
        if add_it != "--" and st.button("Add interest"):
            prefs["interests"].append({"name": add_it, "weight": 1})
            append_event({"event":"add_interest","interest":add_it})
            save_prefs(prefs)
            st.rerun()

        # Add custom, free-text interest
        st.markdown("**Add a custom interest**")
        custom2 = st.text_input("New interest", key="custom_interest_settings")
        if st.button("Add custom interest"):
            cn = (custom2 or "").strip().lower()
            if not cn:
                st.warning("Type something first.")
            elif cn in [x["name"] for x in prefs["interests"]]:
                st.info("That interest is already in your list.")
            else:
                prefs["interests"].append({"name": cn, "weight": 1})
                append_event({"event":"add_interest","interest":cn})
                save_prefs(prefs)
                st.success(f"Added “{cn}”.")
                st.rerun()

        diff_map = {"Just right": None, "Easier": "easier", "Harder": "harder"}
        inv = {v:k for k,v in diff_map.items()}
        cur = inv.get(prefs.get("difficulty_pref"), "Just right")
        diff_pref = st.radio("Difficulty preference", options=list(diff_map.keys()),
                             index=list(diff_map.keys()).index(cur))
        prefs["difficulty_pref"] = diff_map[diff_pref]
        save_prefs(prefs)

        if st.button("Reset onboarding"):
            prefs["interests"] = []
            save_prefs(prefs)
            st.rerun()

def header_block(prefs, df_events):
    # random motivational quote (changes daily)
    st.markdown(f"### 🔥 Quote of the Day")
    st.caption(f"_{QUOTES[hash(dt.date.today()) % len(QUOTES)]}_")

    # tasks completed today
    today = dt.date.today()
    done_today = 0
    if not df_events.empty:
        done_today = int(df_events[(df_events["date"] == today) & (df_events["event"] == "done")].shape[0])
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"**Daily budget:** {prefs['minutes']} min")
    with c2:
        st.markdown(f"**Streak:** 🔥 {prefs.get('streak',0)} day(s)")
    with c3:
        st.markdown(f"**Tasks today:** {done_today}")

    # celebratory note after 3 tasks
    if done_today >= 3:
        st.success("Day completed, champ! 🎉 See you tomorrow — or keep exploring if you’re in flow.")

def _enforce_budget(plan, total_minutes):
    """Take a list of items (each has duration_min) and return a trimmed list whose cumulative sum <= total_minutes."""
    cum = 0
    kept = []
    for it in plan:
        dur = int(it.get("duration_min", 0))
        if dur <= 0:
            continue
        if cum + dur <= total_minutes:
            kept.append(it)
            cum += dur
        # stop early if exactly matched
        if cum == total_minutes:
            break
    return kept, cum

def render_today(df: pd.DataFrame, prefs: dict):
    # Hint if some selected interests have no items yet
    selected = [i["name"] for i in prefs["interests"]]
    has_items = set(df["interest"].str.lower().unique())
    missing = [s for s in selected if s.lower() not in has_items]
    if missing:
        st.info("No content yet for: " + ", ".join(missing) + ". Add rows to data/capsules.csv to start seeing them.")

    # Build the "seen" set once
    seen = set(prefs["history"]["done_ids"] + prefs["history"]["skipped_ids"])

    # --- STRICT BUDGET ENFORCEMENT ---
    budget = int(prefs["minutes"])
    df_budget = df[df["duration_min"] <= budget].copy()

    if df_budget.empty:
        st.warning(f"No items fit within your daily budget of {budget} minute(s). Add some shorter items or increase your budget in Settings.")
        return

    # Build plan
    plan = build_plan_for_today(
        df=df_budget,
        total_minutes=budget,
        interests=prefs["interests"],
        seen_ids=seen,
        difficulty_pref=prefs.get("difficulty_pref")
    )

    if not plan:
        st.info("No items match your interests yet (within budget). Add more rows to data/capsules.csv or tweak interests.")
        return

    # --- NEW: Remove any already-seen items in case the planner didn't filter them ---
    plan = [it for it in plan if str(it.get("id")) not in seen]

    # Enforce cumulative budget
    trimmed, total = _enforce_budget(plan, budget)

    if not trimmed:
        st.info("All recommended items exceeded your remaining budget. Add some shorter items to your catalog or increase your daily minutes.")
        return

    if len(trimmed) < len(plan):
        st.caption(f"Showing a time-boxed plan: {total}/{budget} min (trimmed to fit your budget).")

    # Render cards
    for item in trimmed:
        ph = st.empty()  # placeholder to allow immediate per-card update

        def render_card():
            with ph.container(border=True):
                st.markdown(f"**{item['title']}**")
                st.caption(f"{item['interest']} • {item['format']} • {int(item['duration_min'])} min • {item['level']}")
                st.write(", ".join(str(item.get('tags','')).split(';')))
                col1, col2, col3, col4 = st.columns([1,1,1,1])
                col1.link_button("Open ▶️", item["url"])
                if col2.button("Done ✅", key=f"done_{item['id']}"):
                    item_id = str(item["id"])
                    if item_id not in prefs["history"]["done_ids"]:
                        prefs["history"]["done_ids"].append(item_id)
                    # remove from saved if present
                    if item_id in prefs["history"]["saved_ids"]:
                        prefs["history"]["saved_ids"].remove(item_id)
                    update_streak_on_done(prefs)
                    save_prefs(prefs)
                    append_event({"event":"done","item_id":item_id})
                    # instantly hide this card
                    ph.empty()
                    st.toast("Marked as Done ✅")
                    st.experimental_rerun()
                if col3.button("Not relevant 🚫", key=f"skip_{item['id']}"):
                    item_id = str(item["id"])
                    if item_id not in prefs["history"]["skipped_ids"]:
                        prefs["history"]["skipped_ids"].append(item_id)
                    save_prefs(prefs)
                    append_event({"event":"skip","item_id":item_id})
                    ph.empty()
                    st.toast("Hidden for now")
                    st.experimental_rerun()
                if col4.button("Save 🔖", key=f"save_{item['id']}"):
                    item_id = str(item["id"])
                    if item_id not in prefs["history"]["saved_ids"]:
                        prefs["history"]["saved_ids"].append(item_id)
                        save_prefs(prefs)
                        append_event({"event":"save","item_id":item_id})
                        st.toast("Saved for later")
                    else:
                        st.toast("Already in Saved")

        render_card()

def render_saved(df: pd.DataFrame, prefs: dict):
    st.caption("Your saved items live here. Filter, sort, finish, or remove.")

    # search + sort controls
    q = st.text_input("Search saved (title/tags)", "")
    sort_by = st.selectbox("Sort by", ["Saved order", "Title (A→Z)", "Duration (short→long)", "Interest (A→Z)"])

    saved_ids = list(dict.fromkeys(prefs["history"]["saved_ids"]))  # keep order
    if not saved_ids:
        st.info("No saved items yet. Tap **Save 🔖** on anything you like and it’ll show up here.")
        return

    saved_df = df[df["id"].isin(saved_ids)].copy()

    # search filtering
    if q.strip():
        qlow = q.lower()
        saved_df = saved_df[
            saved_df["title"].str.lower().str.contains(qlow, na=False) |
            saved_df["tags"].astype(str).str.lower().str.contains(qlow, na=False)
        ]

    # sorting
    if sort_by == "Saved order":
        order_map = {sid: idx for idx, sid in enumerate(saved_ids)}
        saved_df["__order"] = saved_df["id"].map(order_map)
        saved_df = saved_df.sort_values("__order").drop(columns="__order")
    elif sort_by == "Title (A→Z)":
        saved_df = saved_df.sort_values("title")
    elif sort_by == "Duration (short→long)":
        saved_df = saved_df.sort_values("duration_min")
    elif sort_by == "Interest (A→Z)":
        saved_df = saved_df.sort_values("interest")

    # grouped render
    for interest, group in saved_df.groupby("interest"):
        st.subheader(f"⭐ {interest}")
        for _, item in group.iterrows():
            with st.container(border=True):
                st.markdown(f"**{item['title']}**")
                st.caption(f"{item['interest']} • {item['format']} • {int(item['duration_min'])} min • {item['level']}")
                st.write(", ".join(str(item.get("tags","")).split(";")))
                c1, c2, c3 = st.columns([1,1,1])
                c1.link_button("Open ▶️", item["url"])
                if c2.button("Done ✅", key=f"saved_done_{item['id']}"):
                    item_id = str(item["id"])
                    if item_id not in prefs["history"]["done_ids"]:
                        prefs["history"]["done_ids"].append(item_id)
                    if item_id in prefs["history"]["saved_ids"]:
                        prefs["history"]["saved_ids"].remove(item_id)
                    update_streak_on_done(prefs)
                    save_prefs(prefs)
                    append_event({"event":"done","item_id":item_id})
                    st.experimental_rerun()
                if c3.button("Remove ❌", key=f"unsave_{item['id']}"):
                    item_id = str(item["id"])
                    if item_id in prefs["history"]["saved_ids"]:
                        prefs["history"]["saved_ids"].remove(item_id)
                        save_prefs(prefs)
                        append_event({"event":"unsave","item_id":item_id})
                        st.experimental_rerun()

    st.divider()
    colA, colB = st.columns([1,1])
    with colA:
        if st.button("Clear all saved ⚠️"):
            prefs["history"]["saved_ids"].clear()
            save_prefs(prefs)
            append_event({"event":"clear_saved"})
            st.experimental_rerun()
    with colB:
        st.caption("This only clears the Saved list (does not affect Done history).")

def render_stats(df_events):
    st.caption("Weekly progress and interest breakdown.")
    if df_events.empty:
        st.info("No activity yet. Complete a few capsules to see your stats here.")
        return

    # Only 'done' events count toward minutes learned
    done_df = df_events[df_events["event"] == "done"].copy()
    if done_df.empty:
        st.info("No completed items yet. Finish some capsules to unlock stats.")
        return

    # Extract last 7 days (including today)
    today = dt.date.today()
    week_days = [today - dt.timedelta(days=i) for i in range(6, -1, -1)]
    day_labels = [d.strftime("%a") for d in week_days]

    # We need durations; we only have item_id in events.
    # Quick join: read capsules to map id -> duration & interest.
    caps = load_capsules()
    id_to_dur = dict(zip(caps["id"], caps["duration_min"]))
    id_to_interest = dict(zip(caps["id"], caps["interest"]))

    done_df["duration"] = done_df["item_id"].map(id_to_dur).fillna(0).astype(int)
    done_df["interest"] = done_df["item_id"].map(id_to_interest).fillna("unknown")
    # aggregate minutes per day
    day_minutes = {d: 0 for d in week_days}
    for _, row in done_df.iterrows():
        d = row.get("date")
        if isinstance(d, dt.date) and d in day_minutes:
            day_minutes[d] += int(row["duration"])

    # ---- Chart 1: Minutes per day (bar) ----
    st.markdown("**Minutes learned (last 7 days)**")
    fig1, ax1 = plt.subplots()
    ax1.bar(day_labels, [day_minutes[d] for d in week_days])
    ax1.set_xlabel("Day")
    ax1.set_ylabel("Minutes")
    ax1.set_title("Daily minutes")
    st.pyplot(fig1)

    # ---- Chart 2: Interest breakdown (pie) ----
    st.markdown("**Interest breakdown (total minutes)**")
    interest_minutes = done_df.groupby("interest")["duration"].sum().sort_values(ascending=False)
    if not interest_minutes.empty:
        fig2, ax2 = plt.subplots()
        ax2.pie(interest_minutes.values, labels=interest_minutes.index, autopct="%1.0f%%")
        ax2.set_title("By interest")
        st.pyplot(fig2)

    # Totals
    st.markdown(f"**Total minutes completed (7d):** {sum(day_minutes.values())}")

def main():
    st.title("⚡ DailySpark")
    st.caption("Small time, big gains. Daily micro-learning tailored to your interests and available time.")

    df = load_capsules()
    if df.empty:
        st.stop()

    prefs = load_prefs()
    # normalize stored ids to str & interests to lowercase for matching
    prefs["history"]["done_ids"]   = [str(x) for x in prefs["history"]["done_ids"]]
    prefs["history"]["skipped_ids"]= [str(x) for x in prefs["history"]["skipped_ids"]]
    prefs["history"]["saved_ids"]  = [str(x) for x in prefs["history"]["saved_ids"]]
    # normalize interest names in prefs for consistency
    for it in prefs["interests"]:
        it["name"] = str(it["name"]).lower()

    if not prefs["interests"]:
        onboarding(df)
        st.stop()

    # Read events once for header + stats
    events = read_events()
    ev_df = events_to_df(events)

    header_block(prefs, ev_df)

    tabs = st.tabs(["Today", "Saved", "Stats", "Settings"])

    with tabs[0]:
        st.subheader("Today’s Capsules")
        render_today(df, prefs)

    with tabs[1]:
        st.subheader("Saved for Later")
        render_saved(df, prefs)

    with tabs[2]:
        st.subheader("Your Stats")
        render_stats(ev_df)

    with tabs[3]:
        settings_panel(df, prefs)

if __name__ == "__main__":
    main()
