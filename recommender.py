# recommender.py  — ε-greedy bandit edition
from __future__ import annotations
import math, json, os, random
import pandas as pd
from typing import Dict, List, Any, Set, Tuple

# ---- Bandit config ----
EPSILON = 0.10             # exploration probability
BANDIT_WEIGHT = 0.20       # how strongly bandit avg reward affects final score
EVENTS_LOG = os.path.join("state", "events.jsonl")

# Reward mapping for user feedback
REWARD_BY_EVENT = {
    "done": 1.0,
    "save": 0.6,
    "skip": 0.0,
    # you can add "open": 0.2 if you start logging it
}

# ---------- Utility: time allocation ----------
def allocate_budgets(total_minutes: int, interests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    weights = [max(0.0, float(i.get("weight", 1))) for i in interests]
    wsum = sum(weights) or 1.0
    base = []
    for i in interests:
        w = max(0.0, float(i.get("weight", 1)))
        budget = int(round(total_minutes * (w / wsum)))
        base.append({"interest": i["name"], "budget": max(1, budget)})

    diff = total_minutes - sum(b["budget"] for b in base)
    step = 1 if diff > 0 else -1
    for k in range(abs(diff)):
        idx = k % len(base)
        if base[idx]["budget"] + step >= 1:
            base[idx]["budget"] += step
    return base

def _fit_duration_score(duration: int, budget: int) -> float:
    if budget <= 0:
        return 0.0
    return max(0.0, 1.0 - abs(duration - budget) / float(max(1, budget)))

# ---------- Bandit: read events → arm stats ----------
def _arm_key_from_row(row: pd.Series) -> str:
    """Define a bandit 'arm' as (interest, subtopic, source)."""
    return f"{str(row.get('interest','')).lower()}|{str(row.get('subtopic','')).lower()}|{str(row.get('source','')).lower()}"

def _load_events() -> List[Dict[str, Any]]:
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

def _build_arm_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Returns:
      { arm_key: {"n": count, "avg_reward": r, "sum_reward": s} }
    Joins events to items via item_id -> row to compute arm stats.
    """
    events = _load_events()
    if not len(events):
        return {}

    # Fast lookup: item_id -> arm_key
    id_to_arm: Dict[str, str] = {}
    for _, row in df.iterrows():
        id_to_arm[str(row["id"])] = _arm_key_from_row(row)

    agg: Dict[str, Dict[str, float]] = {}
    for ev in events:
        item_id = str(ev.get("item_id", ""))
        event = str(ev.get("event", "")).lower()
        if item_id not in id_to_arm:
            continue
        arm = id_to_arm[item_id]
        reward = REWARD_BY_EVENT.get(event)
        if reward is None:
            continue
        if arm not in agg:
            agg[arm] = {"n": 0.0, "sum_reward": 0.0}
        agg[arm]["n"] += 1.0
        agg[arm]["sum_reward"] += reward

    for arm, v in agg.items():
        n = max(1.0, v["n"])
        v["avg_reward"] = v["sum_reward"] / n
    return agg

# ---------- Scoring ----------
def _score_item_base(
    row: pd.Series,
    budget: int,
    seen_ids: Set[str],
    difficulty_pref: str | None = None,
    interest_boost: float = 1.0,
) -> float:
    quality = float(row.get("quality_score", 0.7))
    duration = int(row.get("duration_min", 5))
    fit_dur = _fit_duration_score(duration, budget)
    novelty = 0.0 if str(row["id"]) in seen_ids else 1.0

    level = str(row.get("level", "beginner")).lower()
    if difficulty_pref == "harder":
        diff_fit = 1.0 if level in ("intermediate", "advanced") else 0.3
    elif difficulty_pref == "easier":
        diff_fit = 1.0 if level == "beginner" else 0.3
    else:
        diff_fit = 0.7

    # base weights (sum to 0.90; remaining 0.10 is interest_boost)
    quality_weight   = 0.35
    duration_weight  = 0.25
    novelty_weight   = 0.15
    difficulty_weight= 0.15
    interest_weight  = 0.10

    base = (
        quality_weight * quality +
        duration_weight * fit_dur +
        novelty_weight * novelty +
        difficulty_weight * diff_fit +
        interest_weight * interest_boost
    )
    return float(base)

def _apply_bandit_bonus(row: pd.Series, base_score: float, arm_stats: Dict[str, Dict[str, float]]) -> float:
    arm = _arm_key_from_row(row)
    arm_avg = arm_stats.get(arm, {}).get("avg_reward", None)
    if arm_avg is None:
        return base_score
    # Blend bandit signal into final score
    return base_score + BANDIT_WEIGHT * float(arm_avg)

# ---------- Selection with ε-greedy ----------
def pick_for_interest(
    df: pd.DataFrame,
    interest: str,
    budget: int,
    seen_ids: Set[str],
    tol: float = 0.25,
    difficulty_pref: str | None = None,
    arm_stats: Dict[str, Dict[str, float]] | None = None,
) -> List[Dict[str, Any]]:
    pool = df[df["interest"].str.lower() == interest.lower()].copy()
    if pool.empty:
        return []

    # ε-greedy: with prob ε, explore by shuffling; otherwise rank by (base + bandit)
    explore = (random.random() < EPSILON)

    if explore:
        pool = pool.sample(frac=1.0, random_state=None)  # random order
        # still cap with duration window
        low = max(1, int(math.floor(budget * (1 - tol))))
        high = max(low, int(math.ceil(budget * (1 + tol))))
        chosen, total = [], 0
        for _, row in pool.iterrows():
            dur = int(row.get("duration_min", 5))
            if total + dur <= high:
                chosen.append(row.to_dict())
                total += dur
                if total >= low:
                    break
        if not chosen:
            chosen = [pool.iloc[0].to_dict()]
        return chosen

    # Exploit: compute base score then add bandit bonus if available
    scores: List[Tuple[float, Dict[str, Any]]] = []
    for _, row in pool.iterrows():
        base = _score_item_base(row, budget, seen_ids, difficulty_pref, interest_boost=1.0)
        final = _apply_bandit_bonus(row, base, arm_stats or {})
        scores.append((final, row.to_dict()))

    scores.sort(key=lambda x: x[0], reverse=True)

    low = max(1, int(math.floor(budget * (1 - tol))))
    high = max(low, int(math.ceil(budget * (1 + tol))))

    chosen, total = [], 0
    for sc, rowd in scores:
        dur = int(rowd.get("duration_min", 5))
        if total + dur <= high:
            chosen.append(rowd)
            total += dur
            if total >= low:
                break
    if not chosen:
        chosen = [scores[0][1]]
    return chosen

def build_plan_for_today(
    df: pd.DataFrame,
    total_minutes: int,
    interests: List[Dict[str, Any]],
    seen_ids: Set[str],
    difficulty_pref: str | None = None
) -> List[Dict[str, Any]]:
    # Build arm stats from your past events (done/skip/save)
    arm_stats = _build_arm_stats(df)

    budgets = allocate_budgets(total_minutes, interests)
    plan: List[Dict[str, Any]] = []
    for b in budgets:
        picks = pick_for_interest(
            df, b["interest"], b["budget"], seen_ids,
            tol=0.25, difficulty_pref=difficulty_pref, arm_stats=arm_stats
        )
        plan.extend(picks)

    seen_local, unique_plan = set(), []
    for it in plan:
        key = str(it["id"])
        if key not in seen_local:
            seen_local.add(key)
            unique_plan.append(it)
    return unique_plan
