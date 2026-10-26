"""T1 — Calibration: CASAS labeled log -> activity episodes -> one row of ADL features per day.

A "day" runs from 12:00 to 12:00 the next day, so one night of sleep is never split at midnight.
Clock times are expressed in hours after midnight of the day's date (e.g. 01:30 next morning = 25.5).

Usage:
    python -m simadl.features hh113
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "labeled"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"

CALIBRATION_DAYS = 365      # first year calibrates the simulator, the rest is held out for T5
MIN_EPISODES_PER_DAY = 10   # below this a day is treated as missing annotation, not as behaviour
NIGHT = (22, 30)            # 22:00-06:00 next day, in day-relative hours
MAX_EPISODE_H = 16          # longer episodes come from a missed "end" marker (e.g. 15-day Entertain_Guests)

MEAL_ACTIVITIES = ("Cook", "Eat", "Wash_Breakfast_Dishes", "Wash_Lunch_Dishes", "Wash_Dinner_Dishes")


def load_episodes(home: str) -> pd.DataFrame:
    """Pair Activity="begin"/"end" markers into episodes (activity, start, end)."""
    df = pd.read_csv(RAW / f"{home}.csv", header=None, names=["date", "time", "sensor", "value", "label"],
                     usecols=["date", "time", "label"], dtype=str)
    df = df[df["label"].str.contains('="(?:begin|end)"', na=False)]
    parts = df["label"].str.extract(r'^(?P<activity>[A-Za-z_]+)="(?P<phase>begin|end)"$')
    df = pd.concat([df, parts], axis=1).dropna(subset=["activity"])
    df["ts"] = pd.to_datetime(df["date"] + " " + df["time"], format="mixed")
    df = df.sort_values("ts")

    open_at, episodes = {}, []
    for activity, phase, ts in zip(df["activity"], df["phase"], df["ts"]):
        if phase == "begin":
            open_at[activity] = ts  # an unmatched earlier begin is dropped
        elif activity in open_at:
            episodes.append((activity, open_at.pop(activity), ts))
    ep = pd.DataFrame(episodes, columns=["activity", "start", "end"])
    ep["duration_min"] = (ep["end"] - ep["start"]).dt.total_seconds() / 60
    ep = ep[ep["duration_min"] <= MAX_EPISODE_H * 60]
    return ep.sort_values("start").reset_index(drop=True)


def _day_of(ts: pd.Series) -> pd.Series:
    return (ts - pd.Timedelta(hours=12)).dt.normalize()


def _hours(ts: pd.Series, day: pd.Series) -> pd.Series:
    """Clock time as hours after midnight of `day` (next morning > 24)."""
    return (ts - day).dt.total_seconds() / 3600


def _meal_slot(activity: str, h: float) -> str:
    for slot in ("Breakfast", "Lunch", "Dinner"):
        if slot in activity:
            return slot.lower()
    h = h % 24
    return "breakfast" if 4 <= h < 11 else "lunch" if 11 <= h < 16 else "dinner"


def daily_features(ep: pd.DataFrame) -> pd.DataFrame:
    ep = ep.copy()
    ep["day"] = _day_of(ep["start"])
    ep["h"] = _hours(ep["start"], ep["day"])
    ep["h_end"] = _hours(ep["end"], ep["day"])
    days = pd.date_range(ep["day"].min(), ep["day"].max(), freq="D")
    out = pd.DataFrame(index=pd.Index(days, name="day"))
    out["n_episodes"] = ep.groupby("day").size()

    # Sleep: night-time Sleep episodes (start between 18:00 and 06:00)
    sleep = ep[(ep["activity"] == "Sleep") & ep["h"].between(18, 30)]
    g = sleep.groupby("day")
    out["sleep_onset_h"] = g["h"].min()
    out["wake_h"] = g["h_end"].max()
    out["sleep_hours"] = g["duration_min"].sum() / 60
    out["sleep_episodes"] = g.size()
    night = ep[ep["activity"].isin(["Bed_Toilet_Transition", "Toilet"]) & ep["h"].between(*NIGHT)]
    out["night_toilet"] = night.groupby("day").size()

    # Eating: Cook_*/Eat_*/Wash_*_Dishes episodes, grouped into breakfast/lunch/dinner slots
    meal = ep[ep["activity"].str.startswith(MEAL_ACTIVITIES)].copy()
    meal["slot"] = [_meal_slot(a, h) for a, h in zip(meal["activity"], meal["h"])]
    # a late breakfast running past noon belongs to the previous day's morning
    late = (meal["slot"] == "breakfast") & (meal["h"] < 24)
    meal.loc[late, "day"] -= pd.Timedelta(days=1)
    meal.loc[late, "h"] += 24
    out["meals"] = meal.groupby("day")["slot"].nunique()
    out["meal_episodes"] = meal.groupby("day").size()
    for slot in ("breakfast", "lunch", "dinner"):
        food = meal[(meal["slot"] == slot) & ~meal["activity"].str.startswith("Wash")]
        out[f"{slot}_h"] = food.groupby("day")["h"].min() % 24

    # Social: outings (Leave_Home -> next Enter_Home), phone calls, guests
    move = ep[ep["activity"].isin(["Leave_Home", "Enter_Home"])].reset_index(drop=True)
    nxt = move.shift(-1)
    trips = move[(move["activity"] == "Leave_Home") & (nxt["activity"] == "Enter_Home")].copy()
    trips["hours_out"] = (nxt.loc[trips.index, "start"] - trips["end"]).dt.total_seconds() / 3600
    trips = trips[trips["hours_out"] < 24]
    out["outings"] = ep[ep["activity"] == "Leave_Home"].groupby("day").size()
    out["time_out_h"] = trips.groupby("day")["hours_out"].sum()
    out["phone_calls"] = ep[ep["activity"] == "Phone"].groupby("day").size()
    guests = ep[ep["activity"] == "Entertain_Guests"].groupby("day")
    out["guest_visits"] = guests.size()
    out["guest_h"] = guests["duration_min"].sum() / 60

    counts = ["n_episodes", "sleep_episodes", "night_toilet", "meals", "meal_episodes",
              "outings", "phone_calls", "guest_visits"]
    out[counts] = out[counts].fillna(0).astype(int)
    out[["sleep_hours", "time_out_h", "guest_h"]] = out[["sleep_hours", "time_out_h", "guest_h"]].fillna(0.0)

    out["valid"] = out["n_episodes"] >= MIN_EPISODES_PER_DAY
    out["weekend"] = out.index.day_of_week >= 5
    first = out.index[out["valid"]].min()
    out["split"] = np.where(out.index < first + pd.Timedelta(days=CALIBRATION_DAYS), "calibration", "holdout")
    return out


FEATURES = ["sleep_onset_h", "wake_h", "sleep_hours", "sleep_episodes", "night_toilet",
            "meals", "meal_episodes", "breakfast_h", "dinner_h",
            "outings", "time_out_h", "phone_calls", "guest_visits", "guest_h"]


def summarize(daily: pd.DataFrame) -> pd.DataFrame:
    cal = daily[daily["valid"] & (daily["split"] == "calibration")]
    rows = {}
    for name, part in [("all", cal), ("weekday", cal[~cal["weekend"]]), ("weekend", cal[cal["weekend"]])]:
        rows[(name, "mean")] = part[FEATURES].mean()
        rows[(name, "std")] = part[FEATURES].std()
    return pd.DataFrame(rows).round(2)


def plot_distributions(daily: pd.DataFrame, path: Path, title: str) -> None:
    cal = daily[daily["valid"] & (daily["split"] == "calibration")]
    fig, axes = plt.subplots(3, 5, figsize=(18, 10))
    for ax, f in zip(axes.flat, FEATURES):
        for label, part in [("weekday", cal[~cal["weekend"]]), ("weekend", cal[cal["weekend"]])]:
            ax.hist(part[f].dropna(), bins=20, alpha=0.6, density=True, label=label)
        ax.set_title(f)
    axes.flat[0].legend()
    for ax in list(axes.flat)[len(FEATURES):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(home: str) -> None:
    ep = load_episodes(home)
    daily = daily_features(ep)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(exist_ok=True)
    ep.to_csv(PROCESSED / f"{home}_episodes.csv", index=False)
    daily.to_csv(PROCESSED / f"{home}_daily.csv")
    summary = summarize(daily)
    summary.to_csv(OUTPUTS / f"t1_summary_{home}.csv")
    plot_distributions(daily, OUTPUTS / f"t1_distributions_{home}.png", f"{home} — calibration days")

    valid = daily["valid"]
    print(f"{home}: {len(ep)} episodes, {len(daily)} days, {valid.sum()} valid "
          f"({(valid & (daily['split'] == 'calibration')).sum()} calibration / "
          f"{(valid & (daily['split'] == 'holdout')).sum()} holdout)")
    print(summary.to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "hh113")
