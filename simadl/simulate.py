"""T2 — Normal behaviour simulator.

Persona parameters (means/stds per weekday/weekend, empirical episode durations) are estimated from
the calibration days of a real home (T1). Each simulated day is built as:
    night:   Sleep segments from onset to wake, split by Bed_Toilet_Transition and silent awakenings
    daytime: meal blocks first, then outing/phone/guest blocks at clock times sampled from the real home;
             a block overlapping an earlier-placed one is pushed later, or dropped if it no longer fits
Output episodes have the same columns as `features.load_episodes`, so the T1 feature code is reused.

`simulate(..., modifier=f)` lets T4 change a day's parameters: f(day_index, date, params) -> params.

Usage:
    python -m simadl.simulate hh113 365
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simadl.features import FEATURES, OUTPUTS, PROCESSED, daily_features

SLOTS = ("breakfast", "lunch", "dinner")
START_DATE = "2024-01-01"
GAP_MIN = 1.0  # minimum gap between consecutive daytime blocks


def _stats(x: pd.Series, default_mean: float = 0.0) -> tuple[float, float]:
    x = x.dropna()
    return (float(x.mean()), float(x.std())) if len(x) > 1 else (default_mean, 0.0)


def _trip_minutes(ep: pd.DataFrame) -> pd.Series:
    """Minutes between each Leave_Home and the next Enter_Home (trips < 16 h)."""
    move = ep[ep["activity"].isin(["Leave_Home", "Enter_Home"])].reset_index(drop=True)
    nxt = move.shift(-1)
    trip = move[(move["activity"] == "Leave_Home") & (nxt["activity"] == "Enter_Home")]
    minutes = (nxt.loc[trip.index, "start"] - trip["end"]).dt.total_seconds() / 60
    return minutes[minutes < 16 * 60]


def estimate_params(home: str) -> dict:
    daily = pd.read_csv(PROCESSED / f"{home}_daily.csv", parse_dates=["day"])
    ep = pd.read_csv(PROCESSED / f"{home}_episodes.csv", parse_dates=["start", "end"])
    cal = daily[daily["valid"] & (daily["split"] == "calibration")]

    params = {}
    for daytype, part in [("weekday", cal[~cal["weekend"]]), ("weekend", cal[cal["weekend"]])]:
        p = {}
        p["sleep_onset"] = _stats(part["sleep_onset_h"])
        p["wake"] = _stats(part["wake_h"])
        p["night_toilet"] = _stats(part["night_toilet"])
        frag = part[part["sleep_episodes"] > 1]
        gap_h = (frag["wake_h"] - frag["sleep_onset_h"] - frag["sleep_hours"]) / (frag["sleep_episodes"] - 1)
        p["awake_gap_min"] = float(gap_h.mean() * 60)
        # awakenings that split sleep without a toilet visit
        p["extra_awakenings"] = max(0.0, float((part["sleep_episodes"] - 1).clip(lower=0).mean()
                                               - part["night_toilet"].mean()))
        for slot in SLOTS:
            p[f"{slot}_p"] = float(part[f"{slot}_h"].notna().mean())
            p[f"{slot}_h"] = _stats(part[f"{slot}_h"], default_mean={"breakfast": 8, "lunch": 12.5, "dinner": 18}[slot])
        p["episodes_per_meal"] = float((part["meal_episodes"] / part["meals"].where(part["meals"] > 0)).mean())
        p["outings"] = _stats(part["outings"])
        p["time_out_mean"] = float(part["time_out_h"].mean())  # baselines used by T4 to size changes
        p["trip_mean_h"] = float(_trip_minutes(ep).mean() / 60)
        # T4 hooks (neutral values = normal behaviour)
        p["trip_scale"] = 1.0      # multiplier on sampled trip durations
        p["contact_scale"] = 1.0   # keep-probability for each phone call / guest visit
        p["repeat_meal_p"] = 0.0   # chance per day of an extra (forgotten-already-eaten) meal
        # zero-inflated counts: resample whole days instead of fitting a normal
        p["phone_calls"] = part["phone_calls"].astype(int).tolist()
        p["guest_visits"] = part["guest_visits"].astype(int).tolist()
        params[daytype] = p

    # empirical durations (minutes), pooled by meal prefix
    dur = {}
    for key, mask in {
        "Bed_Toilet_Transition": ep["activity"].isin(["Bed_Toilet_Transition", "Toilet"]),
        "Cook": ep["activity"].str.startswith("Cook"),
        "Eat": ep["activity"].str.startswith("Eat"),
        "Wash": ep["activity"].str.startswith("Wash"),
        "Phone": ep["activity"] == "Phone",
        "Entertain_Guests": ep["activity"] == "Entertain_Guests",
        "Leave_Home": ep["activity"] == "Leave_Home",
        "Enter_Home": ep["activity"] == "Enter_Home",
    }.items():
        dur[key] = ep.loc[mask, "duration_min"].round(2).tolist()
    dur["trip"] = _trip_minutes(ep).round(2).tolist()
    params["durations"] = dur
    # empirical clock hours at which social episodes start
    hour = ep["start"].dt.hour + ep["start"].dt.minute / 60
    params["start_hours"] = {a: hour[ep["activity"] == a].round(2).tolist()
                             for a in ("Leave_Home", "Phone", "Entertain_Guests")}
    return params


def _normal(rng, ms, lo=-np.inf, hi=np.inf):
    return float(np.clip(rng.normal(ms[0], ms[1]), lo, hi))


def _count(rng, ms):
    """ms is (mean, std) for a rounded normal, or a list of observed daily counts to resample."""
    if isinstance(ms, list):
        return int(rng.choice(ms))
    return int(max(0, round(rng.normal(ms[0], ms[1]))))


def _night(rng, p, dur, onset, wake):
    """Sleep from onset to wake (absolute Timestamps), split by toilet visits and silent awakenings."""
    span = (wake - onset).total_seconds() / 60
    breaks = [("Bed_Toilet_Transition", t) for t in rng.uniform(15, span - 15, _count(rng, p["night_toilet"]))]
    breaks += [(None, t) for t in rng.uniform(15, span - 15, rng.poisson(p["extra_awakenings"]))]
    eps, cursor = [], onset
    for activity, t in sorted(breaks, key=lambda b: b[1]):
        t0 = onset + pd.Timedelta(minutes=t)
        if t0 <= cursor:
            continue
        length = rng.choice(dur["Bed_Toilet_Transition"]) if activity else rng.exponential(p["awake_gap_min"])
        t1 = t0 + pd.Timedelta(minutes=length)
        if t1 >= wake:
            break
        eps.append(("Sleep", cursor, t0))
        if activity:
            eps.append((activity, t0, t1))
        cursor = t1
    eps.append(("Sleep", cursor, wake))
    return eps


def _meal_parts(rng, p, dur, slot):
    """Real logs fragment mostly the cooking part: Cook x (n-2), then Eat, then Wash."""
    n = 1 + rng.poisson(max(p["episodes_per_meal"] - 1, 0))
    kinds = ["Cook"] * max(n - 2, 1) + (["Eat"] if n >= 2 else []) + (["Wash"] if n >= 3 else [])
    parts, offset = [], 0.0
    for kind in kinds:
        d = rng.choice(dur[kind])
        name = f"Wash_{slot.title()}_Dishes" if kind == "Wash" else f"{kind}_{slot.title()}"
        parts.append((name, offset, d))
        offset += d + rng.uniform(0.5, 5)
    return parts


def _daytime_blocks(rng, p, params, day):
    """Each block: (priority, start, [(activity, offset_min, duration_min), ...]). Meals have priority 0."""
    dur, hours = params["durations"], params["start_hours"]
    blocks = []
    slots = [s for s in SLOTS if rng.random() < p[f"{s}_p"]]
    # sorted so that breakfast < lunch < dinner even when sampled times cross
    times = sorted(_normal(rng, p[f"{s}_h"], 0, 23.9) for s in slots)
    for slot, hour in zip(slots, times):
        blocks.append((0, day + pd.Timedelta(hours=hour), _meal_parts(rng, p, dur, slot)))
    if slots and rng.random() < p["repeat_meal_p"]:
        slot = rng.choice(slots)
        hour = max(times) + rng.uniform(0.5, 3)
        blocks.append((0, day + pd.Timedelta(hours=min(hour, 23.9)), _meal_parts(rng, p, dur, slot)))

    n_out = _count(rng, p["outings"])
    for _ in range(n_out):
        leave, enter = rng.choice(dur["Leave_Home"]), rng.choice(dur["Enter_Home"])
        away = rng.choice(dur["trip"]) * p["trip_scale"]
        parts = [("Leave_Home", 0.0, leave), ("Enter_Home", leave + away, enter)]
        blocks.append((1, day + pd.Timedelta(hours=rng.choice(hours["Leave_Home"])), parts))
    for activity, key in [("Phone", "phone_calls"), ("Entertain_Guests", "guest_visits")]:
        for _ in range(rng.binomial(_count(rng, p[key]), p["contact_scale"])):
            blocks.append((1, day + pd.Timedelta(hours=rng.choice(hours[activity])),
                           [(activity, 0.0, rng.choice(dur[activity]))]))
    return blocks


def _place(blocks, awake_from, awake_to):
    """Place blocks by (priority, start); push a block past any overlap. If a social block no longer
    fits before bedtime, retry from the start of the day; otherwise drop it (meals are never moved
    to the morning)."""
    eps, taken, gap = [], [], pd.Timedelta(minutes=GAP_MIN)

    def first_free(start, length):
        clash = True
        while clash:
            clash = next((t for t in taken if start < t[1] + gap and t[0] < start + length + gap), None)
            if clash:
                start = clash[1] + gap
        return start

    for priority, start, parts in sorted(blocks, key=lambda b: (b[0], b[1])):
        length = pd.Timedelta(minutes=max(off + d for _, off, d in parts))
        start = first_free(max(start, awake_from), length)
        if start + length > awake_to and priority > 0:
            start = first_free(awake_from, length)
            if start + length > awake_to:
                continue
        taken.append((start, start + length))
        eps += [(a, start + pd.Timedelta(minutes=off), start + pd.Timedelta(minutes=off + d)) for a, off, d in parts]
    return eps


def simulate(params: dict, n_days: int, seed: int = 0, start: str = START_DATE, modifier=None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dur = params["durations"]
    days = pd.date_range(start, periods=n_days, freq="D")
    eps = []
    first = params["weekend" if days[0].day_of_week >= 5 else "weekday"]
    awake_from = days[0] + pd.Timedelta(hours=_normal(rng, first["wake"]) - 24)
    for i, day in enumerate(days):
        p = dict(params["weekend" if day.day_of_week >= 5 else "weekday"])
        if modifier is not None:
            p = modifier(i, day, p)
        onset = day + pd.Timedelta(hours=_normal(rng, p["sleep_onset"], 19, 28))
        wake = day + pd.Timedelta(hours=_normal(rng, p["wake"], 26, 36))
        if wake - onset < pd.Timedelta(hours=2):
            wake = onset + pd.Timedelta(hours=2)
        eps += _place(_daytime_blocks(rng, p, params, day), awake_from + pd.Timedelta(minutes=15),
                      onset - pd.Timedelta(minutes=15))
        eps += _night(rng, p, dur, onset, wake)
        awake_from = wake
    out = pd.DataFrame(eps, columns=["activity", "start", "end"])
    out["duration_min"] = (out["end"] - out["start"]).dt.total_seconds() / 60
    return out.sort_values("start").reset_index(drop=True)


def compare_plot(real: pd.DataFrame, sim: pd.DataFrame, path, title: str) -> None:
    fig, axes = plt.subplots(3, 5, figsize=(18, 10))
    for ax, f in zip(axes.flat, FEATURES):
        ax.hist(real[f].dropna(), bins=20, alpha=0.6, density=True, label="real")
        ax.hist(sim[f].dropna(), bins=20, alpha=0.6, density=True, label="simulated")
        ax.set_title(f)
    axes.flat[0].legend()
    for ax in list(axes.flat)[len(FEATURES):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(home: str, n_days: int) -> None:
    params = estimate_params(home)
    (OUTPUTS / f"t2_params_{home}.json").write_text(json.dumps(
        {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, list) or len(vv) < 3}
         for k, v in params.items() if k in ("weekday", "weekend")}, indent=2))
    sim = simulate(params, n_days, seed=0)
    sim_daily = daily_features(sim, min_episodes=0).iloc[1:-1]  # drop partial first/last day
    sim.to_csv(PROCESSED / f"sim_{home}_episodes.csv", index=False)
    sim_daily.to_csv(PROCESSED / f"sim_{home}_daily.csv")

    real = pd.read_csv(PROCESSED / f"{home}_daily.csv")
    real = real[real["valid"] & (real["split"] == "calibration")]
    compare_plot(real, sim_daily, OUTPUTS / f"t2_real_vs_sim_{home}.png", f"{home}: real calibration days vs simulated")
    table = pd.DataFrame({"real_mean": real[FEATURES].mean(), "sim_mean": sim_daily[FEATURES].mean(),
                          "real_std": real[FEATURES].std(), "sim_std": sim_daily[FEATURES].std()}).round(2)
    print(f"{home}: simulated {len(sim)} episodes over {len(sim_daily)} days")
    print(table.to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "hh113", int(sys.argv[2]) if len(sys.argv) > 2 else 365)
