"""T4 — Anomaly injection and labelled synthetic dataset.

Each condition from docs/symptom_catalog.md becomes a function that edits one day's simulator parameters,
given a severity s in [0, 1] = intensity x time profile (ramp up, plateau, ramp down).
A run = one simulated persona-year containing either no anomaly (normal) or episodes of one condition.

Usage:
    python -m simadl.anomalies 8      # 8 runs per persona x condition
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simadl.features import OUTPUTS, PROCESSED, daily_features
from simadl.simulate import SLOTS, estimate_params, simulate

PERSONAS = ("hh113", "hh115")
CONDITIONS = ("normal", "uti", "depression", "cognitive")
RUN_DAYS = 365
BASELINE_DAYS = 60  # anomaly-free start of every run, so detectors can learn the person's baseline
DATASET = PROCESSED / "dataset"


def _shift(ms, delta=0.0, scale=1.0):
    """Change a (mean, std) pair."""
    return (ms[0] + delta, ms[1] * scale)


def _scale_meals(p, factor):
    for slot in SLOTS:
        p[f"{slot}_p"] *= factor


# Fewer outings free time in crowded days, so outings that would have been dropped now fit and part of
# the reduction is lost. Measured on 2 x 1000-day runs per persona, only ~55-80 % of the target was
# realised, so the reduction is scaled up. Realised vs target is reported in outputs/t4_summary.txt.
TIME_OUT_GAIN = 1.4


def _reduce_time_out(p, hours):
    """Cut daily time out of home by `hours` through fewer outings (lower probability of leaving home)."""
    fewer = TIME_OUT_GAIN * hours / p["trip_mean_h"]
    p["outings"] = (max(0.0, p["outings"][0] - fewer), p["outings"][1])


def uti(p, s):
    p["night_toilet"] = _shift(p["night_toilet"], delta=1.5 * s)
    _scale_meals(p, 1 - 0.15 * s)


def depression(p, s):
    _reduce_time_out(p, 0.52 * s)
    p["contact_scale"] *= 1 - 0.30 * s
    p["sleep_onset"] = _shift(p["sleep_onset"], delta=0.75 * s)
    p["extra_awakenings"] += 0.5 * s
    _scale_meals(p, 1 - 0.25 * s)


def cognitive(p, s):
    _reduce_time_out(p, 1.69 * s)
    for key in ("sleep_onset", "wake", *(f"{slot}_h" for slot in SLOTS)):
        p[key] = _shift(p[key], scale=1 + s)
    p["extra_awakenings"] += 1.0 * s
    p["repeat_meal_p"] = 0.2 * s
    p["contact_scale"] *= 1 - 0.20 * s


APPLY = {"uti": uti, "depression": depression, "cognitive": cognitive}


def sample_episodes(condition: str, rng, n_days: int = RUN_DAYS) -> list[dict]:
    """Onset (day index), duration, ramp lengths and intensity for each episode in a run."""
    eps = []
    if condition == "uti":
        # 2-3 separate infections per year, at least 30 days apart
        for onset in sorted(rng.choice(np.arange(BASELINE_DAYS, n_days - 15, 35), rng.integers(2, 4), replace=False)):
            onset += int(rng.integers(0, 20))
            eps.append(dict(onset=int(onset), duration=int(rng.integers(4, 10)), ramp_up=int(rng.integers(1, 3)),
                            ramp_down=1))
    elif condition == "depression":
        onset = int(rng.integers(BASELINE_DAYS, n_days - 70))
        eps.append(dict(onset=onset, duration=int(rng.integers(28, 57)), ramp_up=14, ramp_down=7))
    elif condition == "cognitive":
        onset = int(rng.integers(BASELINE_DAYS, n_days - 150))
        eps.append(dict(onset=onset, duration=n_days - onset, ramp_up=int(rng.integers(90, 181)), ramp_down=0))
    for e in eps:
        e["intensity"] = float(rng.uniform(0.5, 1.0))
    return eps


def severity(e: dict, i: int) -> float:
    """Trapezoid profile: 0 before onset, linear ramp up, plateau at intensity, linear ramp down."""
    t = i - e["onset"]
    if t < 0 or t >= e["duration"]:
        return 0.0
    up = min(1.0, (t + 1) / e["ramp_up"])
    down = min(1.0, (e["duration"] - t) / e["ramp_down"]) if e["ramp_down"] else 1.0
    return e["intensity"] * min(up, down)


def perturb_params(params: dict, rng) -> dict:
    """Small per-run persona variation so that two runs of one home are not identical people."""
    out = {k: v for k, v in params.items()}
    for daytype in ("weekday", "weekend"):
        p = dict(params[daytype])
        shift = rng.normal(0, 0.3)  # same chronotype shift (h) for sleep and wake
        p["sleep_onset"] = _shift(p["sleep_onset"], delta=shift)
        p["wake"] = _shift(p["wake"], delta=shift)
        p["outings"] = (p["outings"][0] * rng.uniform(0.85, 1.15), p["outings"][1])
        _scale_meals(p, rng.uniform(0.9, 1.1))
        out[daytype] = p
    return out


def make_run(run_id: int, persona: str, condition: str, base: dict, seed: int):
    rng = np.random.default_rng(seed)
    params = perturb_params(base, rng)
    eps = sample_episodes(condition, rng) if condition != "normal" else []

    def modifier(i, day, p):
        s = max((severity(e, i) for e in eps), default=0.0)
        if s > 0:
            APPLY[condition](p, s)
        return p

    events = simulate(params, RUN_DAYS, seed=seed, modifier=modifier)
    daily = daily_features(events, min_episodes=0).iloc[1:-1]
    start = pd.Timestamp(events["start"].min().normalize())
    idx = (daily.index - start).days
    daily["severity"] = [max((severity(e, i) for e in eps), default=0.0) for i in idx]
    daily["label"] = np.where(daily["severity"] > 0, condition, "normal")
    daily.insert(0, "run_id", run_id)
    daily.insert(1, "persona", persona)
    labels = [dict(run_id=run_id, persona=persona, condition=condition,
                   onset=(start + pd.Timedelta(days=e["onset"])).date(), duration_days=e["duration"],
                   ramp_up_days=e["ramp_up"], intensity=round(e["intensity"], 3)) for e in eps]
    return events, daily.drop(columns=["valid", "split"]), labels


def plot_examples(daily: pd.DataFrame, path) -> None:
    feats = [("night_toilet", "night toilet visits"), ("time_out_h", "hours out of home"),
             ("sleep_onset_h", "sleep onset (h)"), ("meals", "meals at home")]
    runs = daily.groupby("run_id").first()
    picks = [runs[(runs["persona"] == "hh113")].index[0]] + \
            [daily[(daily["label"] == c) & (daily["persona"] == "hh113")]["run_id"].iloc[0] for c in CONDITIONS[1:]]
    fig, axes = plt.subplots(len(feats), len(picks), figsize=(18, 10), sharex="col")
    for col, rid in enumerate(picks):
        d = daily[daily["run_id"] == rid]
        cond = d["label"][d["label"] != "normal"].iloc[0] if (d["label"] != "normal").any() else "normal"
        for row, (f, name) in enumerate(feats):
            ax = axes[row, col]
            ax.plot(d.index, d[f], lw=0.6, color="grey")
            ax.plot(d.index, d[f].rolling(7, center=True).mean(), lw=1.5)
            ax.fill_between(d.index, 0, 1, where=d["severity"] > 0, transform=ax.get_xaxis_transform(),
                            color="red", alpha=0.15)
            if col == 0:
                ax.set_ylabel(name)
            if row == 0:
                ax.set_title(f"run {rid}: {cond}")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(runs_per_cell: int) -> None:
    DATASET.mkdir(parents=True, exist_ok=True)
    all_daily, all_labels, run_id = [], [], 0
    for persona in PERSONAS:
        base = estimate_params(persona)
        for condition in CONDITIONS:
            for k in range(runs_per_cell):
                events, daily, labels = make_run(run_id, persona, condition, base, seed=1000 + run_id)
                if k == 0:
                    events.to_csv(DATASET / f"events_run{run_id:03d}_{persona}_{condition}.csv", index=False)
                all_daily.append(daily)
                all_labels += labels
                run_id += 1
            print(f"{persona} {condition}: {runs_per_cell} runs done")
    daily = pd.concat(all_daily)
    labels = pd.DataFrame(all_labels)
    daily.to_csv(DATASET / "daily.csv")
    labels.to_csv(DATASET / "labels.csv", index=False)
    labels.to_csv(OUTPUTS / "t4_labels.csv", index=False)
    plot_examples(daily, OUTPUTS / "t4_examples_hh113.png")

    print(f"\n{run_id} runs, {len(daily)} days, {len(labels)} anomaly episodes")
    print(daily["label"].value_counts().to_string())
    feats = ["night_toilet", "sleep_onset_h", "sleep_episodes", "meals", "outings", "time_out_h", "phone_calls"]
    print("\nmean feature value on normal days vs anomalous days (severity > 0.5):")
    table = {"normal": daily[daily["label"] == "normal"][feats].mean()}
    for c in CONDITIONS[1:]:
        table[c] = daily[(daily["label"] == c) & (daily["severity"] > 0.5)][feats].mean()
    print(pd.DataFrame(table).round(2).to_string())

    # realised effect at full plateau, within the same run (anomalous days vs that run's normal days)
    targets = {("uti", "night_toilet"): 1.5, ("depression", "time_out_h"): -0.52,
               ("depression", "sleep_onset_h"): 0.75, ("cognitive", "time_out_h"): -1.69}
    rows = []
    for (cond, feat), target in targets.items():
        for persona in PERSONAS:
            d = daily[(daily["persona"] == persona) & daily["run_id"].isin(labels.loc[labels["condition"] == cond, "run_id"])]
            plateau = d[d["severity"] > 0]
            plateau = plateau[plateau["severity"] >= 0.9 * plateau.groupby("run_id")["severity"].transform("max")]
            base = d[d["label"] == "normal"].groupby("run_id")[feat].mean()
            # rescale to intensity 1 so it is comparable with the catalogue value
            realised = ((plateau[feat] - plateau["run_id"].map(base)) / plateau["severity"]).mean()
            rows.append(dict(condition=cond, feature=feat, persona=persona, target=target, realised=round(realised, 2)))
    print("\ntarget vs realised change (per unit severity, plateau days vs same-run normal days):")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
