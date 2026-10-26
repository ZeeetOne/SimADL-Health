"""T6 — Detect *when* behaviour changes, per person, against that person's own baseline.

Signal: 7-day rolling means of the daily features (+ 7-day rolling std of sleep onset, for irregularity),
z-scored with the run's first BASELINE_DAYS days. Three detectors give a daily score and an alarm:
  - zscore  : Shahid et al. 2022 style, alarm if any feature's weekly mean leaves baseline mean +/- 2 sigma
  - iforest : Isolation Forest fitted on the baseline weeks, alarm above the 99th pct of baseline scores
  - pelt    : PELT change points (ruptures) on the whole run (retrospective, as when a report is prepared
              before a consultation); a segment raises an alarm if its mean deviates from baseline
Evaluation is per anomaly episode: detected or not, delay from onset, and false alarms on normal days.
Methods are compared twice: with their published/default thresholds, and with thresholds calibrated on the
"normal" runs to the same false-alarm rate (TARGET_FA_PER_30D), which is the fair comparison.

Usage:
    python -m simadl.detect
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt
from sklearn.ensemble import IsolationForest

from simadl.anomalies import BASELINE_DAYS, DATASET
from simadl.features import OUTPUTS

DETECT_FEATURES = ["sleep_onset_h", "wake_h", "sleep_hours", "sleep_episodes", "night_toilet",
                   "meals", "meal_episodes", "breakfast_h", "outings", "time_out_h", "phone_calls", "guest_visits"]
WINDOW = 7
TOLERANCE_BEFORE = 3   # an alarm up to 3 days before onset still counts (cf. +/-3 day UTI labels in Capstick's follow-up)
TOLERANCE_AFTER = 7    # weekly smoothing lags the end of an episode by up to a week
ZSCORE_THRESHOLD = 2.0
IFOREST_QUANTILE = 0.99
PELT_PENALTY = 10.0
PELT_THRESHOLD = 1.0   # segment mean |z| (RMS over features) that counts as a deviation
METHODS = ("zscore", "iforest", "pelt")
TARGET_FA_PER_30D = 0.5  # about one false alarm every two months, between two GP visits


def weekly_signal(run: pd.DataFrame) -> pd.DataFrame:
    """Rolling weekly features z-scored against the run's baseline period."""
    x = run[DETECT_FEATURES].copy()
    # a missing meal time means "no such meal", fill with baseline mean so it does not dominate
    x = x.fillna(x.iloc[:BASELINE_DAYS].mean())
    w = x.rolling(WINDOW, min_periods=3).mean()
    w["sleep_onset_sd"] = x["sleep_onset_h"].rolling(WINDOW, min_periods=3).std()
    base = w.iloc[WINDOW:BASELINE_DAYS]
    z = (w - base.mean()) / base.std().replace(0, 1)
    return z.fillna(0.0)


def score_zscore(z: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    s = z.abs().max(axis=1).to_numpy()
    return s, s > ZSCORE_THRESHOLD


def score_iforest(z: pd.DataFrame, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    base = z.iloc[WINDOW:BASELINE_DAYS]
    model = IsolationForest(n_estimators=200, random_state=seed).fit(base)
    s = -model.score_samples(z)
    thr = np.quantile(-model.score_samples(base), IFOREST_QUANTILE)
    return s, s > thr


def score_pelt(z: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[int]]:
    bkps = rpt.Pelt(model="l2", min_size=WINDOW, jump=1).fit(z.to_numpy()).predict(pen=PELT_PENALTY * z.shape[1])
    s = np.zeros(len(z))
    start = 0
    for end in bkps:
        seg = z.iloc[start:end]
        s[start:end] = np.sqrt((seg.mean() ** 2).mean())  # RMS of the segment's mean z over features
        start = end
    return s, s > PELT_THRESHOLD, [b for b in bkps[:-1]]


def run_detectors(daily: pd.DataFrame) -> pd.DataFrame:
    out = []
    for run_id, run in daily.groupby("run_id", sort=False):
        run = run.reset_index()
        z = weekly_signal(run)
        res = run[["run_id", "persona", "day", "label", "severity"]].copy()
        res["zscore_score"], res["zscore_alarm"] = score_zscore(z)
        res["iforest_score"], res["iforest_alarm"] = score_iforest(z)
        res["pelt_score"], res["pelt_alarm"], bkps = score_pelt(z)
        res["pelt_changepoint"] = False
        res.loc[bkps, "pelt_changepoint"] = True
        out.append(res)
    return pd.concat(out, ignore_index=True)


def _alarm_onsets_per_30d(scores: pd.DataFrame, alarm: np.ndarray) -> float:
    """Rate of alarm onsets (first day of a run of alarm days), after the baseline period."""
    day = pd.to_datetime(scores["day"])
    post = ((day - day.groupby(scores["run_id"]).transform("min")).dt.days >= BASELINE_DAYS).to_numpy()
    prev = pd.Series(alarm, index=scores.index).groupby(scores["run_id"]).shift(1, fill_value=False).to_numpy()
    starts = alarm & ~prev & post
    return 30 * starts.sum() / post.sum()


def calibrate(scores: pd.DataFrame) -> dict:
    """Per method, the lowest score threshold giving <= TARGET_FA_PER_30D on the all-normal runs."""
    normal = scores[scores.groupby("run_id")["label"].transform(lambda s: (s == "normal").all())]
    thresholds = {}
    for m in METHODS:
        s = normal[f"{m}_score"].to_numpy()
        thresholds[m] = float(s.max())
        for thr in np.quantile(s, np.linspace(0.5, 0.9999, 400)):
            if _alarm_onsets_per_30d(normal, s > thr) <= TARGET_FA_PER_30D:
                thresholds[m] = float(thr)
                break
    return thresholds


def evaluate(scores: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Episode recall, detection delay and false alarms per method, condition and persona."""
    scores = scores.copy()
    scores["day"] = pd.to_datetime(scores["day"])
    labels = labels.copy()
    labels["onset"] = pd.to_datetime(labels["onset"])
    rows = []
    for method in METHODS:
        alarm = f"{method}_alarm"
        # an episode counts as detected only if a *new* alarm is raised inside its window; an alarm that was
        # already on before the window (e.g. a long PELT segment) gets no credit
        rising = scores[alarm] & ~scores.groupby("run_id")[alarm].shift(1, fill_value=False)
        for ep in labels.itertuples():
            run = scores[scores["run_id"] == ep.run_id]
            lo = ep.onset - pd.Timedelta(days=TOLERANCE_BEFORE)
            hi = ep.onset + pd.Timedelta(days=ep.duration_days + TOLERANCE_AFTER)
            hits = run[(run["day"] >= lo) & (run["day"] <= hi) & rising[run.index]]
            delay = (hits["day"].min() - ep.onset).days if len(hits) else np.nan
            rows.append(dict(method=method, persona=ep.persona, condition=ep.condition, kind="episode",
                             detected=len(hits) > 0, delay_days=delay))
        # false alarms: alarm onsets (first day of a run of alarm days) on days far from any episode
        for run_id, run in scores[scores["day"] >= scores.groupby("run_id")["day"].transform("min")
                                  + pd.Timedelta(days=BASELINE_DAYS)].groupby("run_id"):
            near = pd.Series(False, index=run.index)
            for ep in labels[labels["run_id"] == run_id].itertuples():
                lo = ep.onset - pd.Timedelta(days=TOLERANCE_BEFORE)
                hi = ep.onset + pd.Timedelta(days=ep.duration_days + TOLERANCE_AFTER)
                near |= (run["day"] >= lo) & (run["day"] <= hi)
            a = run[alarm].to_numpy()
            starts = a & ~np.r_[False, a[:-1]]
            far = ~near.to_numpy()
            rows.append(dict(method=method, persona=run["persona"].iloc[0], condition="(normal days)",
                             kind="normal", false_alarms=int((starts & far).sum()), normal_days=int(far.sum())))
    r = pd.DataFrame(rows)
    ep = r[r["kind"] == "episode"].groupby(["method", "condition", "persona"]).agg(
        episodes=("detected", "size"), recall=("detected", "mean"), median_delay_days=("delay_days", "median"))
    fa = r[r["kind"] == "normal"].groupby(["method", "persona"])[["false_alarms", "normal_days"]].sum()
    fa["false_alarms_per_30d"] = 30 * fa["false_alarms"] / fa["normal_days"]
    return ep.round(2), fa.round(3)


def onset_dating(scores: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """PELT as an onset-dating tool: days between each true onset and the nearest change point."""
    day = pd.to_datetime(scores["day"])
    rows = []
    for ep in labels.itertuples():
        cps = day[(scores["run_id"] == ep.run_id) & scores["pelt_changepoint"]]
        err = (cps - pd.Timestamp(ep.onset)).dt.days
        nearest = err.iloc[err.abs().argmin()] if len(err) else np.nan
        rows.append(dict(condition=ep.condition, persona=ep.persona, error_days=nearest))
    r = pd.DataFrame(rows)
    return r.groupby(["condition", "persona"])["error_days"].agg(
        median_error="median", median_abs_error=lambda e: e.abs().median(),
        within_7_days=lambda e: (e.abs() <= 7).mean()).round(2)


def plot_run(scores: pd.DataFrame, run_id: int, path) -> None:
    run = scores[scores["run_id"] == run_id]
    day = pd.to_datetime(run["day"])
    fig, axes = plt.subplots(len(METHODS), 1, figsize=(14, 8), sharex=True)
    for ax, m in zip(axes, METHODS):
        ax.plot(day, run[f"{m}_score"], lw=1)
        ax.scatter(day[run[f"{m}_alarm"]], run.loc[run[f"{m}_alarm"], f"{m}_score"], s=6, color="black", zorder=3)
        ax.fill_between(day, 0, 1, where=run["severity"] > 0, transform=ax.get_xaxis_transform(), color="red", alpha=0.15)
        if m == "pelt":
            for d in day[run["pelt_changepoint"]]:
                ax.axvline(d, color="green", ls="--", lw=0.8)
        ax.set_ylabel(m)
    cond = run.loc[run["label"] != "normal", "label"]
    axes[0].set_title(f"run {run_id} ({run['persona'].iloc[0]}, {cond.iloc[0] if len(cond) else 'normal'}): "
                      "score, alarms (dots), true anomaly (red), PELT change points (green)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    daily = pd.read_csv(DATASET / "daily.csv")
    labels = pd.read_csv(DATASET / "labels.csv")
    scores = run_detectors(daily)

    for setting in ("default", "calibrated"):
        if setting == "calibrated":
            thresholds = calibrate(scores)
            for m in METHODS:
                scores[f"{m}_alarm"] = scores[f"{m}_score"] > thresholds[m]
            print(f"\n=== calibrated thresholds (target {TARGET_FA_PER_30D} false alarms / 30 days on normal runs):",
                  {m: round(t, 3) for m, t in thresholds.items()})
        else:
            print("=== default thresholds (zscore 2 sigma, iforest 99th pct of baseline, pelt RMS z > 1)")
        episodes, false_alarms = evaluate(scores, labels)
        episodes.to_csv(OUTPUTS / f"t6_episodes_{setting}.csv")
        false_alarms.to_csv(OUTPUTS / f"t6_false_alarms_{setting}.csv")
        print("episode detection:")
        print(episodes.to_string())
        print("false alarms on normal days (alarm onsets per 30 days):")
        print(false_alarms.to_string())
    scores.to_csv(DATASET / "scores.csv", index=False)  # alarms = calibrated
    dating = onset_dating(scores, labels)
    dating.to_csv(OUTPUTS / "t6_onset_dating_pelt.csv")
    print("\nPELT onset dating (nearest change point - true onset, days):")
    print(dating.to_string())
    for cond in ("uti", "depression", "cognitive"):
        rid = labels.loc[(labels["condition"] == cond) & (labels["persona"] == "hh113"), "run_id"].iloc[0]
        plot_run(scores, rid, OUTPUTS / f"t6_scores_{cond}_hh113.png")


if __name__ == "__main__":
    main()
