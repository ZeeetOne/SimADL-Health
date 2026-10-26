"""Train once on synthetic data, save the models used to monitor any person (simadl.report).

Saves to models/:
  attribution.joblib  random forest {normal, uti, depression, cognitive} (T7) + its input columns
  meta.json           detection thresholds and training provenance:
                        thresholds         day level, calibrated to TARGET_FA_PER_30D (T6, research evaluation)
                        report_thresholds  report level, per reference mode: chosen so that only
                                           TARGET_FALSE_REPORTS of normal 35-day reports show a change
The example runs whose raw events are kept in data/processed/dataset/ are excluded from training,
so the demo reports are made for people the model has never seen.

Usage:
    python -m simadl.train
"""
import json
from datetime import date

import joblib
import numpy as np
import pandas as pd

from simadl.anomalies import DATASET
from simadl.attribute import build_windows, feature_cols, make_model, training_rows
from simadl.detect import TARGET_FA_PER_30D, calibrate, run_detectors
from simadl.features import MODELS
from simadl.report import DAILY_COLS, PERIOD_DAYS, REFERENCE_DAYS, period_scores

TARGET_FALSE_REPORTS = 0.2  # share of normal reports allowed to show a (false) change


def example_run_ids() -> list[int]:
    return sorted(int(p.name.split("_")[1][3:]) for p in DATASET.glob("events_run*.csv"))


def report_thresholds(daily: pd.DataFrame) -> dict:
    """Per reference mode, thresholds such that <= TARGET_FALSE_REPORTS of the 35-day reports of normal runs
    contain a change. Each detector gets half of the budget (they are combined with OR)."""
    normal = daily[daily.groupby("run_id")["label"].transform(lambda s: (s == "normal").all())]
    q = 1 - TARGET_FALSE_REPORTS / 2
    out = {}
    for mode in ("rolling", "initial"):
        peaks = {"zscore": [], "iforest": []}
        for _, run in normal.groupby("run_id"):
            person = run.reset_index(drop=True)[["day", *DAILY_COLS]].assign(valid=True)
            for p0 in range(REFERENCE_DAYS, len(person) - PERIOD_DAYS + 1, PERIOD_DAYS):
                _, off, zs, ifs = period_scores(person, p0, p0 + PERIOD_DAYS, mode)
                peaks["zscore"].append(zs[off:].max())
                peaks["iforest"].append(ifs[off:].max())
        out[mode] = {k: round(float(np.quantile(v, q)), 4) for k, v in peaks.items()}
        out[mode]["normal_reports"] = len(peaks["zscore"])
    return out


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    daily = pd.read_csv(DATASET / "daily.csv")
    held_out = example_run_ids()
    daily = daily[~daily["run_id"].isin(held_out)]

    thresholds = calibrate(run_detectors(daily))
    w = training_rows(build_windows(daily))
    cols = feature_cols(w)
    model = make_model().fit(w[cols], w["label"])
    joblib.dump({"model": model, "columns": cols}, MODELS / "attribution.joblib", compress=3)
    meta = dict(trained_on=str(date.today()), runs=int(daily["run_id"].nunique()), held_out_example_runs=held_out,
                training_days=len(w), classes=list(model.classes_),
                thresholds={k: round(v, 4) for k, v in thresholds.items() if k in ("zscore", "iforest")},
                target_false_alarms_per_30d=TARGET_FA_PER_30D,
                report_thresholds=report_thresholds(daily), target_false_reports=TARGET_FALSE_REPORTS)
    (MODELS / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
