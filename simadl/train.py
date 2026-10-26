"""Train once on synthetic data, save the models used to monitor any person (simadl.report).

Saves to models/:
  attribution.joblib  random forest {normal, uti, depression, cognitive} (T7) + its input columns
  meta.json           detection thresholds calibrated on normal runs (T6) and training provenance
The example runs whose raw events are kept in data/processed/dataset/ are excluded from training,
so the demo reports are made for people the model has never seen.

Usage:
    python -m simadl.train
"""
import json
from datetime import date
from pathlib import Path

import joblib
import pandas as pd

from simadl.anomalies import DATASET
from simadl.attribute import build_windows, feature_cols, make_model, training_rows
from simadl.detect import TARGET_FA_PER_30D, calibrate, run_detectors

MODELS = Path(__file__).resolve().parent.parent / "models"


def example_run_ids() -> list[int]:
    return sorted(int(p.name.split("_")[1][3:]) for p in DATASET.glob("events_run*.csv"))


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
                target_false_alarms_per_30d=TARGET_FA_PER_30D)
    (MODELS / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
