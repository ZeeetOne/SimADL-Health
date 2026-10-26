"""T7 — Attribute a behaviour change to the most likely condition (UTI, depression, cognitive decline).

Input per day: the weekly z-scores of T6 (relative to the person's own baseline) plus their change over
the previous 14 days (sudden vs gradual). A random forest predicts {normal, uti, depression, cognitive}.
Training uses days with severity >= MIN_SEVERITY for conditions and post-baseline days of normal runs.

Evaluation:
  - cross-persona: train on one home, test on the other (unseen person), the main result
  - within-persona: GroupKFold by run, as a reference
  - episode level: for each injected episode, average the condition probabilities over its days
SHAP (TreeExplainer) explains which features push a prediction, for the clinician report (T8).

Usage:
    python -m simadl.attribute
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

from simadl.anomalies import BASELINE_DAYS, DATASET
from simadl.detect import weekly_signal
from simadl.features import OUTPUTS

CLASSES = ["normal", "uti", "depression", "cognitive"]
CONDITIONS = CLASSES[1:]
DELTA_DAYS = 14
MIN_SEVERITY = 0.5
SHORT_FEATURES = ["night_toilet", "sleep_episodes"]


def build_windows(daily: pd.DataFrame) -> pd.DataFrame:
    """One row per day: z (this week vs baseline) and dz (this week vs two weeks earlier)."""
    rows = []
    for run_id, run in daily.groupby("run_id", sort=False):
        run = run.reset_index()
        z = weekly_signal(run)
        dz = (z - z.shift(DELTA_DAYS)).add_prefix("d_")
        # short 3-day window for night signals: a 4-9 day UTI is diluted in a weekly mean
        night = run[SHORT_FEATURES].rolling(3, min_periods=2).mean()
        base = night.iloc[:BASELINE_DAYS]
        s3 = ((night - base.mean()) / base.std().replace(0, 1)).add_prefix("s3_")
        x = pd.concat([z.add_prefix("z_"), dz, s3], axis=1)
        x[["run_id", "persona", "day", "label", "severity"]] = run[["run_id", "persona", "day", "label", "severity"]]
        x["t"] = np.arange(len(run))
        rows.append(x.iloc[BASELINE_DAYS:])
    return pd.concat(rows, ignore_index=True).fillna(0.0)


def training_rows(w: pd.DataFrame) -> pd.DataFrame:
    normal_runs = w.groupby("run_id")["label"].transform(lambda s: (s == "normal").all())
    keep = (w["severity"] >= MIN_SEVERITY) | normal_runs
    return w[keep]


def feature_cols(w: pd.DataFrame) -> list[str]:
    return [c for c in w.columns if c.startswith(("z_", "d_", "s3_"))]


def make_model(seed: int = 0) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                                  random_state=seed, n_jobs=-1)


def episode_attribution(model, w: pd.DataFrame, labels: pd.DataFrame, cols) -> pd.DataFrame:
    """Most likely condition per injected episode = argmax of mean condition probability over its days."""
    rows = []
    for ep in labels.itertuples():
        d = w[(w["run_id"] == ep.run_id) & (w["severity"] > 0)]
        d = d[pd.to_datetime(d["day"]).between(pd.Timestamp(ep.onset), pd.Timestamp(ep.onset) + pd.Timedelta(days=ep.duration_days))]
        if d.empty:
            continue
        proba = pd.DataFrame(model.predict_proba(d[cols]), columns=model.classes_)[CONDITIONS].mean()
        rows.append(dict(run_id=ep.run_id, persona=ep.persona, true=ep.condition, pred=proba.idxmax(),
                         confidence=round(proba.max() / proba.sum(), 3)))
    return pd.DataFrame(rows)


def main() -> None:
    daily = pd.read_csv(DATASET / "daily.csv")
    labels = pd.read_csv(DATASET / "labels.csv")
    w = build_windows(daily)
    cols = feature_cols(w)
    train_all = training_rows(w)
    lines, ep_results = [], []

    # within-persona reference: GroupKFold by run
    for persona in ("hh113", "hh115"):
        d = train_all[train_all["persona"] == persona]
        pred = np.empty(len(d), dtype=object)
        for tr, te in GroupKFold(n_splits=4).split(d, groups=d["run_id"]):
            pred[te] = make_model().fit(d.iloc[tr][cols], d.iloc[tr]["label"]).predict(d.iloc[te][cols])
        f1 = f1_score(d["label"], pred, average="macro")
        lines.append(dict(setting="within-persona (GroupKFold by run)", train=persona, test=persona, day_macro_F1=f1))

    # cross-persona: the main result
    models = {}
    for train_p, test_p in (("hh113", "hh115"), ("hh115", "hh113")):
        tr = train_all[train_all["persona"] == train_p]
        te = train_all[train_all["persona"] == test_p]
        model = make_model().fit(tr[cols], tr["label"])
        models[train_p] = model
        pred = model.predict(te[cols])
        lines.append(dict(setting="cross-persona", train=train_p, test=test_p,
                          day_macro_F1=f1_score(te["label"], pred, average="macro")))
        cm = confusion_matrix(te["label"], pred, labels=CLASSES, normalize="true")
        ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(values_format=".2f", cmap="Blues")
        plt.title(f"day-level, train {train_p} -> test {test_p}")
        plt.tight_layout()
        plt.savefig(OUTPUTS / f"t7_confusion_{train_p}_to_{test_p}.png", dpi=110)
        plt.close()
        er = episode_attribution(model, w[w["persona"] == test_p], labels[labels["persona"] == test_p], cols)
        er["train"] = train_p
        ep_results.append(er)

    summary = pd.DataFrame(lines).round(3)
    episodes = pd.concat(ep_results, ignore_index=True)
    episodes.to_csv(OUTPUTS / "t7_episode_attribution.csv", index=False)
    summary.to_csv(OUTPUTS / "t7_summary.csv", index=False)
    print(summary.to_string(index=False))
    print("\nepisode-level attribution on the unseen persona (accuracy per true condition):")
    acc = episodes.assign(correct=episodes["true"] == episodes["pred"]).groupby(["train", "true"]).agg(
        episodes=("correct", "size"), accuracy=("correct", "mean"), mean_confidence=("confidence", "mean"))
    print(acc.round(2).to_string())
    print("\nconfusion (rows = true, cols = predicted), both directions pooled:")
    print(pd.crosstab(episodes["true"], episodes["pred"]).reindex(index=CONDITIONS, columns=CONDITIONS, fill_value=0))

    # SHAP: global importance per class for the model trained on hh113 (explaining hh115 days)
    model = models["hh113"]
    te = train_all[train_all["persona"] == "hh115"].sample(1500, random_state=0)
    sv = shap.TreeExplainer(model).shap_values(te[cols])
    sv = np.stack(sv, axis=-1) if isinstance(sv, list) else sv  # -> (n, features, classes)
    imp = pd.DataFrame({c: np.abs(sv[:, :, list(model.classes_).index(c)]).mean(axis=0) for c in CONDITIONS},
                       index=cols)
    top = imp.loc[imp.sum(axis=1).sort_values(ascending=False).index[:12]]
    top.to_csv(OUTPUTS / "t7_shap_importance.csv")
    print("\nmean |SHAP| per condition, top features (model hh113, explained on hh115):")
    print(top.round(3).to_string())
    top[::-1].plot.barh(figsize=(9, 7))
    plt.xlabel("mean |SHAP value| (contribution to class probability)")
    plt.title("What drives each condition prediction (train hh113, explain hh115)")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "t7_shap_importance.png", dpi=110)
    plt.close()


if __name__ == "__main__":
    main()
