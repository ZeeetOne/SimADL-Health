"""T5 — Realism of simulated normal behaviour.

Compares daily-feature distributions of simulated *normal* runs with real days, using:
  - the two-sample Kolmogorov-Smirnov statistic D per feature (0 = identical distributions), and
  - a classifier two-sample test: a random forest tries to tell real from simulated days
    (cross-validated AUC 0.5 = indistinguishable, 1.0 = trivially separable).
Following SynSys, simulated-vs-real scores are read against reference pairs: the same real home in
another period (how much reality differs from itself) and a different real home.

Usage:
    python -m simadl.validate
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from simadl.anomalies import DATASET
from simadl.features import FEATURES, OUTPUTS, PROCESSED

# breakfast/dinner times are missing on days without that meal; the classifier sees them as -1
CLF_FEATURES = FEATURES


def real_days(home: str, split: str | None = None) -> pd.DataFrame:
    d = pd.read_csv(PROCESSED / f"{home}_daily.csv")
    d = d[d["valid"]]
    return d[d["split"] == split] if split else d


def sim_normal_days(persona: str) -> pd.DataFrame:
    d = pd.read_csv(DATASET / "daily.csv")
    runs = d.groupby("run_id")["label"].transform(lambda s: (s == "normal").all())
    return d[runs & (d["persona"] == persona)]


def ks_table(a: pd.DataFrame, b: pd.DataFrame) -> pd.Series:
    """KS D per feature; NaN when one side never has the feature (e.g. no dinners at home in a period)."""
    return pd.Series({f: ks_2samp(a[f].dropna(), b[f].dropna()).statistic
                      if a[f].notna().sum() > 1 and b[f].notna().sum() > 1 else np.nan for f in FEATURES})


def _xy(a, b, seed):
    n = min(len(a), len(b))  # balance classes
    a, b = a.sample(n, random_state=seed), b.sample(n, random_state=seed)
    X = pd.concat([a[CLF_FEATURES], b[CLF_FEATURES]]).fillna(-1).to_numpy()
    return X, np.r_[np.zeros(n), np.ones(n)]


def discriminator_importance(a: pd.DataFrame, b: pd.DataFrame, seed: int = 0) -> pd.Series:
    """Which features let the discriminator tell the two sets apart."""
    X, y = _xy(a, b, seed)
    clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=seed, n_jobs=-1).fit(X, y)
    return pd.Series(clf.feature_importances_, index=CLF_FEATURES).sort_values(ascending=False)


def discriminator_auc(a: pd.DataFrame, b: pd.DataFrame, seed: int = 0) -> float:
    X, y = _xy(a, b, seed)
    clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=seed, n_jobs=-1)
    return float(cross_val_score(clf, X, y, cv=5, scoring="roc_auc").mean())


def main() -> None:
    pairs = {
        # name: (A, B, kind)
        "sim hh113 vs real hh113 calibration": (sim_normal_days("hh113"), real_days("hh113", "calibration"), "fit"),
        "sim hh113 vs real hh113 holdout": (sim_normal_days("hh113"), real_days("hh113", "holdout"), "test"),
        "sim hh115 vs real hh115": (sim_normal_days("hh115"), real_days("hh115"), "fit"),
        "real hh113 calibration vs holdout": (real_days("hh113", "calibration"), real_days("hh113", "holdout"), "reference"),
        "real hh113 vs real hh115": (real_days("hh113"), real_days("hh115"), "reference"),
        "sim hh113 vs real hh115": (sim_normal_days("hh113"), real_days("hh115"), "reference"),
    }
    ks, rows = {}, []
    for name, (a, b, kind) in pairs.items():
        ks[name] = ks_table(a, b)
        rows.append(dict(pair=name, kind=kind, n_a=len(a), n_b=len(b), mean_KS_D=ks[name].mean(),
                         discriminator_AUC=discriminator_auc(a, b)))
    summary = pd.DataFrame(rows).round(3)
    ks = pd.DataFrame(ks).round(3)
    summary.to_csv(OUTPUTS / "t5_summary.csv", index=False)
    ks.to_csv(OUTPUTS / "t5_ks_per_feature.csv")
    print(summary.to_string(index=False))
    print("\nKS statistic D per feature:")
    print(ks.to_string())
    imp = pd.DataFrame({name: discriminator_importance(*pairs[name][:2]) for name in
                        ("sim hh113 vs real hh113 calibration", "sim hh115 vs real hh115")}).round(3)
    imp.to_csv(OUTPUTS / "t5_discriminator_importance.csv")
    print("\ndiscriminator feature importance (what gives simulated days away):")
    print(imp.to_string())

    fig, ax = plt.subplots(figsize=(12, 6))
    cols = ["sim hh113 vs real hh113 holdout", "real hh113 calibration vs holdout", "real hh113 vs real hh115"]
    ks[cols].plot.bar(ax=ax, width=0.8)
    ax.set_ylabel("KS statistic D (lower = more similar)")
    ax.set_title("Simulated vs real daily features, against real-vs-real references")
    fig.tight_layout()
    fig.savefig(OUTPUTS / "t5_ks.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
