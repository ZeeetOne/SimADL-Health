"""T8 — Monitor any person and produce a pre-consultation report (FR / EN) for their GP.

Input: the person's event list (activity, start, end), i.e. what an ambient/audio recognition layer outputs,
covering at least BASELINE_DAYS + 2 weeks before the consultation. The first BASELINE_DAYS days are taken
as the person's usual habits. Models come from `python -m simadl.train` (trained on synthetic data only).

The report shows:
  1. per-ADL summary: last 4 weeks vs the person's own baseline, flagged when unusual
  2. a 90-day chart of key signals with the detected periods
  3. detected change periods (calibrated T6 alarms), each with the most likely condition, a confidence
     level and the behaviours that support it (signed SHAP of the T7 model, stated as plain changes)
All sentences are templates filled with computed numbers (no generative AI), so nothing is invented.

Usage:
    python -m simadl.report --events patient_events.csv --consult 2026-11-20 --person "Mme D." --lang fr
    python -m simadl.report --demo     # held-out synthetic runs + the two real CASAS residents
"""
import argparse
import base64
import io
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import shap

from simadl.anomalies import BASELINE_DAYS, DATASET
from simadl.attribute import CONDITIONS, person_windows
from simadl.detect import score_iforest, score_zscore, weekly_signal
from simadl.features import OUTPUTS, PROCESSED, daily_features
from simadl.train import MODELS

REPORTS = OUTPUTS / "reports"
RECENT_DAYS = 28
CHART_DAYS = 90
MERGE_GAP_DAYS = 7
FLAG_Z = 2.0
SUPPORT_MIN_Z = 1.0  # a supporting behaviour must differ from habits by >= 1 weekly SD
MIN_EVENTS_PER_DAY = 5  # fewer recorded events = sensor/recognition outage, treated as missing, not as behaviour

# feature -> (ADL, FR label, EN label, kind) ; kind "clock" is printed hh:mm, "h" hours, "n" count
FEATURES = {
    "sleep_onset_h": ("sleep", "Heure du coucher", "Bedtime", "clock"),
    "sleep_hours": ("sleep", "Durée de sommeil (h)", "Sleep duration (h)", "h"),
    "sleep_episodes": ("sleep", "Épisodes de sommeil / nuit", "Sleep episodes / night", "n"),
    "night_toilet": ("sleep", "Passages aux toilettes la nuit", "Night-time toilet visits", "n"),
    "meals": ("eating", "Repas préparés à domicile / jour", "Meals prepared at home / day", "n"),
    "breakfast_h": ("eating", "Heure du petit-déjeuner", "Breakfast time", "clock"),
    "outings": ("social", "Sorties / jour", "Outings / day", "n"),
    "time_out_h": ("social", "Temps hors du domicile (h/jour)", "Time out of home (h/day)", "h"),
    "phone_calls": ("social", "Appels téléphoniques / jour", "Phone calls / day", "n"),
    "guest_visits": ("social", "Visites reçues / jour", "Visitors / day", "n"),
}
EXTRA = {"sleep_onset_sd": ("sleep", "Irrégularité de l'heure du coucher (h)", "Bedtime irregularity (h)", "h")}
DAILY_COLS = ["sleep_onset_h", "wake_h", "sleep_hours", "sleep_episodes", "night_toilet", "meals", "meal_episodes",
              "breakfast_h", "lunch_h", "dinner_h", "outings", "time_out_h", "phone_calls", "guest_visits", "guest_h"]

T = {
    "fr": dict(
        title="Rapport pré-consultation — activités de la vie quotidienne",
        patient="Personne", consult="Consultation prévue le", baseline="Référence (habitudes)",
        missing="jours sans données (exclus)",
        disclaimer="Outil d'aide à la discussion, pas un diagnostic. Les changements sont comparés aux habitudes "
                   "propres de la personne ; les hypothèses viennent d'un modèle entraîné sur des données simulées.",
        s1="1. Synthèse des 4 dernières semaines", s2="2. Évolution sur 90 jours", s3="3. Périodes de changement détectées",
        adl={"sleep": "Sommeil", "eating": "Alimentation", "social": "Vie sociale"},
        cols=("Indicateur", "Habituel", "4 dernières semaines", "Écart"),
        unusual="inhabituel", none="Aucun changement notable détecté sur les 90 derniers jours.",
        period="Du {a} au {b} ({n} j)", likely="Hypothèse la plus compatible : <b>{c}</b> (confiance {conf}, {p:.0%})",
        support="Comportements à l'appui :", other="Autres hypothèses : {o}",
        conf={"high": "élevée", "moderate": "modérée", "low": "faible"},
        cond={"uti": "infection urinaire (nycturie)", "depression": "épisode dépressif",
              "cognitive": "déclin cognitif", "normal": "pas de profil spécifique"},
        unclear="Changement détecté mais sans profil spécifique.",
        change="{f} : {v1} → {v2}", truth="Vérité terrain (données simulées, démonstration uniquement)",
        chart=("Coucher (h)", "Toilettes la nuit", "Repas à domicile", "Heures dehors"),
    ),
    "en": dict(
        title="Pre-consultation report — activities of daily living",
        patient="Person", consult="Consultation on", baseline="Baseline (usual habits)",
        missing="days without data (excluded)",
        disclaimer="Decision-support for the conversation, not a diagnosis. Changes are measured against the "
                   "person's own habits; hypotheses come from a model trained on simulated data.",
        s1="1. Last 4 weeks at a glance", s2="2. 90-day trends", s3="3. Detected change periods",
        adl={"sleep": "Sleep", "eating": "Eating", "social": "Social life"},
        cols=("Indicator", "Usual", "Last 4 weeks", "Change"),
        unusual="unusual", none="No notable change detected in the last 90 days.",
        period="From {a} to {b} ({n} d)", likely="Most compatible hypothesis: <b>{c}</b> ({conf} confidence, {p:.0%})",
        support="Supporting behaviours:", other="Other hypotheses: {o}",
        conf={"high": "high", "moderate": "moderate", "low": "low"},
        cond={"uti": "urinary tract infection (nocturia)", "depression": "depressive episode",
              "cognitive": "cognitive decline", "normal": "no specific pattern"},
        unclear="Change detected, but no specific pattern.",
        change="{f}: {v1} → {v2}", truth="Ground truth (simulated data, demonstration only)",
        chart=("Bedtime (h)", "Night toilet visits", "Meals at home", "Hours out"),
    ),
}


# ---------------------------------------------------------------- analysis

def load_models() -> tuple[dict, dict]:
    bundle = joblib.load(MODELS / "attribution.joblib")
    meta = json.loads((MODELS / "meta.json").read_text())
    bundle["explainer"] = shap.TreeExplainer(bundle["model"])
    return bundle, meta


def person_daily(events: pd.DataFrame, consult: pd.Timestamp) -> pd.DataFrame:
    """Daily features up to the consultation; outage days become missing values."""
    events = events.copy()
    events["start"], events["end"] = pd.to_datetime(events["start"]), pd.to_datetime(events["end"])
    if "duration_min" not in events:
        events["duration_min"] = (events["end"] - events["start"]).dt.total_seconds() / 60
    daily = daily_features(events, min_episodes=MIN_EVENTS_PER_DAY)
    daily = daily[daily.index < consult]
    daily.loc[~daily["valid"], DAILY_COLS] = np.nan
    if len(daily) < BASELINE_DAYS + 14:
        raise ValueError(f"need at least {BASELINE_DAYS + 14} days of history, got {len(daily)}")
    return daily.reset_index()


def periods(alarm: np.ndarray) -> list[tuple[int, int]]:
    """Merge alarm days separated by <= MERGE_GAP_DAYS into (start, end) index periods."""
    out = []
    for i in np.flatnonzero(alarm):
        if out and i - out[-1][1] <= MERGE_GAP_DAYS:
            out[-1][1] = i
        else:
            out.append([i, i])
    return [tuple(p) for p in out]


def detect(person: pd.DataFrame, meta: dict) -> list[tuple[int, int]]:
    z = weekly_signal(person)
    zs, _ = score_zscore(z)
    ifs, _ = score_iforest(z)
    alarm = (zs > meta["thresholds"]["zscore"]) | (ifs > meta["thresholds"]["iforest"])
    alarm[:BASELINE_DAYS] = False
    return [p for p in periods(alarm) if p[1] >= len(person) - CHART_DAYS]


def explain(bundle: dict, windows: pd.DataFrame, person: pd.DataFrame, a: int, b: int):
    model, cols = bundle["model"], bundle["columns"]
    rows = windows.iloc[a:b + 1]
    mean = pd.DataFrame(model.predict_proba(rows[cols]), columns=model.classes_).mean()
    cond_p = mean[CONDITIONS] / mean[CONDITIONS].sum()
    if mean["normal"] > mean[CONDITIONS].sum():
        return "normal", float(mean["normal"]), cond_p, []
    pred = cond_p.idxmax()
    sv = bundle["explainer"].shap_values(rows[cols])
    sv = np.stack(sv, axis=-1) if isinstance(sv, list) else sv
    contrib = pd.Series(sv[:, :, list(model.classes_).index(pred)].mean(axis=0), index=cols)
    # model inputs (z_, d_, s3_) grouped by the behaviour they describe; keep what pushes towards `pred`
    grouped = contrib.groupby(lambda c: c.split("_", 1)[1]).sum().sort_values(ascending=False)
    base, per = person.iloc[:BASELINE_DAYS], person.iloc[a:b + 1]
    support = []
    for f in grouped.index:
        if grouped[f] <= 0 or len(support) == 3:
            break
        if f == "sleep_onset_sd":
            weekly = base["sleep_onset_h"].rolling(7, min_periods=3).std()
            v1, v2, sd = weekly.mean(), per["sleep_onset_h"].std(), weekly.std()
        elif f in FEATURES:
            weekly = base[f].rolling(7, min_periods=3).mean()
            v1, v2, sd = base[f].mean(), per[f].mean(), weekly.std()
        else:
            continue
        # SHAP can favour a class because of what did NOT change (e.g. no extra toilet visits makes UTI less
        # likely); a clinician should only see behaviours that actually moved away from the person's habits
        if sd > 0 and abs(v2 - v1) >= SUPPORT_MIN_Z * sd:
            support.append((f, v1, v2))
    return pred, float(cond_p[pred]), cond_p, support


# ---------------------------------------------------------------- rendering

def fmt(value: float, kind: str) -> str:
    if pd.isna(value):
        return "—"
    if kind == "clock":
        minutes = int(round((value % 24) * 60)) % (24 * 60)
        return f"{minutes // 60:02d}:{minutes % 60:02d}"
    return f"{value:.1f}"


def confidence_level(p: float) -> str:
    return "high" if p >= 0.6 else "moderate" if p >= 0.45 else "low"


def summary_rows(person: pd.DataFrame, t: dict, name) -> list[str]:
    base, recent = person.iloc[:BASELINE_DAYS], person.iloc[-RECENT_DAYS:]
    weekly_sd = base[list(FEATURES)].rolling(7, min_periods=3).mean().std()
    rows = []
    for adl in ("sleep", "eating", "social"):
        rows.append(f'<tr class="adl"><td colspan="4">{t["adl"][adl]}</td></tr>')
        for f, (f_adl, _fr, _en, kind) in FEATURES.items():
            if f_adl != adl:
                continue
            b, r = base[f].mean(), recent[f].mean()
            delta = r - b
            z = delta / weekly_sd[f] if weekly_sd[f] > 0 else 0.0
            arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
            dtxt = f"{arrow} {abs(delta) * 60:.0f} min" if kind == "clock" else f"{arrow} {abs(delta):.1f}"
            flag = f' <span class="flag">{t["unusual"]}</span>' if abs(z) >= FLAG_Z else ""
            rows.append(f"<tr><td>{name(f)}</td><td>{fmt(b, kind)}</td><td>{fmt(r, kind)}</td><td>{dtxt}{flag}</td></tr>")
    return rows


def chart(person: pd.DataFrame, found: list, t: dict) -> str:
    lo = max(0, len(person) - CHART_DAYS)
    d, base = person.iloc[lo:], person.iloc[:BASELINE_DAYS]
    day = pd.to_datetime(d["day"])
    fig, axes = plt.subplots(4, 1, figsize=(9, 7.5), sharex=True)
    for ax, f, label in zip(axes, ["sleep_onset_h", "night_toilet", "meals", "time_out_h"], t["chart"]):
        ax.plot(day, d[f], color="#9aa5b1", lw=0.8)
        ax.plot(day, d[f].rolling(7, min_periods=3).mean(), color="#1f5f99", lw=2)
        m, s = base[f].mean(), base[f].rolling(7, min_periods=3).mean().std()
        ax.axhspan(m - 2 * s, m + 2 * s, color="#2e7d32", alpha=0.08)
        ax.axhline(m, color="#2e7d32", lw=0.8, ls="--")
        for a, b in found:
            ax.axvspan(pd.to_datetime(person["day"].iloc[max(a, lo)]), pd.to_datetime(person["day"].iloc[b]),
                       color="#c62828", alpha=0.12)
        ax.set_ylabel(label, fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def render(person: pd.DataFrame, person_id: str, consult: pd.Timestamp, found: list, explained: list,
           lang: str, truth: list[str] | None = None) -> str:
    t = T[lang]
    labels = {**FEATURES, **EXTRA}
    name = lambda f: labels[f][1 if lang == "fr" else 2]
    datefmt = "%d/%m/%Y" if lang == "fr" else "%Y-%m-%d"
    day = lambda i: pd.to_datetime(person["day"].iloc[i]).strftime(datefmt)
    blocks = []
    for (a, b), (pred, p, cond_p, support) in zip(found, explained):
        lines = [f'<h3>{t["period"].format(a=day(a), b=day(b), n=b - a + 1)}</h3>']
        if pred == "normal":
            lines.append(f"<p>{t['unclear']}</p>")
        else:
            lines.append(f"<p>{t['likely'].format(c=t['cond'][pred], conf=t['conf'][confidence_level(p)], p=p)}</p>")
            items = [t["change"].format(f=name(f), v1=fmt(v1, labels[f][3]), v2=fmt(v2, labels[f][3]))
                     for f, v1, v2 in support]
            if items:
                lines.append(f"<p>{t['support']}</p><ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>")
            others = ", ".join(f"{t['cond'][c]} {cond_p[c]:.0%}" for c in cond_p.sort_values(ascending=False).index if c != pred)
            lines.append(f'<p class="muted">{t["other"].format(o=others)}</p>')
        blocks.append('<div class="period">' + "".join(lines) + "</div>")
    if not blocks:
        blocks.append(f"<p>{t['none']}</p>")
    missing = int(person["sleep_onset_h"].isna().sum() if "valid" not in person else (~person["valid"]).sum())
    truth_html = (f"<details><summary>{t['truth']}</summary><ul>" + "".join(f"<li>{x}</li>" for x in truth)
                  + "</ul></details>") if truth is not None else ""
    return f"""<!doctype html><html lang="{lang}"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{t['title']}</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,sans-serif;max-width:820px;margin:24px auto;padding:0 16px;color:#1d2733;background:#fff}}
h1{{font-size:1.4em;margin-bottom:4px}} h2{{font-size:1.1em;border-bottom:1px solid #d5dbe1;padding-bottom:4px;margin-top:28px}}
h3{{font-size:1em;margin:0 0 6px}} table{{border-collapse:collapse;width:100%;font-size:.92em}}
td,th{{padding:5px 8px;border-bottom:1px solid #eef1f4;text-align:left}} tr.adl td{{font-weight:600;background:#f4f6f8}}
.flag{{background:#fdecea;color:#b71c1c;border-radius:4px;padding:1px 6px;font-size:.85em}}
.period{{border-left:4px solid #c62828;background:#fbf7f7;padding:10px 14px;margin:12px 0}}
.muted{{color:#5f6b78;font-size:.9em}} .disc{{background:#f4f6f8;padding:8px 12px;font-size:.88em;color:#3d4a57}}
img{{max-width:100%}} details{{margin-top:28px;font-size:.88em;color:#5f6b78}}
</style></head><body>
<h1>{t['title']}</h1>
<p>{t['patient']}: <b>{person_id}</b> · {t['consult']} <b>{consult.strftime(datefmt)}</b><br>
{t['baseline']}: {day(0)} – {day(BASELINE_DAYS - 1)} · {missing} {t['missing']}</p>
<p class="disc">{t['disclaimer']}</p>
<h2>{t['s1']}</h2><table><tr>{''.join(f'<th>{c}</th>' for c in t['cols'])}</tr>{''.join(summary_rows(person, t, name))}</table>
<h2>{t['s2']}</h2><img alt="{t['s2']}" src="data:image/png;base64,{chart(person, found, t)}">
<h2>{t['s3']}</h2>{''.join(blocks)}
{truth_html}
</body></html>"""


# ---------------------------------------------------------------- entry points

def generate_report(events: pd.DataFrame, consult, person_id: str, langs=("fr", "en"), truth=None,
                    bundle=None, meta=None, out_stem: str | None = None) -> dict:
    """Full pipeline for one person: events -> daily features -> detection -> attribution -> HTML files."""
    if bundle is None:
        bundle, meta = load_models()
    consult = pd.Timestamp(consult)
    person = person_daily(events, consult)
    found = detect(person, meta)
    windows = person_windows(person)
    explained = [explain(bundle, windows, person, a, b) for a, b in found]
    REPORTS.mkdir(parents=True, exist_ok=True)
    stem = out_stem or "".join(c if c.isalnum() else "_" for c in person_id)
    files = {}
    for lang in langs:
        path = REPORTS / f"report_{stem}_{lang}.html"
        path.write_text(render(person, person_id, consult, found, explained, lang, truth), encoding="utf-8")
        files[lang] = path
    return dict(periods=[(str(person["day"].iloc[a].date()), str(person["day"].iloc[b].date())) for a, b in found],
                hypotheses=[(e[0], round(e[1], 2)) for e in explained], files=files)


def demo() -> None:
    bundle, meta = load_models()
    labels = pd.read_csv(DATASET / "labels.csv")
    cases = []
    # held-out synthetic people (never seen in training), consultation a few days after the last episode
    for path in sorted(DATASET.glob("events_run*.csv")):
        run_id = int(path.name.split("_")[1][3:])
        events = pd.read_csv(path, parse_dates=["start", "end"])
        lab = labels[labels["run_id"] == run_id]
        if len(lab) and lab["condition"].iloc[0] in ("uti", "depression"):
            last = lab.iloc[-1]
            consult = pd.Timestamp(last["onset"]) + pd.Timedelta(days=int(last["duration_days"]) + 5)
        else:
            consult = events["start"].max().normalize()
        truth = [f"{r.condition}, onset {r.onset}, {r.duration_days} d, intensity {r.intensity}"
                 for r in lab.itertuples()] or ["normal (no anomaly injected)"]
        cases.append((path.stem.replace("events_", "synthetic_"), events, consult, truth))
    # the two real CASAS residents: no ground truth, sim-to-real check
    for home in ("hh113", "hh115"):
        events = pd.read_csv(PROCESSED / f"{home}_episodes.csv", parse_dates=["start", "end"])
        cases.append((f"real_casas_{home}", events, events["start"].max().normalize(), None))

    for name, events, consult, truth in cases:
        info = generate_report(events, consult, name, truth=truth, bundle=bundle, meta=meta, out_stem=name)
        print(f"{name} (consult {consult.date()}): periods {info['periods']} -> {info['hypotheses']}")
        if truth:
            print(f"    truth: {truth}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-consultation ADL report for one person.")
    ap.add_argument("--events", help="CSV with columns activity,start,end (one row per recognised activity)")
    ap.add_argument("--consult", help="consultation date YYYY-MM-DD (default: day after the last event)")
    ap.add_argument("--person", default="patient", help="name or identifier printed on the report")
    ap.add_argument("--lang", default="fr,en", help="fr, en or fr,en")
    ap.add_argument("--demo", action="store_true", help="reports for held-out synthetic runs and the real CASAS homes")
    args = ap.parse_args()
    if args.demo:
        return demo()
    if not args.events:
        ap.error("--events is required (or use --demo)")
    events = pd.read_csv(args.events, parse_dates=["start", "end"])
    consult = args.consult or (events["start"].max().normalize() + pd.Timedelta(days=1))
    info = generate_report(events, consult, args.person, langs=tuple(args.lang.split(",")))
    print(info)


if __name__ == "__main__":
    main()
