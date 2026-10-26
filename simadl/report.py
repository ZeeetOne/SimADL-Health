"""T8 — Monitor any person and produce pre-consultation reports (English) for their GP.

Input: the person's event list (activity, start, end), i.e. what an ambient/audio recognition layer outputs.
Models come from `python -m simadl.train` (trained on synthetic data only).

A report covers one PERIOD (default: the 35 days before the consultation = 5 of each weekday, as in the
Montreal monthly-report study) and compares it with a REFERENCE of 60 days:
  - "rolling": the 60 days just before the period (robust to slow drift of habits or of the sensing)
  - "initial": the first 60 days of monitoring (keeps slow decline visible against the original habits)
The long-term chart always shows the initial reference, so slow decline stays visible in both modes.

Output: an HTML page (screen) and an A4 PDF (to send, print or attach to the medical record).
Each report contains: data sufficiency (>= 16 valid days per 30, as the Medicare RPM rule), a per-ADL table,
a long-term chart, and the change periods detected in the period with a hypothesis (or "indeterminate"),
its confidence, the behaviours that changed, and "ongoing since <date>" when a change continues from the
previous report. Sentences are templates filled with computed numbers (no generative AI).

Usage:
    python -m simadl.report --events patient.csv --person "Mrs D." [--consult 2026-11-20] [--since 2026-10-16]
    python -m simadl.report --events patient.csv --person "Mrs D." --series      # a report every 35 days
    python -m simadl.report --demo
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
from simadl.features import MODELS, OUTPUTS, PROCESSED, daily_features

REPORTS = OUTPUTS / "reports"
REFERENCE_DAYS = BASELINE_DAYS
PERIOD_DAYS = 35            # 5 of each weekday
MIN_VALID_SHARE = 16 / 30   # data sufficiency, as Medicare RPM (>= 16 days of data per 30)
MERGE_GAP_DAYS = 7
FLAG_Z = 2.0
SUPPORT_MIN_Z = 1.0         # a supporting behaviour must differ from the reference by >= 1 weekly SD
ABSTAIN_BELOW = 0.45        # below this share of condition probability the hypothesis is "indeterminate"
ONGOING_TOLERANCE = 3       # days: a change touching both period borders continues the previous one
PERSIST_DAYS = 3           # a detector score must stay above threshold 3 days in a row to count
MIN_EVENTS_PER_DAY = 5      # fewer recorded events = sensing outage, treated as missing, not as behaviour
CHART_HISTORY_DAYS = 365

# feature -> (ADL, label, kind) ; kind "clock" is printed hh:mm, "h" hours, "n" count
FEATURES = {
    "sleep_onset_h": ("sleep", "Bedtime", "clock"),
    "sleep_hours": ("sleep", "Sleep duration (h)", "h"),
    "sleep_episodes": ("sleep", "Sleep episodes / night", "n"),
    "night_toilet": ("sleep", "Night-time toilet visits", "n"),
    "meals": ("eating", "Meals prepared at home / day", "n"),
    "breakfast_h": ("eating", "Breakfast time", "clock"),
    "outings": ("social", "Outings / day", "n"),
    "time_out_h": ("social", "Time out of home (h/day)", "h"),
    "phone_calls": ("social", "Phone calls / day", "n"),
    "guest_visits": ("social", "Visitors / day", "n"),
}
EXTRA = {"sleep_onset_sd": ("sleep", "Bedtime irregularity (h)", "h")}
DAILY_COLS = ["sleep_onset_h", "wake_h", "sleep_hours", "sleep_episodes", "night_toilet", "meals", "meal_episodes",
              "breakfast_h", "lunch_h", "dinner_h", "outings", "time_out_h", "phone_calls", "guest_visits", "guest_h"]
ADL = {"sleep": "Sleep", "eating": "Eating", "social": "Social life"}
COND = {"uti": "urinary tract infection (nocturia)", "depression": "depressive episode",
        "cognitive": "cognitive decline"}
# expected direction of change per condition, from docs/symptom_catalog.md (+1 up, -1 down, 0 either way);
# a behaviour is shown as "supporting" only if it moved in this direction
EXPECTED = {
    "uti": {"night_toilet": 1, "sleep_episodes": 1, "sleep_hours": -1, "meals": -1},
    "depression": {"sleep_onset_h": 1, "sleep_hours": 0, "sleep_episodes": 1, "meals": -1, "outings": -1,
                   "time_out_h": -1, "phone_calls": -1, "guest_visits": -1},
    "cognitive": {"sleep_onset_sd": 1, "sleep_episodes": 1, "outings": -1, "time_out_h": -1, "phone_calls": -1,
                  "guest_visits": -1, "breakfast_h": 0},
}


# ---------------------------------------------------------------- analysis

def load_models() -> tuple[dict, dict]:
    bundle = joblib.load(MODELS / "attribution.joblib")
    meta = json.loads((MODELS / "meta.json").read_text())
    bundle["explainer"] = shap.TreeExplainer(bundle["model"])
    return bundle, meta


def person_daily(events: pd.DataFrame) -> pd.DataFrame:
    """Daily features of the whole history; outage days become missing values."""
    events = events.copy()
    events["start"], events["end"] = pd.to_datetime(events["start"]), pd.to_datetime(events["end"])
    if "duration_min" not in events:
        events["duration_min"] = (events["end"] - events["start"]).dt.total_seconds() / 60
    daily = daily_features(events, min_episodes=MIN_EVENTS_PER_DAY)
    daily.loc[~daily["valid"], DAILY_COLS] = np.nan
    return daily.reset_index()[["day", "valid", *DAILY_COLS]]


def _periods(alarm: np.ndarray) -> list[list[int]]:
    """Merge alarm days separated by <= MERGE_GAP_DAYS into [start, end] index periods."""
    out = []
    for i in np.flatnonzero(alarm):
        if out and i - out[-1][1] <= MERGE_GAP_DAYS:
            out[-1][1] = i
        else:
            out.append([i, i])
    return out


def _explain(bundle: dict, windows: pd.DataFrame, frame: pd.DataFrame, a: int, b: int) -> dict:
    model, cols = bundle["model"], bundle["columns"]
    rows = windows.iloc[a:b + 1]
    mean = pd.DataFrame(model.predict_proba(rows[cols]), columns=model.classes_).mean()
    cond_p = mean[CONDITIONS] / mean[CONDITIONS].sum()
    pred = cond_p.idxmax()
    if mean["normal"] > mean[CONDITIONS].sum():
        status = "unspecific"
    elif cond_p[pred] < ABSTAIN_BELOW:
        status = "indeterminate"
    else:
        status = "hypothesis"
    support = []
    if status == "hypothesis":
        sv = bundle["explainer"].shap_values(rows[cols])
        sv = np.stack(sv, axis=-1) if isinstance(sv, list) else sv
        contrib = pd.Series(sv[:, :, list(model.classes_).index(pred)].mean(axis=0), index=cols)
        # model inputs (z_, d_, s3_) grouped by the behaviour they describe; keep what pushes towards `pred`
        grouped = contrib.groupby(lambda c: c.split("_", 1)[1]).sum().sort_values(ascending=False)
        ref, per = frame.iloc[:REFERENCE_DAYS], frame.iloc[a:b + 1]
        for f in grouped.index:
            if grouped[f] <= 0 or len(support) == 3:
                break
            if f == "sleep_onset_sd":
                # same 7-day rolling SD on both sides (a 3-day change would otherwise look "regular")
                rolling_sd = frame["sleep_onset_h"].rolling(7, min_periods=3).std()
                weekly = rolling_sd.iloc[:REFERENCE_DAYS]
                v1, v2, sd = weekly.mean(), rolling_sd.iloc[a:b + 1].mean(), weekly.std()
            elif f in FEATURES:
                weekly = ref[f].rolling(7, min_periods=3).mean()
                v1, v2, sd = ref[f].mean(), per[f].mean(), weekly.std()
            else:
                continue
            # SHAP may favour a class because of what did NOT change (e.g. no extra toilet visits argues against UTI);
            # a clinician should only see behaviours that moved, in the direction the symptom catalogue expects
            expected = EXPECTED[pred].get(f)
            moved = sd > 0 and abs(v2 - v1) >= SUPPORT_MIN_Z * sd
            if moved and expected is not None and (expected == 0 or np.sign(v2 - v1) == expected):
                support.append((f, v1, v2))
    return dict(status=status, condition=pred, confidence=float(cond_p[pred]), cond_p=cond_p, support=support)


def period_scores(person: pd.DataFrame, p0: int, p1: int, mode: str):
    """Frame (reference + period), offset of the period, and persistent daily scores of both detectors.
    A score only counts when it stays high PERSIST_DAYS days in a row (min over a 3-day window)."""
    ref0 = 0 if mode == "initial" else p0 - REFERENCE_DAYS
    if ref0 < 0 or p0 < REFERENCE_DAYS:
        raise ValueError(f"need {REFERENCE_DAYS} days of history before the report period")
    frame = person.iloc[ref0:p1].reset_index(drop=True)
    z = weekly_signal(frame)
    zs, _ = score_zscore(z)
    ifs, _ = score_iforest(z)
    persist = lambda s: pd.Series(s).rolling(PERSIST_DAYS, min_periods=1).min().to_numpy()
    return frame, p0 - ref0, persist(zs), persist(ifs)


def analyse(person: pd.DataFrame, p0: int, p1: int, mode: str, bundle: dict, meta: dict, prev: dict | None = None) -> dict:
    """Analyse the period person[p0:p1] against a 60-day reference ("rolling" or "initial")."""
    frame, off, zs, ifs = period_scores(person, p0, p1, mode)
    thr = meta["report_thresholds"][mode]
    alarm = (zs > thr["zscore"]) | (ifs > thr["iforest"])
    alarm[:off] = False
    windows = person_windows(frame)
    changes = []
    for a, b in _periods(alarm):
        a = max(off, a - PERSIST_DAYS + 1)  # an alarm needs PERSIST_DAYS high days: the change starts at the first
        c = _explain(bundle, windows, frame, a, b)
        c.update(start=frame["day"].iloc[a], end=frame["day"].iloc[b], since=frame["day"].iloc[a],
                 at_period_start=a - off <= ONGOING_TOLERANCE, at_period_end=(len(frame) - 1) - b <= ONGOING_TOLERANCE)
        if prev and c["at_period_start"]:
            carried = [pc for pc in prev["changes"] if pc["at_period_end"]]
            if carried:
                c["since"] = carried[0]["since"]
        changes.append(c)
    period = frame.iloc[off:]
    return dict(mode=mode, frame=frame, off=off, person=person, p0=p0, p1=p1,
                period_start=frame["day"].iloc[off], period_end=frame["day"].iloc[-1],
                valid_share=float(period["valid"].mean()), reference_valid_share=float(frame.iloc[:off]["valid"].mean()),
                changes=changes)


# ---------------------------------------------------------------- rendering

def fmt(value: float, kind: str) -> str:
    if pd.isna(value):
        return "—"
    if kind == "clock":
        minutes = int(round((value % 24) * 60)) % (24 * 60)
        return f"{minutes // 60:02d}:{minutes % 60:02d}"
    return f"{value:.1f}"


def _date(d) -> str:
    return pd.Timestamp(d).strftime("%d %b %Y")


# Report content (shared by the HTML and the PDF renderers) -------------------

def summary_data(r: dict) -> list[dict]:
    """Per-ADL table: one header row per ADL, then one row per indicator."""
    frame, off, person = r["frame"], r["off"], r["person"]
    ref, period, initial = frame.iloc[:off], frame.iloc[off:], person.iloc[:REFERENCE_DAYS]
    weekly_sd = ref[list(FEATURES)].rolling(7, min_periods=3).mean().std()
    rows = []
    for adl, title in ADL.items():
        rows.append(dict(header=title))
        for f, (f_adl, label, kind) in FEATURES.items():
            if f_adl != adl:
                continue
            i, b, p = initial[f].mean(), ref[f].mean(), period[f].mean()
            delta = p - b
            z = delta / weekly_sd[f] if weekly_sd[f] > 0 else 0.0
            arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
            dtxt = f"{arrow} {abs(delta) * 60:.0f} min" if kind == "clock" else f"{arrow} {abs(delta):.1f}"
            rows.append(dict(label=label, initial=fmt(i, kind), reference=fmt(b, kind), period=fmt(p, kind),
                             change=dtxt, unusual=abs(z) >= FLAG_Z))
    return rows


def table_columns(r: dict) -> tuple:
    return ("Indicator", "Initial habits", "Reference", "This period", "Change") if r["mode"] == "rolling" else \
           ("Indicator", "Reference", "This period", "Change")


def header_lines(r: dict, person_id: str) -> list[str]:
    consult = pd.Timestamp(r["period_end"]) + pd.Timedelta(days=1)
    ref = r["frame"].iloc[:r["off"]]
    span = f"{_date(ref['day'].iloc[0])} – {_date(ref['day'].iloc[-1])}"
    ref_txt = f"the 60 days before this period ({span})" if r["mode"] == "rolling" else \
              f"the first 60 days of monitoring ({span})"
    return [f"Person: {person_id} · Consultation: {_date(consult)}",
            f"Period covered: {_date(r['period_start'])} – {_date(r['period_end'])} "
            f"({len(r['frame']) - r['off']} days, {r['valid_share']:.0%} with data)",
            f"Compared with: {ref_txt}"]


DISCLAIMER = ("Decision support for the conversation, not a diagnosis. Changes are measured against the person's own "
              "habits; hypotheses come from a model trained on simulated data.")
NO_CHANGE = ("No change in daily routine was detected in this period. This does not mean the person is clinically "
             "stable: the system only sees sleep, eating and social routines.")


def warnings_text(r: dict) -> list[str]:
    out = []
    if r["valid_share"] < MIN_VALID_SHARE:
        out.append(f"Insufficient data: only {r['valid_share']:.0%} of days in this period have usable recordings "
                   f"(need ≥ {MIN_VALID_SHARE:.0%}). Interpret with caution.")
    if r["reference_valid_share"] < MIN_VALID_SHARE:
        out.append(f"The reference period itself has only {r['reference_valid_share']:.0%} usable days.")
    return out


def change_text(c: dict) -> dict:
    """Plain-text content of one detected change: heading, sentences, supporting behaviours, all hypotheses."""
    labels = {**FEATURES, **EXTRA}
    head = f"{_date(c['start'])} – {_date(c['end'])} ({(c['end'] - c['start']).days + 1} d)"
    if c["since"] < c["start"]:
        head += f" · ongoing since {_date(c['since'])} (continues from the previous report)"
    others = ", ".join(f"{COND[k]} {c['cond_p'][k]:.0%}" for k in c["cond_p"].sort_values(ascending=False).index)
    support = [f"{labels[f][1]}: {fmt(v1, labels[f][2])} → {fmt(v2, labels[f][2])}" for f, v1, v2 in c["support"]]
    if c["status"] == "unspecific":
        lead = ("A change in daily routine was detected, but it does not match any of the patterns the system knows "
                "(UTI, depression, cognitive decline).")
    elif c["status"] == "indeterminate":
        lead = f"Indeterminate: a change was detected, but no hypothesis is clearly supported ({others}). Worth discussing with the person."
    else:
        level = "high" if c["confidence"] >= 0.6 else "moderate"
        lead = f"Most compatible hypothesis: {COND[c['condition']]} ({level} confidence, {c['confidence']:.0%})."
    return dict(head=head, lead=lead, support=support, others=others if c["status"] == "hypothesis" else "",
                condition=COND.get(c["condition"], ""), status=c["status"])


def chart_figure(r: dict, figsize=(9, 7.2)):
    person = r["person"].iloc[:r["p1"]]
    person = person.iloc[max(0, len(person) - CHART_HISTORY_DAYS):]
    initial = r["person"].iloc[:REFERENCE_DAYS]
    day = pd.to_datetime(person["day"])
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    for ax, f, label in zip(axes, ["sleep_onset_h", "night_toilet", "meals", "time_out_h"],
                            ("Bedtime (h)", "Night toilet visits", "Meals at home", "Hours out")):
        ax.plot(day, person[f], color="#c3cad3", lw=0.6)
        ax.plot(day, person[f].rolling(7, min_periods=3).mean(), color="#1f5f99", lw=1.8)
        m, s = initial[f].mean(), initial[f].rolling(7, min_periods=3).mean().std()
        ax.axhspan(m - 2 * s, m + 2 * s, color="#2e7d32", alpha=0.08)
        ax.axhline(m, color="#2e7d32", lw=0.8, ls="--")
        ax.axvspan(pd.Timestamp(r["period_start"]), pd.Timestamp(r["period_end"]), color="#5f6b78", alpha=0.08)
        for c in r["changes"]:
            ax.axvspan(pd.Timestamp(c["start"]), pd.Timestamp(c["end"]), color="#c62828", alpha=0.15)
        ax.set_ylabel(label, fontsize=8)
    axes[0].set_title("Weekly mean (blue) vs initial habits (green band); grey = this period, red = detected change",
                      fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


# HTML renderer -----------------------------------------------------------------

def chart(r: dict) -> str:
    fig = chart_figure(r)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def summary_rows(r: dict) -> list[str]:
    rows = []
    for row in summary_data(r):
        if "header" in row:
            rows.append(f'<tr class="adl"><td colspan="5">{row["header"]}</td></tr>')
            continue
        flag = ' <span class="flag">unusual</span>' if row["unusual"] else ""
        first = f"<td class='muted'>{row['initial']}</td>" if r["mode"] == "rolling" else ""
        rows.append(f"<tr><td>{row['label']}</td>{first}<td>{row['reference']}</td><td>{row['period']}</td>"
                    f"<td>{row['change']}{flag}</td></tr>")
    return rows


def change_block(c: dict) -> str:
    t = change_text(c)
    head = t["head"].replace(" · ongoing since", " · <b>ongoing since").replace(" (continues", "</b> (continues")
    lead = t["lead"].replace(t["condition"], f"<b>{t['condition']}</b>") if t["status"] == "hypothesis" else \
        t["lead"].replace("Indeterminate:", "<b>Indeterminate</b>:")
    lines = [f"<h3>{head}</h3><p>{lead}</p>"]
    if t["support"]:
        lines.append("<p>Behaviours that changed:</p><ul>" + "".join(f"<li>{s}</li>" for s in t["support"]) + "</ul>")
    if t["others"]:
        lines.append(f'<p class="muted">All hypotheses: {t["others"]}</p>')
    return '<div class="period">' + "".join(lines) + "</div>"


def render(r: dict, person_id: str, truth: list[str] | None = None) -> str:
    h = header_lines(r, person_id)
    warn = "".join(f'<p class="warn">{w}</p>' for w in warnings_text(r))
    blocks = "".join(change_block(c) for c in r["changes"]) or \
        f"<p>{NO_CHANGE.replace('This does not mean the person is clinically stable', '<b>This does not mean the person is clinically stable</b>')}</p>"
    truth_html = ("<details><summary>Ground truth (simulated data, demonstration only)</summary><ul>"
                  + "".join(f"<li>{x}</li>" for x in truth) + "</ul></details>") if truth is not None else ""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Pre-consultation report</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,sans-serif;max-width:820px;margin:24px auto;padding:0 16px;color:#1d2733;background:#fff}}
h1{{font-size:1.4em;margin-bottom:4px}} h2{{font-size:1.1em;border-bottom:1px solid #d5dbe1;padding-bottom:4px;margin-top:28px}}
h3{{font-size:1em;margin:0 0 6px}} table{{border-collapse:collapse;width:100%;font-size:.92em}}
td,th{{padding:5px 8px;border-bottom:1px solid #eef1f4;text-align:left}} tr.adl td{{font-weight:600;background:#f4f6f8}}
.flag{{background:#fdecea;color:#b71c1c;border-radius:4px;padding:1px 6px;font-size:.85em}}
.warn{{background:#fff4e5;border-left:4px solid #ef6c00;padding:8px 12px}}
.period{{border-left:4px solid #c62828;background:#fbf7f7;padding:10px 14px;margin:12px 0}}
.muted{{color:#5f6b78;font-size:.9em}} .disc{{background:#f4f6f8;padding:8px 12px;font-size:.88em;color:#3d4a57}}
img{{max-width:100%}} details{{margin-top:28px;font-size:.88em;color:#5f6b78}}
</style></head><body>
<h1>Pre-consultation report — activities of daily living</h1>
<p>{h[0]}<br>{h[1]}<br>{h[2]}</p>
<p class="disc">{DISCLAIMER}</p>
{warn}
<h2>1. This period at a glance</h2><table><tr>{''.join(f'<th>{c}</th>' for c in table_columns(r))}</tr>{''.join(summary_rows(r))}</table>
<h2>2. Long-term trends</h2><img alt="Long-term trends" src="data:image/png;base64,{chart(r)}">
<h2>3. Changes detected in this period</h2>{blocks}
{truth_html}
</body></html>"""


# PDF renderer (A4, matplotlib only: no LaTeX or extra dependency, so it also runs on a small home device) ------

A4 = (8.27, 11.69)
INK, MUTED, RED, ORANGE = "#1d2733", "#5f6b78", "#b71c1c", "#ef6c00"


class _Page:
    """Top-down text cursor on an A4 matplotlib figure (coordinates in figure fraction)."""

    def __init__(self, pdf, title=None):
        self.pdf, self.fig, self.y = pdf, plt.figure(figsize=A4), 0.955
        if title:
            self.text(title, size=15, weight="bold", gap=0.03)

    def text(self, s, size=9.5, color=INK, weight="normal", gap=0.018, indent=0.07, width=105):
        import textwrap
        for line in textwrap.wrap(s, width=width) or [""]:
            if self.y < 0.05:
                self.new()
            self.fig.text(indent, self.y, line, size=size, color=color, weight=weight, va="top")
            self.y -= gap * size / 9.5
        return self

    def space(self, h=0.01):
        self.y -= h
        return self

    def new(self):
        self.pdf.savefig(self.fig)
        plt.close(self.fig)
        self.fig, self.y = plt.figure(figsize=A4), 0.955

    def close(self):
        self.pdf.savefig(self.fig)
        plt.close(self.fig)


def render_pdf(r: dict, person_id: str, path, truth: list[str] | None = None) -> None:
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(path, metadata={"Title": f"Pre-consultation report — {person_id}", "Creator": "SimADL-Health"}) as pdf:
        page = _Page(pdf, "Pre-consultation report — activities of daily living")
        for line in header_lines(r, person_id):
            page.text(line)
        page.space().text(DISCLAIMER, size=8.5, color=MUTED)
        for w in warnings_text(r):
            page.text("⚠ " + w, color=ORANGE, weight="bold")

        page.space(0.015).text("1. This period at a glance", size=11.5, weight="bold", gap=0.022)
        cols = table_columns(r)
        x = [0.07, 0.42, 0.56, 0.70, 0.83] if len(cols) == 5 else [0.07, 0.50, 0.66, 0.82]
        for xi, c in zip(x, cols):
            page.fig.text(xi, page.y, c, size=8.5, weight="bold", va="top")
        page.y -= 0.02
        for row in summary_data(r):
            if "header" in row:
                page.fig.text(0.07, page.y, row["header"], size=9, weight="bold", color=MUTED, va="top")
            else:
                vals = [row["label"], row["initial"], row["reference"], row["period"], row["change"]]
                if len(cols) == 4:
                    vals.pop(1)
                for xi, v in zip(x, vals):
                    page.fig.text(xi, page.y, v, size=8.5, va="top")
                if row["unusual"]:
                    page.fig.text(x[-1] + 0.075, page.y, "unusual", size=8, color=RED, weight="bold", va="top")
            page.y -= 0.017

        page.space(0.015).text("2. Changes detected in this period", size=11.5, weight="bold", gap=0.022)
        if not r["changes"]:
            page.text(NO_CHANGE)
        for c in r["changes"]:
            t = change_text(c)
            page.space(0.006).text(t["head"], weight="bold", color=RED)
            page.text(t["lead"])
            if t["support"]:
                page.text("Behaviours that changed:")
                for s in t["support"]:
                    page.text("• " + s, indent=0.09)
            if t["others"]:
                page.text("All hypotheses: " + t["others"], size=8.5, color=MUTED)
        if truth is not None:
            page.space().text("Ground truth (simulated data, demonstration only): " + "; ".join(truth), size=8, color=MUTED)
        page.close()

        fig = chart_figure(r, figsize=A4)
        fig.suptitle("3. Long-term trends", x=0.07, ha="left", fontsize=11.5, fontweight="bold")
        fig.subplots_adjust(top=0.93)
        pdf.savefig(fig)
        plt.close(fig)


# ---------------------------------------------------------------- entry points

def report_once(events, person_id, consult=None, since=None, mode="rolling", bundle=None, meta=None,
                truth=None, out_dir=None) -> dict:
    """One report for the period [since, consult) (default: the PERIOD_DAYS days before the consultation)."""
    if bundle is None:
        bundle, meta = load_models()
    person = person_daily(events)
    day = pd.to_datetime(person["day"])
    consult = pd.Timestamp(consult) if consult is not None else day.iloc[-1] + pd.Timedelta(days=1)
    since = pd.Timestamp(since) if since is not None else consult - pd.Timedelta(days=PERIOD_DAYS)
    p0, p1 = int((day < since).sum()), int((day < consult).sum())
    r = analyse(person, p0, p1, mode, bundle, meta)
    out = (out_dir or REPORTS) / f"report_{_slug(person_id)}_{consult.date()}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(r, person_id, truth), encoding="utf-8")
    render_pdf(r, person_id, out.with_suffix(".pdf"), truth)
    return dict(result=r, file=out, pdf=out.with_suffix(".pdf"))


def series(events_or_person, mode="rolling", every=PERIOD_DAYS, bundle=None, meta=None) -> list[dict]:
    """A report every `every` days after the first REFERENCE_DAYS of monitoring (one per consultation)."""
    if bundle is None:
        bundle, meta = load_models()
    person = events_or_person if "valid" in events_or_person else person_daily(events_or_person)
    out, prev = [], None
    for p0 in range(REFERENCE_DAYS, len(person) - every + 1, every):
        prev = analyse(person, p0, p0 + every, mode, bundle, meta, prev)
        out.append(prev)
    return out


def write_series(results: list[dict], person_id: str, truth=None) -> list:
    folder = REPORTS / _slug(person_id)
    folder.mkdir(parents=True, exist_ok=True)
    files, items = [], []
    for k, r in enumerate(results, 1):
        consult = pd.Timestamp(r["period_end"]) + pd.Timedelta(days=1)
        path = folder / f"report_{k:02d}_{consult.date()}.html"
        path.write_text(render(r, person_id, truth), encoding="utf-8")
        render_pdf(r, person_id, path.with_suffix(".pdf"), truth)
        files.append(path)
        summary = "; ".join(
            (f"{COND[c['condition']]} ({c['confidence']:.0%})" if c["status"] == "hypothesis" else c["status"])
            + (f", ongoing since {_date(c['since'])}" if c["since"] < c["start"] else "") for c in r["changes"]) or "no change"
        items.append(f"<tr><td><a href='{path.name}'>Report {k}</a> · <a href='{path.with_suffix('.pdf').name}'>PDF</a>"
                     f"</td><td>{_date(r['period_start'])} – "
                     f"{_date(r['period_end'])}</td><td>{summary}</td></tr>")
    (folder / "index.html").write_text(
        f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{person_id}</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:900px;margin:24px auto;padding:0 16px;background:#fff;color:#1d2733}"
        "td,th{padding:5px 8px;border-bottom:1px solid #eef1f4;text-align:left}table{border-collapse:collapse;width:100%}</style>"
        f"</head><body><h1>{person_id}: reports every {PERIOD_DAYS} days</h1><table><tr><th>Report</th><th>Period</th>"
        f"<th>Changes</th></tr>{''.join(items)}</table></body></html>", encoding="utf-8")
    return files


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text)


def demo() -> None:
    bundle, meta = load_models()
    labels = pd.read_csv(DATASET / "labels.csv")
    cases = []
    for run_id, name in ((16, "synthetic_run016_hh113_depression"), (8, "synthetic_run008_hh113_uti"),
                         (24, "synthetic_run024_hh113_cognitive")):
        events = pd.read_csv(next(DATASET.glob(f"events_run{run_id:03d}_*.csv")), parse_dates=["start", "end"])
        lab = labels[labels["run_id"] == run_id]
        truth = [f"{r.condition}, onset {r.onset}, {r.duration_days} d, intensity {r.intensity}" for r in lab.itertuples()]
        cases.append((name, events, truth))
    for home in ("hh113", "hh115"):
        cases.append((f"real_casas_{home}", pd.read_csv(PROCESSED / f"{home}_episodes.csv", parse_dates=["start", "end"]), None))
    for name, events, truth in cases:
        results = series(events, bundle=bundle, meta=meta)
        write_series(results, name, truth)
        print(f"{name}: {len(results)} reports")
        for k, r in enumerate(results, 1):
            ch = [(c["condition"] if c["status"] == "hypothesis" else c["status"]) + ("*" if c["since"] < c["start"] else "")
                  for c in r["changes"]]
            print(f"   {k:2d} {r['period_start'].date()}..{r['period_end'].date()} data {r['valid_share']:.0%}: {ch or '-'}")
        if truth:
            print("   truth:", truth)


def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-consultation ADL report(s) for one person.")
    ap.add_argument("--events", help="CSV with columns activity,start,end (one row per recognised activity)")
    ap.add_argument("--person", default="patient", help="name or identifier printed on the report")
    ap.add_argument("--consult", help="consultation date YYYY-MM-DD (default: day after the last event)")
    ap.add_argument("--since", help="start of the report period, e.g. the previous consultation (default: 35 days before)")
    ap.add_argument("--reference", choices=("rolling", "initial"), default="rolling")
    ap.add_argument("--series", action="store_true", help=f"one report every {PERIOD_DAYS} days over the whole history")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        return demo()
    if not args.events:
        ap.error("--events is required (or use --demo)")
    events = pd.read_csv(args.events, parse_dates=["start", "end"])
    if args.series:
        files = write_series(series(events, mode=args.reference), args.person)
        print(f"{len(files)} reports in {files[0].parent if files else REPORTS}")
    else:
        print(report_once(events, args.person, args.consult, args.since, args.reference)["file"])


if __name__ == "__main__":
    main()
