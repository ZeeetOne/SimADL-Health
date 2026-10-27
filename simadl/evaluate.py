"""T9 — Report-level evaluation: does each 35-day pre-consultation report tell the GP the right thing?

Fresh synthetic people are simulated with new seeds (never used for training or calibration). For each person a
report is made every 35 days after the first 60 days, as in clinical follow-up. Per report:
  truth     = conditions active (severity > 0) on >= MIN_TRUE_DAYS days of the period, else "normal"
  predicted = conditions stated as a hypothesis in the report (indeterminate / unspecific changes are not claims)
Metrics, per reference mode (rolling vs initial) and per condition:
  - sensitivity      : share of reports with condition C where some change is reported
  - correct          : share of reports with condition C that name C
  - clean normal     : share of normal reports with no change at all
  - wrong claim      : share of reports that name a condition that is not active
Also, on the two real CASAS residents (no ground truth): how many reports name a condition.

Usage:
    python -m simadl.evaluate 4      # 4 fresh people per home x condition
"""
import sys

import pandas as pd

from simadl.anomalies import CONDITIONS as RUN_CONDITIONS, PERSONAS, make_run
from simadl.features import OUTPUTS, PROCESSED
from simadl.report import load_models, person_daily, series
from simadl.simulate import estimate_params

MIN_TRUE_DAYS = 3
SEED_OFFSET = 50_000  # far from the seeds used to build the training dataset
MODES = ("rolling", "initial")


def report_rows(results: list[dict], daily: pd.DataFrame | None, run_id, persona, run_condition, mode) -> list[dict]:
    rows = []
    for k, r in enumerate(results, 1):
        claims = {c["condition"] for c in r["changes"] if c["status"] == "hypothesis"}
        truth = "normal"
        if daily is not None:
            d = daily[(daily.index >= r["period_start"]) & (daily.index <= r["period_end"])]
            active = d.loc[d["severity"] > 0, "label"].value_counts()
            active = active[active >= MIN_TRUE_DAYS]
            truth = active.index[0] if len(active) else "normal"
        rows.append(dict(mode=mode, run_id=run_id, persona=persona, run_condition=run_condition, report=k,
                         truth=truth, any_change=bool(r["changes"]), claims=",".join(sorted(claims)),
                         correct=truth in claims, wrong_claim=bool(claims - {truth}),
                         valid_share=round(r["valid_share"], 2)))
    return rows


def main(n_per_cell: int) -> None:
    bundle, meta = load_models()
    rows, run_id = [], 0
    for persona in PERSONAS:
        base = estimate_params(persona)
        for condition in RUN_CONDITIONS:
            for _ in range(n_per_cell):
                events, daily, _labels = make_run(run_id, persona, condition, base, seed=SEED_OFFSET + run_id)
                person = person_daily(events)
                for mode in MODES:
                    rows += report_rows(series(person, mode=mode, bundle=bundle, meta=meta), daily, run_id,
                                        persona, condition, mode)
                run_id += 1
            print(f"{persona} {condition}: done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUTS / "t9_report_level.csv", index=False)

    out = []
    for mode, g in df.groupby("mode"):
        for truth, t in g.groupby("truth"):
            out.append(dict(mode=mode, truth=truth, reports=len(t),
                            sensitivity=t["any_change"].mean() if truth != "normal" else None,
                            correct=t["correct"].mean() if truth != "normal" else None,
                            clean_normal=(~t["any_change"]).mean() if truth == "normal" else None,
                            wrong_claim=t["wrong_claim"].mean()))
    summary = pd.DataFrame(out).round(2)
    summary.to_csv(OUTPUTS / "t9_report_level_summary.csv", index=False)
    print(f"\n{run_id} fresh synthetic people, {len(df) // len(MODES)} reports per mode")
    print(summary.to_string(index=False))

    # real residents: no ground truth, count condition claims per report
    real = []
    for home in ("hh113", "hh115"):
        person = person_daily(pd.read_csv(PROCESSED / f"{home}_episodes.csv", parse_dates=["start", "end"]))
        for mode in MODES:
            for r in report_rows(series(person, mode=mode, bundle=bundle, meta=meta), None, home, home, "real", mode):
                real.append(r)
    real = pd.DataFrame(real)
    real.to_csv(OUTPUTS / "t9_real_reports.csv", index=False)
    print("\nreal CASAS residents (no ground truth): share of reports naming a condition, and which")
    agg = real.groupby(["persona", "mode"]).agg(reports=("report", "size"), any_change=("any_change", "mean"),
                                                 claims=("claims", lambda s: ", ".join(x for x in s if x) or "-"))
    print(agg.round(2).to_string())


def latex_table() -> None:
    """paper/report_table.tex from the saved CSVs, so the paper always shows the latest numbers."""
    s = pd.read_csv(OUTPUTS / "t9_report_level_summary.csv")
    real = pd.read_csv(OUTPUTS / "t9_real_reports.csv")
    real["claims"] = real["claims"].fillna("")
    pct = lambda v: "--" if pd.isna(v) else f"{v:.0%}".replace("%", r"\%")
    lines = [r"\begin{tabular}{llcccc}", r"\toprule",
             r"Reference & Truth (reports) & Sensitivity & Correct & Clean normal & Wrong claim \\", r"\midrule"]
    for mode in MODES:
        for row in s[s["mode"] == mode].itertuples():
            lines.append(f"{mode} & {row.truth} ({row.reports}) & {pct(row.sensitivity)} & {pct(row.correct)} & "
                         f"{pct(row.clean_normal)} & {pct(row.wrong_claim)} \\\\")
        lines.append(r"\midrule")
    for (persona, mode), g in real.groupby(["persona", "mode"]):
        named = (g["claims"] != "").mean()
        cog = g["claims"].str.contains("cognitive").sum()
        lines.append(f"{mode} & real {persona} ({len(g)}) & \\multicolumn{{4}}{{l}}{{{named:.0%} of reports name a "
                     f"condition; {cog} name cognitive decline}} \\\\".replace("%", r"\%"))
    lines += [r"\bottomrule", r"\end{tabular}"]
    path = OUTPUTS.parent / "paper" / "report_table.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--table":
        latex_table()
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
        latex_table()
