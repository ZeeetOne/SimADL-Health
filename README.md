# SimADL-Health

**From a list of daily-living events to a report the GP can read before the consultation.**

SimADL-Health simulates long-term activity logs (sleep, eating, social life) of older adults living alone, injects
health anomalies taken from clinical descriptions (urinary tract infection, depression, mild cognitive decline),
detects *when* behaviour changes against the person's own habits, attributes *which* condition is most compatible,
and writes a periodic pre-consultation report (HTML + PDF).

It is a one-week prototype of the three objectives of the ObsAVQ research project (IMT Atlantique):
**O1** a synthetic-dataset generator, **O2** ML methods that produce the report, **O3** an evaluation against prior work.
Technical report: [`paper/main.tex`](paper/main.tex). Design choices are checked against the literature in
[`docs/methodology_review.md`](docs/methodology_review.md).

```
CASAS hh113/hh115 ─► simulator (normal habits) ─► symptom injection ─► detection vs own baseline ─► attribution (RF+SHAP) ─► 35-day report
   real calibration      O1                       known ground truth          O2                          O2                    HTML + PDF
```

## Key results
| Question | Result |
|---|---|
| Is simulated behaviour realistic? (O1) | Simulated hh113 vs a **held-out real year**: mean KS *D* **0.188**. Real hh113 year 1 vs its own later period: 0.199. Another real home: 0.379. A classifier still separates simulated from real days (AUC 0.89 hh113, 0.64 hh115), mainly through time out of home. |
| Can changes be detected per person? (O2) | At 0.5 false alarms / 30 days (thresholds calibrated on the *other* home), Isolation Forest finds every cognitive-decline episode (median 8 / 30 days) and every hh113 depression episode (6 days). ±σ catches 79 % of hh113 UTI episodes (3 days). UTI is hard for hh115, whose baseline is already 2 night toilet visits. |
| Can the cause be attributed on an unseen person? (O2) | Training on one home and testing on the other gives day macro-F1 **0.58 / 0.59**. Cognitive decline: 16/16 episodes. Depression: 9/16. UTI: 23/38. SHAP explanations match the symptom catalogue. |
| Is a 35-day report useful? (O3) | With report-level calibration, ~80 % of normal reports are clean, but report sensitivity is modest (see `outputs/t9_report_level_summary.csv`). On real residents, a fixed baseline turns habit drift into "cognitive decline", so the rolling reference is the default. |

These are synthetic results and therefore optimistic. Prior work on real nurse-labelled events (Isudra) reports low
scores. Limitations are listed in Section 5 of the paper.

## Use it for a person
Models are trained once on synthetic data, then applied to anyone's event list:
```bash
python -m simadl.train                                   # -> models/ (thresholds + attribution model)
python -m simadl.report --events patient.csv --person "Mrs D." --consult 2026-11-20 --since 2026-10-16
python -m simadl.report --events patient.csv --person "Mrs D." --series   # one report every 35 days
python -m simadl.report --demo                           # held-out synthetic people + two real CASAS residents
```
`patient.csv` needs columns `activity,start,end`. Activity names follow CASAS labels (`Sleep`, `Bed_Toilet_Transition`,
`Toilet`, `Cook_*`, `Eat_*`, `Leave_Home`, `Enter_Home`, `Phone`, `Entertain_Guests`, …).

Each report covers one period (default: the 35 days before the consultation, i.e. 5 of each weekday) and compares it with
a 60-day reference: `rolling` (the 60 days just before the period, default) or `initial` (the first 60 days of monitoring).
It warns when fewer than 16 of 30 days have data, marks changes that continue from the previous report ("ongoing since"),
says "indeterminate" when no hypothesis is clearly supported, lists only behaviours that changed in the direction the
symptom catalogue expects, and reminds the reader that no detected change does not mean clinical stability.
Each report is written as an English HTML page (screen) and an A4 PDF (to send, print or attach to the medical record),
from templates and computed numbers. No generative AI is used. Example series: `outputs/reports/*/index.html`.

## Reproduce the study
```bash
python -m simadl.features hh113 && python -m simadl.features hh115   # T1 calibration
python -m simadl.simulate hh113 365                                   # T2 simulator check
python -m simadl.anomalies 8                                          # T4 labelled synthetic dataset
python -m simadl.validate                                             # T5 realism
python -m simadl.detect                                               # T6 change detection
python -m simadl.attribute                                            # T7 condition attribution
python -m simadl.train                                                # models for the reports
python -m simadl.evaluate 4                                           # T9 report-level evaluation (+ paper table)
python -m simadl.report --demo                                        # example report series
```
The paper compiles with `pdflatex main && bibtex main && pdflatex main && pdflatex main` in `paper/`, or on Overleaf
(upload the `paper/` folder).

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows  (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
```
Data: CASAS smart home datasets, [Zenodo record 15708568](https://zenodo.org/records/15708568). Place them under
`data/raw/` (not tracked by git). Only `labeled/hh113.csv` and `labeled/hh115.csv` are used.

## Layout
```
simadl/      features (T1) · simulate (T2) · anomalies (T4) · validate (T5) · detect (T6) · attribute (T7)
             train · report (T8) · evaluate (T9)
docs/        symptom_catalog.md (T3) · literature_notes.md · methodology_review.md
paper/       technical report (LaTeX)
outputs/     figures, result tables, example reports
```
