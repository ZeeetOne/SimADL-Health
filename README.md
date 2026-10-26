# SimADL-Health

Simulating long-term ADL logs (sleep, eating, social interaction) of an older adult living alone,
injecting health anomalies derived from clinical descriptions, then detecting *when* behaviour changes,
attributing *which* condition likely caused it, and producing a pre-consultation report (FR/EN) for a GP.

> Work in progress — a compact prototype of a synthetic-data → anomaly-detection → clinical-report pipeline.

## Pipeline
1. **Calibrate** daily features on CASAS Aruba (real smart-home data).
2. **Simulate** months of normal behaviour as `(timestamp, activity)` events.
3. **Inject** symptoms of UTI/nocturia, depression and mild cognitive decline (see `docs/symptom_catalog.md`).
4. **Validate** realism of synthetic vs real daily features (KS test).
5. **Detect** change (±2σ baseline, Isolation Forest, PELT change-point).
6. **Attribute** the likely condition (Random Forest + SHAP).
7. **Report** a weekly/monthly pre-consultation summary.
8. **Evaluate**: detection F1, detection delay, attribution accuracy, unseen-persona robustness.

## Use it for a person
The models are trained once on synthetic data, then applied to anyone's activity log:
```bash
python -m simadl.train                       # -> models/ (detection thresholds + attribution model)
python -m simadl.report --events patient.csv --consult 2026-11-20 --person "Mme D." --lang fr,en
python -m simadl.report --demo               # held-out synthetic people + two real CASAS residents
```
`patient.csv` needs columns `activity,start,end`. Activity names follow CASAS labels (`Sleep`, `Bed_Toilet_Transition`,
`Toilet`, `Cook_*`, `Eat_*`, `Leave_Home`, `Enter_Home`, `Phone`, `Entertain_Guests`, …). At least 74 days of history
are required: the first 60 days are taken as the person's usual habits. The report is a self-contained HTML file in
`outputs/reports/`, written from templates and computed numbers (no generative AI).

## Reproduce the study
```bash
python -m simadl.features hh113 && python -m simadl.features hh115   # T1 calibration
python -m simadl.simulate hh113 365                                   # T2 simulator check
python -m simadl.anomalies 8                                          # T4 labelled synthetic dataset
python -m simadl.validate                                             # T5 realism
python -m simadl.detect                                               # T6 change detection
python -m simadl.attribute                                            # T7 condition attribution
```

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```
Place the CASAS datasets under `data/raw/` (not tracked by git).

## Layout
```
simadl/      one module per pipeline stage
docs/        symptom catalogue, literature notes
notebooks/   quick exploration
outputs/     figures, result tables, example reports
```
