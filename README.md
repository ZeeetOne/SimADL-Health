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
