# Methodology review against prior work

Purpose: check each stage of SimADL-Health against published work that does the same thing, so our choices are not
just our own opinion. Full notes and links for each paper are in `docs/literature_notes.md`.

## Closest prior work (end-to-end)

| Work | What it does | Closest to our stage(s) |
|---|---|---|
| **Tanaka et al. 2023/2024** (arXiv 2312.16852, 2411.13153; PhD thesis Hokkaido 2025) | Simulator of an older adult living alone (activities with normal start/duration, Poisson counts) + 6 injected anomalies (semi-bedridden, housebound, forgetting, wandering, 2 fall types), severity driven by a latent MMSE. Classifiers trained on 9 simulated years, tested on 9 other simulated years. | **Whole pipeline** (T2, T4, T6, T7, T9). Most similar published design. |
| **Dahmen & Cook 2019, SynSys** (Sensors 19:1181) | Synthetic smart-home data: HMM for activity sequence + ridge regression for durations, trained on 10 real CASAS homes of older adults. Realism measured by distance vs random data, another home, another period. | T1, T2, T5 |
| **Dahmen 2019 PhD thesis; Dahmen & Cook 2021, Isudra** (ACM TIST 12(2)) | Synthetic data with realistic health events (thesis). Isudra detects nurse-labelled health events (falls, nocturia, depression signs, weakness) in 5 real CASAS homes, with IF / LOF / OCSVM / PCA / KNN detectors chosen by Bayesian optimisation. | T4, T6 |
| **Sprint, Cook, Fritz, Schmitter-Edgecombe 2016** (SMARTCOMP / IEEE Computer) | Behaviour-change detection comparing activity timing/duration between time windows, with significance testing, on real health events (cancer treatment, insomnia). | T6 |
| **Cejudo et al. 2025** (PMC12106144) | Per-user daily activity vectors from 30 CASAS single-resident homes. Next-day prediction error as a z-score, plus per-activity boxplot rule to explain which activities deviate. No labels. | T6, T8 |
| **Alshammari et al. 2018, SIMADL** (Data 3:11) | OpenSHS virtual-home simulation by 7 participants. Anomalies are defined by the participants themselves, not clinically. | T4 (contrast) |
| **Fritz et al. 2022; Ugwu, Fritz, Cook, Doppa (AAAI)** | Nurse-in-the-loop: health events align with deviation from the person's own baseline; clinician-facing output. | T6, T8 |
| **Capstick et al. 2024** (npj Digit Med) | UTI detection from in-home monitoring in dementia; bathroom-visit features; ±3-day label windows (follow-up arXiv 2502.17484). | T3, T6 |

## Stage-by-stage check

Verdicts: ✅ consistent with prior work · ⚠️ deviation, justified or acknowledged · 🔧 gap that we fixed

| Stage | What we do | What prior work does | Verdict |
|---|---|---|---|
| **T1 Calibration** | Daily ADL features from activity-labelled real homes of single older adults (CASAS hh113/hh115). | SynSys calibrates on activity-labelled CASAS homes of older adults. Tanaka uses a national time-use survey. Cejudo builds daily per-activity vectors from CASAS single-resident homes. | ✅ |
| **T2 Simulator** | Schedule-based parametric model: sleep onset/wake and meal times ~ mean + std·residual, counts rounded-normal or resampled, empirical durations, weekday/weekend. | Tanaka: fundamental activities with normal start/duration, necessary activities with Poisson counts, random activities with probabilities. This is the same family as ours. SynSys uses HMM + regression; ADLGen uses Transformers (generative SOTA). | ✅ Same model family as Tanaka. ⚠️ We model daily ADL events, not raw sensor firings. This is intended, because ObsAVQ assumes an event list. |
| **T3 Symptom catalogue** | Per condition: changed parameters, onset profile, duration, intensity, each tagged SOURCED / DIRECTION / ASSUMPTION. | Tanaka: housebound = fewer outings **and fewer phone calls** for ~14 days; semi-bedridden = more naps/rest for ~30 days. Frequencies and durations come partly from literature and partly from the authors, and they are not validated on real anomalies. Isudra's nurse labels list "less time out of home", "not sleeping well", "more time in chair" as depression signs, and nocturia as an event. | ✅ Our depression signs (less time out, fewer contacts, sleep disturbance) match Isudra's nurse-labelled depression signs and Tanaka's housebound model. ⚠️ Like Tanaka, several magnitudes are assumptions. We tag them explicitly, which is stricter than Tanaka. |
| **T4 Injection** | Modify simulator parameters per day with a trapezoid severity (ramp up, plateau, ramp down), intensity U[0.5, 1], durations from the catalogue; 60 anomaly-free baseline days; ground truth per episode. | Tanaka: anomalies change simulator behaviour with frequency/duration control and a latent severity (MMSE). SIMADL lets participants define anomalies (subjective). Zenodo CASAS-derived test sets inject anomalies per area and label anomalous days. | ✅ Same principle as Tanaka (parameter-level injection, severity variable, known ground truth). |
| **T5 Realism** | KS per daily feature + classifier two-sample test (AUC), against real-vs-real references: same home in another period, another home. | SynSys: Euclidean and DTW distance of one week vs real, with baselines random / another home / another time period / simpler generator. Tanaka: Levenshtein day-to-day similarity of activity sequences vs Kasteren and CASAS Aruba. | ✅ Same reference logic as SynSys (other home, other period). ⚠️ No "random" baseline. We compare daily features, not event sequences. Our classifier test is stricter than a distance. |
| **T6 Detection** | Weekly rolling features z-scored vs the person's own first 60 days; ±2σ, Isolation Forest, PELT; alarm thresholds calibrated to a false-alarm rate. Episode recall, detection delay, false alarms per 30 days. | Shahid 2022: ±2σ on activity duration. Tanaka: mean ± c·σ over consecutive days for housebound/semi-bedridden. **Sensitivity = anomaly intervals overlapping a detected interval; false alarm rate = detected intervals with no overlap, per day.** Isudra: IF / LOF / OCSVM etc. on windows, G-mean, 2-week windows around events. Cejudo: per-user baseline from the first 70 % of days, 15-day window best. Capstick follow-up: ±3-day label windows. Sprint: compare windows before/after. | ✅ Detectors and metrics match prior work. Our recall/false-alarm definitions are essentially Tanaka's sensitivity/FAR. 🔧 **Fixed:** thresholds were calibrated on the same normal runs used to count false alarms. They are now calibrated on the *other* home (train/test separation as in Tanaka). ⚠️ Prior work uses 7–15 day windows, and ours is 7 days. |
| **T7 Attribution** | Random forest on baseline-relative features → {normal, UTI, depression, cognitive}; cross-persona test; SHAP. | Tanaka: random forest / decision tree / HMM per anomaly type, trained on simulation. Isudra: detector chosen per event type. Cejudo: per-activity boxplot rule to explain which activities deviate. | ✅ Same idea as Tanaka ("train on simulation, classify anomaly type"). Our cross-persona test goes beyond Tanaka's single resident profile. |
| **T8 Report** | Template FR/EN report: per-ADL summary vs habits, 90-day chart, detected periods, hypothesis + confidence + behaviours that changed; "not a diagnosis". | Cejudo: flagged days presented as prompts for clinical follow-up, not diagnoses. Fritz / Ugwu: clinician-in-the-loop, deviations from the person's baseline. | ✅ |
| **T9 Evaluation** | Synthetic test (held-out runs, unseen persona) + qualitative run on 2 real residents without labels. | Tanaka: trains and tests on simulation only, and states that validity depends on simulation realism. Isudra: real nurse-labelled events, low absolute G-mean (≈0.05–0.4). | ⚠️ Same limitation as Tanaka. Our real-resident run already shows a sim-to-real gap (long "cognitive" periods). Isudra's low real-data scores warn that synthetic results are optimistic. |

## Report generation (T8): how prior work presents results to clinicians

No published work found (Oct 2026) that turns ambient smart-home data into a **narrative pre-consultation report for a GP**.
That gap is ObsAVQ's own objective. Prior systems present results in four ways:

| Approach | Example | What the clinician gets | Lessons |
|---|---|---|---|
| Rule / threshold alerts vs personal routine | Community-in-the-loop prototype, JMIR Aging 2026 (46 older adults, 6 months); TIHM for Dementia (NHS, 408 people) | Alerts by email/SMS or on a colour-coded, priority-ranked dashboard watched by a monitoring team | 4,719 alerts and 93 % needed no action, so alert fatigue is a problem: merge overlapping alerts and suppress repeats. Make the logic and thresholds transparent. **State that "no alert" does not mean clinically stable.** Use non-judgemental wording. |
| ML risk score + calibrated uncertainty + abstention | Ugwu, Fritz, Cook, Doppa, AAAI 2026 (UTI flare-ups, 8 homes) | Daily probability with a conformal interval, and "UTI / no UTI / **I don't know**" when uncertain | 42 nurses rated the calibrated-interval plots positively for clarity, trust, decision influence and workflow fit. Nurses **asked to see the contributing factors**. Features were chosen with SHAP. |
| Statistical deviation + "what changed" explanation | Cejudo et al. 2025 | Flagged days plus the activities outside the boxplot range | Presented as prompts for clinical follow-up, not diagnoses. |
| LLM-written explanations | Fiori, Civitarese, Choudhary, Bettini 2025 (ACM TIoT); PH-LLM (Nature Medicine 2025); Health-LLM 2024 | Natural-language explanations or insights | Users rated LLM explanations higher than template/heuristic ones. **Main risk is over-reliance**: convincing explanations for wrong predictions and rationales not supported by the data. Proposed mitigations: give the LLM a structured textual representation of the data, check every rationale against the data (e.g. a verifier agent), and evaluate with expert rubrics (PH-LLM). |

**Where SimADL-Health stands:** our T8 pipeline (trained models → computed facts → template text) belongs to the
"risk score + explanation" family. It already shows the contributing behaviours that nurses asked for in Ugwu et al., and it
is a "decision support, not diagnosis" report like Cejudo et al. Using templates instead of an LLM is the safe choice the
literature supports for clinical text.

Gaps compared with prior work:
1. Our confidence is a raw random-forest probability, with no calibrated uncertainty and no explicit abstention (Ugwu: conformal + "I don't know").
2. The report does not say that "no detected change ≠ clinically stable" (community-in-the-loop lesson).
3. There is no evaluation by clinicians. Ugwu used a 42-nurse Likert survey and PH-LLM used expert rubrics.
4. LLM verbalisation is optional future work. If used, it should only rephrase the computed facts, with each sentence checked against them (Fiori et al. mitigation).

## Real-world deployments: report period, data sufficiency, where computation runs

| System | Status | Where it runs | What is produced, for whom, how often | Reference period |
|---|---|---|---|---|
| **MinderCare** (UK DRI / NHS North West London Virtual Hospital) | NHS pilot, ~100 homes, 4-person clinical monitoring team | Back-end servers | Alerts checked by a monitoring team, then a clinical team. **Regular "Minder Reports"** summarising data and alerts go to the clinical team, **or to the GP on request** | n/a |
| **TIHM for Dementia** (NHS Surrey) | RCT, 408 people with dementia | Back-end; ML on integrated data | Alerts prioritised and colour-coded on a dashboard watched 24/7; the team phones the carer | n/a |
| **Sensara** (NL, TNO spin-off, 2013) | Commercial (home care, nursing homes) | Cloud | Self-learning per-person patterns, deviation notifications to carers / family | person's own learned pattern |
| **Montreal AAL single case** (JMIR Aging 2020, PMC7695528) | Public home-care service, 490 days | Custom sensors + server | **Monthly report emailed to the care professional.** 490 days were split into **14 periods of 35 days (5 of each weekday)**. Sleep, outings, low mobility, cooking, hygiene, appliance use; trend lines with significant periods shaded; table of means and 95 % CI | **First monitoring month = baseline.** Linear mixed models (month, weekday). Events flagged at 2 SD above the person's mean |
| **DomoCare** (12-month study) | Research | Server | **Weekly activity graphs** emailed to nurses when activity deviated. 76 % favourable to ambient sensors | n/a |
| **Community-in-the-loop** (JMIR Aging 2026) | 46 older adults, 6 months | Server, rules | Email/SMS alerts, then a CHW phone call; 4,719 alerts, 93 % needed no action | the person's established routine |
| **US Medicare RPM** (CPT 99454 / 99445, 2026) | Reimbursement rule | — | Billing per **30-day period**: 99454 requires data on **≥ 16 of 30 days**; 99445 covers 2–15 days | — (data sufficiency rule) |

**Edge / local LLM evidence**
- Deployed systems process on servers and use **no LLM**. Text is produced by clinicians or monitoring teams.
- Gaitán-Guerrero et al., UCAmI 2025: small local LLMs on **Raspberry Pi 4 (8 GB)** can turn sensor data into readable activity summaries. Small models use the hardware heavily, so tasks had to be **distributed across devices** and **prompts split into focused segments**. Feasible but resource-bound.
- DESAMO (UIST 2025): an audio LLM fully on **Jetson Orin Nano** (Whisper-large-v3 encoder at 16-bit, LLM at 4-bit, GGUF), with ~5.3 s latency, for voice commands and ambient events (falls, calls for help).
- Edge acoustic scene recognition for elderly routines (PMC11946438): quantised models on device, motivated by privacy and latency.
- France: health data is GDPR Art. 9 special-category data, and hosting personal health data for care requires an **HDS-certified host** with storage in the EEA. Keeping raw audio and processing on the device reduces what must be hosted.

**Implications for SimADL-Health / ObsAVQ**
1. **Report period.** Deployed systems report monthly (Montreal: 35-day periods of whole weeks) or weekly. A GP pre-consultation report should cover **the period since the last consultation**, or by default the last 5 whole weeks.
2. **Reference period.** Montreal uses the first month as a fixed baseline. Our real-resident results showed that a fixed old baseline drifts out of date. Options are a fixed baseline (good for slow decline) or a rolling baseline just before the report period (robust to drift). This should be a parameter, and the report should state which one was used.
3. **Data sufficiency.** The report should refuse or warn when the period has too few valid days. The Medicare RPM threshold (≥ 16 of 30 days) is a real-world reference.
4. **Where to compute.** Audio → events must stay on the device for privacy, as ObsAVQ's audio sensing implies. Our features + detection + attribution are light (RF model 5.6 MB, a few seconds per person on a laptop CPU), so they can run on a home box or an HDS server. Template text needs no LLM. A local LLM is **optional**, only to rephrase verified facts, and the edge evidence shows it is feasible with small quantised models but costly.

## What changed after this review
1. **T6 threshold leakage fixed**: thresholds are calibrated on the other home's normal runs. False alarms on the held-out
   home are 0.27–0.77 per 30 days, against a target of 0.5. Thresholds transfer only roughly between people, which argues for
   per-person calibration in deployment.
2. Recorded as limitations (no code change): no random baseline in T5; daily-feature rather than sequence-level realism;
   no real labelled health events (unlike Isudra); a 7-day window where prior work also uses 15 days.

## Positioning summary
"The pipeline follows the design of simulator-based anomaly detection for older adults living alone (Tanaka et al.),
calibrated on real CASAS homes as in SynSys, with realism checked against real-vs-real references (SynSys), detectors used in
Isudra and Shahid et al., and the sensitivity / false-alarm metrics of Tanaka et al. Our additions are: symptom parameters
tagged by evidence level, a per-person baseline, a cross-person test, explanations for the clinician, and a first sim-to-real
check on real residents."
