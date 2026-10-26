# Literature notes

Running log of every paper/source consulted: what it says that matters to us, the link, and how well we checked it.

**Status legend**
- `READ`: page/full text fetched and the numbers extracted directly.
- `SNIPPET`: seen only via search-result summaries or abstract. Verify before quoting numbers.
- `MEMORY`: cited from prior knowledge, not opened in this project. Verify before the final report.
- `TODO`: worth reading, not done yet.

Where it is used: `T3` = symptom catalogue, `T6`/`T7` = methods, `POS` = background / related work.

---

## UTI / nocturia

### Capstick A. et al., 2024 — *Digital remote monitoring for screening and early detection of urinary tract infections*, npj Digital Medicine — `SNIPPET` — T3, POS
- Link: https://ukdri.ac.uk/publications/digital-remote-monitoring-screening-and-early-detection-urinary-tract-infections-0 · PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC10787784/ (blocked by captcha when fetched)
- 117 people living with dementia, 27,828 person-days, low-cost in-home devices (activity, sleep, nocturnal physiology).
- On unseen participants: sensitivity 65.3 %, specificity 70.9 %. After risk stratification: 74.7 % (67.9–81.5) / 87.9 % (85.0–90.9).
- Top contributors: **bathroom visit statistics**, night-time respiratory rate, number of previous UTIs.
- Engineered features "representing symptoms used for diagnosing a UTI". **No effect size reported** for how much bathroom visits rise.
- Infections in dementia often go unreported, which is the motivation for early flagging.

### Follow-up: *UTI detection in digital remote monitoring: participant-specific prediction complexity* (arXiv 2502.17484) — `SNIPPET` — T3, T6
- Link: https://arxiv.org/pdf/2502.17484
- Features: activation frequency per room (bathroom, bedroom, hallway, kitchen, lounge), nocturnal heart/respiratory rate, nocturnal awake occurrences. Day/night bathroom frequency with **moving averages and percentage changes**, plus time to reach the bathroom.
- **Label windowing: a confirmed UTI day is extended ±3 days.** This is useful for our detection-tolerance window in T6/T9.

### *Clinician-in-the-loop smart home system to detect UTI flare-ups via uncertainty-aware decision support* (arXiv 2511.18334) — `SNIPPET` — T3, T8
- Link: https://arxiv.org/pdf/2511.18334
- One year of ambient sensors (motion, door, light, temperature) in older adults' homes.
- UTI symptoms in older adults are often **atypical** (confusion, dizziness, falls), not just urinary frequency.
- Likely related to Ugwu, Fritz, Cook, Doppa (AAAI), the reference for the T8 report format.

### Enshaeifar S. et al., 2019 — TIHM for Dementia, PLoS ONE 14(1): e0209909 — `SNIPPET` — T6
- Links: https://eprints.gla.ac.uk/217130 · https://doaj.org/article/6cfaad9dcb14471abff0cccdbace6e84
- IoT + in-home sensors (PIR etc.) in more than 150 homes of people with dementia. NMF latent factors → clustering to flag possible UTIs.
- UTIs ≈ 9 % of hospital admissions of people with dementia in the UK (top 5 reasons).
- Li et al. 2020 (arXiv 2011.13916): semi-supervised UTI risk score from the same data. https://arxiv.org/pdf/2011.13916v3

### Rantz M. et al. — TigerPlace early illness detection — `SNIPPET` / `TODO` — T3
- News 2015: https://munewsarchives.missouri.edu/news-releases/2015/1207-older-adults-live-independently-longer-when-monitored-by-care-team-and-technology
- Thesis "Sensor network for early illness detection in the elderly": https://mospace.umsystem.edu/xmlui/handle/10355/15958
- Health problems were detected **1–2 weeks earlier** with sensors (news-release claim). UTIs, pneumonia/URI and CHF were among the illnesses detected early. Sensors: motion, bed (pulse, respiration, restlessness), gait.
- **TODO:** Rantz et al. *"Using sensor networks to detect urinary tract infections in older adults"*. This may give the UTI effect size we are missing.

### Nuqoba B., Tan H.-P., 2020 — nocturia prediction, IEEE BigData — `SNIPPET` — T3
- Link: https://ink.library.smu.edu.sg/sis_research/5909
- Motion sensors per apartment zone + door sensors, 39 older adults living alone. Sensor-based nocturia classification.
- Older adults with nocturia are more likely to have poor sleep quality and conditions related to physical frailty.

### Hoffmann T. et al., 2020 — natural history of uncomplicated UTI without antibiotics, BJGP 70(699): e714 — `SNIPPET` — T3 (UTI duration)
- Link: https://bjgp.org/content/70/699/e714
- Most improvement happens in the first 9 days. About ⅓ of women are symptom-free spontaneously within 7–10 days. Not specific to older women.

### Little P. et al., 2010 — natural course of UTI symptoms, BMJ — `SNIPPET` — T3
- Severe symptoms last ~3.3 days with sensitive-organism antibiotic treatment, and ~62 % longer without antibiotics. Women aged 18–70.

### Vogel T. et al., 2004 — 3- vs 7-day ciprofloxacin in women ≥65, CMAJ 170(4): 469 — `SNIPPET`
- Link: https://www.cmaj.ca/content/170/4/469 — about treatment length, only indirectly relevant.

### Loeb M. et al., 2001 — minimum criteria for antibiotics in long-term care, Infect Control Hosp Epidemiol — `MEMORY` — T3
- New-onset urgency/frequency as UTI criteria.

### van Kerrebroeck P. et al., 2002 — ICS standardisation of nocturia terminology, Neurourol Urodyn — `MEMORY` — T3
- Nocturia = waking ≥1 time at night to void. ≥2 is commonly used as the clinically bothersome threshold.

### Withings blog: *Sleep as an early warning system… UTIs* — `SNIPPET`, secondary
- https://www.withings.com/us/en/health-solutions/insight-hub/sleep-as-an-early-warning-system-detecting-hidden-health-risks-like-urinary-tract-infections
- Claims that nocturnal HR/breathing/movement change days before diagnosis. Marketing summary of Capstick, so don't cite it as evidence.

---

## Depression / mood

### Petersen J., Austin D., Mattek N., Kaye J., 2015 — *Time out-of-home and cognitive, physical, and emotional wellbeing of older adults: a longitudinal mixed effects model*, PLoS ONE 10(10): e0139643 — `READ` — T3 (key source), POS
- Link: https://pmc.ncbi.nlm.nih.gov/articles/PMC4593630/
- 85 independent older adults aged 65–96, **living alone**, from ISAAC/ORCATECH. In-home sensors for up to 1 year.
- Mean time out-of-home **4.2 ± 2.7 h/day**.
- Coefficients (h/day time out):
  - **Low-mood week: −0.520** (p < .001), ≈ 25 min less per day out.
  - **Cognition CDR (0.5 = MCI vs 0): −1.687** (p < .001). Probability of leaving home on a given day is ≈ 13 % lower (text says 12 %).
  - Loneliness (UCLA, per point): −0.046, so the loneliest vs least lonely spend 1.29 h less out per day out.
  - Pain (per unit 0–10): −0.123.
  - Normalised in-home walking speed: +0.115 per SD.
- Caveats noted when reading: the text and table are inconsistent in places (walking speed, pain). The mood measure is a weekly yes/no item, not GDS. These are **population coefficients**, while we apply them within one person.

### JMIR 2026 — longitudinal passive sensing in 75 depressed older adults (e92348) — `SNIPPET` — POS
- Link: https://www.jmir.org/2026/1/e92348
- More time away from home was associated with *lower* self-reported behavioural activation in the active-control group (preliminary, not multiplicity-adjusted). The direction is counter-intuitive, which shows mood ↔ mobility is not simple.

### Gerontechnology 2012 — PIR motion density maps and mental-health change — `SNIPPET`
- Link: https://pmc.ncbi.nlm.nih.gov/articles/PMC3773874
- Asks whether density maps + average time out are sensitive to mental-health change over time.

### BMC Psychiatry, 2020 — *Circadian rhythm and sleep alterations in older people with lifetime depression* — `SNIPPET` — T3
- Link: https://doaj.org/article/024c6b04f56e43c9b90f66e687330f3f
- 34 depressed vs 30 controls: **longer sleep latency** and REM latency, with altered sleep timing.
- Same search (separate study, not this paper): in community-dwelling older women, depressive symptoms are linked to **objective sleep fragmentation and napping**. In older men, there was only a modest link with sleep latency.

### Rotterdam Study — actigraphy sleep / 24-h rhythm ↔ depressive symptoms, bidirectional — `SNIPPET`
- 1,734 participants, mean age 62.3.

### American Psychiatric Association — DSM-5 Major Depressive Episode — `MEMORY` — T3
- ≥2 weeks. Criteria include diminished interest (A2), appetite change (A3), insomnia/hypersomnia (A4).

---

## Mild cognitive impairment / decline

### *Early detection of MCI with in-home sensors… Singapore: cross-sectional feasibility study*, JMIR Aging 2020 — `READ` — T3
- Link: https://pmc.ncbi.nlm.nih.gov/articles/PMC7238076/
- 59 seniors living alone, 2 months. MCI n = 28 vs healthy n = 21. **No difference was significant.**
- Time away from home 267 vs 300 min/day (p = .44). Outings 1 vs 1/day. Sleep 427 vs 440 min. **Sleep interruptions 2 vs 1/night (p = .27).** Steps 3,407 vs 4,033.
- Cites prior work: walking-speed CoV in MCI is about **2× controls** (Dodge et al., Neurology 2012).

### Kaye J. et al. / ORCATECH — `SNIPPET` — T3, POS
- Alzforum: https://alzforum.org/node/36824 — **day-to-day variability** is one of the earliest signs of decline, and in-home sensing detects MCI change more sensitively than clinical tests (group's claim).
- Methods: https://pmc.ncbi.nlm.nih.gov/articles/PMC4462097 · https://frontiersin.org/journals/aging-neuroscience/articles/10.3389/fnagi.2015.00102/full — more than 480 homes, up to 8 years of data.
- Hayes T. et al., 2008, *Unobtrusive assessment of activity patterns associated with MCI*, Alzheimer's & Dementia (PMID 19012864) — `TODO`.
- Dodge H. et al., 2012, Neurology — in-home walking speed and its variability (CoV) — `TODO`.
- Fall-related: day-to-day variability of walking speed was lower before a fall. https://alzforum.org/node/1226746

---

## Smart-home health events, anomaly detection, simulation (from the project plan)

### Fritz R., Wuestney K., Dermody G., Cook D., 2022 — *Nurse-in-the-loop smart home detection of health events*, Int J Nursing Studies Advances — `SNIPPET` — T3, T8
- Link: https://doaj.org/article/3a707206d43d420dbc2ed2753fd107ad
- 25 homes. Cases: 2 CHF exacerbations, 1 UTI, 2 bowel-inflammation flares, 4 sleep interruption. Health events aligned with **changes from the person's own baseline** in sensor-derived behaviour.

### Dahmen J., Cook D., 2021 — *Indirectly supervised anomaly detection of clinically meaningful health events from smart home data (Isudra)*, ACM TIST 12(2) — `MEMORY` — T6
- Link: https://ir.webis.de/anthology/2021.tist_journal-ir0anthology0volumeA12A2.4
- Validated on more than 2M sensor readings from 5 homes with 26 health events (falls, nocturia, depression, weakness). Beats supervised and unsupervised baselines.

### Dahmen J., 2019 — *Automated health event detection in smart homes* (PhD thesis, WSU) — `MEMORY` / `TODO` — T4
- Synthetic smart-home data with realistic synthetic health events, which is the closest prior work to our O1.

### Dahmen J., Cook D., 2019 — *SynSys*, Sensors 19(5): 1181 — `MEMORY` — T2, T5
- HMM-based synthetic smart-home data trained on real homes. Realism measured by time-series distance vs random data, other homes and other periods.

### Alshammari T. et al., 2018 — *SIMADL*, Data — `MEMORY` — T4
- Simulated ADL dataset with injected anomaly patterns. Dataset is on Zenodo.

### Tanaka et al. — simulator-based detection of typical anomalies — `MEMORY` — T7
- Classifier trained on simulator data, with parameters tuned per anomaly (semi-bedridden, not leaving home, forgetting, wandering, falls). This is "train in simulation, detect anomaly type", the same as our T7.

### Shahid Z. K., Saguna S., Åhlund C., 2022 — JMIR Aging — `MEMORY` — T6 baseline
- About 2 years of smart-home data from older adults. Anomaly if activity duration is outside [μ − 2σ, μ + 2σ] or the activity is missing.

### Ugwu, Fritz, Cook, Doppa — clinician-in-the-loop UTI recurrence (AAAI) — `MEMORY` — T8
- Clinically relevant behavioural markers from ambient sensors. Reference for the clinician-facing report format.

### You W. et al. — *ADLGen* — `MEMORY` — POS (cited in the ObsAVQ project description)
- Code: https://github.com/wyou-uga/ADLGen — decoder-only Transformer with symbolic temporal encoding, context/layout-aware sampling, and an LLM generate–evaluate–refine loop.

### Bouchabou D., Nguyen S. M., Lohr C., LeDuc B., Kanellos I., 2021 — HAR survey, Sensors 21: 6037 — `MEMORY` — POS
### Bouchabou D., Grosset J., Nguyen S. M., Lohr C., Puig X., 2023 — *A smart home digital twin to support the recognition of ADL* (Virtual Smart Home), Sensors — `MEMORY` — POS
### Fleury A., Vacher M., Noury N. — *SVM-based multimodal classification of ADL in health smart homes*, IEEE TITB — `MEMORY` — POS
### Vacher M., Portet F., Fleury A., Noury N., 2011 — audio sensing for AAL, IJEHMC — `MEMORY` — POS
### AgentSense, 2025 — LLM agents living routines in virtual homes, evaluated on CASAS Aruba/Milan/Kyoto7/Cairo and Orange4Home — `MEMORY` — POS

---

## Dataset

### CASAS smart home datasets — Zenodo record 15708568 — `READ`
- Link: https://zenodo.org/records/15708568 · community: https://zenodo.org/communities/casas
- 189 homes, 2007–2024, PIR motion + magnetic door sensors named by location. Some homes are activity-labelled.
- hh101–106, hh108–120 and hh122–130 have 1 resident each: older adults living independently in a retirement community. **hh113 and hh115 are used** (see README, Setup).
