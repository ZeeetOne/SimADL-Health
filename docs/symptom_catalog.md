# Symptom catalogue (Stage T3)

How each condition changes the simulator parameters of T2. Full notes and links for every reference: `docs/literature_notes.md`.
Every change is tagged:

- **SOURCED**: the direction *and* the magnitude come from a cited study.
- **DIRECTION**: the direction is sourced, but the magnitude is our choice.
- **ASSUMPTION**: no source found. The value is plausible and stated openly, so treat it as a limitation.

Magnitudes are given at full intensity (`intensity = 1`). T4 samples `intensity ∈ [0.5, 1]` per episode and scales
every change linearly, so detection is tested on milder cases as well.
Changes are **relative to the persona's own baseline** (hh113 vs hh115 night-toilet baselines differ 8x, see T1).

## 1. Urinary tract infection (UTI) with nocturia — acute

| Parameter | Change at full intensity | Onset / time course | Tag | Basis |
|---|---|---|---|---|
| `night_toilet` | **+1.5 visits/night** (additive) | ramp up over 1–2 days, plateau, then resolve | DIRECTION | UTI → frequency/urgency/nocturia [L1][C1]; bathroom-visit statistics were the top predictors of UTI in home monitoring [C1]. Clinically, nocturia counts from ≥1 void/night [I1] and ≥2 is the usual bothersome threshold. +1.5 takes hh113 (0.26) above 1.5 and hh115 (2.0) to 3.5. |
| sleep fragmentation | follows from the extra toilet visits (no separate parameter) | — | (mechanistic) | Each toilet visit splits a Sleep episode in the simulator. |
| `breakfast_p`, `lunch_p`, `dinner_p` | **−15 %** | same as above | ASSUMPTION | Fatigue and malaise in atypical UTI presentations of older adults [C1][A1]. No quantitative source. |
| social (outings, phone, guests) | unchanged | — | ASSUMPTION | Kept fixed so the condition is mainly a night-time signal. |
| duration | **4–9 days** | — | SOURCED | Most symptom improvement happens in the first 9 days, with ~⅓ resolving spontaneously in 7–10 days [H1]. Severe symptoms last ~3.3 days when treated, and ~62 % longer without antibiotics [L2]. |

## 2. Depression — gradual

| Parameter | Change at full intensity | Onset / time course | Tag | Basis |
|---|---|---|---|---|
| total time out of home | **−0.52 h/day** | linear ramp over 14 days, then plateau | SOURCED | Weeks with low mood: β = −0.52 h/day time out-of-home (85 older adults living alone, in-home sensors, up to 1 year) [P1]. Implemented by scaling `outings` and `trip_scale` so the persona's daily total drops by 0.52 h. |
| `phone_calls`, `guest_visits` | **−30 %** | same ramp | DIRECTION | Loss of interest and social withdrawal (DSM-5 criterion A2) [D1]. Loneliness also lowers time out [P1]. Magnitude is ours. |
| `sleep_onset` | **+45 min later** | same ramp | DIRECTION | Longer sleep latency in older adults with depression [B1], and insomnia/hypersomnia in DSM-5 criterion A4 [D1]. We model the insomnia variant. Magnitude is ours. |
| `extra_awakenings` | **+0.5 /night** | same ramp | DIRECTION | Depressive symptoms are associated with objective sleep fragmentation in older adults [B1]. |
| `breakfast_p`, `lunch_p`, `dinner_p` | **−25 %** (skipped meals) | same ramp | DIRECTION | Decreased appetite (DSM-5 criterion A3) [D1]. Magnitude is ours. |
| duration | **28–56 days** | — | DIRECTION | DSM-5 requires symptoms for ≥2 weeks [D1]. 4–8 weeks is chosen so the plateau is visible in weekly windows. |

## 3. Mild cognitive decline (MCI-like) — very gradual, persistent

| Parameter | Change at full intensity | Onset / time course | Tag | Basis |
|---|---|---|---|---|
| total time out of home | **−1.69 h/day** | linear ramp over 90–180 days, persists until the end | SOURCED | CDR 0.5 (MCI) vs CDR 0: β = −1.69 h/day time out-of-home, ~13 % lower probability of leaving home on a given day [P1]. Same singleton-persona sensor platform as our target. |
| day-to-day variability of `sleep_onset`, `wake`, meal times | **std × 2** | same ramp | DIRECTION | Day-to-day variability is one of the earliest in-home signs of decline [K1]. Walking-speed CoV in MCI is about twice that of controls [S1 citing Dodge 2012]. We transfer the ×2 from walking speed to routine timing, which is our assumption. |
| sleep interruptions (`extra_awakenings`) | **+1 /night** | same ramp | DIRECTION (weak) | MCI 2 vs healthy 1 interruptions/night, n.s. (p = .27, n = 49) [S1]. |
| repeated meal | **+20 % chance/day of an extra meal block** | same ramp | ASSUMPTION | Forgetting that one has eaten is commonly described, but we found no sensor-based rate. |
| `phone_calls`, `guest_visits` | **−20 %** | same ramp | ASSUMPTION | Social disengagement accompanies cognitive decline. Magnitude is ours. |

## 4. Normal

No parameter changes. Normal personas still vary day to day, weekday vs weekend, so "normal" is not constant.

## Known weaknesses (for the report)
- The UTI magnitude is not sourced; its effect rests on the clinical nocturia threshold. [C1] reports the features that matter, not effect sizes. The next step would be to read the full text of [C1] and of Rantz et al. "Using sensor networks to detect UTIs in older adults".
- [P1] effects are population regression coefficients. We apply them as within-person changes.
- Conditions are injected one at a time, never overlapping. In reality, comorbidity is common.
- The CASAS homes have no physiological sensors. Night-time respiratory rate, a key UTI predictor in [C1], cannot be simulated.

## References
- **[C1]** Capstick A. et al. *Digital remote monitoring for screening and early detection of urinary tract infections.* npj Digital Medicine, 2024. 117 people with dementia, 27,828 person-days. Top predictors: bathroom visit statistics, night-time respiratory rate, previous UTIs. https://ukdri.ac.uk/publications/digital-remote-monitoring-screening-and-early-detection-urinary-tract-infections-0
- **[A1]** *Clinician-in-the-loop smart home system to detect UTI flare-ups* (arXiv 2511.18334). Atypical UTI symptoms in older adults (confusion, dizziness, falls). https://arxiv.org/pdf/2511.18334
- **[L1]** Loeb M. et al. *Development of minimum criteria for the initiation of antibiotics in residents of long-term-care facilities.* Infect Control Hosp Epidemiol, 2001. New-onset frequency/urgency as UTI criteria.
- **[I1]** van Kerrebroeck P. et al. *The standardisation of terminology in nocturia (ICS).* Neurourol Urodyn, 2002. Nocturia = waking ≥1 time at night to void.
- **[H1]** Hoffmann T. et al. *Natural history of uncomplicated UTI without antibiotics: systematic review.* BJGP, 2020. https://bjgp.org/content/70/699/e714
- **[L2]** Little P. et al. *Presentation, pattern and natural course of severe symptoms… of UTI.* BMJ, 2010.
- **[P1]** Petersen J., Austin D., Mattek N., Kaye J. *Time out-of-home and cognitive, physical, and emotional wellbeing of older adults: a longitudinal mixed effects model.* PLoS ONE 10(10): e0139643, 2015. https://pmc.ncbi.nlm.nih.gov/articles/PMC4593630/
- **[S1]** *Early detection of MCI with in-home sensors in community-dwelling senior citizens in Singapore: cross-sectional feasibility study.* JMIR Aging, 2020. https://pmc.ncbi.nlm.nih.gov/articles/PMC7238076/
- **[K1]** Kaye J. et al., ORCATECH. Day-to-day variability as an early sign of decline. https://alzforum.org/node/36824 ; Hayes T. et al. *Unobtrusive assessment of activity patterns associated with MCI.* Alzheimer's & Dementia, 2008 (PMID 19012864).
- **[B1]** *Circadian rhythm and sleep alterations in older people with lifetime depression: a case-control study.* BMC Psychiatry, 2020. Longer sleep latency, plus objective fragmentation in older women with depressive symptoms. https://doaj.org/article/024c6b04f56e43c9b90f66e687330f3f
- **[D1]** American Psychiatric Association. *DSM-5*, Major Depressive Episode criteria A1–A9, ≥2 weeks.
- Context: Fritz R. et al. *Nurse-in-the-loop smart home detection of health events* (Int J Nurs Stud Adv, 2022). Health events (UTI, CHF, sleep interruption) align with deviations from the person's own sensor baseline.
