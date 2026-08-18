# Hospital Readmission Prediction (30-day) + HSE Cost Analysis

Predicting whether a diabetic patient will be readmitted to hospital within 30 days, then
turning that prediction into a Low/Medium/High risk score, mapping each tier to an Irish
**HSE** follow-up pathway, and estimating the cost that could be saved.

*MSc Data Analytics — Domain Applications project.*

## Overview

Unplanned 30-day readmissions are a common measure of care quality and a big cost to health
systems. This project builds a machine-learning pipeline to flag high-risk diabetic patients
at discharge, so follow-up care can be targeted where it matters most.

The focus is on the **applied side**, not just the model score: the predictions are converted
into a three-tier risk framework, linked to real HSE Enhanced Community Care actions, and
followed by a transparent cost-impact analysis.

## Data

- **Source:** UCI *Diabetes 130-US Hospitals (1999–2008)* — [dataset link](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
- **Size:** 101,766 encounters × 50 columns
- After cleaning (removing death/hospice discharges and keeping the first encounter per
  patient), the analytic cohort is **69,990 patients**.

> Download `diabetic_data.csv` from the link above and place it in the same folder as the
> notebook before running.

## Approach

1. **EDA** — target balance, missing values, and what drives readmission (age, prior inpatient visits, discharge type)
2. **Preprocessing** — group ICD-9 diagnoses, encode age, one-hot encode categoricals, leakage-safe train/test split
3. **Feature selection** — Recursive Feature Elimination (139 → 40 features)
4. **Imbalance handling** — SMOTE on the training set only (~9% positive class)
5. **Modelling** — LightGBM (main) + Logistic Regression (baseline), with threshold tuning for F1
6. **Risk stratification** — Low / Medium / High tiers mapped to HSE pathways
7. **Cost analysis** — estimated bed-day savings, with a sensitivity analysis

## Key results

| Metric | Result |
|---|---|
| 30-day readmission rate | 8.98% |
| LightGBM AUC-ROC | ≈ 0.646 |
| Logistic Regression AUC-ROC | ≈ 0.638 |
| Risk tiers (observed readmission) | Low 6.4% · Medium 11.3% · High 26.8% |
| Cost impact (High + Medium tiers) | ≈ €207k net saving in the test cohort |

The High tier captures patients readmitted at roughly **3× the cohort average**, which is what
makes targeted follow-up worthwhile. AUC ≈ 0.65 is about the realistic ceiling for this dataset,
so the value of the project is in the decision-support and cost layers built on top of the model.

## Tech stack

Python · pandas · NumPy · scikit-learn · imbalanced-learn (SMOTE) · LightGBM · Matplotlib · seaborn

## Repository structure

```
.
├── hospital_readmission_analysis.ipynb   # full analysis (run top to bottom)
├── report/                               # written report
├── data/                                 # place diabetic_data.csv here (see Data section)
└── README.md
```

## How to run

```bash
pip install pandas numpy scikit-learn imbalanced-learn lightgbm matplotlib seaborn joblib
```

Put `diabetic_data.csv` in the folder, open the notebook, and run all cells top to bottom.
A fixed random seed makes the results reproducible.

## Author

**Ya Wai Thone** — MSc Data Analytics, National College of Ireland
[LinkedIn](https://www.linkedin.com/in/ya-wai-thone/) · [GitHub](https://github.com/yawaithone)

