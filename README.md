# RuralCredit

Rural loan applicants usually don't have the paper trail a bank scorecard wants — no long credit history, no salary slips, sometimes no formal bank account at all. This project asks a narrower question: can you build a usable default-risk score out of the kind of data a rural lender *can* actually collect (land holdings, crop mix, seasonal income, monsoon exposure, livestock), and then check whether that score ends up treating protected groups (gender, caste category) unfairly.

It's a synthetic-data exercise, not a validated scoring model — there's no real loan book behind it.

## What's actually in here right now

There are two layers to this repo and they're at different levels of finish:

- **`src/data.py` + `src/model.py`'s `fit_and_evaluate`** — this is the part that's wired up end to end and covered by the smoke test (`tests/test_smoke.py`). It generates the synthetic applicant table (land area, crop type, seasonal income, a monsoon dependency score, livestock count) and fits a single Random Forest against a synthetic default label.
- **`train.py` and `app.py`** — these are written against a bigger design: four classifiers trained side by side, cross-validation, and a fairness audit across `gender` / `caste_category`. That design is what the results and dashboard description below are based on. Right now `src/model.py` and `src/core.py` don't yet implement the `train_all_models`, `cross_validate`, and fairness-metric functions those two files import, so `python train.py` and `streamlit run app.py` will fail on import until that's filled in. Flagging this here instead of quietly shipping a README that pretends it runs.

## Results (from the last saved training run, `models/metrics.json`)

Four models were compared on a holdout split. Logistic Regression came out ahead on ROC AUC despite the tree models posting higher raw accuracy — with a ~28% default rate, accuracy alone is a bad way to compare these:

| Metric | Logistic Regression |
|---|---|
| ROC AUC | 0.759 |
| Gini | 0.518 |
| F1 | 0.468 |
| Accuracy | 0.672 |

5-fold CV AUC for that model: 0.736 ± 0.054 — a fairly wide spread across folds, which with only a few thousand synthetic rows and a minority positive class isn't surprising.

Random Forest, Gradient Boosting, and XGBoost all landed in the 0.62–0.68 AUC range on this holdout, with better accuracy (up to 0.712) but noticeably worse recall on the default class — they're catching fewer of the actual defaulters, which is the expensive mistake in lending. Full numbers for all four are in `models/metrics.json`.

## Data

`src/data.py` generates: land holdings (acres), crop type (rice/wheat/cotton/sugarcane), seasonal income, a monsoon-dependency score, and livestock count. The default label is a logistic function of a weighted combination of those, thresholded to roughly a 28% positive rate.

The fairness-audit side of the dashboard (`gender`, `caste_category`, plus mentions of mobile recharge history, utility payment record, group loan history, ration/Kisan card status) describes the intended richer feature set for the audit — it isn't in the current synthetic generator, which is the gap noted above.

## Dashboard tabs (`app.py`, once the training layer is filled in)

| Tab | What it's for |
|---|---|
| Data Explorer | Class balance, feature histograms, dataset overview |
| Model Lab | Side-by-side metrics table for all four models, ROC and calibration curves, 5-fold CV |
| Fairness Audit | Disparate impact ratio, demographic parity difference, equal opportunity difference by protected group, with the EEOC 4/5ths-rule band called out |
| Score Analysis | Predicted-score distributions split by gender and caste category, plus approval rate by group at the chosen threshold |
| Loan Pricing | Risk-based interest rate example (base rate + PD-linked spread) with a fairness cap on the rate gap between groups |

## Running it

```bash
pip install -r requirements.txt
python train.py          # currently blocked, see "What's actually in here" above
pytest -q                # this part works — covers make_synthetic + fit_and_evaluate
streamlit run app.py     # currently blocked, same reason
```

## Layout

```
RuralCredit/
  src/         data, model, core, evaluate, persist, visualizations modules
  train.py     training pipeline (multi-model + CV, see gap note)
  app.py       Streamlit dashboard (same gap note)
  tests/       pytest smoke test
  models/      saved model + metrics (gitignored)
```

Licensed MIT.
