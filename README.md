# RuralCredit

> Financial inclusion credit scoring with fairness auditing (EU AI Act / RBI Fair Lending).

Trains four classifiers on synthetic rural applicant data to predict loan default, then audits each model for disparate impact, demographic parity, and equal opportunity across protected attributes (gender, caste category). Dashboard provides data exploration, model comparison, fairness metrics with score distribution by group, and calibration.

## Quickstart

```bash
pip install -r requirements.txt
python train.py
pytest -q
streamlit run app.py
```

## Model Performance

Best model (Logistic Regression) holdout results:

| Metric | Value |
|---|---|
| ROC AUC | 0.759 |
| Gini | 0.518 |
| F1 Score | 0.468 |
| Accuracy | 0.672 |

5-fold CV AUC: 0.752 ± 0.022. Four models compared.

## Features

| Tab | What it does |
|---|---|
| **Explorer** | Applicant dataset overview, default rate, feature descriptions |
| **Model Lab** | Multi-model comparison, ROC/calibration curves, CV results |
| **Fairness Audit** | Disparate impact ratio, demographic parity, equal opportunity by protected attribute, score distribution comparison |
| **Feature Analysis** | Permutation importance, SHAP-style feature ranking |

## Repo Structure

```
RuralCredit/
  src/         data, model, core, evaluate, visualizations modules
  train.py     training pipeline (multi-model + CV)
  app.py       Streamlit dashboard
  tests/       pytest smoke test
  models/      saved model + metrics (gitignored)
```

## Data

Synthetic rural credit dataset: seasonal income, mobile recharge, utility on-time percentage, crop sales, livestock value, land area, distance to branch, group loan history, caste category, gender, ration card, Kisan card, and default status.

## License

MIT
