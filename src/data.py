from __future__ import annotations
import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "seasonal_income", "mobile_recharge_avg", "utility_ontime_pct",
    "crop_sales", "livestock_value", "land_area_acres",
    "distance_to_branch", "group_loan_history", "family_size",
    "education_level", "gender", "caste_category",
    "has_ration_card", "has_kisan_card", "num_dependents",
]
PROTECTED_ATTRIBUTES = ["gender", "caste_category"]
CATEGORICAL_FEATURES = [
    "education_level", "gender", "caste_category",
    "has_ration_card", "has_kisan_card", "group_loan_history",
]
NUMERICAL_FEATURES = [
    "seasonal_income", "mobile_recharge_avg", "utility_ontime_pct",
    "crop_sales", "livestock_value", "land_area_acres",
    "distance_to_branch", "family_size", "num_dependents",
]


def make_synthetic(n=8000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "seasonal_income": rng.lognormal(mean=10.5, sigma=0.6, size=n).clip(5000, 500000).astype(int),
        "mobile_recharge_avg": rng.uniform(50, 800, size=n).round(1),
        "utility_ontime_pct": rng.beta(6 + rng.uniform(0, 4, size=n), 3, size=n).clip(0.1, 1.0).round(3),
        "crop_sales": rng.lognormal(mean=10, sigma=0.8, size=n).clip(0, 300000).astype(int),
        "livestock_value": rng.lognormal(mean=9, sigma=0.7, size=n).clip(0, 150000).astype(int),
        "land_area_acres": rng.lognormal(mean=1.2, sigma=0.9, size=n).clip(0.1, 50).round(2),
        "distance_to_branch": rng.uniform(0.5, 50, size=n).round(1),
        "group_loan_history": rng.choice(["yes", "no"], size=n, p=[0.4, 0.6]),
        "family_size": rng.poisson(lam=5, size=n).clip(1, 15),
        "education_level": rng.choice(
            ["none", "primary", "secondary", "higher_secondary", "graduate"],
            size=n, p=[0.30, 0.25, 0.22, 0.13, 0.10],
        ),
        "gender": rng.choice(["male", "female"], size=n, p=[0.52, 0.48]),
        "caste_category": rng.choice(
            ["general", "obc", "sc", "st", "ews"],
            size=n, p=[0.25, 0.30, 0.20, 0.15, 0.10],
        ),
        "has_ration_card": rng.choice(["yes", "no"], size=n, p=[0.65, 0.35]),
        "has_kisan_card": rng.choice(["yes", "no"], size=n, p=[0.25, 0.75]),
        "num_dependents": rng.poisson(lam=2.5, size=n).clip(0, 8),
    })
    utility = df["utility_ontime_pct"].values
    seasonal_income = np.log(df["seasonal_income"].values / 1000)
    mobile = df["mobile_recharge_avg"].values / 500
    distance = np.clip(df["distance_to_branch"].values / 30, 0, 2)
    livestock = np.log1p(df["livestock_value"].values / 1000)
    land = np.clip(df["land_area_acres"].values / 10, 0, 3)
    has_group = (df["group_loan_history"] == "yes").astype(int).values
    caste_sc = (df["caste_category"] == "sc").astype(int).values
    caste_st = (df["caste_category"] == "st").astype(int).values
    gender_female = (df["gender"] == "female").astype(int).values
    log_odds = (
        -3.0
        + 1.0 * utility
        + 0.15 * seasonal_income
        + 0.2 * mobile
        - 0.3 * distance
        + 0.1 * livestock
        + 0.1 * land
        + 0.3 * has_group
        - 0.3 * caste_sc
        - 0.4 * caste_st
        - 0.15 * gender_female
        + rng.normal(0, 0.5, size=n)
    )
    prob = 1.0 / (1.0 + np.exp(-log_odds))
    y = (prob > 0.50).astype(np.float64)
    if y.mean() < 0.10 or y.mean() > 0.40:
        targeted = np.percentile(prob, 75)
        y = (prob > targeted).astype(np.float64)
    return {
        "X": df, "y": y, "features": FEATURE_NAMES,
        "df": df.assign(default=y),
        "categorical_features": CATEGORICAL_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "protected_attributes": PROTECTED_ATTRIBUTES,
        "n_samples": n, "n_features": len(FEATURE_NAMES),
        "positive_rate": y.mean(),
    }
