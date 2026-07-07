# Rural credit data: crop type, land, monsoon dependency
import numpy as np
import pandas as pd

FEATURES = [
    "land_acres", "crop_type", "seasonal_income",
    "monsoon_score", "livestock_count",
]

def make_synthetic(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    land = rng.lognormal(1.5, 0.5, n).clip(0.5, 50).round(1)
    crop = rng.choice(["rice", "wheat", "cotton", "sugarcane"], n, p=[0.4, 0.3, 0.2, 0.1])
    income = rng.lognormal(10.5, 0.6, n).round(0).astype(int)
    monsoon = rng.beta(4, 3, n).round(2)
    livestock = rng.poisson(5, n).clip(0, 30).astype(int)
    df = pd.DataFrame({
        "land_acres": land, "crop_type": crop,
        "seasonal_income": income, "monsoon_score": monsoon,
        "livestock_count": livestock,
    })
    score = (
        0.3 * (land / 50) + 0.3 * monsoon
        + 0.2 * np.clip(income / 500000, 0, 1)
        + 0.1 * (livestock / 30)
        + 0.1 * (crop == "sugarcane").astype(float)
    )
    prob = 1 / (1 + np.exp(-(3 * score - 1.5)))
    y = (prob > np.percentile(prob, 72)).astype(float)
    return {
        "X": df, "y": y, "features": FEATURES,
        "categorical_features": ["crop_type"],
        "numerical_features": ["land_acres", "seasonal_income", "monsoon_score", "livestock_count"],
        "n_samples": n, "positive_rate": float(y.mean()),
    }