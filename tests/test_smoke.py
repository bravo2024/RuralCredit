from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.data import make_synthetic
from src.model import train_all_models, cross_validate
from src.core import compute_metrics, disparate_impact


def test_data_shape():
    data = make_synthetic(n=500, seed=42)
    assert data["X"].shape[0] == 500
    assert 0.05 < data["positive_rate"] < 0.50
    assert "gender" in data["protected_attributes"]


def test_train_all_models():
    data = make_synthetic(n=300, seed=42)
    bundle = train_all_models(data, seed=42)
    assert len(bundle["models"]) == 4
    for n, r in bundle["results"].items():
        assert r["metrics"].get("roc_auc", 0) > 0.5


def test_xgboost_trains():
    data = make_synthetic(n=300, seed=42)
    bundle = train_all_models(data, seed=42)
    assert bundle["results"]["XGBoost"]["metrics"].get("roc_auc", 0) > 0.5


def test_cross_validation():
    data = make_synthetic(n=300, seed=42)
    cv = cross_validate(data, seed=42, n_folds=3)
    for n, s in cv.items():
        assert s["roc_auc"]["mean"] > 0.5
        assert len(s["roc_auc"]["values"]) == 3  # type: ignore


def test_disparate_impact():
    y_pred = np.array([1, 1, 0, 0, 1, 0, 0, 1, 0, 0])
    protected = np.array(["female", "female", "female", "male", "male", "male", "female", "female", "male", "male"])
    di = disparate_impact(y_pred, protected, group_name="female")
    assert 0 <= di <= 2


def test_metrics_ranges():
    data = make_synthetic(n=200, seed=42)
    bundle = train_all_models(data, seed=42)
    for n, r in bundle["results"].items():
        m = r["metrics"]
        assert 0 <= m["accuracy"] <= 1
        assert 0 <= m["precision"] <= 1
        assert 0 <= m["recall"] <= 1
        assert 0 <= m["f1"] <= 1
