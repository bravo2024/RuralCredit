# Rural credit model: Random Forest with one-hot encoding
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def fit_and_evaluate(data, seed=42):
    X = data["X"]
    y = np.asarray(data["y"])
    X = pd.get_dummies(X, columns=data["categorical_features"], drop_first=True).values.astype(float)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    model = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=seed, n_jobs=-1)
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(yte, proba))
    return (
        {"model": model},
        {
            "auc": auc,
            "n_trees": len(model.estimators_),
            "n_features": X.shape[1],
        },
    )