from __future__ import annotations
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb


def compute_metrics(y_true, y_pred, y_proba=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    return metrics


def build_models(X_train, y_train, seed=42):
    lr = LogisticRegression(C=0.1, class_weight="balanced", solver="liblinear", random_state=seed, max_iter=1000)
    lr.fit(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=20, class_weight="balanced", random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)
    gbt = GradientBoostingClassifier(n_estimators=200, max_depth=5, min_samples_leaf=20, learning_rate=0.05, subsample=0.8, random_state=seed)
    gbt.fit(X_train, y_train)
    xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=seed)
    xgb_model.fit(X_train, y_train)
    return {"Logistic Regression": lr, "Random Forest": rf, "Gradient Boosting": gbt, "XGBoost": xgb_model}


def disparate_impact(y_pred, protected, group_name="female"):
    mask = (protected == group_name)
    approval_advantaged = (y_pred[~mask] == 0).mean()
    approval_disadvantaged = (y_pred[mask] == 0).mean()
    di_ratio = approval_disadvantaged / approval_advantaged if approval_advantaged > 0 else 0
    return di_ratio


def demographic_parity(y_pred, protected, group_name="female"):
    mask = (protected == group_name)
    rate_advantaged = y_pred[mask].mean()
    rate_disadvantaged = y_pred[~mask].mean()
    return rate_advantaged - rate_disadvantaged


def equal_opportunity(y_true, y_pred, protected, group_name="female"):
    mask = (protected == group_name)
    tpr_advantaged = (y_pred[mask & (y_true == 1)]).sum() / max((mask & (y_true == 1)).sum(), 1)
    tpr_disadvantaged = (y_pred[~mask & (y_true == 1)]).sum() / max((~mask & (y_true == 1)).sum(), 1)
    return tpr_advantaged - tpr_disadvantaged
