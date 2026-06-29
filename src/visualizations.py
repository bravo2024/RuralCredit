from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

THEME = {"bg": "#0e1117", "fg": "#ffffff", "grid": "#1a1f2e", "cyan": "#22d3ee", "violet": "#a78bfa", "orange": "#f97316", "rose": "#f43f5e", "amber": "#fbbf24", "green": "#22c55e"}


def _style():
    plt.rcParams.update({"figure.facecolor": THEME["bg"], "axes.facecolor": THEME["bg"], "axes.edgecolor": THEME["grid"], "axes.labelcolor": THEME["fg"], "text.color": THEME["fg"], "xtick.color": THEME["fg"], "ytick.color": THEME["fg"], "grid.color": THEME["grid"], "grid.alpha": 0.3, "legend.facecolor": "#1a1f2e", "legend.edgecolor": THEME["grid"], "legend.labelcolor": THEME["fg"]})


def plot_roc_curve(y_true, y_proba_dict):
    _style()
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [THEME["cyan"], THEME["violet"], THEME["orange"], THEME["rose"]]
    for (name, y_proba), c in zip(y_proba_dict.items(), colors):
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = np.trapz(tpr, fpr)
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], ":", color="#555", lw=1.5)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC Curves", color=THEME["fg"])
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    return fig


def plot_fairness_metrics(fairness_df):
    _style()
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(fairness_df))
    width = 0.25
    metrics = ["Demographic\nParity Diff", "Equal\nOpportunity Diff", "Disparate\nImpact"]
    colors_m = [THEME["cyan"], THEME["violet"], THEME["amber"]]
    for i, metric in enumerate(metrics):
        vals = fairness_df.iloc[:, i].values
        bars = ax.bar(x + i * width, vals, width, color=colors_m[i], alpha=0.8, label=metric)
    ax.axhline(0, color="#555", lw=1)
    ax.axhline(0.8, color=THEME["green"], ls="--", lw=1, alpha=0.5)
    ax.axhline(1.25, color=THEME["green"], ls="--", lw=1, alpha=0.5)
    ax.set_xticks(x + width)
    ax.set_xticklabels(fairness_df.index, fontsize=8)
    ax.set_title("Fairness Metrics by Protected Group", color=THEME["fg"])
    ax.legend(fontsize=7); ax.grid(True, alpha=0.2, axis="y")
    return fig


def plot_score_by_group(y_proba, group_labels, group_name="Gender"):
    _style()
    fig, ax = plt.subplots(figsize=(8, 4))
    unique_groups = sorted(set(group_labels))
    for i, g in enumerate(unique_groups):
        mask = np.array(group_labels) == g
        scores = y_proba[mask]
        ax.hist(scores, bins=30, alpha=0.5, label=f"{g} (n={mask.sum()})", density=True)
    ax.set_xlabel("Predicted Default Probability")
    ax.set_ylabel("Density")
    ax.set_title(f"Score Distribution by {group_name}", color=THEME["fg"])
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    return fig


def plot_feature_importance(importances, features, color=None):
    _style()
    imp = np.array([v["mean"] if isinstance(v, dict) else v for v in importances])
    idx = np.argsort(np.abs(imp))
    fig, ax = plt.subplots(figsize=(8, max(4, len(features) * 0.3)))
    ax.barh(range(len(features)), imp[idx], color=color or THEME["cyan"], alpha=0.8)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels([features[i] for i in idx], fontsize=8)
    ax.set_xlabel("Importance"); ax.set_title("Feature Importance", color=THEME["fg"])
    ax.grid(True, alpha=0.2, axis="x")
    return fig


def plot_calibration_curve(y_true, y_proba_dict):
    _style()
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [THEME["cyan"], THEME["violet"], THEME["orange"], THEME["rose"]]
    for (name, y_proba), c in zip(y_proba_dict.items(), colors):
        bins = np.linspace(0, 1, 11)
        bin_idx = np.digitize(y_proba, bins[1:-1])
        mean_pred = np.array([y_proba[bin_idx == i].mean() if (bin_idx == i).sum() > 0 else bins[i] for i in range(10)])
        frac_pos = np.array([y_true[bin_idx == i].mean() if (bin_idx == i).sum() > 0 else 0 for i in range(10)])
        ax.plot(mean_pred, frac_pos, "o-", color=c, lw=2, ms=4, label=name)
    ax.plot([0, 1], [0, 1], ":", color="#555", lw=1.5)
    ax.set_xlabel("Mean Predicted Probability"); ax.set_ylabel("Fraction Positive")
    ax.set_title("Calibration Curve", color=THEME["fg"])
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    return fig
