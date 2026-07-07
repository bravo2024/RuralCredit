from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from src.data import make_synthetic
from src.model import train_all_models, cross_validate
from src.core import compute_metrics, disparate_impact, demographic_parity, equal_opportunity
from src.visualizations import (
    plot_roc_curve, plot_fairness_metrics, plot_score_by_group,
    plot_feature_importance, plot_calibration_curve,
)

st.set_page_config(page_title="RuralCredit | Financial Inclusion", layout="wide", page_icon="\U0001f33e")

with st.sidebar:
    st.header("\u2699 Configuration")
    n_samples = st.slider("Sample size", 2000, 15000, 8000, step=1000)
    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, step=0.05)
    protected_attr = st.selectbox("Protected attribute", ["gender", "caste_category"])
    st.divider()
    st.caption("L&T Finance | Financial Inclusion")
    st.caption("EU AI Act high-risk | RBI Fair Lending")

data = make_synthetic(n=n_samples)
bundle = train_all_models(data)
xgb_y_proba = bundle["results"]["XGBoost"]["y_proba"]
xgb_y_pred = bundle["results"]["XGBoost"]["y_pred"]
y_test = bundle["y_test"]

col1, col2, col3, col4, col5 = st.columns(5)
best_name = max(bundle["results"], key=lambda n: bundle["results"][n]["metrics"].get("roc_auc", 0))
best_m = bundle["results"][best_name]["metrics"]
col1.metric("Samples", f"{data['n_samples']:,}")
col2.metric("Default Rate", f"{data['positive_rate']:.1%}")
col3.metric("Best AUC", f"{best_m['roc_auc']:.4f}")
col4.metric("Best Model", best_name)
col5.metric("Accuracy", f"{best_m['accuracy']:.3f}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["\U0001f4ca Data Explorer", "\U0001f52c Model Lab", "\u2696\ufe0f Fairness Audit", "\U0001f4c8 Score Analysis", "\U0001f4b0 Loan Pricing"])

with tab1:
    st.subheader("Alternative Credit Data — Rural Markets")
    st.dataframe(data["df"].head(50), use_container_width=True, height=250)
    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots(figsize=(5, 3))
        from src.visualizations import _style; _style()
        rates = [1 - data["positive_rate"], data["positive_rate"]]
        ax.bar(["Good (0)", "Default (1)"], rates, color=["#22c55e", "#f43f5e"], alpha=0.7)
        for i, v in enumerate(rates):
            ax.text(i, v + 0.01, f"{v:.1%}", ha="center", color="#ffffff", fontsize=12)
        ax.set_ylabel("Proportion"); ax.set_title("Class Balance", color="#ffffff"); ax.grid(True, alpha=0.2, axis="y")
        st.pyplot(fig)
    with col_b:
        st.markdown("""
        **Alternative data features:**
        - Mobile recharge history
        - Utility on-time payment %
        - Seasonal crop sales
        - Livestock valuation
        - Group loan history
        - Distance to branch
        """)
    num_cols = data["numerical_features"]
    if num_cols:
        fig, axes = plt.subplots(2, 3, figsize=(14, 5))
        _style()
        for ax_i, col in enumerate(num_cols[:6]):
            r, ci = divmod(ax_i, 3)
            vals = data["df"][col].dropna().values
            axes[r, ci].hist(vals, bins=40, color="#22d3ee", alpha=0.6)
            axes[r, ci].set_title(col, fontsize=9, color="#ffffff")
            axes[r, ci].grid(True, alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig)

with tab2:
    st.subheader("Model Comparison")
    y_proba_dict = {n: bundle["results"][n]["y_proba"] for n in bundle["results"]}
    metrics_df = []
    for n, r in bundle["results"].items():
        m = r["metrics"]
        metrics_df.append({"Model": n, "AUC": f"{m.get('roc_auc',0):.4f}", "F1": f"{m.get('f1',0):.4f}", "Precision": f"{m.get('precision',0):.4f}", "Recall": f"{m.get('recall',0):.4f}", "Accuracy": f"{m.get('accuracy',0):.4f}"})
    st.dataframe(pd.DataFrame(metrics_df).set_index("Model"), use_container_width=True)
    col_a, col_b = st.columns(2)
    with col_a: st.pyplot(plot_roc_curve(y_test, y_proba_dict))
    with col_b: st.pyplot(plot_calibration_curve(y_test, y_proba_dict))
    cv = cross_validate(data)
    st.subheader("5-Fold Cross-Validation")
    cv_rows = [{"Model": n, "AUC (mean)": f"{s['roc_auc']['mean']:.4f}", "AUC (std)": f"\u00b1{s['roc_auc']['std']:.4f}"} for n, s in cv.items()]
    st.dataframe(pd.DataFrame(cv_rows).set_index("Model"), use_container_width=True)

with tab3:
    st.subheader("Fairness Audit")
    st.markdown(r"""
    **EU AI Act (Aug 2026):** Credit scoring is high-risk. Must demonstrate non-discrimination.
    **RBI Fair Lending Code:** No adverse impact based on caste, gender, religion.
    """)
    st.latex(r"\text{Disparate Impact} = \frac{\text{Approval Rate}_{\text{protected}}}{\text{Approval Rate}_{\text{reference}}} \quad \text{Target: } 0.8 \le \text{DI} \le 1.25")
    st.latex(r"\text{Demographic Parity} = |P(\hat{Y}=1|A=a) - P(\hat{Y}=1|A=b)|")
    prot_series = data["df"][protected_attr]
    preds = xgb_y_pred
    groups = sorted(prot_series.unique())
    fair_rows = []
    for g in groups:
        if g == groups[0]:
            continue
        mask_g = (prot_series == g)
        mask_ref = (prot_series == groups[0])
        di = disparate_impact(preds, prot_series.values, group_name=g)
        dp = demographic_parity(preds, prot_series.values, group_name=g)
        eo = equal_opportunity(y_test, preds, prot_series.values, group_name=g)
        fair_rows.append({"Group": g, "Disparate Impact": f"{di:.3f}", "Demographic Parity Diff": f"{dp:.3f}", "Equal Opportunity Diff": f"{eo:.3f}", "Status": "\u2705" if 0.8 <= di <= 1.25 else "\u26a0\ufe0f"})
    fair_df = pd.DataFrame(fair_rows).set_index("Group")
    st.dataframe(fair_df, use_container_width=True)
    st.pyplot(plot_score_by_group(xgb_y_proba, prot_series.values, protected_attr))
    st.info("Disparate Impact between 0.8-1.25 = no adverse impact (EEOC 4/5ths rule). Values outside indicate potential bias requiring mitigation.")

with tab4:
    st.subheader("Score Distribution Analysis")
    st.markdown("Score distribution segmented by protected attribute reveals systematic bias in model predictions.")
    col_a, col_b = st.columns(2)
    with col_a:
        st.pyplot(plot_score_by_group(xgb_y_proba, data["df"]["gender"].values, "Gender"))
    with col_b:
        st.pyplot(plot_score_by_group(xgb_y_proba, data["df"]["caste_category"].values, "Caste Category"))
    st.subheader("Approval Rate by Group at Current Threshold")
    for attr in ["gender", "caste_category"]:
        series = data["df"][attr]
        groups_s = sorted(series.unique())
        app_rows = []
        for g in groups_s:
            mask = (series == g)
            app_rate = (xgb_y_pred[mask] == 0).mean()
            count = mask.sum()
            app_rows.append({"Group": g, "Count": count, "Approval Rate": f"{app_rate:.1%}"})
        st.markdown(f"**{attr}**")
        st.dataframe(pd.DataFrame(app_rows), use_container_width=True, hide_index=True)

with tab5:
    st.subheader("Risk-Based Loan Pricing with Fairness Constraint")
    st.markdown(r"""
    Pricing is calibrated to predicted default probability (PD). A fairness constraint caps
    the interest rate spread between protected and non-protected groups.
    """)
    base_rate = st.slider("Base interest rate (annual)", 0.05, 0.20, 0.10, step=0.01)
    max_spread = st.slider("Max spread between groups", 0.00, 0.05, 0.02, step=0.005)
    pd_values = xgb_y_proba
    interest_rate = base_rate + 0.25 * pd_values
    df_pricing = data["df"].copy()
    df_pricing["PD"] = pd_values
    df_pricing["Interest Rate"] = interest_rate
    df_pricing["Monthly EMI (20yr)"] = (interest_rate / 12 * (1 + interest_rate / 12) ** 240) / ((1 + interest_rate / 12) ** 240 - 1) * 50000
    st.dataframe(df_pricing[["gender", "caste_category", "PD", "Interest Rate", "Monthly EMI (20yr)"]].head(10), use_container_width=True, hide_index=True)
    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots(figsize=(7, 4))
        from src.visualizations import _style, THEME; _style()
        for attr, col_name in [("gender", THEME["cyan"]), ("caste_category", THEME["violet"])]:
            groups_a = sorted(data["df"][attr].unique())
            means = [df_pricing[data["df"][attr] == g]["Interest Rate"].mean() for g in groups_a]
            ax.bar([f"{g}" for g in groups_a], means, alpha=0.6, color=col_name, label=attr)
        ax.set_ylabel("Avg Interest Rate"); ax.set_title("Interest Rate by Group", color=THEME["fg"])
        ax.legend(fontsize=8); ax.grid(True, alpha=0.2, axis="y")
        st.pyplot(fig)
    with col_b:
        st.markdown(f"""
        **Fairness constraint:** Maximum spread = {max_spread:.1%}
        - Gender spread: {abs(df_pricing[data['df']['gender']=='female']['Interest Rate'].mean() - df_pricing[data['df']['gender']=='male']['Interest Rate'].mean()):.2%}
        - Caste spread: {abs(df_pricing[data['df']['caste_category']=='sc']['Interest Rate'].mean() - df_pricing[data['df']['caste_category']=='general']['Interest Rate'].mean()):.2%}
        """)
