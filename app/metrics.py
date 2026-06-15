import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)


def compute_and_show_metrics(y_true, y_prob, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    with st.expander("📊 Metrics", expanded=True):
        col_roc, col_pr = st.columns(2)

        # ROC-AUC
        with col_roc:
            fig_roc, ax_roc = plt.subplots(figsize=(4, 3))
            ax_roc.plot(fpr, tpr, lw=2, color="#1f77b4", label=f"ROC AUC = {roc_auc:.3f}")
            ax_roc.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.set_title("ROC Curve")
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc, width="stretch")
            plt.close(fig_roc)

        # PR-AUC
        with col_pr:
            fig_pr, ax_pr = plt.subplots(figsize=(4, 3))
            ax_pr.plot(
                recall, precision, lw=2, color="#ff7f0e", label=f"AP (PR AUC) = {pr_auc:.3f}"
            )
            ax_pr.plot([0, 1], [1, 0], color="gray", lw=1, linestyle="--")
            ax_pr.set_xlabel("Recall")
            ax_pr.set_ylabel("Precision")
            ax_pr.set_title("Precision-Recall Curve")
            ax_pr.legend(loc="lower left")
            st.pyplot(fig_pr, width="stretch")
            plt.close(fig_pr)
