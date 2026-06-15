"""Apply the AdenoPredict model to a local dataset and write a PDF report.

This is a thin command-line wrapper around the :mod:`adenopredict` library: it
reuses :func:`adenopredict.load_model` and :func:`adenopredict.predict_dataframe`
instead of re-implementing preprocessing, then renders a one-page PDF with a
metrics table, confusion matrix, ROC curve and precision-recall curve.

Usage
-----
    python examples/model_apply.py \
        --input examples/df_example.csv \
        --model examples/best_model_svm.pkl \
        --output report_metrics_and_plot.pdf

Requires the project to be installed (``pip install -e .``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    recall_score,
    roc_curve,
)

from adenopredict import load_model, map_target, predict_dataframe
from adenopredict.constants import TARGET_COLUMN

_DEFAULT_DIR = Path(__file__).resolve().parent


def specificity_sensitivity(conf_matrix) -> tuple[float, float]:
    """Return ``(specificity, sensitivity)`` from a 2x2 confusion matrix."""
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    return specificity, sensitivity


def build_report(y_true, y_prob, y_pred) -> plt.Figure:
    """Build the one-page metrics-and-plots report figure."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    conf_matrix = confusion_matrix(y_true, y_pred)
    specificity, _ = specificity_sensitivity(conf_matrix)

    metrics_df = pd.DataFrame(
        {
            "ROC AUC": [roc_auc],
            "PR AUC": [pr_auc],
            "Sensitivity": [recall_score(y_true, y_pred)],
            "Specificity": [specificity],
            "F1 Score": [f1_score(y_true, y_pred)],
            "MCC": [matthews_corrcoef(y_true, y_pred)],
        }
    )

    fig = plt.figure(figsize=(15, 15))
    fig.suptitle(
        "REPORT - AdenoPredict model on local dataset (CSV)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    plt.figtext(0.5, 0.89, "Created by: Davi Ferreira, MD., MSc.", fontsize=12, ha="center")

    ax_table = fig.add_subplot(4, 1, 1)
    ax_table.axis("off")
    ax_table.text(0.5, 0.8, "Table: Metrics Summary", fontsize=14, ha="center", fontweight="bold")
    table = ax_table.table(
        cellText=metrics_df.round(3).values,
        colLabels=metrics_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.scale(1, 3)
    ax_table.text(
        0.5,
        0.1,
        "Confusion Matrix, ROC Curve and PR-Curve",
        fontsize=14,
        ha="center",
        fontweight="bold",
    )

    ax_cm = fig.add_subplot(4, 3, 5)
    ConfusionMatrixDisplay(confusion_matrix=conf_matrix).plot(ax=ax_cm, cmap=plt.cm.Blues)
    ax_cm.set_title("Confusion Matrix\n", fontsize=12)

    ax_roc = fig.add_subplot(2, 2, 3)
    ax_roc.plot(fpr, tpr, lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    ax_roc.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve", fontsize=12)
    ax_roc.legend(loc="lower right")

    ax_pr = fig.add_subplot(2, 2, 4)
    ax_pr.plot(recall, precision, lw=2, label=f"AP (AUC = {pr_auc:.2f})")
    ax_pr.plot([1, 0], [0, 1], color="gray", lw=1, linestyle="--")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curve", fontsize=12)
    ax_pr.legend(loc="lower left")

    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=_DEFAULT_DIR / "df_example.csv",
        help="Input CSV with age, sex, diameter, adc and consistency columns.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=_DEFAULT_DIR / "best_model_svm.pkl",
        help="Path to the serialized model (.pkl).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("report_metrics_and_plot.pdf"),
        help="Path for the generated PDF report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input)
    if TARGET_COLUMN not in df.columns:
        raise SystemExit(f"Input must contain the '{TARGET_COLUMN}' column to evaluate metrics.")

    model = load_model(str(args.model))
    predictions = predict_dataframe(model, df)
    y_true = map_target(df[TARGET_COLUMN]).astype(int)

    fig = build_report(
        y_true,
        predictions["proba_non_soft"].to_numpy(),
        predictions["predicted_label"].to_numpy(),
    )
    with PdfPages(args.output) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()
