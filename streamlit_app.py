import os
import base64
import pandas as pd
import streamlit as st
from adenopredict.inference import load_model, predict_dataframe
import matplotlib.pyplot as plt
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    recall_score,
    roc_curve,
    average_precision_score,
)
 


st.set_page_config(page_title="Adeno Predict", layout="wide")

# Header with logo and title
cols = st.columns([1, 0.02, 6])
with cols[0]:
    try:
        st.image("images/logo_adeno-predict-repositorio.png", width='stretch')
    except Exception:
        st.write("")
with cols[1]:
    # Vertical separator (semi-transparent)
    st.markdown("<div style='height:100%; border-left:1px solid rgba(128,128,128,0.3);'></div>", unsafe_allow_html=True)
with cols[2]:
    st.markdown("""
        <h1 style="margin-bottom: 4px;">Adeno Predict</h1>
    """, unsafe_allow_html=True)
    st.markdown("""
        <p style="margin-top: 0; color: rgba(0,0,0,0.7);">
        Prediction of pituitary macroadenoma consistency based on demographic data and brain MRI parameters
        </p>
    """, unsafe_allow_html=True)

 


def _compute_and_show_metrics(y_true, y_prob, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    with st.expander("📊 Metrics", expanded=True):
        col_roc, col_pr = st.columns(2)

        # ROC (blue)
        with col_roc:
            fig_roc, ax_roc = plt.subplots(figsize=(4, 3))
            ax_roc.plot(fpr, tpr, lw=2, color="#1f77b4", label=f"ROC AUC = {roc_auc:.3f}")
            ax_roc.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.set_title("ROC Curve")
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc, width='stretch')
            plt.close(fig_roc)

        # PR (orange)
        with col_pr:
            fig_pr, ax_pr = plt.subplots(figsize=(4, 3))
            ax_pr.plot(recall, precision, lw=2, color="#ff7f0e", label=f"AP (PR AUC) = {pr_auc:.3f}")
            ax_pr.set_xlabel("Recall")
            ax_pr.set_ylabel("Precision")
            ax_pr.set_title("Precision-Recall Curve")
            ax_pr.legend(loc="lower left")
            st.pyplot(fig_pr, width='stretch')
            plt.close(fig_pr)


@st.cache_resource(show_spinner=False)
def _load_model_cached(path: str):
    return load_model(path)


default_model_path = "examples/best_model_svm.pkl"

tab_onboarding, tab_predict = st.tabs(["📘 Onboarding", "⚙️ Run model"]) 

with tab_onboarding:
    st.subheader("How to use")
    with st.expander("ℹ️ Instructions", expanded=True):
        st.markdown(
            """
            **Instructions**

            1. **Prepare** a CSV with columns: `age, sex, diameter, adc` (optional `consistency`).
            2. **Check** the usage example below.
            3. If `consistency` is provided, it will be used to compute metrics by comparing to model predictions.
            4. Go to the **run model** tab to upload your CSV and get predictions.
            """
        )

    # Example section
    with st.expander("🧪 Example", expanded=True):
        try:
            example_df = pd.read_csv("dataset/dataset_example.csv")
            st.markdown("#### 🗂️ Example dataset (first 10 rows)")
            st.dataframe(example_df.head(10), width='stretch')

            # Load model and run a quick demo output
            model = _load_model_cached(default_model_path)
            example_out = predict_dataframe(model, example_df)

            # Save example results
            os.makedirs("results", exist_ok=True)
            example_out.to_csv("results/df_prediction-results.csv", index=False)

            st.markdown("#### 📈 Adeno Predict - Results")
            st.dataframe(example_out.head(10), width='stretch')
            st.info("Saved results to results/df_prediction-results.csv")

            # Metrics if ground-truth is available
            if "consistency" in example_df.columns:
                gt_map = {"soft": 0, "non-soft": 1}
                y_true = example_df["consistency"].map(gt_map).fillna(example_df["consistency"]).astype(int)
                y_prob = example_out["proba_non_soft"].values
                y_pred = example_out["predicted_label"].values
                _compute_and_show_metrics(y_true, y_prob, y_pred)
        except Exception as e:
            st.info(f"Example preview not available: {e}")

with tab_predict:
    uploaded = st.file_uploader("Upload CSV file", type=["csv"]) 

if uploaded:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    with st.spinner("Loading model..."):
        try:
            model = _load_model_cached(default_model_path)
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.stop()

    st.subheader("CSV preview")
    st.dataframe(df.head(20), width='stretch')

    try:
        output = predict_dataframe(model, df)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    st.subheader("Results")
    st.dataframe(output, width='stretch')

    # Save user results
    try:
        os.makedirs("results", exist_ok=True)
        output.to_csv("results/df_prediction-results.csv", index=False)
        st.info("Saved results to results/df_prediction-results.csv")
    except Exception:
        pass

    # Metrics if ground-truth is available
    if "consistency" in df.columns:
        try:
            gt_map = {"soft": 0, "non-soft": 1}
            y_true = df["consistency"].map(gt_map).fillna(df["consistency"]).astype(int)
            y_prob = output["proba_non_soft"].values
            y_pred = output["predicted_label"].values
            _compute_and_show_metrics(y_true, y_prob, y_pred)
        except Exception as e:
            st.warning(f"Could not compute metrics: {e}")

    # Download
    csv_out = output.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download results (.csv)",
        data=csv_out,
        file_name="adenopredict_results.csv",
        mime="text/csv",
    )

# Author and article (above footer, left-aligned, boxed)
with st.container(border=True):
    st.markdown(
        """
        <div style="text-align:center;">
          <p style="margin: 0 0 6px 0;">
            <strong>🧑‍💻 Created by</strong> Davi Ferreira, MD., MSc.
            <a href="https://orcid.org/0000-0003-1151-9652" target="_blank" style="margin-left:6px;">
              <img src="https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png" alt="ORCID" style="height:16px; vertical-align:middle;"/>
            </a>
          </p>
          <p style="margin: 0 0 6px 0;"><strong>✉️</strong> davi.ferreira.soares@gmail.com</p>
          <p style="margin: 0;"><strong>Article</strong> <a href="https://doi.org/10.1007/s10278-025-01417-6" target="_blank"><img alt="DOI" src="https://img.shields.io/badge/DOI-10.1007%2Fs10278--025--01417--6-blue"/></a></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Footer with institutional logos (centered; data-URI images + vertical bar + links)
st.markdown("---")

def _img_to_data_uri(path: str) -> str:
    try:
        with open(path, "rb") as f:
            data = f.read()
        mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
        b64 = base64.b64encode(data).decode("utf-8")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return ""

unicamp_src = _img_to_data_uri("images/logo_unicamp-institutional.png")
ita_src = _img_to_data_uri("images/logo_ita-institutional.jpg")

footer_html = f"""
<div style=\"display:flex; justify-content:center; align-items:center; gap: 24px;\">
  <div style=\"display:flex; align-items:center; gap:10px;\">
    <a href=\"https://portal.fcm.unicamp.br/\" target=\"_blank\">
      <img src=\"{unicamp_src}\" alt=\"UNICAMP\" style=\"height:40px;\"/>
    </a>
    <a href=\"https://portal.fcm.unicamp.br/\" target=\"_blank\" style=\"text-decoration:none; color:inherit;\">
      School of Medical Sciences - UNICAMP
    </a>
  </div>
  <div style=\"width:1px; height:28px; background: rgba(128,128,128,0.3);\"></div>
  <div style=\"display:flex; align-items:center; gap:10px;\">
    <a href=\"http://www.ita.br/\" target=\"_blank\">
      <img src=\"{ita_src}\" alt=\"ITA\" style=\"height:40px;\"/>
    </a>
    <a href=\"http://www.ita.br/\" target=\"_blank\" style=\"text-decoration:none; color:inherit;\">
      Aeronautics Institute of Technology - ITA
    </a>
  </div>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)
