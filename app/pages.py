import os
import streamlit as st
from adenopredict.inference import predict_dataframe
from .components import instructions_box, table_section, results_download_button
from .model import load_model_cached
from .data import load_example_dataframe, save_results
from .metrics import compute_and_show_metrics
from .config import DEFAULT_MODEL_PATH, EXAMPLE_CSV_PATH, RESULTS_PATH


def render_onboarding_tab():
    st.subheader("How to use")
    instructions_box()

    with st.expander("🧪 Example", expanded=True):
        try:
            example_df = load_example_dataframe(EXAMPLE_CSV_PATH)
            table_section("#### 🗂️ Example dataset (first 10 rows)", example_df.head(10))

            model = load_model_cached(DEFAULT_MODEL_PATH)
            example_out = predict_dataframe(model, example_df)

            save_results(example_out, RESULTS_PATH)
            table_section("#### 📈 Adeno Predict - Results", example_out.head(10))
            st.info(f"Saved results to {RESULTS_PATH}")

            if "consistency" in example_df.columns:
                gt_map = {"soft": 0, "non-soft": 1}
                y_true = example_df["consistency"].map(gt_map).fillna(example_df["consistency"]).astype(int)
                y_prob = example_out["proba_non_soft"].values
                y_pred = example_out["predicted_label"].values
                compute_and_show_metrics(y_true, y_prob, y_pred)
        except Exception as e:
            st.info(f"Example preview not available: {e}")


def render_run_model_tab():
    uploaded = st.file_uploader("Upload CSV file", type=["csv"]) 
    if not uploaded:
        return

    try:
        import pandas as pd
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    with st.spinner("Loading model..."):
        try:
            model = load_model_cached(DEFAULT_MODEL_PATH)
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.stop()

    st.subheader("CSV preview")
    table_section("", df.head(20))

    try:
        output = predict_dataframe(model, df)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    st.subheader("Results")
    table_section("", output)

    try:
        save_results(output, RESULTS_PATH)
        st.info(f"Saved results to {RESULTS_PATH}")
    except Exception:
        pass

    if "consistency" in df.columns:
        try:
            gt_map = {"soft": 0, "non-soft": 1}
            y_true = df["consistency"].map(gt_map).fillna(df["consistency"]).astype(int)
            y_prob = output["proba_non_soft"].values
            y_pred = output["predicted_label"].values
            compute_and_show_metrics(y_true, y_prob, y_pred)
        except Exception as e:
            st.warning(f"Could not compute metrics: {e}")

    results_download_button(output)
