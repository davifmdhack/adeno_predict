import pandas as pd
import streamlit as st

from adenopredict.inference import predict_dataframe

from .components import instructions_box, results_download_button, table_section
from .config import DEFAULT_MODEL_PATH, EXAMPLE_CSV_PATH, RESULTS_PATH
from .data import load_example_dataframe, save_results
from .metrics import compute_and_show_metrics
from .model import load_model_cached


def render_onboarding_tab():
    st.subheader("Welcome to Adeno Predict!")
    instructions_box()

    with st.expander("🧪 Example Workflow", expanded=True):
        st.markdown(
            """
            **Step-by-step Example:**

            1. **Dataset Prediction:**
                - Download the example CSV from the [examples/df_example.csv](../examples/df_example.csv) file.
                - Go to the **⚙️Run Model in Dataset** tab and upload the file.
                - View the predictions and download the results.

            2. **Individual Prediction:**
                - Go to the **🙋🏻‍♀️ Individual Patient Analysis** tab.
                - Enter the data for a single patient and click **Predict**.
                - Instantly receive the probability and predicted class.

            3. **Metrics:**
                - If your data includes the `consistency` column, the app will show performance metrics (ROC AUC, PR AUC).

            ---
            """
        )
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
                y_true = (
                    example_df["consistency"]
                    .map(gt_map)
                    .fillna(example_df["consistency"])
                    .astype(int)
                )
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


def render_individual_patient_tab():
    import streamlit as st

    from adenopredict.inference import predict_dataframe

    from .config import DEFAULT_MODEL_PATH
    from .model import load_model_cached

    st.subheader("Individual Patient Analysis")
    st.markdown("Enter patient data below to predict tumor consistency.")

    with st.form("individual_form"):
        age = st.number_input("Age (years)", min_value=18, max_value=120, value=40)
        sex = st.selectbox("Sex", options=["F", "M"])
        diameter = st.number_input(
            "Tumor diameter (cm)",
            min_value=0.1,
            max_value=10.0,
            value=2.0,
            format="%.2f",
        )
        adc = st.number_input(
            "ADC (10⁻³ mm²/s)", min_value=0.1, max_value=10.0, value=2.0, format="%.3f"
        )
        submitted = st.form_submit_button("Predict")

    if submitted:
        # Prepare single-row DataFrame
        input_df = pd.DataFrame(
            {
                "age": [age],
                "sex": [sex],
                "diameter": [diameter],
                "adc": [adc],
            }
        )
        try:
            model = load_model_cached(DEFAULT_MODEL_PATH)
            result = predict_dataframe(model, input_df)
            st.success(
                f"Probability of non-soft consistency: {result['proba_non_soft'].iloc[0]:.2%}"
            )
            st.info(f"Predicted consistency: {result['predicted_consistency'].iloc[0]}")
        except Exception as e:
            st.error(f"Prediction error: {e}")
