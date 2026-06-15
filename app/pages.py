import pandas as pd
import streamlit as st

from .components import instructions_box, results_download_button, table_section
from .config import DEFAULT_MODEL_PATH, EXAMPLE_CSV_PATH, RESULTS_PATH
from .data import load_example_dataframe, save_results
from .metrics import compute_and_show_metrics
from .model import load_model_cached
from .service import PredictionResult, run_prediction


def _save_predictions(predictions: pd.DataFrame) -> None:
    """Persist predictions to disk, surfacing failures instead of hiding them."""
    try:
        save_results(predictions, RESULTS_PATH)
        st.info(f"Saved results to {RESULTS_PATH}")
    except OSError as error:
        st.warning(f"Could not save results to {RESULTS_PATH}: {error}")


def _show_metrics_if_available(result: PredictionResult) -> None:
    """Render evaluation metrics when the input carried a ground-truth column."""
    if not result.has_ground_truth:
        return
    try:
        compute_and_show_metrics(
            result.ground_truth,
            result.predictions["proba_non_soft"].to_numpy(),
            result.predictions["predicted_label"].to_numpy(),
        )
    except (ValueError, TypeError) as error:
        st.warning(f"Could not compute metrics: {error}")


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
            result = run_prediction(model, example_df)

            _save_predictions(result.predictions)
            table_section("#### 📈 Adeno Predict - Results", result.predictions.head(10))
            _show_metrics_if_available(result)
        except (FileNotFoundError, ValueError) as error:
            st.info(f"Example preview not available: {error}")


def render_run_model_tab():
    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    if not uploaded:
        return

    try:
        df = pd.read_csv(uploaded)
    except (ValueError, pd.errors.ParserError) as error:
        st.error(f"Failed to read CSV: {error}")
        st.stop()

    with st.spinner("Loading model..."):
        try:
            model = load_model_cached(DEFAULT_MODEL_PATH)
        except (FileNotFoundError, OSError) as error:
            st.error(f"Failed to load model: {error}")
            st.stop()

    st.subheader("CSV preview")
    table_section("", df.head(20))

    try:
        result = run_prediction(model, df)
    except ValueError as error:
        st.error(f"Prediction error: {error}")
        st.stop()

    st.subheader("Results")
    table_section("", result.predictions)

    _save_predictions(result.predictions)
    _show_metrics_if_available(result)
    results_download_button(result.predictions)


def render_individual_patient_tab():
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

    if not submitted:
        return

    input_df = pd.DataFrame({"age": [age], "sex": [sex], "diameter": [diameter], "adc": [adc]})
    try:
        model = load_model_cached(DEFAULT_MODEL_PATH)
        prediction = run_prediction(model, input_df).predictions
    except (FileNotFoundError, OSError, ValueError) as error:
        st.error(f"Prediction error: {error}")
        return

    st.success(f"Probability of non-soft consistency: {prediction['proba_non_soft'].iloc[0]:.2%}")
    st.info(f"Predicted consistency: {prediction['predicted_consistency'].iloc[0]}")
