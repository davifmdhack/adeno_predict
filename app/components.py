import streamlit as st


def instructions_box():
    with st.expander("ℹ️ Instructions", expanded=True):
        st.markdown(
            """
            **How to Use Adeno Predict**

            1. **Dataset Prediction:**
                - Go to the **⚙️Run Model in Dataset** tab.
                - Upload a CSV file with the following columns: `age`, `sex` (F/M), `diameter` (cm), `adc` (10⁻³ mm²/s). Optionally, include `consistency` for metric evaluation.
                - The app will display predictions for all patients in your dataset, including probability and predicted class. If `consistency` is present, performance metrics will be shown.
                - Download the results as a **CSV file** (.csv).

            2. **Individual Patient Analysis:**
                - Go to the **🙋🏻‍♀️ Individual Patient Analysis** tab.
                - Enter the patient's data manually in the form.
                - Click **Predict** to receive the probability and predicted tumor consistency for that individual.

            3. **Onboarding & Example:**
                - The **📘 Onboarding** tab provides a step-by-step example and further details about the platform.

            **Note:**
            - The same machine learning model is used for both dataset and individual predictions, ensuring consistent results.
            - For more information, see the README or the onboarding example below.
            """
        )


def table_section(title: str, dataframe):
    st.markdown(title)
    st.dataframe(dataframe, width="stretch")


def results_download_button(output_df):
    csv_out = output_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download results (.csv)",
        data=csv_out,
        file_name="adenopredict_results.csv",
        mime="text/csv",
    )
