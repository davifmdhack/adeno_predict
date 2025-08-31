import streamlit as st


def instructions_box():
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


def table_section(title: str, dataframe):
    st.markdown(title)
    st.dataframe(dataframe, width='stretch')


def results_download_button(output_df):
    csv_out = output_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download results (.csv)",
        data=csv_out,
        file_name="adenopredict_results.csv",
        mime="text/csv",
    )
