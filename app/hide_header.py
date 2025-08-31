import streamlit as st

def hide_header():
    st.markdown(
        """
        <style>
        [data-testid="stHeader"] { display: none; }
        [data-testid="stToolbar"] { display: none; }
        .stDeployButton { display: none; }
        #MainMenu { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
)
