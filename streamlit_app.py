import streamlit as st
from app import render_header, render_created_by_box, render_footer, hide_header
from app.pages import render_onboarding_tab, render_run_model_tab

# Set page config
st.set_page_config(
    page_title="Adeno Predict",
    page_icon="images/logo_adeno-predict-repositorio.png",
    layout="wide",
)

# Hide Streamlit top header/toolbar (Deploy and overflow menu)
hide_header()

# Render header
render_header()

# Create tabs
tab_onboarding, tab_predict = st.tabs(["📘 Onboarding", "⚙️ Run model"]) 

# Render onboarding and predict tabs
with tab_onboarding:
    render_onboarding_tab()
with tab_predict:
    render_run_model_tab()

# Render final box and footer
render_created_by_box()
render_footer()
