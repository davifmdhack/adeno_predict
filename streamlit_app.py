import streamlit as st

from app import hide_header, render_created_by_box, render_footer, render_header
from app.pages import (
    render_individual_patient_tab,
    render_onboarding_tab,
    render_run_model_tab,
)

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
tab_onboarding, tab_predict, tab_individual = st.tabs(
    ["📘 Onboarding", "⚙️Run Model in Dataset", "🙋🏻‍♀️ Individual Patient Analysis"]
)

# Render onboarding, predict, and individual analysis tabs
with tab_onboarding:
    render_onboarding_tab()
with tab_predict:
    render_run_model_tab()
with tab_individual:
    render_individual_patient_tab()

# Render final box and footer
render_created_by_box()
render_footer()
