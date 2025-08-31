import streamlit as st
from adenopredict.inference import load_model

@st.cache_resource(show_spinner=False)
def load_model_cached(path: str):
    return load_model(path)
