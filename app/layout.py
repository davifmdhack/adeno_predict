import streamlit as st
from .utils import image_to_data_uri


def render_header():
    try:
        logo_src = image_to_data_uri("images/logo_adeno-predict-repositorio.png")
    except Exception:
        logo_src = ""
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap: 12px;">
            <img src="{logo_src}" alt="Adeno Predict" style="height:110px;"/>
            <div style="width:1px; height:100px; background: rgba(128,128,128,0.35);"></div>
            <div style="display:flex; flex-direction:column; justify-content:center; gap:0;">
                <h1 style="margin:0; line-height:1.05;">Adeno Predict</h1>
                <p style="margin:0; line-height:1.1; color: rgba(0,0,0,0.7);">Prediction of pituitary macroadenoma consistency based on demographic data and brain MRI parameters</p>
            </div>
        </div>
        </br>
        """,
        unsafe_allow_html=True,
    )


def render_created_by_box():
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
            </br>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_footer():
    unicamp_src = image_to_data_uri("images/logo_unicamp-institutional.png")
    ita_src = image_to_data_uri("images/logo_ita-institutional.jpg")
    footer_html = f"""
    <div style=\"display:flex; justify-content:center; align-items:center; gap: 24px;\">
      <div style=\"display:flex; align-items:center; gap:10px;\">
        <a href=\"https://portal.fcm.unicamp.br/\" target=\"_blank\">
          <img src=\"{unicamp_src}\" alt=\"UNICAMP\" style=\"height:40px;\"/>
        </a>
        <a href=\"https://portal.fcm.unicamp.br/\" target=\"_blank\" style=\"text-decoration:none; color:inherit;\">School of Medical Sciences - UNICAMP</a>
      </div>
      <div style=\"width:1px; height:28px; background: rgba(128,128,128,0.3);\"></div>
      <div style=\"display:flex; align-items:center; gap:10px;\">
        <a href=\"http://www.ita.br/\" target=\"_blank\">
          <img src=\"{ita_src}\" alt=\"ITA\" style=\"height:40px;\"/>
        </a>
        <a href=\"http://www.ita.br/\" target=\"_blank\" style=\"text-decoration:none; color:inherit;\">Aeronautics Institute of Technology - ITA</a>
      </div>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)


