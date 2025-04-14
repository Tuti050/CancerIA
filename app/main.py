import streamlit as st
import pickle as pickle
import pandas as pd
def main ():
    st.set_page_config( 
        page_title="CancerIA",
        layout="wide",
        page_icon= "🧬",
        initial_sidebar_state="expanded"
        
        )
    
    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    
    with st.container():
        st.title("Welcome to CancerIA")
        st.write("Please connect this app to your cytology lab to help diagnose breast cancer form your tissue sample. This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your cytology lab. You can also update the measurements by hand using the sliders in the sidebar")

    
        
    