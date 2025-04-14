import streamlit as st
import pickle as pickle
import pandas as pd

def main ():
    st.set_page_config(
        page_title="Breast Cancer Prediction", layout="wide",
        page_icon= ":female-doctor:",
        
        )
    
st.write("Breast Cancer Prediction" )


if __name__ == '__main__':
    main()  