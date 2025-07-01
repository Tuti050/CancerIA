import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from PIL import Image
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.mainmodel import extrair_medidas_da_imagem
from fpdf import FPDF
import base64
import sklearn
print(sklearn.__version__)
from streamlit.runtime.scriptrunner import RerunException
from streamlit.runtime.scriptrunner import get_script_run_ctx
import streamlit as st


st.set_page_config(
    page_title="MamamIA",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def get_clean_data():
    data = pd.read_csv("data/data.csv")
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data


def add_sidebar():
    st.sidebar.header("🔬 Ajuste das Medidas")
    data = get_clean_data()

    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    input_dict = {}

    with st.sidebar.expander("📊 Médias (Mean)"):
        for label, key in slider_labels:
            if key.endswith("_mean"):
                value = st.session_state.get(key, float(data[key].mean()))
                input_dict[key] = st.slider(
                    label,
                    min_value=0.0,
                    max_value=float(data[key].max()),
                    value=value
                )

    with st.sidebar.expander("📈 Erro Padrão (SE)"):
        for label, key in slider_labels:
            if key.endswith("_se"):
                value = st.session_state.get(key, float(data[key].mean()))
                input_dict[key] = st.slider(
                    label,
                    min_value=0.0,
                    max_value=float(data[key].max()),
                    value=value
                )

    with st.sidebar.expander("⚠️ Piores Casos (Worst)"):
        for label, key in slider_labels:
            if key.endswith("_worst"):
                value = st.session_state.get(key, float(data[key].mean()))
                input_dict[key] = st.slider(
                    label,
                    min_value=0.0,
                    max_value=float(data[key].max()),
                    value=value
                )

    return input_dict


def atualizar_measurements(medidas: dict):
    st.session_state.radius_mean = medidas["radius_mean"]
    st.session_state.texture_mean = medidas["texture_mean"]
    st.session_state.perimeter_mean = medidas["perimeter_mean"]

    for key, value in medidas.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_scaled_values(input_dict):
    data = get_clean_data()
    X = data.drop(['diagnosis'], axis=1)

    scaled_dict = {}

    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value

    return scaled_dict


def get_radar_chart(input_data):
    input_data = get_scaled_values(input_data)

    categories = ['Radius', 'Texture', 'Perimeter', 'Area',
                'Smoothness', 'Compactness',
                'Concavity', 'Concave Points',
                'Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
            input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
            input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
            input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
            input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
            input_data['concave points_se'], input_data['symmetry_se'], input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
            input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
            input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
            input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True
    )

    return fig


def add_predictions(input_data):
    model = joblib.load('model/model.pkl')
    scaler = joblib.load( 'model/scaler.pkl')    
    input_df = pd.DataFrame([input_data])
    input_array_scaled = scaler.transform(input_df)
    prediction = model.predict(input_array_scaled)

    st.subheader("Cell cluster prediction")
    st.write("The cell cluster is:")

    if prediction[0] == 0:
        st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malicious'>Malicious</span>", unsafe_allow_html=True)

    st.write("Probability of being benign: ", model.predict_proba(input_array_scaled)[0][0])
    st.write("Probability of being malicious: ", model.predict_proba(input_array_scaled)[0][1])
    st.write("This app can assist medical professionals in making a diagnosis, "
            "but should not be used as a substitute for a professional diagnosis.")

def gerar_pdf(input_data, predicao, prob_benigno, prob_maligno):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Relatório MamamIA", ln=True, align="C")
    pdf.ln(10)

    pdf.cell(200, 10, txt=f"Predição: {'Benigno' if predicao == 0 else 'Maligno'}", ln=True)
    pdf.cell(200, 10, txt=f"Probabilidade Benigno: {prob_benigno:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Probabilidade Maligno: {prob_maligno:.2f}", ln=True)

    pdf.ln(10)
    pdf.cell(200, 10, txt="Medições:", ln=True)
    for key, val in input_data.items():
        pdf.cell(200, 10, txt=f"{key}: {val:.2f}", ln=True)

    file_path = "relatorio_mamamia.pdf"
    pdf.output(file_path)
    return file_path




def main():
    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    st.title("MamamIA")
    st.write("Please connect this app to your cytology lab to help diagnose breast cancer from your tissue sample. "
            "This app predicts using a machine learning model whether a breast mass is benign or malignant based "
            "on the measurements it receives from your cytosis lab. You can also update the measurements by hand "
            "using the sliders in the sidebar.")

    uploaded_image = st.file_uploader("Send a mamography image", type=["jpg", "png", "jpeg"])
    if uploaded_image:
        st.image(uploaded_image, caption="Mamografia enviada", use_column_width=True)


    if st.button("Use image", key="stFloatingButton"):
        if uploaded_image:
            image = Image.open(uploaded_image)
            with st.spinner("Processando imagem e extraindo medidas..."):
                try:
                    medidas_extraidas = extrair_medidas_da_imagem(image)
                    atualizar_measurements(medidas_extraidas)
                    try:
                        medidas_extraidas = extrair_medidas_da_imagem(image)
                        atualizar_measurements(medidas_extraidas)
                        st.success(f"Medições extraídas: {medidas_extraidas}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erro ao processar a imagem: {e}")
                    st.success(f"Medições extraídas: {medidas_extraidas}")
                except Exception as e:
                    st.error(f"Erro ao processar a imagem: {e}")
        else:
            st.warning("Por favor, envie uma imagem antes de usar esse botão.")


    input_data = add_sidebar()

    col1, col2 = st.columns([4, 1])

    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    with col2:
        add_predictions(input_data)
        model = joblib.load('model/model.pkl')
    scaler = joblib.load('model/scaler.pkl')
    input_df = pd.DataFrame([input_data])
    input_array_scaled = scaler.transform(input_df)
    prediction = model.predict(input_array_scaled)[0]
    prob_benigno = model.predict_proba(input_array_scaled)[0][0]
    prob_maligno = model.predict_proba(input_array_scaled)[0][1]

    if st.button("Gerar relatório PDF"):
        pdf_path = gerar_pdf(input_data, prediction, prob_benigno, prob_maligno)
        with open(pdf_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        href = f'<a href="data:application/octet-stream;base64,{base64_pdf}" download="relatorio_mamamia.pdf">📄 Baixar Relatório PDF</a>'
        st.markdown(href, unsafe_allow_html=True)



if __name__ == '__main__':
    main()