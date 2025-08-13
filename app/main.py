# Bibliotecas padrão
import os
import sys
import base64
import tempfile
import sqlite3
from io import BytesIO
from datetime import datetime
import getpass

# Bibliotecas de terceiros
import streamlit as st
from streamlit.runtime.scriptrunner import RerunException, get_script_run_ctx
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from PIL import Image  # Usando o nome padrão
from PIL import Image as PILImage  # Se você realmente precisar de ambos, pode manter assim
import sklearn
from fpdf import FPDF
from reportlab.lib.pagesizes import letter, A4
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet

# Módulo local
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.mainmodel import extrair_medidas_da_imagem

# Exibir a versão do sklearn
print(sklearn.__version__)




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
    st.sidebar.header("Measurements adjustments")
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

    with st.sidebar.expander("Mean Values"):
        for label, key in slider_labels:
            if key.endswith("mean"):
                value = st.session_state.get(key, float(data[key].mean()))
                input_dict[key] = st.slider(label, 0.0, float(data[key].max()), value)

    with st.sidebar.expander("Standard Error"):
        for label, key in slider_labels:
            if key.endswith("se"):
                value = st.session_state.get(key, float(data[key].mean()))
                input_dict[key] = st.slider(label, 0.0, float(data[key].max()), value)

    with st.sidebar.expander("Worst Cases"):
        for label, key in slider_labels:
            if key.endswith("worst"):
                value = st.session_state.get(key, float(data[key].mean()))
                input_dict[key] = st.slider(label, 0.0, float(data[key].max()), value)

        
    return input_dict


def atualizar_measurements(medidas: dict):
    for key, value in medidas.items():
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
    
    categories = ['Radius', 'Texture', 'Perimeter', 'Area',
                'Smoothness', 'Compactness',
                'Concavity', 'Concave Points',
                'Symmetry', 'Fractal Dimension']

    
    mean_vals = [
        input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
        input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
        input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
        input_data['fractal_dimension_mean']
    ]
    se_vals = [
        input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
        input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
        input_data['concave points_se'], input_data['symmetry_se'], input_data['fractal_dimension_se']
    ]
    worst_vals = [
        input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
        input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
        input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
        input_data['fractal_dimension_worst']
    ]

    
    def normalize_list(lst):
        max_val = max(lst)
        min_val = min(lst)
        return [(v - min_val) / (max_val - min_val) if max_val != min_val else 0 for v in lst]

    mean_scaled = normalize_list(mean_vals)
    se_scaled = normalize_list(se_vals)
    worst_scaled = normalize_list(worst_vals)

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=mean_scaled,
        theta=categories,
        fill='toself',
        name='Mean Value',
        hovertemplate = [f"{cat}: {val:.4f}" for cat, val in zip(categories, mean_vals)] + ["<extra></extra>"]
    ))

    fig.add_trace(go.Scatterpolar(
        r=se_scaled,
        theta=categories,
        fill='toself',
        name='Standard Error',
        hovertemplate = [f"{cat}: {val:.4f}" for cat, val in zip(categories, se_vals)] + ["<extra></extra>"]
    ))

    fig.add_trace(go.Scatterpolar(
        r=worst_scaled,
        theta=categories,
        fill='toself',
        name='Worst Value',
        hovertemplate = [f"{cat}: {val:.4f}" for cat, val in zip(categories, worst_vals)] + ["<extra></extra>"]
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
        st.write("<span class='diagnosis malignant'>Malignant</span>", unsafe_allow_html=True)

    st.write("Probability of being benign: ", model.predict_proba(input_array_scaled)[0][0])
    st.write("Probability of being malignant: ", model.predict_proba(input_array_scaled)[0][1])
    st.write("This app can assist medical professionals in making a diagnosis, "
            "but should not be used as a substitute for a professional diagnosis.")

# ========== Banco de dados ==========
def conectar_banco():
    conn = sqlite3.connect("banco.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS relatorios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome_paciente TEXT,
            idade INTEGER,
            tipo_sanguineo TEXT,
            nome_medico TEXT,
            resultado TEXT,
            diagnostico TEXT,
            data_criacao TEXT,
            nome_arquivo TEXT
        )
    """)
    conn.commit()
    return conn

def salvar_em_banco(dados, nome_pdf):
    conn = conectar_banco()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO relatorios (
            nome_paciente, idade, tipo_sanguineo, nome_medico, resultado,
            diagnostico, data_criacao, nome_arquivo
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        dados.get('nome_paciente', ''),
        dados.get('idade', None),
        dados.get('tipo_sanguineo', ''),
        dados.get('nome_medico', ''),
        dados.get('resultado', ''),
        dados.get('diagnostico', ''),
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        nome_pdf
    ))
    conn.commit()
    conn.close()

def buscar_relatorios(nome):
    # Conectar ao banco de dados
    conn = conectar_banco()
    cursor = conn.cursor()
    
    # Realizar a busca no banco
    cursor.execute("SELECT nome_paciente, nome_arquivo, data_criacao FROM relatorios WHERE nome_paciente LIKE ?", (f"%{nome}%",))
    resultados = cursor.fetchall()
    conn.close()

    return resultados

def verificar_senha():
    # Aqui você pode usar uma senha fixa ou verificar contra um banco de dados de usuários
    senha_correta = "senha123"  # Defina aqui a senha para acesso

    # Solicitar a senha ao usuário
    senha = getpass.getpass("Digite a senha para acessar o relatório: ")

    # Verificar se a senha está correta
    if senha == senha_correta:
        return True
    else:
        print("Senha incorreta. Acesso negado.")
        return False
    
def acessar_relatorio(nome_paciente):
    # Buscar relatórios do paciente
    relatorios = buscar_relatorios(nome_paciente)

    if not relatorios:
        print("Nenhum relatório encontrado para o paciente.")
        return

    print(f"Relatórios encontrados para {nome_paciente}:")
    for index, relatorio in enumerate(relatorios, 1):
        nome_paciente_db, nome_arquivo, data_criacao = relatorio
        print(f"{index}. Nome do paciente: {nome_paciente_db}, Arquivo: {nome_arquivo}, Data: {data_criacao}")

    # Solicitar que o usuário escolha um relatório para acessar
    escolha = int(input("Escolha o número do relatório para acessar: ")) - 1

    if 0 <= escolha < len(relatorios):
        relatorio = relatorios[escolha]
        nome_arquivo = relatorio[1]

        # Verificar senha para acessar o PDF
        if verificar_senha():
            # Lógica para o download ou visualização do PDF
            print(f"Você tem acesso ao relatório: {nome_arquivo}")
            # Aqui você pode implementar o código para servir o PDF
            # Exemplo: download_pdf(nome_arquivo)
        else:
            print("Acesso negado ao relatório.")
    else:
        print("Opção inválida.")


def gerar_pdf(dados):
    """Gera PDF estilo relatório, salva no diretório pdfs e grava no DB."""
    nome_pdf = f"{dados.get('nome_paciente','Paciente') .replace(' ', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
    caminho_pdf = os.path.join("pdfs", nome_pdf)

    doc = SimpleDocTemplate(caminho_pdf, pagesize=A4)
    estilos = getSampleStyleSheet()
    conteudo = []

    conteudo.append(Paragraph("Relatório da Mamografia", estilos['Title']))
    conteudo.append(Spacer(1, 12))

    conteudo.append(Paragraph(f"<b>Nome da paciente:</b> {dados.get('nome_paciente','')}", estilos['BodyText']))
    conteudo.append(Paragraph(f"<b>Idade:</b> {dados.get('idade','')}", estilos['BodyText']))
    conteudo.append(Paragraph(f"<b>Tipo sanguíneo:</b> {dados.get('tipo_sanguineo','')}", estilos['BodyText']))
    conteudo.append(Paragraph(f"<b>Data de geração:</b> {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", estilos['BodyText']))
    conteudo.append(Paragraph(f"<b>Médico responsável:</b> Dr(a). {dados.get('nome_medico','')}", estilos['BodyText']))
    conteudo.append(Paragraph(f"<b>Resultado:</b> {dados.get('resultado','')}", estilos['BodyText']))
    conteudo.append(Paragraph(f"<b>Diagnóstico:</b> {dados.get('diagnostico','')}", estilos['BodyText']))
    conteudo.append(Spacer(1, 12))

    # Medidas
    conteudo.append(Paragraph("<b>Medições extraídas:</b>", estilos['Heading3']))
    medidas = dados.get('medidas', {})
    for key, val in medidas.items():
        try:
            conteudo.append(Paragraph(f"{key}: {float(val):.4f}", estilos['BodyText']))
        except:
            conteudo.append(Paragraph(f"{key}: {val}", estilos['BodyText']))
    conteudo.append(Spacer(1, 12))

    # Imagens: criar temp files para ReportLab
    imagens = dados.get('imagens', [])
    for idx, img_path in enumerate(imagens):
        try:
            pil_img = PILImage.open(img_path)
            pil_img.thumbnail((800, 800))
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            pil_img.save(tmp.name, format='JPEG')
            tmp.close()
            conteudo.append(Image(tmp.name, width=300, height=300))
            conteudo.append(Spacer(1, 12))
        except Exception as e:
            conteudo.append(Paragraph(f"Erro ao inserir imagem {img_path}: {e}", estilos['BodyText']))

    doc.build(conteudo)

    # salvar no DB
    salvar_em_banco(dados, nome_pdf)
    return caminho_pdf

# ========== Funções de dados e visualização ==========
def get_clean_data():
    try:
        data = pd.read_csv("data/data.csv")
        data = data.drop(['Unnamed: 32', 'id'], axis=1)
        data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
        return data
    except Exception as e:
        st.warning("Não foi possível carregar data/data.csv. Algumas funcionalidades (sliders automáticos) podem falhar.")
        raise

def carregar_modelo():
    model = joblib.load('model/model.pkl')
    scaler = joblib.load('model/scaler.pkl')
    return model, scaler

def add_predictions_display(input_data):
    try:
        model, scaler = carregar_modelo()
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
        return None, None, None

    input_df = pd.DataFrame([input_data])
    try:
        input_array_scaled = scaler.transform(input_df)
    except Exception as e:
        st.error(f"Erro ao aplicar scaler: {e}")
        return None, None, None

    prediction = model.predict(input_array_scaled)[0]
    prob = model.predict_proba(input_array_scaled)[0]
    prob_benigno = prob[0]
    prob_maligno = prob[1]

    st.subheader("Predição")
    st.write("Cluster celular previsto:")
    if prediction == 0:
        st.markdown("<span style='color:green;font-weight:bold'>Benigno</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span style='color:red;font-weight:bold'>Maligno</span>", unsafe_allow_html=True)

    st.write(f"Probabilidade Benigno: {prob_benigno:.4f}")
    st.write(f"Probabilidade Maligno: {prob_maligno:.4f}")

    st.info("Este sistema é suporte — não substitui um laudo médico profissional.")
    return prediction, prob_benigno, prob_maligno

def main():
    # Carregar estilo CSS
    if os.path.exists("assets/style.css"):
        with open("assets/style.css") as f:
            st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    st.title("🧬 MamamIA - Diagnóstico e Relatórios")

    # Menu lateral
    aba = st.sidebar.radio("Navegação", ["Novo exame", "Buscar relatórios"])

    # ----------------- ABA BUSCAR RELATÓRIOS -----------------
    if aba == "Buscar relatórios":
        st.header("Buscar relatórios por paciente")
        nome_busca = st.text_input("Nome da paciente para buscar")
        if st.button("Buscar"):
            resultados = buscar_relatorios(nome_busca)
            if resultados:
                for r in resultados:
                    st.write(f"Paciente: {r[0]} | Data: {r[2]}")
                    pdf_file = os.path.join("pdfs", r[1])
                    if os.path.exists(pdf_file):
                        with open(pdf_file, "rb") as f:
                            st.download_button(label=f"📄 Baixar {r[1]}", data=f, file_name=r[1], mime="application/pdf")
                    else:
                        st.write("Arquivo PDF não encontrado no diretório `pdfs/`.")
            else:
                st.warning("Nenhum relatório encontrado.")
        return

    # ----------------- ABA NOVO EXAME -----------------
    st.write("Envie uma imagem ou ajuste manualmente as medidas na barra lateral.")
    col_left, col_right = st.columns([2, 1])

    # Sliders na sidebar
    try:
        input_data = add_sidebar()
    except Exception:
        st.error("Erro ao construir sliders: verifique data/data.csv")
        input_data = {}

    # ----------------- COLUNA ESQUERDA -----------------
    with col_left:
        st.subheader("Imagem")

        uploaded_image = st.file_uploader("Envie a mamografia (jpg/png/jpeg)", type=["jpg", "png", "jpeg"])

        if uploaded_image and st.button("Visualizar imagem enviada"):
            st.image(uploaded_image, caption="Imagem carregada", use_column_width=True)

        if st.button("Usar imagem para extrair medidas"):
            if not uploaded_image:
                st.warning("Primeiro envie uma imagem.")
            else:
                img_path = os.path.join("imagens", uploaded_image.name)
                with open(img_path, "wb") as f:
                    f.write(uploaded_image.getbuffer())
                try:
                    pil_img = Image.open(img_path)
                    medidas_extraidas = extrair_medidas_da_imagem(pil_img)
                    atualizar_measurements(medidas_extraidas)
                    st.success("Medições extraídas e atualizadas nos sliders.")
                    input_data = medidas_extraidas
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro ao extrair medidas da imagem: {e}")

        # Radar chart
        if input_data:
            try:
                fig = get_radar_chart(input_data)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Erro ao gerar radar chart: {e}")

    # ----------------- COLUNA DIREITA -----------------
    with col_right:
        st.subheader("Paciente & Relatório")
        nome_paciente = st.text_input("Nome da paciente", value=st.session_state.get("nome_paciente", ""))
        idade = st.number_input("Idade", min_value=0, max_value=120, value=int(st.session_state.get("idade", 0) or 0))
        tipo_sanguineo = st.text_input("Tipo sanguíneo", value=st.session_state.get("tipo_sanguineo", ""))
        nome_medico = st.text_input("Nome do médico", value=st.session_state.get("nome_medico", ""))

        st.markdown("---")
        st.write("Predição com o modelo (usando as medidas atuais)")
        pred, pb, pm = add_predictions_display(input_data) if input_data else (None, None, None)

        # Botão gerar PDF
        if st.button("Gerar relatório PDF"):
            if not input_data:
                st.warning("Não há medidas para gerar o relatório.")
            else:
                # criar dict de dados para o PDF
                imagens_list = []
                # se houve upload, priorizar a imagem salva; tentar achar última imagem na pasta 'imagens' com nome do upload
                if uploaded_image:
                    img_path = os.path.join("imagens", uploaded_image.name)
                    if os.path.exists(img_path):
                        imagens_list.append(img_path)

                dados_pdf = {
                    "nome_paciente": nome_paciente or "Paciente_sem_nome",
                    "idade": int(idade) if idade else None,
                    "tipo_sanguineo": tipo_sanguineo,
                    "nome_medico": nome_medico,
                    "resultado": "Benigno" if pred == 0 else ("Maligno" if pred == 1 else "Desconhecido"),
                    "diagnostico": "Maligno" if pred == 1 else ("Benigno" if pred == 0 else "Indeterminado"),
                    "medidas": input_data,
                    "imagens": imagens_list
                }

                try:
                    pdf_path = gerar_pdf(dados_pdf)
                    st.success(f"PDF gerado: {pdf_path}")
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            label="📄 Baixar PDF",
                            data=f,
                            file_name=os.path.basename(pdf_path),
                            mime="application/pdf"
                        )
                except Exception as e:
                    st.error(f"Erro ao gerar PDF: {e}")

if __name__ == '__main__':
    main()
    #usar no terminal "python -m streamlit run app/main.py" para inicializar o codigo