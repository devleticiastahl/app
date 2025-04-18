import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from io import BytesIO

# ============ CONFIGURAÇÃO DA PÁGINA ============
st.set_page_config(
    page_title="Analytics Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ FUNÇÕES AUXILIARES ============
@st.cache_data
def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            return pd.read_excel(uploaded_file, engine='openpyxl')
    except Exception as e:
        st.error(f"Erro ao ler arquivo: {str(e)}")
        return None

def is_datetime_column(series):
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    
    col_name = series.name.lower()
    date_keywords = ['date', 'time', 'hora', 'dia', 'ano', 'mes']
    if any(keyword in col_name for keyword in date_keywords):
        try:
            pd.to_datetime(series)
            return True
        except:
            return False
    return False

# ============ PÁGINA INICIAL ============
def show_homepage():
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://i.imgur.com/7kMk3Zz.png", width=300)
    with col2:
        st.markdown("""
        ## Bem-vindo ao Analytics Pro!
        **Sistema completo para análise exploratória de dados**
        
        ✅ Suporta arquivos **CSV** e **Excel** (XLSX)  
        📊 Gera visualizações automáticas  
        🔍 Fornece estatísticas descritivas completas  
        """)
    
    st.markdown("---")
    st.markdown("""
    ### Como usar:
    1. **Carregue seu arquivo** no menu lateral ➡️
    2. **Explore** as visualizações automáticas
    3. **Analise** os insights gerados
    """)

# ============ SIDEBAR ============
with st.sidebar:
    st.title("📊 Analytics Pro")
    st.markdown("<hr>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Carregue seu arquivo de dados", 
        type=["csv", "xlsx"],
        help="Formatos suportados: .csv, .xlsx"
    )

# ============ PÁGINA PRINCIPAL ============
if not uploaded_file:
    show_homepage()
    st.stop()

# Carregamento de dados
df = load_data(uploaded_file)
if df is None:
    st.error("❌ Não foi possível carregar o arquivo. Verifique o formato e tente novamente.")
    st.stop()

st.session_state['df'] = df
st.success(f"✅ Arquivo '{uploaded_file.name}' carregado com sucesso!")

# ... (restante do código mantido igual) ...
