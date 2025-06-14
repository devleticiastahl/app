import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from fpdf import FPDF
import base64
from tempfile import NamedTemporaryFile
import os

# ============ CONFIGURAÇÃO DA PÁGINA ============
st.set_page_config(
    page_title="Gerador de Relatórios",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ CONFIGURAÇÃO DE GRÁFICOS ============
plt.style.use('default')
sns.set_theme(style="whitegrid")
sns.set_palette("viridis")

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

def create_pdf_report(df, logo_path, filename="relatorio.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Adicionar logo se existir
    if logo_path:
        pdf.image(logo_path, x=10, y=8, w=30)
    
    # Título do relatório
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 20, "Relatório de Análise de Dados", ln=1, align='C')
    pdf.ln(10)
    
    # Informações básicas
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, f"Data do relatório: {datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=1)
    pdf.cell(0, 10, f"Total de registros: {len(df)}", ln=1)
    pdf.cell(0, 10, f"Total de colunas: {len(df.columns)}", ln=1)
    pdf.ln(10)
    
    # Amostra dos dados
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Amostra dos Dados", ln=1)
    pdf.set_font("Arial", '', 10)
    
    # Criar tabela com amostra dos dados
    cols = df.columns.tolist()
    rows = df.head().values.tolist()
    
    # Configurar largura das colunas
    col_width = pdf.w / (len(cols) + 1)
    
    # Cabeçalho da tabela
    pdf.set_fill_color(200, 220, 255)
    for col in cols:
        pdf.cell(col_width, 10, str(col)[:15], border=1, fill=True)
    pdf.ln()
    
    # Dados da tabela
    pdf.set_fill_color(255, 255, 255)
    for row in rows:
        for item in row:
            pdf.cell(col_width, 10, str(item)[:15], border=1)
        pdf.ln()
    
    # Salvar o PDF
    pdf.output(filename)
    return filename

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

# ============ PÁGINA INICIAL ============
def show_homepage():
    col1, col2 = st.columns([1, 2])
    with col2:
        st.markdown("""
        ## Bem-vindo ao Gerador de Relatórios!
        **Sistema para criação de relatórios em PDF personalizados**
        
        ✅ Suporta arquivos **CSV** e **Excel** (XLSX)  
        📊 Gera visualizações automáticas  
        📑 Cria relatórios em PDF com sua logo  
        """)
    
    st.markdown("---")
    st.markdown("""
    ### Como usar:
    1. **Carregue seu arquivo de dados** no menu lateral
    2. **Carregue sua logo** (opcional)
    3. **Explore** as visualizações automáticas
    4. **Gere o relatório** em PDF
    """)

# ============ SIDEBAR ============
with st.sidebar:
    st.title("📊 Gerador de Relatórios")
    st.markdown("<hr>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Carregue seu arquivo de dados", 
        type=["csv", "xlsx"],
        help="Formatos suportados: .csv, .xlsx"
    )
    
    logo_file = st.file_uploader(
        "Carregue sua logo (opcional)", 
        type=["png", "jpg", "jpeg"],
        help="Formatos suportados: .png, .jpg, .jpeg"
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

# Salvar logo temporariamente se for carregada
logo_path = None
if logo_file:
    with NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
        tmp_file.write(logo_file.getvalue())
        logo_path = tmp_file.name
    st.success(f"✅ Logo carregada com sucesso!")

# ============ VISÃO GERAL ============
st.header("Visão Geral")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total de Registros", len(df))
with col2:
    st.metric("Total de Colunas", len(df.columns))
with col3:
    st.metric("Valores Faltantes", df.isnull().sum().sum())
with col4:
    st.metric("Tipos de Dados", df.dtypes.nunique())

st.subheader("Amostra dos Dados")
st.dataframe(df.head(), height=250, use_container_width=True)

# ============ ANÁLISE NUMÉRICA ============
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
if numerical_cols:
    st.header("Análise Numérica")
    num_col = st.selectbox("Selecione coluna numérica", numerical_cols)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.histplot(df[num_col], kde=True, ax=ax1, color='royalblue')
    ax1.set_title(f'Distribuição de {num_col}')
    
    sns.boxplot(x=df[num_col], ax=ax2, color='lightgreen')
    ax2.set_title(f'Boxplot de {num_col}')
    
    st.pyplot(fig)

# ============ ANÁLISE CATEGÓRICA ============
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
if categorical_cols:
    st.header("Análise Categórica")
    cat_col = st.selectbox("Selecione coluna categórica", categorical_cols)
    
    top_n = st.slider("Mostrar top N valores", 5, 20, 10)
    counts = df[cat_col].value_counts().nlargest(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=counts.values, y=counts.index, ax=ax, palette='viridis')
    plt.title(f'Top {top_n} Valores em {cat_col}')
    st.pyplot(fig)

# ============ BOTÃO PARA GERAR RELATÓRIO ============
if st.button("Gerar Relatório em PDF"):
    with st.spinner("Criando relatório..."):
        report_path = create_pdf_report(df, logo_path)
        st.success("Relatório gerado com sucesso!")
        
        # Mostrar link para download
        st.markdown(get_binary_file_downloader_html(report_path, 'Relatório PDF'), unsafe_allow_html=True)
        
        # Limpar arquivo temporário da logo
        if logo_path and os.path.exists(logo_path):
            os.unlink(logo_path)
