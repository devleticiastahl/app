import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# ============ CONFIGURAÇÃO DA PÁGINA (DEVE SER O PRIMEIRO COMANDO) ============
st.set_page_config(
    page_title="Analytics Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ TEMA DARK COMPLETO ============
DARK_THEME = """
<style>
/* Cores principais */
:root {
    --primary-bg: #141414;
    --secondary-bg: #191919;
    --tertiary-bg: #262626;
    --accent: #FFD119;
    --text: #E0E0E0;
}

/* Configuração global */
.stApp {
    background-color: var(--primary-bg) !important;
    color: var(--text) !important;
}

/* Textos */
p, div, span, pre {
    color: var(--text) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: var(--secondary-bg) !important;
}

/* Widgets */
.stSelectbox, .stSlider, .stRadio, .stTextInput {
    background-color: var(--tertiary-bg) !important;
    color: var(--text) !important;
    border-color: #404040 !important;
}

/* Tabelas */
.stDataFrame {
    background-color: var(--tertiary-bg) !important;
}

/* Cabeçalhos */
h1, h2, h3, h4, h5, h6 {
    color: var(--accent) !important;
}

/* Botões */
.stButton>button {
    background-color: var(--tertiary-bg) !important;
    color: var(--accent) !important;
    border-color: var(--accent) !important;
}

.stButton>button:hover {
    background-color: var(--accent) !important;
    color: var(--primary-bg) !important;
}

/* Abas */
[data-testid="stTab"] {
    background-color: var(--secondary-bg) !important;
}

/* Gráficos */
[data-testid="stPlotlyChart"], [data-testid="stPyplot"] {
    background-color: var(--secondary-bg) !important;
}

/* Métricas */
[data-testid="stMetric"] {
    background-color: var(--tertiary-bg) !important;
    border-radius: 8px;
    padding: 15px;
}

/* Dataframe */
.dataframe {
    background-color: var(--tertiary-bg) !important;
    color: var(--text) !important;
}

/* Linhas da tabela */
.dataframe tr:nth-child(even) {
    background-color: var(--secondary-bg) !important;
}

/* Células da tabela */
.dataframe th, .dataframe td {
    color: var(--text) !important;
    border-color: #404040 !important;
}
</style>
"""
st.markdown(DARK_THEME, unsafe_allow_html=True)

# ============ CONFIGURAÇÃO DE GRÁFICOS ============
plt.style.use('dark_background')
sns.set_theme(style="darkgrid")
custom_palette = ["#FFD119", "#404040", "#262626", "#191919"]
sns.set_palette(custom_palette)

# ============ FUNÇÕES AUXILIARES ============
@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

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

# ============ SIDEBAR ============
with st.sidebar:
    st.title("📊 Analytics Pro")
    st.markdown("<hr style='border:1px solid #FFD119'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Carregue seu arquivo CSV", type="csv")

# ============ PÁGINA PRINCIPAL ============
if not uploaded_file:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://i.imgur.com/7kMk3Zz.png", width=300)
    with col2:
        st.markdown("""
        ## Bem-vindo ao Analytics Pro!
        **Sistema completo para análise exploratória de dados**
        
        1. 📤 Carregue seu CSV no menu lateral  
        2. 📊 Explore visualizações automáticas  
        3. 🔍 Analise estatísticas descritivas  
        """)
    st.stop()

# Carregamento de dados
try:
    df = load_data(uploaded_file)
    st.session_state['df'] = df
    st.success("✅ Arquivo carregado com sucesso!")
except Exception as e:
    st.error(f"❌ Erro ao ler arquivo: {str(e)}")
    st.stop()

# ============ VISÃO GERAL ============
st.header("📋 Visão Geral")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total de Registros", len(df))
with col2:
    st.metric("Total de Colunas", len(df.columns))
with col3:
    st.metric("Valores Faltantes", df.isnull().sum().sum())

st.dataframe(df.head(), height=250, use_container_width=True)

# ============ ANÁLISE TEMPORAL ============
datetime_cols = [col for col in df.columns if is_datetime_column(df[col])]
if datetime_cols:
    st.header("⏰ Análise Temporal")
    date_col = st.selectbox("Selecione coluna temporal", datetime_cols)
    
    try:
        df[date_col] = pd.to_datetime(df[date_col])
        
        tab1, tab2 = st.tabs(["📈 Série Temporal", "🗓️ Distribuição"])
        with tab1:
            freq = st.radio("Frequência", ["Diária", "Mensal"], horizontal=True)
            if freq == "Diária":
                temp_df = df[date_col].dt.floor('D').value_counts().sort_index()
            else:
                temp_df = df.groupby(df[date_col].dt.to_period('M')).size()
                temp_df.index = temp_df.index.to_timestamp()
            
            st.line_chart(temp_df)
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Por Hora do Dia")
                df['__hour__'] = df[date_col].dt.hour
                st.bar_chart(df['__hour__'].value_counts())
            
            with col2:
                st.subheader("Por Dia da Semana")
                df['__weekday__'] = df[date_col].dt.weekday
                st.bar_chart(df['__weekday__'].value_counts())
    except Exception as e:
        st.warning(f"⚠️ Não foi possível analisar a coluna temporal: {str(e)}")

# ============ ANÁLISE NUMÉRICA ============
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
if numerical_cols:
    st.header("🔢 Análise Numérica")
    num_col = st.selectbox("Selecione coluna numérica", numerical_cols)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.histplot(df[num_col], kde=True, ax=ax1, color='#FFD119')
    ax1.set_title(f'Distribuição de {num_col}', color='white')
    ax1.tick_params(colors='white')
    ax1.xaxis.label.set_color('white')
    ax1.yaxis.label.set_color('white')
    
    sns.boxplot(x=df[num_col], ax=ax2, color='#FFD119')
    ax2.set_title(f'Boxplot de {num_col}', color='white')
    ax2.tick_params(colors='white')
    ax2.xaxis.label.set_color('white')
    
    fig.patch.set_facecolor('#141414')
    st.pyplot(fig)

# ============ ANÁLISE CATEGÓRICA ============
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
if categorical_cols:
    st.header("📌 Análise Categórica")
    cat_col = st.selectbox("Selecione coluna categórica", categorical_cols)
    
    top_n = st.slider("Mostrar top N valores", 5, 20, 10)
    counts = df[cat_col].value_counts().nlargest(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=counts.values, y=counts.index, ax=ax, palette=custom_palette)
    plt.title(f'Top {top_n} Valores em {cat_col}', color='white')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    fig.patch.set_facecolor('#141414')
    st.pyplot(fig)

# ============ CORRELAÇÕES ============
if len(numerical_cols) > 1:
    st.header("🔗 Correlações Numéricas")
    fig, ax = plt.subplots(figsize=(12, 8))
    mask = np.triu(np.ones_like(df[numerical_cols].corr(), dtype=bool))
    sns.heatmap(df[numerical_cols].corr(), annot=True, fmt=".2f", 
                cmap='coolwarm', mask=mask, ax=ax, center=0,
                annot_kws={"color": "white"})
    ax.tick_params(colors='white')
    fig.patch.set_facecolor('#141414')
    st.pyplot(fig)
