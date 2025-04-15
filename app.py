import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# ============ CONFIGURAÇÃO DE TEMA ============
DARK_THEME = """
<style>
/* Cores principais */
:root {
    --primary-bg: #141414;
    --secondary-bg: #191919;
    --tertiary-bg: #262626;
    --accent: #FFD119;
    --text: #FFFFFF;
}

/* Estilos gerais */
html, body, [class*="css"] {
    color: var(--text);
    background-color: var(--primary-bg);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: var(--secondary-bg) !important;
}

/* Widgets */
.stSelectbox, .stSlider, .stRadio {
    background-color: var(--tertiary-bg);
    border-radius: 8px;
    padding: 10px;
}

/* Gráficos */
svg {
    background-color: var(--secondary-bg) !important;
}

/* Tabelas */
[data-testid="stDataFrame"] {
    background-color: var(--tertiary-bg) !important;
    color: var(--text) !important;
}

/* Headers */
h1, h2, h3, h4, h5, h6 {
    color: var(--accent) !important;
}

/* Botões e hover */
.stButton>button {
    background-color: var(--tertiary-bg) !important;
    color: var(--accent) !important;
    border: 1px solid var(--accent) !important;
}

.stButton>button:hover {
    background-color: var(--accent) !important;
    color: var(--primary-bg) !important;
}

/* Ajustes de Matplotlib */
div.stPlotlyChart, div.stPyplot {
    background-color: var(--secondary-bg) !important;
}
</style>
"""

st.markdown(DARK_THEME, unsafe_allow_html=True)

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

# Configuração de estilo para gráficos
plt.style.use('dark_background')
sns.set_theme(style="darkgrid", palette="viridis")
custom_palette = ["#FFD119", "#404040", "#262626", "#191919", "#141414"]
sns.set_palette(custom_palette)

# ============ SIDEBAR ============
with st.sidebar:
    st.title("📊 Analytics Pro")
    st.markdown("<hr style='border:1px solid #FFD119'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Carregue seu arquivo CSV", type="csv")

# ============ PÁGINA PRINCIPAL ============
if not uploaded_file:
    st.markdown("""
    ## Bem-vindo ao Analytics Pro!
    Um sistema completo para análise exploratória de dados.

    1. **Carregue seu CSV** usando o menu lateral ➡️
    2. **Explore as estatísticas descritivas** 📈
    3. **Analise as visualizações automáticas** 📊
    """)
    st.image("https://i.imgur.com/7kMk3Zz.png", width=400)
    st.stop()

try:
    df = load_data(uploaded_file)
    st.session_state['df'] = df
    st.success("✅ Arquivo carregado com sucesso!")
except Exception as e:
    st.error(f"❌ Erro ao ler arquivo: {str(e)}")
    st.stop()

# ============ VISUALIZAÇÃO DE DADOS ============
st.header("📋 Visão Geral dos Dados")

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"<div style='background-color:#191919; padding:20px; border-radius:10px;'>"
                f"<h3 style='color:#FFD119;'>Total de Registros</h3>"
                f"<h1 style='color:#FFD119; text-align:center;'>{len(df)}</h1></div>", 
                unsafe_allow_html=True)

with col2:
    st.markdown(f"<div style='background-color:#191919; padding:20px; border-radius:10px;'>"
                f"<h3 style='color:#FFD119;'>Total de Colunas</h3>"
                f"<h1 style='color:#FFD119; text-align:center;'>{len(df.columns)}</h1></div>", 
                unsafe_allow_html=True)

# ... (o restante do código mantém a mesma estrutura, mas com as cores aplicadas)

# Exemplo de ajuste em um gráfico
def plot_custom_chart(data, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax = sns.histplot(data, kde=True, color='#FFD119')
    plt.title(title, color='#FFD119', fontsize=14)
    ax.set_facecolor('#141414')
    fig.patch.set_facecolor('#141414')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    return fig

# ============ DEMONSTRAÇÃO DE USO ============
# Substitua todos os gráficos existentes por versões usando a função plot_custom_chart
# Exemplo na seção de distribuição numérica:
if numerical_cols:
    st.subheader("📉 Distribuição Numérica")
    num_col = st.selectbox("Selecione coluna numérica", numerical_cols)
    st.pyplot(plot_custom_chart(df[num_col], f'Distribuição de {num_col}'))
