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

# ============ CONFIGURAÇÃO DE GRÁFICOS ============
plt.style.use('default')
sns.set_theme(style="whitegrid")
sns.set_palette("viridis")

# ============ FUNÇÕES AUXILIARES ============
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        return pd.read_excel(uploaded_file)
    else:
        st.error("Formato de arquivo não suportado")
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

# ============ PÁGINA INICIAL MELHORADA ============
def show_homepage():
    st.markdown("""
    <style>
    .home-header {
        font-size: 2.5rem !important;
        color: #2c3e50 !important;
        margin-bottom: 0.5rem;
    }
    .feature-card {
        padding: 1.5rem;
        border-radius: 10px;
        background: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://i.imgur.com/7kMk3Zz.png", width=300)
    with col2:
        st.markdown('<div class="home-header">📊 Analytics Pro</div>', unsafe_allow_html=True)
        st.markdown("""
        **Sistema avançado para análise exploratória de dados (EDA)**
        
        Transforme seus dados em insights poderosos com visualizações automáticas
        """)
    
    st.markdown("---")
    
    cols = st.columns(3)
    with cols[0]:
        st.markdown('<div class="feature-card">'
                    '<h3>📤 Importação Fácil</h3>'
                    '<p>Suporte a arquivos CSV e Excel</p>'
                    '</div>', unsafe_allow_html=True)
    with cols[1]:
        st.markdown('<div class="feature-card">'
                    '<h3>📊 Visualizações Inteligentes</h3>'
                    '<p>Gráficos automáticos por tipo de dado</p>'
                    '</div>', unsafe_allow_html=True)
    with cols[2]:
        st.markdown('<div class="feature-card">'
                    '<h3>🔍 Análise Profunda</h3>'
                    '<p>Estatísticas descritivas completas</p>'
                    '</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    with st.expander("ℹ️ Como usar"):
        st.markdown("""
        1. **Carregue seu arquivo** (CSV ou Excel) no menu lateral
        2. **Explore as visualizações automáticas**
        3. **Analise as estatísticas** descritivas
        4. **Exporte os resultados** para sua análise
        """)
    
    st.markdown("""
    <div style="text-align: center; margin-top: 2rem;">
    <p>Comece carregando seu arquivo no menu lateral ➡️</p>
    </div>
    """, unsafe_allow_html=True)

# ============ SIDEBAR ============
with st.sidebar:
    st.title("📊 Analytics Pro")
    st.markdown("<hr>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Carregue seu arquivo de dados", 
        type=["csv", "xlsx"],
        help="Formatos suportados: CSV e Excel"
    )

# ============ PÁGINA PRINCIPAL ============
if not uploaded_file:
    show_homepage()
    st.stop()

# Carregamento de dados
try:
    df = load_data(uploaded_file)
    if df is None:
        st.stop()
    
    st.session_state['df'] = df
    st.success(f"✅ Arquivo {uploaded_file.name} carregado com sucesso!")
    
    # Mostrar informações básicas do arquivo
    with st.expander("🔍 Detalhes do arquivo"):
        st.write(f"**Nome do arquivo:** {uploaded_file.name}")
        st.write(f"**Tamanho:** {uploaded_file.size / 1024:.2f} KB")
        st.write(f"**Formato:** {uploaded_file.type}")
        
except Exception as e:
    st.error(f"❌ Erro ao ler arquivo: {str(e)}")
    st.stop()

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

# ============ ANÁLISE TEMPORAL ============
datetime_cols = [col for col in df.columns if is_datetime_column(df[col])]
if datetime_cols:
    st.header("Análise Temporal")
    date_col = st.selectbox("Selecione coluna temporal", datetime_cols)
    
    try:
        df[date_col] = pd.to_datetime(df[date_col])
        
        tab1, tab2 = st.tabs(["📈 Série Temporal", "🗓️ Distribuição"])
        with tab1:
            freq = st.radio("Frequência", ["Diária", "Mensal", "Anual"], horizontal=True)
            if freq == "Diária":
                temp_df = df[date_col].dt.floor('D').value_counts().sort_index()
            elif freq == "Mensal":
                temp_df = df.groupby(df[date_col].dt.to_period('M')).size()
                temp_df.index = temp_df.index.to_timestamp()
            else:
                temp_df = df.groupby(df[date_col].dt.to_period('Y')).size()
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
    st.header("Análise Numérica")
    num_col = st.selectbox("Selecione coluna numérica", numerical_cols)
    
    tab1, tab2 = st.tabs(["📊 Distribuição", "📈 Tendência"])
    
    with tab1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        sns.histplot(df[num_col], kde=True, ax=ax1, color='royalblue')
        ax1.set_title(f'Distribuição de {num_col}')
        
        sns.boxplot(x=df[num_col], ax=ax2, color='lightgreen')
        ax2.set_title(f'Boxplot de {num_col}')
        
        st.pyplot(fig)
    
    with tab2:
        if datetime_cols:
            date_col = st.selectbox("Selecione coluna temporal para análise", datetime_cols)
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                trend_df = df.groupby(df[date_col].dt.to_period('M'))[num_col].mean()
                trend_df.index = trend_df.index.to_timestamp()
                st.line_chart(trend_df)
            except:
                st.warning("Não foi possível criar gráfico de tendência")

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

# ============ CORRELAÇÕES ============
if len(numerical_cols) > 1:
    st.header("🔗 Correlações Numéricas")
    fig, ax = plt.subplots(figsize=(12, 8))
    mask = np.triu(np.ones_like(df[numerical_cols].corr(), dtype=bool))
    sns.heatmap(df[numerical_cols].corr(), annot=True, fmt=".2f", 
                cmap='coolwarm', mask=mask, ax=ax, center=0,
                annot_kws={"size": 10})
    st.pyplot(fig)

# ============ EXPORTAÇÃO DE RESULTADOS ============
st.markdown("---")
st.header("📤 Exportar Resultados")

export_format = st.selectbox("Formato de exportação", ["CSV", "Excel"])
filename = st.text_input("Nome do arquivo", "analytics_results")

if st.button("Exportar Dados"):
    try:
        if export_format == "CSV":
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Baixar CSV",
                data=csv,
                file_name=f"{filename}.csv",
                mime="text/csv"
            )
        else:
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False)
            st.download_button(
                label="Baixar Excel",
                data=output.getvalue(),
                file_name=f"{filename}.xlsx",
                mime="application/vnd.ms-excel"
            )
    except Exception as e:
        st.error(f"Erro ao exportar: {str(e)}")
