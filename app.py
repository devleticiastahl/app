import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    """Verifica se uma coluna parece ser datetime."""
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
    uploaded_file = st.file_uploader("Carregue seu arquivo CSV", type="csv")

# ============ CARREGAMENTO DE DADOS ============
if not uploaded_file:
    st.markdown("""
    ## Bem-vindo ao Analytics Pro!
    Um sistema completo para análise exploratória de dados.

    1. **Carregue seu CSV** usando o menu lateral
    2. **Explore as estatísticas descritivas**
    3. **Analise as visualizações automáticas**
    """)
    st.image("https://i.imgur.com/7kMk3Zz.png", width=400)
    st.stop()

try:
    df = load_data(uploaded_file)
    st.success("✅ Arquivo carregado com sucesso!")
except Exception as e:
    st.error(f"❌ Erro ao ler arquivo: {str(e)}")
    st.stop()

# ============ ANÁLISE EXPLORATÓRIA ============
st.header("Análise Exploratória")

# Métricas básicas
col1, col2 = st.columns(2)
with col1:
    st.metric("Total de Registros", len(df))
with col2:
    st.metric("Total de Colunas", len(df.columns))

# Primeiras linhas do dataframe
st.subheader("Primeiras Linhas")
st.dataframe(df.head(), height=250)

# Estatísticas descritivas
with st.expander("Estatísticas Descritivas"):
    st.subheader("Tipos de Dados")
    st.write(df.dtypes.astype(str))
    
    st.subheader("Valores Faltantes")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        st.bar_chart(missing[missing > 0])
    else:
        st.success("✅ Nenhum valor faltante encontrado!")
    
    st.subheader("Estatísticas Numéricas")
    st.write(df.describe())

# Análise temporal
datetime_cols = [col for col in df.columns if is_datetime_column(df[col])]
if datetime_cols:
    date_col = st.selectbox("Selecione coluna temporal", datetime_cols)
    
    try:
        df[date_col] = pd.to_datetime(df[date_col])
        df['__temp_date__'] = df[date_col].dt.floor('D')
        
        st.subheader(f"Análise Temporal: {date_col}")
        tab1, tab2 = st.tabs(["Série Temporal", "Distribuição Temporal"])
        
        with tab1:
            freq = st.radio("Frequência", ["Diária", "Mensal"], horizontal=True)
            if freq == "Diária":
                temp_df = df['__temp_date__'].value_counts().sort_index()
            else:
                temp_df = df.groupby(df[date_col].dt.to_period('M')).size()
                temp_df.index = temp_df.index.to_timestamp()
            
            st.line_chart(temp_df)
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Distribuição por Hora")
                df['__hour__'] = df[date_col].dt.hour
                st.bar_chart(df['__hour__'].value_counts())
            
            with col2:
                st.subheader("Distribuição por Dia da Semana")
                df['__weekday__'] = df[date_col].dt.weekday
                st.bar_chart(df['__weekday__'].value_counts())
        
        del df['__temp_date__'], df['__hour__'], df['__weekday__']
    except Exception as e:
        st.error(f"⚠️ Erro na análise temporal: {str(e)}")

# Análise numérica
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
if numerical_cols:
    st.subheader("Distribuição Numérica")
    num_col = st.selectbox("Selecione coluna numérica", numerical_cols)
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        sns.histplot(df[num_col], kde=True, ax=ax, color='skyblue')
        plt.title(f'Histograma de {num_col}')
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots()
        sns.boxplot(x=df[num_col], ax=ax, color='lightgreen')
        plt.title(f'Boxplot de {num_col}')
        st.pyplot(fig)

# Análise categórica
# Análise categórica
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
if categorical_cols:
    st.subheader("Análise Categórica")
    cat_col = st.selectbox("Selecione coluna categórica", categorical_cols)
    
    top_n = st.slider("Mostrar top N valores", 5, 20, 10)
    counts = df[cat_col].value_counts().nlargest(top_n)
    
    # Paleta customizada com destaque para top 3
    palette = [
        '#1f77b4' if i < 3 else '#aec7e8'  # Azul mais forte para top 3, suave para o restante
        for i in range(len(counts))
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x=counts.values, 
        y=counts.index, 
        ax=ax, 
        palette=palette,
        linewidth=0  # Remove completamente as bordas
    )
    
    plt.title(f'Top {top_n} Valores em {cat_col}', fontsize=14, pad=20)
    plt.xlabel('Contagem', fontsize=12)
    plt.ylabel('')
    sns.despine()
    st.pyplot(fig)


# Mapa de calor de correlação
if len(numerical_cols) > 1:
    st.subheader("Mapa de Correlação")
    fig, ax = plt.subplots(figsize=(12, 8))
    mask = np.triu(np.ones_like(df[numerical_cols].corr(), dtype=bool))
    sns.heatmap(df[numerical_cols].corr(), annot=True, fmt=".2f", 
                cmap='coolwarm', mask=mask, ax=ax)
    st.pyplot(fig)
