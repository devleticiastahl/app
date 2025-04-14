import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    r2_score, mean_squared_error, roc_curve, auc
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.cluster import KMeans
import joblib

# ============ CONFIGURAÇÃO DA PÁGINA ============
st.set_page_config(
    page_title="ML Analytics Pro",
    page_icon="🤖",
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

def preprocess_data(df, target_column):
    # Pré-processamento automático
    df_clean = df.dropna()
    
    # Codificação de variáveis categóricas
    le = LabelEncoder()
    for col in df_clean.select_dtypes(include=['object']):
        if col != target_column:
            df_clean[col] = le.fit_transform(df_clean[col])
    
    # Separação de features e target
    X = df_clean.drop(columns=[target_column])
    y = df_clean[target_column]
    
    # Normalização
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, le

def plot_roc_curve(y_true, y_pred_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    return fig

# ============ SIDEBAR ============
with st.sidebar:
    st.title("🤖 ML Analytics Pro")
    uploaded_file = st.file_uploader("Carregue seu CSV", type="csv")
    st.header("Configurações de ML")
    ml_task = st.selectbox("Tipo de Tarefa", ["Classificação", "Regressão", "Clustering"])
    model_type = st.selectbox("Algoritmo", {
        "Classificação": ["Logistic Regression", "Random Forest", "XGBoost"],
        "Regressão": ["Linear Regression", "Random Forest", "XGBoost"],
        "Clustering": ["K-Means"]
    }[ml_task])

# ============ CARREGAMENTO DE DADOS ============
if not uploaded_file:
    st.markdown("""
    ## Bem-vindo ao ML Analytics Pro!
    Um sistema completo para análise de dados e machine learning.
    """)
    st.image("https://i.imgur.com/7kMk3Zz.png", width=400)
    st.stop()

try:
    df = load_data(uploaded_file)
    st.success("✅ Arquivo carregado com sucesso!")
except Exception as e:
    st.error(f"❌ Erro ao ler arquivo: {str(e)}")
    st.stop()

# ============ SELEÇÃO DE VARIÁVEL ALVO ============
if ml_task != "Clustering":
    target_col = st.selectbox("Selecione a variável alvo", df.columns)
    X, y, scaler, le = preprocess_data(df, target_col)
else:
    X, _, scaler, _ = preprocess_data(df, df.columns[0])

# ============ TREINAMENTO DO MODELO ============
if st.button("🎯 Treinar Modelo"):
    st.header("Resultados do Modelo")
    
    try:
        if ml_task == "Clustering":
            model = KMeans(n_clusters=3)
            clusters = model.fit_predict(X)
            
            fig = plt.figure(figsize=(10, 6))
            sns.scatterplot(x=X[:,0], y=X[:,1], hue=clusters, palette="viridis")
            st.pyplot(fig)
            
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Seleção do modelo
            if model_type == "Logistic Regression":
                model = LogisticRegression()
            elif model_type == "Random Forest":
                model = RandomForestClassifier() if ml_task == "Classificação" else RandomForestRegressor()
            elif model_type == "XGBoost":
                model = XGBClassifier() if ml_task == "Classificação" else XGBRegressor()
            else:
                model = LinearRegression()
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Métricas de avaliação
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Métricas de Desempenho")
                if ml_task == "Classificação":
                    st.metric("Acurácia", f"{accuracy_score(y_test, y_pred):.2%}")
                    st.write(classification_report(y_test, y_pred))
                    
                    # Matriz de Confusão
                    fig, ax = plt.subplots()
                    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', ax=ax)
                    ax.set_title('Matriz de Confusão')
                    st.pyplot(fig)
                    
                else:
                    st.metric("R² Score", f"{r2_score(y_test, y_pred):.2f}")
                    st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
            
            with col2:
                st.subheader("Visualizações")
                if ml_task == "Classificação":
                    y_pred_prob = model.predict_proba(X_test)[:,1]
                    roc_fig = plot_roc_curve(y_test, y_pred_prob)
                    st.pyplot(roc_fig)
                else:
                    fig, ax = plt.subplots()
                    ax.scatter(y_test, y_pred, alpha=0.3)
                    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
                    ax.set_xlabel('Valores Reais')
                    ax.set_ylabel('Previsões')
                    ax.set_title('Previsão vs Real')
                    st.pyplot(fig)
            
            # Download do modelo
            joblib.dump(model, 'model.pkl')
            with open('model.pkl', 'rb') as f:
                st.download_button("⬇️ Baixar Modelo Treinado", f, file_name="modelo_treinado.pkl")
    
    except Exception as e:
        st.error(f"Erro no treinamento: {str(e)}")

# ============ ANÁLISE EXPLORATÓRIA ============
st.header("🔍 Análise Exploratória")

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
with st.expander("📈 Estatísticas Descritivas"):
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
    date_col = st.selectbox("Selecione coluna temporal", datetime_cols, key="temp_analysis")
    
    try:
        df[date_col] = pd.to_datetime(df[date_col])
        df['__temp_date__'] = df[date_col].dt.floor('D')
        
        st.subheader(f"Análise Temporal: {date_col}")
        tab1, tab2 = st.tabs(["Série Temporal", "Distribuição Temporal"])
        
        with tab1:
            freq = st.radio("Frequência", ["Diária", "Mensal"], 
                          horizontal=True, key="temp_freq")
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
        
        # Limpeza de colunas temporárias
        del df['__temp_date__'], df['__hour__'], df['__weekday__']
    except Exception as e:
        st.error(f"⚠️ Erro na análise temporal: {str(e)}")

# Análise numérica
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
if numerical_cols:
    st.subheader("📉 Distribuição Numérica")
    num_col = st.selectbox("Selecione coluna numérica", numerical_cols, key="num_dist")
    
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
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
if categorical_cols:
    st.subheader("📊 Análise Categórica")
    cat_col = st.selectbox("Selecione coluna categórica", categorical_cols, key="cat_analysis")
    
    top_n = st.slider("Mostrar top N valores", 5, 20, 10, key="top_n")
    counts = df[cat_col].value_counts().nlargest(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=counts.values, y=counts.index, ax=ax, palette='viridis')
    plt.title(f'Top {top_n} Valores em {cat_col}')
    st.pyplot(fig)

# Mapa de calor de correlação
if len(numerical_cols) > 1:
    st.subheader("🔥 Mapa de Correlação")
    fig, ax = plt.subplots(figsize=(12, 8))
    mask = np.triu(np.ones_like(df[numerical_cols].corr(), dtype=bool))
    sns.heatmap(df[numerical_cols].corr(), annot=True, fmt=".2f", 
                cmap='coolwarm', mask=mask, ax=ax)
    st.pyplot(fig)
