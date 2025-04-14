import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from fpdf import FPDF
import base64
import tempfile

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

def create_pdf_report(df, figures, stats):
    """Cria um relatório PDF com os resultados da análise"""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Capa
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Relatório Analytics Pro', 0, 1, 'C')
    pdf.ln(10)
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f'Gerado em: {datetime.now().strftime("%d/%m/%Y %H:%M")}', 0, 1, 'C')
    
    # Sumário
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Sumário Executivo', 0, 1)
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 10, f"""
    - Total de registros: {stats['total_records']}
    - Total de colunas: {stats['total_columns']}
    - Valores faltantes: {stats['missing_values']}
    - Colunas numéricas: {stats['numerical_cols']}
    - Colunas categóricas: {stats['categorical_cols']}
    """)
    
    # Gráficos
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Visualizações', 0, 1)
    
    for idx, fig in enumerate(figures):
        img_path = f"temp_fig_{idx}.png"
        fig.savefig(img_path, bbox_inches='tight')
        pdf.image(img_path, x=10, y=None, w=190)
        pdf.ln(85)
    
    # Salva o PDF
    pdf_path = "analytics_report.pdf"
    pdf.output(pdf_path)
    return pdf_path

# ============ SIDEBAR ============
with st.sidebar:
    st.title("📊 Analytics Pro")
    uploaded_file = st.file_uploader("Carregue seu arquivo CSV", type="csv")
    
    # Botão de exportação
    if uploaded_file:
        if st.button("📤 Exportar Relatório PDF"):
            with st.spinner("Gerando relatório..."):
                try:
                    # Carrega dados
                    df = load_data(uploaded_file)
                    
                    # Gera estatísticas básicas
                    stats = {
                        'total_records': len(df),
                        'total_columns': len(df.columns),
                        'missing_values': df.isnull().sum().sum(),
                        'numerical_cols': len(df.select_dtypes(include=np.number).columns),
                        'categorical_cols': len(df.select_dtypes(include=['object']).columns)
                    }
                    
                    # Gera todas as figuras
                    figures = []
                    
                    # Gráfico temporal
                    datetime_cols = [col for col in df.columns if is_datetime_column(df[col])]
                    if datetime_cols:
                        date_col = datetime_cols[0]
                        df[date_col] = pd.to_datetime(df[date_col])
                        fig, ax = plt.subplots(figsize=(10, 4))
                        df[date_col].dt.date.value_counts().plot(kind='line', ax=ax)
                        plt.title('Série Temporal')
                        figures.append(fig)
                    
                    # Gráfico numérico
                    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
                    if numerical_cols:
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                        sns.histplot(df[numerical_cols[0]], kde=True, ax=ax1)
                        sns.boxplot(x=df[numerical_cols[0]], ax=ax2)
                        plt.suptitle('Distribuição Numérica')
                        figures.append(fig)
                    
                    # Gráfico categórico
                    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                    if categorical_cols:
                        fig = plt.figure(figsize=(10, 4))
                        counts = df[categorical_cols[0]].value_counts().nlargest(10)
                        sns.barplot(x=counts.values, y=counts.index, palette='Blues_r')
                        plt.title('Top 10 Valores Categóricos')
                        figures.append(fig)
                    # Mapa de calor de correlação
                    if len(numerical_cols) > 1:
                        st.subheader("Mapa de Correlação")
                        fig, ax = plt.subplots(figsize=(10, 6))  # TAMANHO AJUSTADO
                        mask = np.triu(np.ones_like(df[numerical_cols].corr(), dtype=bool))
                        sns.heatmap(
                            df[numerical_cols].corr(), 
                            annot=True, 
                            fmt=".2f", 
                            cmap='coolwarm', 
                            mask=mask, 
                            ax=ax,
                            annot_kws={'size': 8}  # FONTE DAS ANOTAÇÕES MENOR
                        )
                        plt.xticks(fontsize=8)
                        plt.yticks(fontsize=8)
                        st.pyplot(fig)
                        
                    # Gera PDF
                    pdf_path = create_pdf_report(df, figures, stats)
                    
                    # Disponibiliza download
                    with open(pdf_path, "rb") as f:
                        pdf_data = f.read()
                        b64 = base64.b64encode(pdf_data).decode()
                        href = f'<a href="data:application/octet-stream;base64,{b64}" download="relatorio_analytics.pdf">Clique aqui para baixar</a>'
                        st.markdown(href, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Erro ao gerar relatório: {str(e)}")
                    
