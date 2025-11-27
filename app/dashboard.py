import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Configuração da Página
st.set_page_config(page_title="FarmTech Solutions - IA", layout="wide")

# Título e Descrição
st.title("🚜 FarmTech Solutions: Agricultura Cognitiva")
st.markdown("""
Este dashboard utiliza **Machine Learning** para prever a produtividade da lavoura com base em sensores IoT.
Insira os dados coletados pelos sensores para obter insights de manejo.
""")

# --- 1. CARREGAMENTO E TREINAMENTO ---
@st.cache_data # Cache para não treinar toda hora que clicar num botão
def carregar_e_treinar():
    try:
        df = pd.read_csv('../data/dados_sensores.csv')
    except FileNotFoundError:
        st.error("Arquivo de dados não encontrado. Gere os dados primeiro.")
        return None, None, None

    # Separação X (Features) e y (Target)
    X = df[['Umidade_Solo', 'Temperatura', 'pH', 'Chuva_mm']]
    y = df['Produtividade_kg_ha']

    # Divisão Treino/Teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modelo de Regressão Linear
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, df, (X_test, y_test)

model, df, test_data = carregar_e_treinar()

if model is not None:
    X_test, y_test = test_data

    # --- 2. BARRA LATERAL (INPUTS) ---
    st.sidebar.header("📡 Simulador de Sensores IoT")
    st.sidebar.markdown("Ajuste os valores abaixo:")
    
    in_umidade = st.sidebar.slider("Umidade do Solo (%)", 0.0, 100.0, 50.0)
    in_ph = st.sidebar.slider("pH do Solo", 0.0, 14.0, 6.5)
    in_temp = st.sidebar.slider("Temperatura (°C)", 0.0, 50.0, 25.0)
    in_chuva = st.sidebar.number_input("Chuva Acumulada (mm)", 0.0, 300.0, 80.0)

    # Botão de Previsão
    if st.sidebar.button("Processar Previsão"):
        # Criar dataframe com os inputs
        input_data = pd.DataFrame([[in_umidade, in_temp, in_ph, in_chuva]], 
                                  columns=['Umidade_Solo', 'Temperatura', 'pH', 'Chuva_mm'])
        
        prediction = model.predict(input_data)[0]
        
        # --- 3. EXIBIÇÃO DOS RESULTADOS ---
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Produtividade Estimada")
            st.metric(label="Kg por Hectare", value=f"{prediction:.2f} kg/ha")
            
        with col2:
            st.subheader("📢 Sugestão de Manejo")
            recomendacoes = []
            
            # Lógica simples de "Sistema Especialista" baseada nos inputs
            if in_ph < 5.5:
                recomendacoes.append("⚠️ **Correção de Acidez:** pH baixo. Recomenda-se aplicar calcário (calagem).")
            elif in_ph > 7.5:
                recomendacoes.append("⚠️ **Correção de Alcalinidade:** pH alto. Avaliar uso de gesso agrícola.")
            else:
                recomendacoes.append("✅ **pH Ideal:** Monitorar periodicamente.")
                
            if in_umidade < 30:
                recomendacoes.append("💧 **Irrigação Crítica:** Umidade muito baixa. Ativar pivô central imediatamente.")
            elif in_umidade > 80:
                recomendacoes.append("🚫 **Drenagem:** Solo encharcado. Risco de fungos.")
            else:
                recomendacoes.append("✅ **Umidade Controlada:** Manter ciclo atual.")

            for rec in recomendacoes:
                st.write(rec)

    # --- 4. ANÁLISE DO MODELO (DASHBOARD GERAL) ---
    st.divider()
    st.header("📊 Performance do Modelo e Dados Históricos")
    
    tab1, tab2, tab3 = st.tabs(["Métricas do Modelo", "Correlações", "Base de Dados"])
    
    with tab1:
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        c1, c2 = st.columns(2)
        c1.metric("R² Score (Precisão)", f"{r2:.2%}")
        c2.metric("Erro Médio (MSE)", f"{mse:.2f}")
        
        st.info("O R² indica o quanto as variáveis (pH, Umidade, etc) explicam a variação da produtividade.")

    with tab2:
        st.write("Relação entre Umidade e Produtividade (Amostra)")
        fig, ax = plt.subplots()
        sns.scatterplot(x=df['Umidade_Solo'], y=df['Produtividade_kg_ha'], ax=ax, color='green', alpha=0.5)
        st.pyplot(fig)
        
    with tab3:
        st.dataframe(df.head(10))
