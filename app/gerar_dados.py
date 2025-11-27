import pandas as pd
import numpy as np
import random

# Configuração da semente para reprodutibilidade
np.random.seed(42)

def gerar_dataset_agricola(n_linhas=1000):
    """
    Gera um dataset simulado para treinamento do modelo de Regressão.
    Variáveis: Umidade, Temperatura, pH, Nitrogênio, Fósforo, Potássio.
    Target: Produtividade (kg/ha).
    """
    data = []
    
    for _ in range(n_linhas):
        # Simulação de variaveis independentes (Features)
        umidade = round(np.random.uniform(20, 90), 2)  # %
        temperatura = round(np.random.uniform(15, 35), 2) # Celsius
        ph = round(np.random.uniform(4.5, 8.5), 1) # Escala de pH
        chuva = round(np.random.uniform(0, 150), 1) # mm
        
        # Simulação simplificada de produtividade baseada em regras biológicas aproximadas
        # (Fórmula fictícia apenas para criar correlação matemática para a IA aprender)
        produtividade_base = 3000 # kg/ha base
        
        # Penalidades e Bônus
        fator_umidade = -10 * abs(umidade - 60) # Umidade ideal ~60%
        fator_ph = -50 * abs(ph - 6.5) # pH ideal ~6.5
        fator_temp = -20 * abs(temperatura - 25) # Temp ideal ~25C
        bonus_chuva = 5 * chuva if chuva < 100 else -5 * (chuva - 100) # Chuva demais atrapalha
        
        produtividade_final = produtividade_base + fator_umidade + fator_ph + fator_temp + bonus_chuva
        
        # Adiciona um pouco de ruído aleatório (mundo real não é perfeito)
        ruido = np.random.normal(0, 150)
        produtividade_final += ruido
        
        # Garantir que não seja negativo
        produtividade_final = max(0, produtividade_final)

        data.append([umidade, temperatura, ph, chuva, round(produtividade_final, 2)])

    df = pd.DataFrame(data, columns=['Umidade_Solo', 'Temperatura', 'pH', 'Chuva_mm', 'Produtividade_kg_ha'])
    
    # Salvar na pasta data
    import os
    os.makedirs('../data', exist_ok=True)
    df.to_csv('../data/dados_sensores.csv', index=False)
    print("Arquivo 'dados_sensores.csv' gerado com sucesso na pasta data!")

if __name__ == "__main__":
    gerar_dataset_agricola()
