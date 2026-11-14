import numpy as np
import pandas as pd
import joblib
from pathlib import Path


# Caminho do modelo treinado
caminho_modelo = Path(__file__).resolve().parent / "models" / "melhor_modelo.joblib"

# Carregando o modelo
modelo = joblib.load(caminho_modelo)

# Variáveis utilizadas no treinamento
colunas = [
    "idade",
    "sexo",
    "tipo_dor_peito",
    "pressao_repouso",
    "colesterol",
    "acucar_sangue_jejum",
    "ecg_repouso",
    "freq_cardiaca_max",
    "angina_exercicio"
]

# Função que realiza a previsão
def prever_doenca(dados_paciente):

    # Verificando se todas as variáveis foram informadas
    for coluna in colunas:
        if coluna not in dados_paciente:
            raise ValueError(f"Campo não informado: {coluna}")

    # Criando o DataFrame com os dados recebidos
    entrada = pd.DataFrame([dados_paciente], columns=colunas)

    # Convertendo os valores para números
    for coluna in colunas:
        entrada[coluna] = pd.to_numeric(
            entrada[coluna],
            errors="coerce"
        )

    
    # Verificando se existem valores inválidos
    if entrada.isna().any().any():
        raise ValueError("Existem valores ausentes ou inválidos.")

    # Valores permitidos para as variáveis categóricas
    categorias_validas = {
        "sexo": [0, 1],
        "tipo_dor_peito": [1, 2, 3, 4],
        "acucar_sangue_jejum": [0, 1],
        "ecg_repouso": [0, 1, 2],
        "angina_exercicio": [0, 1]
    }

    # Verificando se as categorias são válidas
    for coluna, valores in categorias_validas.items():
        valor = entrada[coluna].iloc[0]

        if valor not in valores:
          raise ValueError(f"Valor inválido para {coluna}")


    # Verificando valores numéricos
    colunas_numericas = [
        "idade",
        "pressao_repouso",
        "colesterol",
        "freq_cardiaca_max"
    ]

    
    for coluna in colunas_numericas:
        valor = entrada[coluna].iloc[0]

        if valor <= 0:
          raise ValueError(f"Valor inválido para {coluna}")

        if not np.isfinite(valor):
          raise ValueError(f"Valor inválido para {coluna}")


    # Executando a previsão
    resultado = modelo.predict(entrada)

    return int(resultado[0])


