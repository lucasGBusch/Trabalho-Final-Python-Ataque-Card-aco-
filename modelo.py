import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# 1. Arquivos e colunas utilizadas
BASE = Path(__file__).resolve().parent
PASTA_MODELOS = BASE / "models"
ARQUIVO_RESULTADOS = PASTA_MODELOS / "resultados_modelos.json"

ARQUIVOS_MODELOS = {
    "LogisticRegression": "logistic_regression.joblib",
    "DecisionTreeClassifier": "decision_tree.joblib",
    "RandomForestClassifier": "random_forest.joblib"
}

COLUNAS = [
    "idade", "sexo", "tipo_dor_peito", "pressao_repouso", "colesterol",
    "acucar_sangue_jejum", "ecg_repouso", "freq_cardiaca_max", "angina_exercicio"
]

NUMERICAS = [
    "idade", "pressao_repouso", "colesterol", "freq_cardiaca_max"
]

CATEGORICAS = [
    "sexo", "tipo_dor_peito", "acucar_sangue_jejum",
    "ecg_repouso", "angina_exercicio"
]

CATEGORIAS_VALIDAS = {
    "sexo": [0, 1],
    "tipo_dor_peito": [1, 2, 3, 4],
    "acucar_sangue_jejum": [0, 1],
    "ecg_repouso": [0, 1, 2],
    "angina_exercicio": [0, 1]
}


# 2. Entrega as informações da análise para o Flask
def avaliar_modelo(nome_modelo):
    if nome_modelo not in ARQUIVOS_MODELOS:
        raise ValueError("Modelo não reconhecido.")

    with open(ARQUIVO_RESULTADOS, encoding="utf-8") as arquivo:
        resultados = json.load(arquivo)

    escolhido = resultados["avaliacoes"][nome_modelo]

    # Formata as porcentagens para o HTML
    comparacao = []

    for item in resultados["comparacao"]:
        comparacao.append({
            "modelo": item["modelo"],
            "acuracia": f"{item['acuracia']:.2%}",
            "precisao": f"{item['precisao']:.2%}",
            "recall": f"{item['recall']:.2%}",
            "f1": f"{item['f1']:.2%}"
        })

    return {
        "modelo": nome_modelo,
        "tamanho_teste": resultados["tamanho_teste"],
        "distribuicao_teste": resultados["distribuicao_teste"],
        "baseline": f"{resultados['baseline']:.2%}",
        "acuracia": f"{escolhido['acuracia']:.2%}",
        "comparacao": comparacao,
        "matriz_confusao": escolhido["matriz_confusao"]
    }


# 3. Recebe os dados do formulário e faz uma previsão
def fazer_previsao(nome_modelo, dados, campos_ativos):
    if nome_modelo not in ARQUIVOS_MODELOS:
        raise ValueError("Modelo não reconhecido.")

    if not campos_ativos:
        raise ValueError("Selecione pelo menos um campo.")

    for campo in campos_ativos:
        if campo not in COLUNAS:
            raise ValueError(f"Campo desconhecido: {campo}")

    paciente = {}

    # Conferindo as nove características
    for coluna in COLUNAS:

        # Campos desmarcados serão preenchidos pelo Pipeline
        if coluna not in campos_ativos:
            paciente[coluna] = np.nan
            continue

        valor = dados.get(coluna)

        if valor is None or str(valor).strip() == "":
            raise ValueError(f"Preencha o campo: {coluna}")

        try:
            numero = float(valor)
        except (TypeError, ValueError):
            raise ValueError(f"Valor inválido para {coluna}") from None

        # Não aceita números infinitos ou NaN
        if not np.isfinite(numero):
            raise ValueError(f"Valor inválido para {coluna}")

        # Valida as categorias
        if coluna in CATEGORIAS_VALIDAS:
            if numero not in CATEGORIAS_VALIDAS[coluna]:
                raise ValueError(f"Categoria inválida para {coluna}")

        # Valida as variáveis numéricas
        if coluna in NUMERICAS:
            if numero <= 0:
                raise ValueError(f"Valor inválido para {coluna}")

        paciente[coluna] = numero

    # Criando uma linha com todas as nove características
    entrada = pd.DataFrame([paciente], columns=COLUNAS)

    # Carregando o algoritmo selecionado
    modelo = joblib.load(
        PASTA_MODELOS / ARQUIVOS_MODELOS[nome_modelo]
    )

    # Realizando uma nova previsão
    resultado = int(modelo.predict(entrada)[0])

    if resultado == 1:
        return "Classe 1 — indicação de doença cardíaca"

    return "Classe 0 — sem indicação de doença cardíaca"
