
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


# 1. Arquivos e colunas utilizadas
BASE = Path(__file__).resolve().parent
PASTA_MODELOS = BASE / "models"
ARQUIVO_CSV = BASE / "data" / "dataset_cleveland_didatico.csv"
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


# 2. Treina os três modelos, testa e salva os resultados
def treinar_modelos():
    df = pd.read_csv(ARQUIVO_CSV)

    X = df[COLUNAS]                     # Características dos pacientes
    y = (df["num"] > 0).astype(int)     # 0 = sem doença, 1 = com doença

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Prepara números e categorias sem usar os dados de teste para aprender
    preparo_numerico = Pipeline([
        ("preencher", SimpleImputer(strategy="median")),
        ("padronizar", StandardScaler())
    ])

    preparo_categorico = Pipeline([
        ("preencher", SimpleImputer(strategy="most_frequent")),
        ("codificar", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessador = ColumnTransformer([
        ("numericas", preparo_numerico, NUMERICAS),
        ("categoricas", preparo_categorico, CATEGORICAS)
    ])

    # Criando os três algoritmos
    algoritmos = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=42
        ),

        "DecisionTreeClassifier": DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=5, random_state=42
        ),

        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=100, max_depth=5,
            min_samples_leaf=3, random_state=42
        )
    }

    PASTA_MODELOS.mkdir(exist_ok=True)

    avaliacoes = {}
    comparacao = []

    print(f"Total de pacientes: {len(df)}")
    print(f"Treinamento: {len(X_train)} | Teste: {len(X_test)}")

    # Repetindo o mesmo treinamento para os três modelos
    for nome, algoritmo in algoritmos.items():
        modelo = Pipeline([
            ("preprocessamento", clone(preprocessador)),
            ("classificador", algoritmo)
        ])

        modelo.fit(X_train, y_train)             # Aprende com o treino
        y_pred = modelo.predict(X_test)          # Prevê os pacientes do teste

        acuracia = accuracy_score(y_test, y_pred)

        matriz = confusion_matrix(
            y_test, y_pred, labels=[0, 1]
        ).tolist()

        # Métricas para comparação dos algoritmos
        comparacao.append({
            "modelo": nome,
            "acuracia": float(acuracia),
            "precisao": float(precision_score(
                y_test, y_pred, zero_division=0
            )),
            "recall": float(recall_score(
                y_test, y_pred, zero_division=0
            )),
            "f1": float(f1_score(
                y_test, y_pred, zero_division=0
            ))
        })

        # Resultado individual de cada modelo
        avaliacoes[nome] = {
            "acuracia": float(acuracia),
            "matriz_confusao": matriz
        }

        # Salvando cada modelo treinado
        joblib.dump(
            modelo,
            PASTA_MODELOS / ARQUIVOS_MODELOS[nome]
        )

        # Mantendo compatibilidade com o modelo.py antigo
        if nome == "LogisticRegression":
            joblib.dump(
                modelo,
                PASTA_MODELOS / "melhor_modelo.joblib"
            )

        print(f"{nome}: acurácia de {acuracia:.2%}")
        print("Matriz de confusão:", matriz)

    # Guardando as informações do conjunto de teste
    resultados = {
        "tamanho_teste": len(y_test),
        "distribuicao_teste": {
            "0": int((y_test == 0).sum()),
            "1": int((y_test == 1).sum())
        },
        "baseline": float(
            y_test.value_counts(normalize=True).max()
        ),
        "comparacao": comparacao,
        "avaliacoes": avaliacoes
    }

    # Salvando métricas para o Flask consultar
    with open(ARQUIVO_RESULTADOS, "w", encoding="utf-8") as arquivo:
        json.dump(resultados, arquivo, ensure_ascii=False, indent=4)

    pd.DataFrame(comparacao).to_csv(
        PASTA_MODELOS / "resultados_modelos.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nTrês modelos e métricas salvos na pasta models!")


# 3. Entrega as informações da análise para o Flask
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


# 4. Recebe os dados do formulário e faz uma previsão
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


# Executa o treinamento somente ao rodar: python treinar.py
if __name__ == "__main__":
    treinar_modelos()
