import json

import joblib
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

from modelo import (
    BASE, PASTA_MODELOS, ARQUIVO_RESULTADOS, ARQUIVOS_MODELOS,
    COLUNAS, NUMERICAS, CATEGORICAS
)


# 1. Arquivo de dados usado no treinamento
ARQUIVO_CSV = BASE / "data" / "dataset_cleveland_didatico.csv"


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


# Executa o treinamento somente ao rodar: python treinar.py
if __name__ == "__main__":
    treinar_modelos()
