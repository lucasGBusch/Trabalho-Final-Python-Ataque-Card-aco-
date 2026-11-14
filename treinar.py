import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from pathlib import Path
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Localizando o dataset
caminho = Path(__file__).resolve().parent / "data" / "dataset_cleveland_didatico.csv"


df = pd.read_csv(caminho)


X = df.drop(columns=["num"])

#
y = (df["num"] > 0).astype(int)


print("Variáveis utilizadas no treinamento:")
print(X.columns.tolist())

print("\nQuantidade de registros e variáveis:")
print(X.shape)

print("\nDistribuição da variável alvo:")
print(y.value_counts())



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nQuantidade de registros para treinamento:")
print(X_train.shape)

print("\nQuantidade de registros para teste:")
print(X_test.shape)

print("\nDistribuição das classes no teste:")
print(y_test.value_counts())



colunas_numericas = [
    "idade",
    "pressao_repouso",
    "colesterol",
    "freq_cardiaca_max"
]

colunas_categoricas = [
    "sexo",
    "tipo_dor_peito",
    "acucar_sangue_jejum",
    "ecg_repouso",
    "angina_exercicio"
]


print("\nValores ausentes no conjunto de treinamento:")
print(X_train.isnull().sum())


print("\nValores ausentes no conjunto de teste:")
print(X_test.isnull().sum())



preprocessamento_numerico = Pipeline(steps=[
    ("imputador", SimpleImputer(strategy="median")),
    ("escalador", StandardScaler())
])


preprocessamento_categorico = Pipeline(steps=[
    ("imputador", SimpleImputer(strategy="most_frequent")),
    ("codificador", OneHotEncoder(handle_unknown="ignore"))
])


preprocessador = ColumnTransformer(transformers=[
    ("numericas", preprocessamento_numerico, colunas_numericas),
    ("categoricas", preprocessamento_categorico, colunas_categoricas)
])

print("\nPre-processamento configurado com sucesso!")

# LogisticRegression
modelo_logistico = Pipeline(steps=[
    ("preprocessamento", preprocessador),
    ("classificador", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])


modelo_logistico.fit(X_train, y_train)


y_pred_logistico = modelo_logistico.predict(X_test)


acuracia_logistica = accuracy_score(y_test, y_pred_logistico)

print("\nResultado da Regressão Logística:")
print(f"Acurácia: {acuracia_logistica:.2%}")

# DecisionTree
modelo_arvore = Pipeline(steps=[
    ("preprocessamento", clone(preprocessador)),
    ("classificador", DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=5,
        random_state=42
    ))
])


modelo_arvore.fit(X_train, y_train)


y_pred_arvore = modelo_arvore.predict(X_test)


acuracia_arvore = accuracy_score(y_test, y_pred_arvore)

print("\nResultado da Árvore de Decisão:")
print(f"Acurácia: {acuracia_arvore:.2%}")

# RandomForest
modelo_floresta = Pipeline(steps=[
    ("preprocessamento", clone(preprocessador)),
    ("classificador", RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=3,
        random_state=42
    ))
])


modelo_floresta.fit(X_train, y_train)


y_pred_floresta = modelo_floresta.predict(X_test)


acuracia_floresta = accuracy_score(y_test, y_pred_floresta)

print("\nResultado do Random Forest:")
print(f"Acurácia: {acuracia_floresta:.2%}")


# Comparando os resultados dos três modelos
resultados = {
    "Regressão Logística": y_pred_logistico,
    "Árvore de Decisão": y_pred_arvore,
    "Random Forest": y_pred_floresta
}

for nome, previsoes in resultados.items():
    print(f"\nAvaliação do modelo: {nome}")

    print(f"Acurácia: {accuracy_score(y_test, previsoes):.2%}")
    print(f"Precisão: {precision_score(y_test, previsoes, zero_division=0):.2%}")
    print(f"Recall: {recall_score(y_test, previsoes, zero_division=0):.2%}")
    print(f"F1-score: {f1_score(y_test, previsoes, zero_division=0):.2%}")

    print("Matriz de confusão:")
    print(confusion_matrix(y_test, previsoes, labels=[0, 1]))


# Definindo o caminho para salvar o modelo
caminho_modelo = Path(__file__).resolve().parent / "models" / "melhor_modelo.joblib"

# Salvando o modelo escolhido
joblib.dump(modelo_logistico, caminho_modelo)

print("\nModelo salvo com sucesso!")
print("Local:", caminho_modelo)
