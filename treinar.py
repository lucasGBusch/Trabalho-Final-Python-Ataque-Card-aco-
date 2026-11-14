from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path

# Localizando o dataset
caminho = Path(__file__).resolve().parent / "data" / "dataset_cleveland_didatico.csv"

# Carregando os dados
df = pd.read_csv(caminho)

# Separando as 9 variáveis de entrada
X = df.drop(columns=["num"])

# Definindo a variável que queremos prever
y = (df["num"] > 0).astype(int)

# Conferindo as entradas
print("Variáveis utilizadas no treinamento:")
print(X.columns.tolist())

print("\nQuantidade de registros e variáveis:")
print(X.shape)

print("\nDistribuição da variável alvo:")
print(y.value_counts())


# Separando os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Conferindo a divisão dos dados
print("\nQuantidade de registros para treinamento:")
print(X_train.shape)

print("\nQuantidade de registros para teste:")
print(X_test.shape)

print("\nDistribuição das classes no teste:")
print(y_test.value_counts())


# Separando as variáveis numéricas e categóricas
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

# Conferindo os valores ausentes no treinamento
print("\nValores ausentes no conjunto de treinamento:")
print(X_train.isnull().sum())

# Conferindo os valores ausentes no teste
print("\nValores ausentes no conjunto de teste:")
print(X_test.isnull().sum())


# Preparação das variáveis numéricas
preprocessamento_numerico = Pipeline(steps=[
    ("imputador", SimpleImputer(strategy="median")),
    ("escalador", StandardScaler())
])

# Preparação das variáveis categóricas
preprocessamento_categorico = Pipeline(steps=[
    ("imputador", SimpleImputer(strategy="most_frequent")),
    ("codificador", OneHotEncoder(handle_unknown="ignore"))
])

# Juntando os dois tipos de preparação
preprocessador = ColumnTransformer(transformers=[
    ("numericas", preprocessamento_numerico, colunas_numericas),
    ("categoricas", preprocessamento_categorico, colunas_categoricas)
])

print("\nPre-processamento configurado com sucesso!")

# Criando o modelo de Regressão Logística
modelo_logistico = Pipeline(steps=[
    ("preprocessamento", preprocessador),
    ("classificador", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

# Treinando o modelo
modelo_logistico.fit(X_train, y_train)

# Fazendo previsões com os dados de teste
y_pred_logistico = modelo_logistico.predict(X_test)

# Calculando a acurácia
acuracia_logistica = accuracy_score(y_test, y_pred_logistico)

print("\nResultado da Regressão Logística:")
print(f"Acurácia: {acuracia_logistica:.2%}")

# Criando o modelo de Árvore de Decisão
modelo_arvore = Pipeline(steps=[
    ("preprocessamento", clone(preprocessador)),
    ("classificador", DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=5,
        random_state=42
    ))
])

# Treinando a Árvore de Decisão
modelo_arvore.fit(X_train, y_train)

# Fazendo previsões com os dados de teste
y_pred_arvore = modelo_arvore.predict(X_test)

# Calculando a acurácia
acuracia_arvore = accuracy_score(y_test, y_pred_arvore)

print("\nResultado da Árvore de Decisão:")
print(f"Acurácia: {acuracia_arvore:.2%}")



