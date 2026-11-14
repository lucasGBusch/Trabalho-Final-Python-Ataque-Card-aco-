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

