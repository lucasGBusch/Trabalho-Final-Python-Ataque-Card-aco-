from flask import Flask, render_template, request
import treinar
import dados
import modelo
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


app = Flask(__name__)


@app.route("/")
def index():
    resultado = dados.rodar("data/dataset_cleveland_didatico.csv")
    graficos = resultado["graficos"]
    return render_template(
        "index.html",
        graficos=graficos,
        diagnostico=resultado["diagnostico"],
        tamanho_depois=len(resultado["df_limpo"]),
        log=resultado["log"],
        achados=resultado["analise"]["achados"],
        amostra=resultado["df_limpo"].head(5).to_dict(orient="records"),
        colunas=list(resultado["df_limpo"].columns),
    )

@app.route("/analisar", methods=["POST"])
def analisar():
    modelo = request.form.get("modelo")
    match modelo:
        case "RandomForestClassifier":
            nome = "Random Forest"
            acuracia = treinar.getAcuraciaFl()
        case "LogisticRegression":
            nome = "Regressão Logística"
            acuracia = treinar.getAcuraciaL()
        case "DecisionTreeClassifier":
            nome = "Árvore de Decisão"
            acuracia = treinar.getAcuraciaArv()
        case _:
            return "Modelo inválido", 400

    resultados = treinar.getResultados()   # {nome: y_pred}
    y_teste = treinar.y_test

    tamanhoTeste = treinar.getRegistrosTestes()[0]
    distribuicao = y_teste.value_counts().to_dict()
    baseline = y_teste.value_counts(normalize=True).max()
    matriz = confusion_matrix(y_teste, resultados[nome], labels=[0, 1]).tolist()

    lista = []
    for n, pred in resultados.items():
        lista.append({
            "modelo": n,
            "acuracia": f"{accuracy_score(y_teste, pred):.2%}",
            "precisao": f"{precision_score(y_teste, pred, zero_division=0):.2%}",
            "recall": f"{recall_score(y_teste, pred, zero_division=0):.2%}",
            "f1": f"{f1_score(y_teste, pred, zero_division=0):.2%}",
        })

    # O resultado final deve ir nos parametros, é preciso: modelo, acurácia, baseline, tamanho_teste, distribuição_teste, matriz_confusão, comparação
    return render_template(
        "analise.html",
        modelo=modelo,
        acuracia=f"{acuracia:.2%}",
        baseline=f"{baseline:.2%}",
        tamanho_teste=tamanhoTeste,
        distribuicao_teste=distribuicao,
        matriz_confusao=matriz,
        comparacao=lista,
    )
@app.route("/previsao", methods=["GET", "POST"])
def previsao():
    modelo = request.values.get("modelo")
    resultado = None
    dados = {}

    if request.method == "POST":
        # Mesmos nomes usados no match da rota /analisar
        pipelines = {
            "RandomForestClassifier": treinar.modelo_floresta,
            "LogisticRegression": treinar.modelo_logistico,
            "DecisionTreeClassifier": treinar.modelo_arvore,
        }
        campos = [
            "idade", "sexo", "tipo_dor_peito", "pressao_repouso", "colesterol",
            "acucar_sangue_jejum", "ecg_repouso", "freq_cardiaca_max", "angina_exercicio",
        ]

        # Campos desmarcados não são enviados pelo navegador: ficam NaN e o pipeline imputa
        ativos = request.form.getlist("usar_campos")
        linha = {c: float("nan") for c in treinar.X_train.columns}
        dados = {c: "" for c in campos}  # "" -> template mostra "Imputado pelo modelo"

        for c in ativos:
            if c in campos:
                bruto = request.form.get(c, "").strip()
                dados[c] = bruto
                if bruto != "":
                    linha[c] = float(bruto)

        pipeline = pipelines.get(modelo)
        if pipeline is None:
            resultado = "Modelo não reconhecido"
        else:
            entrada = pd.DataFrame([linha], columns=treinar.X_train.columns)
            classe = int(pipeline.predict(entrada)[0])
            prob = pipeline.predict_proba(entrada)[0][1]
            diagnostico = "Com doença cardíaca" if classe == 1 else "Sem doença cardíaca"
            resultado = f"{diagnostico} (probabilidade: {prob:.1%})"

    # O resultado final deve ir nos parametros, é preciso: modelo, resultado, dados
    return render_template("previsao.html", modelo=modelo, resultado=resultado, dados=dados)

if __name__ == "__main__":
    app.run(debug=True)
