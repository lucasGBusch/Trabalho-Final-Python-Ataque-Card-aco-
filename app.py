from pathlib import Path

from flask import Flask, render_template, request

import dados
from treinar import avaliar_modelo, fazer_previsao, COLUNAS


app = Flask(__name__)

BASE = Path(__file__).resolve().parent


# Página inicial
@app.route("/")
def index():

    caminho = BASE / "data" / "dataset_cleveland_didatico.csv"

    resultado = dados.rodar(caminho)
    graficos = resultado["graficos"]

    return render_template(
        "index.html",
        graficos=graficos
    )


# Página de análise dos modelos
@app.route("/analisar", methods=["POST"])
def analisar():

    # Recebe o modelo selecionado no HTML
    nome_modelo = request.form.get("modelo")

    try:
        # Busca as métricas calculadas no treinar.py
        avaliacao = avaliar_modelo(nome_modelo)

    except ValueError as erro:
        return str(erro), 400

    # Envia os sete parâmetros para o HTML
    return render_template(
        "analise.html",
        **avaliacao
    )


# Página de previsão
@app.route("/previsao", methods=["GET", "POST"])
def previsao():

    # Recebe o nome do modelo
    nome_modelo = request.values.get("modelo")

    # GET: apenas mostra o formulário
    if request.method == "GET":
        return render_template(
            "previsao.html",
            modelo=nome_modelo
        )

    # POST: recebe os campos selecionados
    campos_ativos = request.form.getlist("usar_campos")

    dados = {}
    dados_tela = {}

    # Percorre as nove características
    for coluna in COLUNAS:

        if coluna in campos_ativos:

            # Campo selecionado recebe o valor do HTML
            valor = request.form.get(coluna)
            dados[coluna] = valor
            dados_tela[coluna] = valor if valor is not None else ""

        else:

            # Campo desmarcado recebe None
            dados[coluna] = None
            dados_tela[coluna] = ""

    resultado = None
    erro = None

    try:
        # Executa a função criada no treinar.py
        resultado = fazer_previsao(
            nome_modelo,
            dados,
            campos_ativos
        )

    except ValueError as excecao:
        erro = str(excecao)

    # Envia o resultado ou erro para o HTML
    return render_template(
        "previsao.html",
        modelo=nome_modelo,
        resultado=resultado,
        dados=dados_tela,
        campos_ativos=campos_ativos,
        erro=erro
    ), 400 if erro else 200


if __name__ == "__main__":
    app.run(debug=True)
