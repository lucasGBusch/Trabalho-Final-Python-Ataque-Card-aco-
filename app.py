from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analisar", methods=["POST"])
def analisar():
    modelo = request.form.get("modelo")
    # Pegar informações e enviar nos argumentos
    # O resultado final deve ir nos parametros, é preciso: modelo, acurácia, baseline, tamanho_teste, distribuição_teste, matriz_confusão, comparação
    return render_template("analise.html", modelo=modelo)

@app.route("/previsao", methods=["GET", "POST"])
def previsao():
    modelo = request.values.get("modelo")
    # Pegar informações e enviar nos argumentos
    # O resultado final deve ir nos parametros, é preciso: modelo, resultado, dados
    return render_template("previsao.html", modelo=modelo)

if __name__ == "__main__":
    app.run(debug=True)
