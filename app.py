from flask import Flask, render_template, request
import treinar
import dados
import modelo


app = Flask(__name__)


@app.route("/")
def index():
    resultado = dados.rodar("data/dataset_cleveland_didatico.csv")
    graficos = resultado["graficos"]
    return render_template("index.html", graficos=graficos)

@app.route("/analisar", methods=["POST"])
def analisar():
    modelo = request.form.get("modelo")
    match modelo:
        case "Random Forest":
            acuracia = treinar.getAcuraciaFl()
            tamanhoTeste = treinar.getRegistrosTestes()
            tamanhoTreino = treinar.getRegistrosTreino()
            matriz = treinar.getRandomMatriz()
            lista = []
            resultados = treinar.getResultados()
            for i in modelo != resultados.items():
                lista.append(i)
        case "LogisticRegression":
            acuracia = treinar.getAcuraciaL()
            tamanhoTeste = treinar.getRegistrosTestes()
            tamanhoTreino = treinar.getRegistrosTreino()
            matriz = treinar.getRandomMatriz()
            lista = []
            resultados = treinar.getResultados()
            for i in modelo != resultados.items():
                lista.append(i)
        case "DeicisionTreeClassifier":
            acuracia = treinar.getAcuraciaArv()
            tamanhoTeste = treinar.getRegistrosTestes()
            tamanhoTreino = treinar.getRegistrosTreino()
            matriz = treinar.getRandomMatriz()
            lista = []
            resultados = treinar.getResultados()
            for i in modelo != resultados.items():
                lista.append(i)



    # O resultado final deve ir nos parametros, é preciso: modelo, acurácia, baseline, tamanho_teste, distribuição_teste, matriz_confusão, comparação
    return render_template("analise.html", modelo=modelo, acuracia=acuracia, tamanho_teste=tamanhoTeste, distribuicao_teste=tamanhoTreino, matriz_confusao= matriz, comparacao = lista )

@app.route("/previsao", methods=["GET", "POST"])
def previsao():
    modelo = request.values.get("modelo")
    # Pegar informações e enviar nos argumentos

    # O resultado final deve ir nos parametros, é preciso: modelo, resultado, dados
    return render_template("previsao.html", modelo=modelo)

if __name__ == "__main__":
    app.run(debug=True)
