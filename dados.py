# dados.py
# Leitura do CSV, validação, limpeza, análise e gráficos.
# A limpeza é feita numa cópia do DataFrame, o arquivo original não muda.

import io
import base64

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # para o matplotlib funcionar dentro do Flask
import matplotlib.pyplot as plt


# colunas usadas pelo modelo (X)
COLUNAS_X = ["idade", "sexo", "tipo_dor_peito", "pressao_repouso",
             "colesterol", "freq_cardiaca_max", "angina_exercicio"]

# o CSV precisa ter as colunas de X e a coluna num (diagnóstico de 0 a 4)
COLUNAS_OBRIGATORIAS = COLUNAS_X + ["num"]

# coluna que a gente cria: 1 se num > 0, senão 0
COLUNA_ALVO = "doenca"

# valores mínimos e máximos aceitos em cada coluna
FAIXAS_VALIDAS = {
    "idade": (18, 100),
    "sexo": (0, 1),
    "tipo_dor_peito": (1, 4),
    "pressao_repouso": (70, 250),
    "colesterol": (100, 700),
    "freq_cardiaca_max": (60, 230),
    "angina_exercicio": (0, 1),
    "num": (0, 4),
}

NOMES_CLASSES = {0: "Sem doença", 1: "Com doença"}


class ErroCSV(Exception):
    # erro com uma mensagem que pode ser mostrada para o usuário
    pass


def ler_csv(caminho):
    try:
        with open(caminho, encoding="utf-8-sig") as arquivo:
            texto = arquivo.read()
    except UnicodeDecodeError:
        raise ErroCSV("O arquivo não é um CSV válido.")

    if texto.strip() == "":
        raise ErroCSV("O arquivo está vazio.")

    # descobre o separador olhando o cabeçalho
    cabecalho = texto.splitlines()[0]
    if cabecalho.count(";") > cabecalho.count(","):
        separador = ";"
    else:
        separador = ","

    try:
        df = pd.read_csv(io.StringIO(texto), sep=separador, na_values=["?", ""])
    except pd.errors.ParserError:
        raise ErroCSV("Algumas linhas do arquivo têm mais colunas que o cabeçalho.")

    if len(df) == 0:
        raise ErroCSV("O arquivo só tem o cabeçalho, sem nenhum paciente.")
    if len(df.columns) == 1:
        raise ErroCSV("Só foi encontrada uma coluna. O separador deve ser vírgula ou ponto e vírgula.")

    df.columns = df.columns.str.strip().str.lower()

    faltando = []
    for coluna in COLUNAS_OBRIGATORIAS:
        if coluna not in df.columns:
            faltando.append(coluna)
    if faltando:
        raise ErroCSV("Faltam as colunas: " + ", ".join(faltando))

    return df


def diagnosticar(df):
    # situação da base antes da limpeza
    numeros = df[COLUNAS_OBRIGATORIAS].apply(pd.to_numeric, errors="coerce")

    impossiveis = {}
    for coluna, (minimo, maximo) in FAIXAS_VALIDAS.items():
        fora = numeros[coluna].notna() & ~numeros[coluna].between(minimo, maximo)
        if fora.sum() > 0:
            impossiveis[coluna] = int(fora.sum())

    ausentes = {}
    for coluna in df.columns:
        n = int(df[coluna].isna().sum())
        if n > 0:
            ausentes[coluna] = n

    return {
        "linhas": len(df),
        "colunas": len(df.columns),
        "tipos": df.dtypes.astype(str).to_dict(),
        "ausentes": ausentes,
        "duplicatas": int(df.duplicated().sum()),
        "impossiveis": impossiveis,
    }


def limpar(df_original):
    df = df_original.copy()
    log = []  # lista com o que foi feito, para mostrar na página

    # textos no lugar de números viram NaN
    df[COLUNAS_OBRIGATORIAS] = df[COLUNAS_OBRIGATORIAS].apply(pd.to_numeric, errors="coerce")

    # os valores vêm com muitas casas decimais (ex: 141.3762945), deixamos só uma
    df["pressao_repouso"] = df["pressao_repouso"].round(1)
    df["colesterol"] = df["colesterol"].round(1)
    df["freq_cardiaca_max"] = df["freq_cardiaca_max"].round(1)
    log.append("Pressão, colesterol e frequência arredondados para 1 casa decimal.")

    antes = len(df)
    df = df.drop_duplicates()
    log.append(f"{antes - len(df)} linhas duplicadas removidas.")

    # valores fora da faixa: como não tem como saber o valor certo, a linha é removida
    for coluna, (minimo, maximo) in FAIXAS_VALIDAS.items():
        fora = df[coluna].notna() & ~df[coluna].between(minimo, maximo)
        if fora.sum() > 0:
            df = df[~fora]
            log.append(f"{fora.sum()} linhas removidas por '{coluna}' estar fora de {minimo} a {maximo}.")

    # sem o diagnóstico a linha não serve para treinar
    sem_num = df["num"].isna().sum()
    if sem_num > 0:
        df = df.dropna(subset=["num"])
        log.append(f"{sem_num} linhas removidas por não terem diagnóstico (num).")

    # os ausentes de X não são preenchidos aqui. Quem preenche é o modelo,
    # com a mediana calculada só nos dados de treino.
    for coluna in COLUNAS_X:
        n = df[coluna].isna().sum()
        if n > 0:
            log.append(f"{n} valores ausentes em '{coluna}' mantidos (o modelo preenche com a mediana).")

    df[COLUNA_ALVO] = (df["num"] > 0).astype(int)
    log.append("Criada a coluna 'doenca' (1 se num > 0, senão 0).")

    if len(df) < 10:
        raise ErroCSV(f"Depois da limpeza sobraram só {len(df)} pacientes. O mínimo é 10.")
    if df[COLUNA_ALVO].nunique() < 2:
        raise ErroCSV("O arquivo só tem pacientes com doença ou só sem doença.")

    return df.reset_index(drop=True), log


def numero_br(valor):
    # 75.9 -> "75,9"
    return f"{valor:.1f}".replace(".", ",")


def analisar(df):
    pct_doenca = df[COLUNA_ALVO].mean() * 100

    medias = df.groupby(COLUNA_ALVO)[["idade", "pressao_repouso", "colesterol", "freq_cardiaca_max"]].mean()
    medias = medias.round(1).rename(index=NOMES_CLASSES)
    com = medias.loc["Com doença"]
    sem = medias.loc["Sem doença"]

    pct_angina = df.groupby("angina_exercicio")[COLUNA_ALVO].mean() * 100
    pct_sexo = df.groupby("sexo")[COLUNA_ALVO].mean() * 100

    achados = [
        f"{numero_br(pct_doenca)}% dos pacientes têm doença. A base é desbalanceada, "
        f"por isso só a acurácia não é suficiente para avaliar o modelo.",

        f"Quem tem doença atinge uma frequência cardíaca máxima menor: em média "
        f"{numero_br(com['freq_cardiaca_max'])} bpm, contra {numero_br(sem['freq_cardiaca_max'])} bpm de quem não tem.",

        f"Os doentes são um pouco mais velhos ({numero_br(com['idade'])} anos contra {numero_br(sem['idade'])}). "
        f"Já o colesterol e a pressão mudam pouco entre os dois grupos.",

        f"Entre quem sente angina no exercício, {numero_br(pct_angina.get(1, 0))}% têm doença. "
        f"Entre quem não sente, são {numero_br(pct_angina.get(0, 0))}%.",

        f"{numero_br(pct_sexo.get(1, 0))}% dos homens têm doença, contra {numero_br(pct_sexo.get(0, 0))}% das mulheres.",
    ]

    contagem = df[COLUNA_ALVO].value_counts().rename(NOMES_CLASSES).to_dict()

    return {
        "contagem": contagem,
        "prop_doenca": round(pct_doenca, 1),
        "medias": medias,
        "achados": achados,
    }


# ---------------- gráficos ----------------
# cada função devolve a imagem em base64 para o HTML mostrar com
# <img src="data:image/png;base64,...">

AZUL = "#2a78d6"
LARANJA = "#eb6834"
CINZA = "#666666"


def figura_para_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode()


def deixar_limpo(ax, titulo, eixo_x, eixo_y):
    # título, nomes dos eixos e tira as bordas que só poluem o gráfico
    ax.set_title(titulo, loc="left", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel(eixo_x, color=CINZA)
    ax.set_ylabel(eixo_y, color=CINZA)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(colors=CINZA, length=0)
    ax.set_axisbelow(True)


def grafico_distribuicao_idade(df):
    # pirâmide etária: sem doença para a esquerda, com doença para a direita
    inicio = int(df["idade"].min()) // 5 * 5
    fim = int(df["idade"].max()) // 5 * 5 + 5
    faixas = pd.cut(df["idade"], bins=range(inicio, fim + 1, 5), right=False)
    tabela = pd.crosstab(faixas, df[COLUNA_ALVO]).reindex(columns=[0, 1], fill_value=0)
    nomes = [f"{f.left}-{f.right - 1}" for f in tabela.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    # o lado "sem doença" usa valores negativos para ir para a esquerda
    sem = ax.barh(nomes, -tabela[0], color=AZUL, height=0.7, label="Sem doença")
    com = ax.barh(nomes, tabela[1], color=LARANJA, height=0.7, label="Com doença")
    # número na ponta de cada barra (sem mostrar os zeros)
    ax.bar_label(sem, labels=[v if v > 0 else "" for v in tabela[0]], padding=4, fontsize=9, color=CINZA)
    ax.bar_label(com, labels=[v if v > 0 else "" for v in tabela[1]], padding=4, fontsize=9, color=CINZA)
    ax.axvline(0, color="#999999", linewidth=0.8)

    # eixo x simétrico e sem o sinal de menos
    limite = tabela.values.max() * 1.15
    ax.set_xlim(-limite, limite)
    marcas = [m for m in ax.get_xticks() if abs(m) <= limite]
    ax.set_xticks(marcas)
    ax.set_xticklabels([str(abs(int(m))) for m in marcas])

    deixar_limpo(ax, "Distribuição da idade por diagnóstico", "Número de pacientes", "Faixa etária")
    ax.grid(axis="x", color="#eeeeee")
    ax.legend(frameon=False, loc="lower right")
    return figura_para_base64(fig)


def grafico_comparacao_dor(df):
    tabela = pd.crosstab(df["tipo_dor_peito"], df[COLUNA_ALVO]).reindex(columns=[0, 1], fill_value=0)
    tabela = tabela.rename(index={1: "Angina típica", 2: "Angina atípica",
                                  3: "Dor não anginosa", 4: "Assintomático"})
    tabela = tabela.rename(columns=NOMES_CLASSES)

    fig, ax = plt.subplots(figsize=(8, 5))
    tabela.plot(kind="bar", ax=ax, color=[AZUL, LARANJA], width=0.7, rot=0, edgecolor="white")
    for barras in ax.containers:
        ax.bar_label(barras, padding=3, fontsize=9, color=CINZA)

    deixar_limpo(ax, "Pacientes por tipo de dor no peito", "Tipo de dor no peito", "Número de pacientes")
    ax.grid(axis="y", color="#eeeeee")
    ax.legend(title="", frameon=False)
    return figura_para_base64(fig)


def grafico_relacao_frequencia(df):
    sem_doenca = df[df[COLUNA_ALVO] == 0]["freq_cardiaca_max"].dropna()
    com_doenca = df[df[COLUNA_ALVO] == 1]["freq_cardiaca_max"].dropna()
    grupos = [sem_doenca, com_doenca]
    cores = [AZUL, LARANJA]

    fig, ax = plt.subplots(figsize=(8, 5))
    caixas = ax.boxplot(grupos, tick_labels=["Sem doença", "Com doença"], widths=0.5,
                        patch_artist=True, showfliers=False, showcaps=False,
                        medianprops={"color": "black", "linewidth": 2},
                        whiskerprops={"color": "#999999"})
    for caixa, cor in zip(caixas["boxes"], cores):
        caixa.set_facecolor(cor)
        caixa.set_alpha(0.25)
        caixa.set_edgecolor(cor)

    # cada paciente vira um ponto; o "espalhamento" na horizontal é só para
    # os pontos não ficarem um em cima do outro
    aleatorio = np.random.default_rng(42)
    for i, (valores, cor) in enumerate(zip(grupos, cores)):
        x = i + 1 + aleatorio.uniform(-0.12, 0.12, len(valores))
        ax.scatter(x, valores, s=10, color=cor, alpha=0.5, zorder=3)
        ax.text(i + 1.3, valores.median(), f"mediana: {valores.median():.0f} bpm",
                va="center", fontsize=9, color=CINZA)

    deixar_limpo(ax, "Frequência cardíaca máxima por diagnóstico", "Diagnóstico", "Frequência máxima (bpm)")
    ax.grid(axis="y", color="#eeeeee")
    return figura_para_base64(fig)


def rodar(caminho):
    # faz tudo de uma vez: lê, diagnostica, limpa, analisa e gera os gráficos.
    # Se o arquivo tiver problema, lança ErroCSV com a mensagem para o usuário.
    df = ler_csv(caminho)
    diagnostico = diagnosticar(df)
    df_limpo, log = limpar(df)

    return {
        "df_original": df,
        "df_limpo": df_limpo,
        "diagnostico": diagnostico,
        "log": log,
        "analise": analisar(df_limpo),
        "graficos": {
            "idade": grafico_distribuicao_idade(df_limpo),
            "dor": grafico_comparacao_dor(df_limpo),
            "frequencia": grafico_relacao_frequencia(df_limpo),
        },
    }


# teste: rodar "python dados.py" no terminal
if __name__ == "__main__":
    df = ler_csv("data/dataset_cleveland_didatico.csv")
    print(diagnosticar(df))

    df_limpo, log = limpar(df)
    print(f"\n{len(df)} linhas antes da limpeza e {len(df_limpo)} depois")
    for passo in log:
        print("-", passo)

    print("\nAchados:")
    for achado in analisar(df_limpo)["achados"]:
        print("-", achado)
