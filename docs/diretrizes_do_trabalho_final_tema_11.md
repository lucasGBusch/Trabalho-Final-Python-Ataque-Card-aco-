# Trabalho Final: Aplicação Web com Análise de Dados e Machine Learning
**Disciplina:** Tópicos Especiais em Ciência da Computação
**Valor:** 5,0 pontos
**Equipe:** Grupos de 3 a 4 alunos

---

## 1. O Desafio do Projeto
Construir uma aplicação web completa que conecte interface, análise de dados e aprendizado de máquina. O foco é justificar decisões, interpretar resultados e demonstrar domínio individual do código.

### Tema 11: Doença Cardíaca (Aprovado)
*   **Tipo de Problema:** Classificação
*   **Dataset:** UCI Heart Disease (recorte Cleveland)
*   **Variáveis de Entrada (X):** Idade, pressão em repouso, colesterol e frequência máxima. *(Nota: O grupo explica por que cada variável ajuda a responder à pergunta. IDs e variáveis que revelam o alvo devem ser excluídos).*
*   **Variável Alvo (y):** Presença de doença (`num > 0`). *(Nota: A coluna `num` original não pode entrar no `X`).*
*   **Instrução Específica:** Tratar valores ausentes (obrigatório para este tema) e apresentar como estudo didático.

---

## 2. Escopo Obrigatório
*   **Integração:** Flask e HTML.
*   **Manipulação de Dados:** Upload de arquivo CSV e análise com biblioteca Pandas.
*   **Gráficos:** Pelo menos 3 visualizações relevantes.
*   **Machine Learning:** Teste e comparação de no mínimo **2 modelos** do scikit-learn.
*   **Interface Final:** Avaliação e previsão via formulário no navegador.

*O que NÃO é exigido (Limites):* Selenium/Playwright, mapas/gráficos interativos avançados, escolha de modelos pelo usuário, retreinamento dinâmico, configuração de hiperparâmetros ou Deep Learning.

---

## 3. Fluxo Visual e Requisitos do Sistema

### A. Upload de CSV e Validação
*   **Formato:** O grupo define e explica as colunas aceitas.
*   **Validação:** Verificar se o arquivo está vazio, formato, separador, colunas obrigatórias e valores incompatíveis. Mostrar mensagens amigáveis em caso de erro.
*   **Resultado:** Exibir amostra dos dados, quantidade de registros e colunas. Os gráficos e análises devem atualizar com base no arquivo enviado.

### B. Leitura, Limpeza e Análise (Pandas)
*   **Diagnóstico e Tratamento:** Verificar tipos, duplicatas e valores ausentes (exigência forte do Tema 11). Explicar e justificar exclusões, conversões e preenchimentos.
*   **Análise:** Usar estatísticas e agrupamentos. Mostrar o tamanho da base antes e depois. Explicar ao menos 3 achados sobre o dataset.

### C. Visualizações (Mínimo de 3 Gráficos)
Cada gráfico precisa de título, eixos legíveis e interpretação textual na página.
1.  **Distribuição:** Ex: histograma de idade ou colesterol (explicar concentração e extremos).
2.  **Comparação:** Ex: barras por classe do alvo (explicar diferenças/desequilíbrios entre doentes e não doentes).
3.  **Relação:** Ex: boxplot por classe (relacionar à pergunta do projeto).
*(Nota: Matriz de confusão não conta como um destes 3 gráficos).*

---

## 4. Machine Learning (Classificação)

### Treinamento e Teste
*   **Separação:** Usar `train_test_split` (ex: 80/20) e registrar a proporção e o `random_state`.
*   **Vazamento de Dados:** Ajustar codificações/preenchimentos aprendidos *somente* no conjunto de treino. Não escolher variáveis para decorar o teste.

### Modelos Permitidos (Escolher no mínimo 2 para comparar)
*   Decision Tree Classifier
*   KNN (K-Nearest Neighbors)
*   Logistic Regression
*   Random Forest Classifier
*   Naive Bayes

### Comparação e Avaliação
*   Usar a mesma divisão de dados para todos os modelos testados.
*   **Acurácia (`accuracy_score`):** Informar tamanho do teste, distribuição das classes e comparar com a proporção da classe mais frequente.
*   **Matriz de Confusão (`confusion_matrix`):** Mostrar nomes das classes, eixos real/previsto e explicar os erros (falsos positivos/negativos) no contexto médico da doença cardíaca.
*   **Escolha Final:** Justificar o modelo escolhido para a aplicação final com base nesses resultados.

---

## 5. Previsão Integrada (Flask + HTML)
*   **Formulário (HTML):** Criar campos para as variáveis `X` (idade, pressão, colesterol, freq. máxima) com rótulos e unidades adequadas.
*   **Processamento (Flask):** Validar valores, construir uma linha com os dados do form e aplicar **exatamente o mesmo preparo** (limpeza/transformação) usado no treinamento.
*   **Saída:** Executar o `predict()` (do modelo fixo carregado ou treinado no app) e exibir se há previsão de doença ou não na tela.

---

## 6. Estrutura Sugerida do Projeto
```text
projeto/
├── app.py              # rotas e integração Flask
├── dados.py            # leitura, limpeza e análise (Pandas)
├── modelo.py           # preparação e previsão
├── treinar.py          # script separado para treino e avaliação (opcional)
├── templates/          
│   ├── index.html
│   ├── analise.html
│   └── previsao.html
├── static/             # CSS, imagens
├── data/               # CSV de exemplo
├── models/             # modelo salvo (ex: .pkl), se houver persistência
├── requirements.txt    # dependências do projeto
└── README.md           # instruções de instalação e uso
```

---

## 7. Entregáveis
1.  **Repositório Completo:** Link do GitHub enviado no Blackboard. Deve conter código organizado/comentado, HTML, `requirements.txt`. O commit avaliado será o último feito dentro do prazo.
2.  **Dados:** CSV de exemplo incluído ou instrução reproduzível para obtê-lo.
3.  **README:** Deve explicar instalação, execução, esquema do CSV, dicionário de dados, X/y, modelos testados, comparação, escolha final, métricas, limitações e listar a participação/responsabilidade de cada integrante.
4.  **Autoria:** Citar fontes, códigos adaptados e ferramentas (incluindo IA).

---

## 8. Critérios de Avaliação (Total: 5,0 pontos)

| Critério | Pontos | Detalhes |
| :--- | :---: | :--- |
| **Aplicação Flask** | 0,75 | Rotas, upload e mensagens funcionam. |
| **Dados e Análise** | 0,75 | Limpeza justificada, análises corretas e achados interpretados. |
| **Visualizações** | 0,50 | Três gráficos relevantes e legíveis. |
| **Machine Learning** | 1,25 | X/y, treino/teste, comparação de $\ge 2$ modelos, avaliação e escolha justificada. |
| **Previsão Integrada** | 0,50 | Formulário validado gera previsão real no navegador. |
| **Organização** | 0,25 | Código organizado, comentado e README reproduzível. |
| **Apresentação/Defesa**| 1,00 | Demonstração (0,25) e Defesa Individual (0,75). |

### Regra Crítica para a Defesa Individual
Na apresentação, o grupo deve mostrar o app rodando (upload, gráficos, avaliação e nova previsão). **Cada integrante** precisará explicar, com as próprias palavras, o código, variáveis, como os modelos funcionam e as métricas. 
**Atenção:** Quem não souber explicar o código na defesa individual terá a nota do trabalho **zerada** ($0,0$ de 5,0).