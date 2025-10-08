# ml_preditivo_bioimpedancia.py

Este repositório contém o arquivo ml_preditivo_bioimpedancia.py, um script Python para treinar e avaliar modelos de classificação de vigor de sementes a partir de dados de bioimpedância.

## Objetivo
O script treina classificadores (Random Forest, Regressão Logística, XGBoost) para prever a coluna classificacao_vigor usando exclusivamente as três colunas de impedância/frequência:

- Z' / ohm
- Z'' / ohm
- Freq / hz

O objetivo é comparar desempenho entre modelos e produzir curvas ROC e relatórios de classificação.

## Pré-requisitos
- Python 3.8+ (testado em 3.11)
- Pacotes (instale com pip install -r requirements.txt):
  - pandas
  - numpy
  - scikit-learn
  - imbalanced-learn
  - xgboost
  - matplotlib

> Observação: se não houver 
equirements.txt nesta pasta, instale as bibliotecas acima manualmente.

## Entradas
Coloque o arquivo bioimpedance_dataset.xlsx na raiz do repositório (já presente neste repositório). O script espera que a planilha tenha, no mínimo, as colunas mencionadas em "Objetivo" e a coluna alvo classificacao_vigor.

## Fluxo do algoritmo (resumo)
1. Lê ioimpedance_dataset.xlsx (mesmo diretório do script).
2. Seleciona as colunas z' / ohm, z'' / ohm, req / hz e a coluna alvo classificacao_vigor.
3. Remove linhas com valores faltantes nas colunas selecionadas.
4. Normaliza (StandardScaler) as três features.
5. Codifica o alvo com LabelEncoder.
6. Divide em treino/teste (25% teste) com estratificação.
7. Executa Validação Cruzada estratificada (5 folds) usando pipelines com SMOTE (oversampling) seguido do modelo  isto evita vazamento de dados na validação.
8. Treina cada modelo no conjunto de treino (com SMOTE dentro do pipeline), avalia no teste e faz:
   - impressão da acurácia e do classification report;
   - cálculo de ROC AUC (quando o modelo fornece probabilidades) e plot da curva ROC.

## Como rodar
1. Ative seu ambiente Python com as dependências instaladas.
2. No diretório deste repositório execute:

`powershell
python ml_preditivo_bioimpedancia.py
`

O script imprimirá métricas no terminal e exibirá (e salva) a figura de comparação ROC.

## Saídas
- Curva ROC exibida em tela e salva como figura (nome depende do script).
- Relatórios de classificação e acurácias impressos no terminal.

## Observações e boas práticas
- O script remove linhas com NaNs nas colunas usadas; se você quiser usar mais features sem perder classes, implemente imputação antes.
- SMOTE está corretamente aplicado dentro do pipeline para evitar data leakage.
- Se precisar adaptar o alvo ou o conjunto de features, crie uma cópia do script e modifique apenas as variáveis eatures e 	arget.

---

Arquivo focado: ml_preditivo_bioimpedancia.py. Este README explica apenas esse script, conforme solicitado.



