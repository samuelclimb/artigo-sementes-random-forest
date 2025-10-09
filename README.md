# Algoritmos de predição BIAS-SEEDs

Este repositório agrupa scripts para treinar e avaliar classificadores de vigor de sementes usando dados de bioimpedância.

Dois scripts principais estão presentes (ou foram preparados para inclusão):

- ml_preditivo_bioimpedancia.py  versão original, utiliza apenas as três features de impedância/frequência.
- ml_preditivo_allfeatures.py  versão estendida que tenta aproveitar features adicionais quando disponíveis (por exemplo 
eg. phase / ° e cs / f).

## Pré-requisitos

- Python 3.8+ (testado com 3.11)
- Dependências principais (instale com pip):
  - pandas
  - numpy
  - scikit-learn
  - imbalanced-learn
  - xgboost
  - matplotlib

Crie um ambiente virtual e instale as dependências antes de rodar os scripts:

`powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib
`

## Dataset

Coloque bioimpedance_dataset.xlsx na raiz do repositório (já incluído aqui). Os scripts esperam pelo menos as colunas:

- z' / ohm
- z'' / ohm
- frequência / hz
- deg. phase / °
- cs / f
- classificacao_vigor (target)


## ml_preditivo_allfeatures.py

Resumo do fluxo:

1. Lê o dataset (.xlsx ou .csv como fallback).
2. Verifica quais das features estendidas existem no arquivo e usa apenas as presentes.
3. Remove linhas com NaNs (ou realiza imputação dependendo da versão do script) nas features selecionadas e no target.
4. Escala, codifica target, aplica SMOTE em pipeline e executa CV e treino final  igual ao script original.

Como executar:

`powershell
python ml_preditivo_allfeatures.py
`

Observações:

- Se adicionar mais features e perder muitas amostras por causa de NaNs, considere implementar imputação (por exemplo SimpleImputer(strategy='median')) antes de dar dropnas.
- SMOTE está configurado dentro do pipeline para evitar vazamento de dados durante a validação.
- O script estendido imprime avisos se alguma coluna esperada não estiver presente.

## Saídas

- Métricas (accuracy, classification report) são impressas no terminal.
- Curvas ROC são plotadas e salvas como imagens.




