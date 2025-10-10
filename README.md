# ml_preditivo_allfeatures.py

Este README descreve especificamente o script ml_preditivo_allfeatures.py 

## Objetivo

Treinar e avaliar classificadores para prever classificacao_vigor usando (z' / ohm, z'' / ohm, / frequencia / hz / deg. phase / ° e cs / f).
O foco é avaliar se a inclusão de mais sinais eletrofísicos melhora a capacidade preditiva sem introduzir vazamento de dados ou viés.

## Dependências

Instale as dependências num ambiente virtual:

`powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib openpyxl
`

( openpyxl é necessário para ler arquivos .xlsx )

## Entradas

- Bioimpedance_dataset.xlsx (coloque na raiz do repositório). O script também faz fallback para bioimpedance_dataset.csv caso o .xlsx não exista.
- Colunas esperadas (pelo menos):
  - z' / ohm
  - z'' / ohm
  - Frequência / hz
  - classificacao_vigor (target)

- Colunas opcionais (serão usadas se existirem):

  - deg. phase / °
  - cs / f

O script verifica automaticamente quais colunas estão presentes e usa somente as que existem.

## Fluxo do algoritmo (resumo técnico)

1. Lê o dataset (.xlsx preferencialmente, .csv como fallback).
2. Verifica a lista de features desejadas e filtra para as presentes no arquivo.
3. Remove linhas com valores faltantes nas features escolhidas e no target (ou, na versão que desejar, implemente imputação antes do drop).
4. Normaliza as features com StandardScaler.
5. Codifica o target com LabelEncoder (gera rótulos inteiros 0..K-1).
6. Divide em treino/teste com estratificação (25% teste).
7. Para validação cruzada usa StratifiedKFold (5 folds) e Pipeline que engloba SMOTE (oversampling) seguido do modelo  assim o oversampling é aplicado somente no treinamento de cada fold (sem vazamento).
8. Testa os modelos: Random Forest, Logistic Regression e XGBoost. Para modelos que retornam probabilidades calcula ROC AUC; em multiclass usa AUC macro (one-vs-rest).
9. Treina o modelo final e imprime relatórios (classification report, accuracy) e plota/salva curvas ROC quando aplicável.

## Como rodar

No repositório (após ativar o venv e instalar dependências):

`powershell
python ml_preditivo_allfeatures.py
`

O script imprime progresso e resultados no terminal. As figuras ROC são exibidas e salvas (se disponível back-end gráfico).

## Onde os resultados ficam

- Gráficos ROC: salvos no diretório do script com nome contendo 
oc_comparacao_modelos.
- Relatórios: impressos no terminal; você pode redirecionar a saída para um arquivo se quiser arquivar.

## Boas práticas e notas

- SMOTE está aplicado dentro da pipeline para evitar data leakage. Não mova o SMOTE para fora do pipeline se pretende fazer validação correta.
- Para XGBoost, passe y já codificado (inteiros). O parâmetro use_label_encoder é obsoleto em versões recentes do XGBoost; recomendamos não usá-lo (o aviso observado é informativo apenas).


## Sugestões de extensão

- Adicionar imputação configurável (por exemplo, via argumento de linha de comando ou variável no topo do script).
- Salvar métricas (por modelo) em CSV/XLSX para comparações reproduzíveis.
- Incluir um notebook de exemplo com visualizações detalhadas por classe.

--


