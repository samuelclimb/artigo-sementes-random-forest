import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import os

# Lê o arquivo Excel
excel_path = os.path.join(os.path.dirname(__file__), 'bioimpedance_dataset.xlsx')
df = pd.read_excel(excel_path)

# Seleciona colunas relevantes
features = ["z' / ohm", "z'' / ohm", "freq / hz"]
# Alvo: classificação_vigor (ou ajuste para outro alvo se quiser)
target = "classificacao_vigor"

# Remove linhas com dados faltantes
df = df.dropna(subset=features + [target])


# Normaliza os dados
scaler = StandardScaler()
X = scaler.fit_transform(df[features])
y = df[target].astype(str)

# Codifica alvo para XGBoost
le = LabelEncoder()
y_num = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_num, test_size=0.25, stratify=y_num, random_state=42)

# Modelos
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
}



kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print('\n===== Validação Cruzada (5 folds com SMOTE em pipeline) =====')
for name, model in models.items():
    print(f'\nModelo: {name}')
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('model', model)
    ])
    scores = cross_validate(pipeline, X, y_num, cv=kf, scoring='accuracy', return_train_score=False)
    print(f'Acurácia média: {scores["test_score"].mean():.3f} (+/- {scores["test_score"].std():.3f})')

plt.figure(figsize=(7,6))
for name, model in models.items():
    print(f'\nTreinando modelo final: {name}')
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('model', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    if hasattr(pipeline.named_steps['model'], "predict_proba"):
        y_proba = pipeline.named_steps['model'].predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")
    else:
        auc = None

    print(f"Acurácia: {accuracy_score(y_test, y_pred):.3f}")
    if auc is not None:
        print(f"ROC AUC: {auc:.3f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC - Modelos com SMOTE')
plt.legend()
plt.show()
