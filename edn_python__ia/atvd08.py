from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix
)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Carregar o dataset
data = load_breast_cancer()
X = data.data
y = data.target  # 0 = maligno, 1 = benigno

# 2. Dividir o dataset (70% treino, 30% teste)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3. Treinar o modelo Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 4. Fazer previsões
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Para AUC-ROC

# 5. Calcular métricas
metrics = {
    "Acurácia": accuracy_score(y_test, y_pred),
    "Precisão": precision_score(y_test, y_pred),
    "Recall (Sensibilidade)": recall_score(y_test, y_pred),
    "F1-Score": f1_score(y_test, y_pred),
    "AUC-ROC": roc_auc_score(y_test, y_pred_proba)
}

# 6. Exibir resultados
print("Relatório de Classificação:\n")
print(classification_report(y_test, y_pred, target_names=data.target_names))

print("Métricas Individuais:")
for nome, valor in metrics.items():
    print(f"{nome}: {valor:.4f}")
