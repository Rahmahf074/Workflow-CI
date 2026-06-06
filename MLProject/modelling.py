import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--n_estimators', type=int, default=200)
parser.add_argument('--max_depth', type=int, default=10)
parser.add_argument('--min_samples_split', type=int, default=5)
args = parser.parse_args()

dagshub.init(
    repo_owner='Rahmahf074',
    repo_name='Eksperimen_SML_RahmaHidayatiFitrah',
    mlflow=True
)

df_train = pd.read_csv('MLProject/diabetes_preprocessing.csv')
df_test  = pd.read_csv('MLProject/diabetes_test.csv')

X_train = df_train.drop('Outcome', axis=1)
y_train = df_train['Outcome']
X_test  = df_test.drop('Outcome', axis=1)
y_test  = df_test['Outcome']

print(f" Train set: {X_train.shape} | Test set: {X_test.shape}")

with mlflow.start_run(run_name="RandomForest_Diabetes_CI"):

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)

    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_param("min_samples_split", args.min_samples_split)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    os.makedirs("artifacts", exist_ok=True)
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix - CI Run (Diabetes)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('artifacts/confusion_matrix.png', dpi=150)
    plt.close()
    mlflow.log_artifact('artifacts/confusion_matrix.png')

    feat_imp = pd.Series(
        model.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    feat_imp.plot(kind='bar', color='steelblue')
    plt.title('Feature Importance - CI Run (Diabetes)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('artifacts/feature_importance.png', dpi=150)
    plt.close()
    mlflow.log_artifact('artifacts/feature_importance.png')

    mlflow.sklearn.log_model(model, "model")

    print(f" Training selesai!")
    print(f"   Accuracy : {acc:.4f}")
    print(f"   F1 Score : {f1:.4f}")
