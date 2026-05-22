import os
import sys
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Añadir el directorio raíz al path de Python para poder importar 'shared'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.data_loader import load_and_preprocess_data

def train_and_evaluate_knn(n_neighbors=5, weights='uniform'):
    """
    Carga los datos, entrena un clasificador KNN y evalúa sus resultados.
    """
    print(f"--- Entrenando KNN con n_neighbors={n_neighbors}, weights='{weights}' ---")
    
    # 1. Cargar datos preprocesados y normalizados (crucial para KNN ya que se basa en distancias)
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(scale_features=True)
    
    # 2. Inicializar y entrenar el modelo
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    knn.fit(X_train, y_train)
    
    # 3. Realizar predicciones
    y_pred = knn.predict(X_test)
    y_pred_proba = knn.predict_proba(X_test)[:, 1]  # Probabilidad de clase positiva (enfermo)
    
    # 4. Evaluación
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Exactitud (Accuracy): {accuracy:.4f}")
    print("\nMatriz de Confusión:")
    print(conf_matrix)
    print("\nReporte de Clasificación:")
    print(report)
    
    return knn, accuracy, conf_matrix, report

def tune_knn():
    """
    Ejemplo simple de ajuste de hiperparámetros probando diferentes valores de K.
    """
    print("--- Iniciando Ajuste de Hiperparámetros (K) ---")
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data(scale_features=True)
    
    best_k = 1
    best_acc = 0
    results = []
    
    # Probar valores impares de K para evitar empates
    for k in [1, 3, 5, 7, 9, 11, 13, 15]:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        acc = accuracy_score(y_test, knn.predict(X_test))
        results.append((k, acc))
        print(f"K = {k:2d} -> Exactitud en prueba: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_k = k
            
    print(f"\nMejor K encontrado: {best_k} con exactitud de {best_acc:.4f}")
    return best_k

if __name__ == '__main__':
    # 1. Ajustar el valor de K
    best_k = tune_knn()
    print()
    # 2. Entrenar el modelo final con el mejor K
    train_and_evaluate_knn(n_neighbors=best_k)
