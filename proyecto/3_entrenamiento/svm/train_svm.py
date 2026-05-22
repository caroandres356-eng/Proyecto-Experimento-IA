import os
import sys
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Añadir el directorio raíz al path de Python para importar 'shared'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from shared.data_loader import load_and_preprocess_data

def train_and_evaluate_svm(C=1.0, kernel='rbf', gamma='scale'):
    """
    Entrena un clasificador de Vectores de Soporte (SVC) y evalúa los resultados.
    """
    print(f"--- Entrenando SVM con C={C}, kernel='{kernel}', gamma='{gamma}' ---")
    
    # 1. Cargar datos preprocesados
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data(scale_features=True)
    
    # 2. Inicializar y entrenar el modelo
    svm = SVC(C=C, kernel=kernel, gamma=gamma, probability=True, random_state=42)
    svm.fit(X_train, y_train)
    
    # 3. Realizar predicciones
    y_pred = svm.predict(X_test)
    
    # 4. Evaluación
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Exactitud (Accuracy): {accuracy:.4f}")
    print("\nMatriz de Confusión:")
    print(conf_matrix)
    print("\nReporte de Clasificación:")
    print(report)
    
    return svm, accuracy, conf_matrix, report

def tune_svm():
    """
    Prueba una cuadrícula simple de hiperparámetros (C y tipo de Kernel).
    """
    print("--- Ajuste de Hiperparámetros para SVM ---")
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data(scale_features=True)
    
    best_acc = 0
    best_params = {}
    
    # Evaluar diferentes combinaciones
    for kernel in ['linear', 'rbf']:
        for c_val in [0.1, 1.0, 10.0]:
            svm = SVC(C=c_val, kernel=kernel, random_state=42)
            svm.fit(X_train, y_train)
            acc = accuracy_score(y_test, svm.predict(X_test))
            print(f"Kernel = {kernel:6s} | C = {c_val:4.1f} -> Exactitud en prueba: {acc:.4f}")
            
            if acc > best_acc:
                best_acc = acc
                best_params = {'C': c_val, 'kernel': kernel}
                
    print(f"\nMejores parámetros: {best_params} con exactitud de {best_acc:.4f}")
    return best_params

if __name__ == '__main__':
    # 1. Ajustar SVM
    best_params = tune_svm()
    print()
    # 2. Entrenar final con mejores parámetros
    train_and_evaluate_svm(C=best_params['C'], kernel=best_params['kernel'])
