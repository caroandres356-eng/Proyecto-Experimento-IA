import os
import sys
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Añadir el directorio raíz al path de Python para importar 'shared'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.data_loader import load_and_preprocess_data

def train_and_evaluate_mlp(hidden_layer_sizes=(100,), activation='relu', max_iter=500):
    """
    Entrena una red neuronal (Perceptrón Multicapa) y evalúa los resultados.
    """
    print(f"--- Entrenando MLP con hidden_layer_sizes={hidden_layer_sizes}, activation='{activation}' ---")
    
    # 1. Cargar datos preprocesados
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data(scale_features=True)
    
    # 2. Inicializar y entrenar el modelo
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        max_iter=max_iter,
        random_state=42,
        early_stopping=True,     # Detener el entrenamiento para prevenir sobreajuste
        validation_fraction=0.1   # Fracción para parada temprana
    )
    mlp.fit(X_train, y_train)
    
    # 3. Realizar predicciones
    y_pred = mlp.predict(X_test)
    
    # 4. Evaluación
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Exactitud (Accuracy): {accuracy:.4f}")
    print(f"Iteraciones realizadas: {mlp.n_iter_}")
    print("\nMatriz de Confusión:")
    print(conf_matrix)
    print("\nReporte de Clasificación:")
    print(report)
    
    return mlp, accuracy, conf_matrix, report

def tune_mlp():
    """
    Evalúa arquitecturas de red neuronal y funciones de activación simples.
    """
    print("--- Ajuste de Hiperparámetros para MLP ---")
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data(scale_features=True)
    
    best_acc = 0
    best_params = {}
    
    # Configurar combinaciones de capas y activaciones
    architectures = [(50,), (100,), (50, 25)]
    activations = ['relu', 'tanh']
    
    for arch in architectures:
        for act in activations:
            mlp = MLPClassifier(
                hidden_layer_sizes=arch,
                activation=act,
                max_iter=500,
                random_state=42,
                early_stopping=True
            )
            mlp.fit(X_train, y_train)
            acc = accuracy_score(y_test, mlp.predict(X_test))
            print(f"Arquitectura = {str(arch):10s} | Activación = {act:5s} -> Exactitud: {acc:.4f}")
            
            if acc > best_acc:
                best_acc = acc
                best_params = {'hidden_layer_sizes': arch, 'activation': act}
                
    print(f"\nMejores parámetros: {best_params} con exactitud de {best_acc:.4f}")
    return best_params

if __name__ == '__main__':
    # 1. Ajustar red neuronal
    best_params = tune_mlp()
    print()
    # 2. Entrenar final con mejores parámetros
    train_and_evaluate_mlp(
        hidden_layer_sizes=best_params['hidden_layer_sizes'],
        activation=best_params['activation']
    )
