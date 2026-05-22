import os
import sys
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Añadir el directorio raíz al path de Python para poder importar 'shared'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.data_loader import load_and_preprocess_data

def train_and_evaluate_tree(max_depth=None, criterion='gini'):
    """
    Entrena un clasificador de Árbol de Decisión y evalúa sus resultados.
    """
    print(f"--- Entrenando Árbol de Decisión con max_depth={max_depth}, criterion='{criterion}' ---")
    
    # 1. Cargar datos preprocesados
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data(scale_features=True)
    
    # 2. Inicializar y entrenar el modelo
    dt = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=42)
    dt.fit(X_train, y_train)
    
    # 3. Realizar predicciones
    y_pred = dt.predict(X_test)
    
    # 4. Evaluación
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Exactitud (Accuracy): {accuracy:.4f}")
    print("\nMatriz de Confusión:")
    print(conf_matrix)
    print("\nReporte de Clasificación:")
    print(report)
    
    # 5. Importancia de características
    importances = dt.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Característica': X_train.columns,
        'Importancia': importances
    }).sort_values(by='Importancia', ascending=False)
    
    print("\nImportancia de las características (Top 5):")
    print(feature_importance_df.head(5).to_string(index=False))
    
    return dt, accuracy, conf_matrix, report

def tune_tree():
    """
    Busca la mejor profundidad máxima del árbol para evitar sobreajuste.
    """
    print("--- Iniciando Ajuste de Hiperparámetros (Profundidad Máxima) ---")
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data(scale_features=True)
    
    best_depth = None
    best_acc = 0
    
    # Probar diferentes profundidades
    for depth in [2, 3, 4, 5, 7, 10, None]:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
        dt.fit(X_train, y_train)
        acc = accuracy_score(y_test, dt.predict(X_test))
        depth_str = str(depth) if depth is not None else "Ilimitado"
        print(f"Profundidad = {depth_str:9s} -> Exactitud en prueba: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_depth = depth
            
    depth_best_str = str(best_depth) if best_depth is not None else "Ilimitado"
    print(f"\nMejor profundidad encontrada: {depth_best_str} con exactitud de {best_acc:.4f}")
    return best_depth

if __name__ == '__main__':
    # 1. Ajustar el árbol
    best_depth = tune_tree()
    print()
    # 2. Entrenar modelo con la mejor profundidad
    train_and_evaluate_tree(max_depth=best_depth)
