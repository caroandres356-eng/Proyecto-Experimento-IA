import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

# Añadir el directorio raíz al path de Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from shared.data_loader import load_and_preprocess_data

def run_comparison():
    print("=== Iniciando Comparación de Modelos del Experimento ===")
    
    # 1. Cargar datos
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data(scale_features=True)
    
    # 2. Definir y entrenar modelos (usando hiperparámetros recomendados por defecto)
    models = {
        'KNN (K=5)': KNeighborsClassifier(n_neighbors=5),
        'Árbol de Decisión (max_depth=3)': DecisionTreeClassifier(max_depth=3, random_state=42)
    }
    
    metrics_summary = []
    confusion_matrices = {}
    
    for name, model in models.items():
        print(f"\nEntrenando {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calcular métricas globales
        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        
        metrics_summary.append({
            'Modelo': name,
            'Exactitud (Accuracy)': round(acc, 4),
            'Precisión (Precision)': round(precision, 4),
            'Sensibilidad (Recall)': round(recall, 4),
            'F1-Score': round(f1, 4)
        })
        
        # Guardar matriz de confusión para graficar
        confusion_matrices[name] = confusion_matrix(y_test, y_pred)
        
    # 3. Mostrar tabla de resultados en consola
    df_metrics = pd.DataFrame(metrics_summary)
    print("\n" + "="*50)
    print("TABLA COMPARATIVA DE RESULTADOS")
    print("="*50)
    print(df_metrics.to_string(index=False))
    print("="*50)
    
    # Guardar tabla a CSV en la carpeta de resultados
    output_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '6_resultados', 'comparacion_metricas_experimento.csv'))
    df_metrics.to_csv(output_csv, index=False)
    print(f"[OK] Tabla guardada en: {output_csv}")
    
    # 4. Graficar y guardar matrices de confusión lado a lado en su carpeta correspondiente
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i, (name, cm) in enumerate(confusion_matrices.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False,
                    xticklabels=['Sano (0)', 'Enfermo (1)'],
                    yticklabels=['Sano (0)', 'Enfermo (1)'])
        axes[i].set_title(f'Matriz de Confusión - {name}')
        axes[i].set_xlabel('Predicho')
        axes[i].set_ylabel('Real')
        
    plt.tight_layout()
    output_img = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '4_matrices_confusion', 'experimento_confusion_matrices.png'))
    plt.savefig(output_img, dpi=300)
    plt.close()
    print(f"[OK] Grafico de matrices de confusion guardado en: {output_img}")
    
if __name__ == '__main__':
    run_comparison()
