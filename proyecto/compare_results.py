import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

# Añadir el directorio raíz al path de Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.data_loader import load_and_preprocess_data

def run_comparison():
    print("=== Iniciando Comparación de Modelos del Proyecto ===")
    
    # 1. Cargar datos
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data(scale_features=True)
    
    # 2. Definir y entrenar modelos (usando hiperparámetros por defecto o recomendados)
    models = {
        'SVM (RBF Kernel, C=1.0)': SVC(C=1.0, kernel='rbf', random_state=42),
        'Red Neuronal MLP (100,)': MLPClassifier(hidden_layer_sizes=(100,), activation='relu', max_iter=500, random_state=42)
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
    print("TABLA COMPARATIVA DE RESULTADOS DEL PROYECTO")
    print("="*50)
    print(df_metrics.to_string(index=False))
    print("="*50)
    
    # Guardar tabla a CSV
    output_csv = os.path.join(os.path.dirname(__file__), 'comparacion_metricas_proyecto.csv')
    df_metrics.to_csv(output_csv, index=False)
    print(f"[OK] Tabla guardada en: {output_csv}")
    
    # 4. Graficar y guardar matrices de confusión lado a lado
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i, (name, cm) in enumerate(confusion_matrices.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=axes[i], cbar=False,
                    xticklabels=['Sano (0)', 'Enfermo (1)'],
                    yticklabels=['Sano (0)', 'Enfermo (1)'])
        axes[i].set_title(f'Matriz de Confusión - {name}')
        axes[i].set_xlabel('Predicho')
        axes[i].set_ylabel('Real')
        
    plt.tight_layout()
    output_img = os.path.join(os.path.dirname(__file__), 'proyecto_confusion_matrices.png')
    plt.savefig(output_img, dpi=300)
    plt.close()
    print(f"[OK] Grafico de matrices de confusion guardado en: {output_img}")
    
if __name__ == '__main__':
    run_comparison()
