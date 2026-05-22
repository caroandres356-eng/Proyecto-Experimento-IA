# Experimento de Machine Learning: KNN y Árbol de Decisión

Este componente contiene el diseño, entrenamiento y comparación de dos algoritmos de clasificación tradicionales sobre el dataset de enfermedad cardíaca:

1. **K-Nearest Neighbors (KNN)**: Algoritmo basado en la distancia a los k vecinos más cercanos.
2. **Árbol de Decisión (Decision Tree)**: Algoritmo no paramétrico basado en reglas jerárquicas de decisión.

---

## 🛠️ Estructura del Módulo

* **`train_knn.py`**: Entrenamiento, optimización del hiperparámetro `k` (número de vecinos) y métricas del modelo KNN.
* **`train_decision_tree.py`**: Entrenamiento, control del sobreajuste mediante la profundidad del árbol (`max_depth`) y análisis de importancia de características.
* **`compare_results.py`**: Script de consolidación para comparar ambos algoritmos, graficar matrices de confusión lado a lado y exportar tablas comparativas.
* **`experimento_notebook.ipynb`**: Notebook de Jupyter para visualizar de forma interactiva todo el proceso del experimento.

---

## 📈 Hiperparámetros a Evaluar

### 1. KNN
* **`n_neighbors`**: Probar valores de $k$ (por ejemplo, impares entre 1 y 15) para evaluar el equilibrio entre sesgo y varianza.
* **`weights`**: Comparar voto uniforme ('uniform') y voto ponderado por distancia ('distance').

### 2. Árbol de Decisión
* **`max_depth`**: Limitar la profundidad máxima (p. ej., 3, 5, 10, None) para evitar el sobreajuste.
* **`criterion`**: Comparar impureza de Gini ('gini') con ganancia de información ('entropy').

---

## 📊 Métricas de Evaluación Requeridas

Para cada modelo se calcularán y presentarán:
* **Exactitud (Accuracy)**
* **Precisión (Precision)**
* **Sensibilidad (Recall)**
* **F1-Score**
* **Matriz de Confusión** (Visualizada como mapa de calor)
