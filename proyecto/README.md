# Proyecto de Machine Learning: SVM y Red Neuronal MLP

Este componente contiene el diseño, entrenamiento y comparación de dos algoritmos de clasificación avanzados sobre el dataset de enfermedad cardíaca:

1. **Support Vector Machines (SVM)**: Algoritmo de margen máximo que busca proyectar los datos a dimensiones superiores mediante funciones kernel para separar las clases de forma óptima.
2. **Red Neuronal - Perceptrón Multicapa (MLP)**: Modelo bioinspirado compuesto por capas de neuronas conectadas, capaz de aproximar fronteras de decisión altamente complejas y no lineales.

---

## 🛠️ Estructura del Módulo

* **`train_svm.py`**: Entrenamiento, selección de tipo de kernel (lineal, polinomial, RBF), y optimización de márgenes y regularización (`C`).
* **`train_mlp.py`**: Entrenamiento de la red neuronal MLP, permitiendo configurar capas ocultas (`hidden_layer_sizes`), funciones de activación, y tasa de aprendizaje.
* **`compare_results.py`**: Script de consolidación para comparar ambos algoritmos, graficar las curvas de pérdida y matrices de confusión lado a lado.
* **`proyecto_notebook.ipynb`**: Notebook de Jupyter para experimentar de manera interactiva con los hiperparámetros de SVM y MLP.

---

## 📈 Hiperparámetros a Evaluar

### 1. Support Vector Machines (SVM)
* **`kernel`**: Evaluar kernels `linear` (lineal) y `rbf` (gaussiano).
* **`C`**: Parámetro de penalización por error (p. ej., 0.1, 1.0, 10.0) para balancear la suavidad de la frontera con la clasificación correcta de entrenamiento.
* **`gamma`**: Coeficiente del kernel RBF (p. ej., 'scale', 'auto', 0.01).

### 2. Multi-Layer Perceptron (MLP)
* **`hidden_layer_sizes`**: Probar arquitecturas de una capa (p. ej., `(100,)`, `(50,)`) y múltiples capas (p. ej., `(50, 25)`).
* **`activation`**: Funciones de activación como `relu`, `tanh` o `logistic` (sigmoide).
* **`max_iter`**: Controlar el número máximo de iteraciones de entrenamiento.

---

## 📊 Métricas de Evaluación Requeridas

Para cada modelo se calcularán y presentarán:
* **Exactitud (Accuracy)**
* **Precisión (Precision)**
* **Sensibilidad (Recall)**
* **F1-Score**
* **Matriz de Confusión** (Visualizada como mapa de calor)
