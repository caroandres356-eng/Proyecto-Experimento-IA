# Proyecto y Experimento Final: Clasificación de Enfermedades Cardíacas

Este repositorio contiene la arquitectura y el código para resolver un problema de clasificación binaria (predicción de enfermedad cardíaca utilizando `heart.csv`) empleando diferentes técnicas de Machine Learning. El repositorio está organizado de forma que el **Experimento** y el **Proyecto** se trabajen de manera independiente.

---

## 📂 Estructura del Repositorio

La arquitectura del proyecto está organizada de la siguiente manera:

* **`data/`**: Contiene la base de datos `heart.csv` con ejemplos de aprendizaje.
* **`shared/`**: Contiene código común para evitar duplicidad, como la carga y el preprocesamiento de datos.
* **`experimento/`**: Implementación de las técnicas para el Experimento (KNN y Árbol de Decisión).
* **`proyecto/`**: Implementación de las técnicas para el Proyecto (SVM y Redes Neuronales MLP).
* **`requirements.txt`**: Archivo con las librerías necesarias.

---

## 🔬 Técnicas Implementadas por Componente

### 1. Experimento
El objetivo del experimento es comparar dos técnicas clásicas de clasificación:
* **K-Nearest Neighbors (KNN)**
* **Árbol de Decisión (Decision Tree)**

### 2. Proyecto
El objetivo del proyecto es utilizar técnicas más avanzadas o con bases teóricas diferentes:
* **Máquinas de Soporte Vectorial (SVM)**
* **Redes Neuronales (Perceptrón Multicapa - MLP)**

---

## ⚙️ Instalación y Requisitos

Para instalar las dependencias necesarias en tu entorno de desarrollo Python, ejecuta:

```bash
pip install -r requirements.txt
```

---

## 🚀 Cómo Ejecutar

1. **Preprocesamiento y Exploración**:
   * Puedes usar los notebooks `experimento/experimento_notebook.ipynb` o `proyecto/proyecto_notebook.ipynb` para visualizar la exploración y análisis de los datos.

2. **Entrenamiento de Modelos del Experimento**:
   ```bash
   python experimento/train_knn.py
   python experimento/train_decision_tree.py
   ```

3. **Comparar Resultados del Experimento**:
   ```bash
   python experimento/compare_results.py
   ```

4. **Entrenamiento de Modelos del Proyecto**:
   ```bash
   python proyecto/train_svm.py
   python proyecto/train_mlp.py
   ```

5. **Comparar Resultados del Proyecto**:
   ```bash
   python proyecto/compare_results.py
   ```
