# Proyecto y Experimento de Machine Learning 2026

Este repositorio contiene la implementación de dos trabajos de clasificación utilizando técnicas de inteligencia computacional, siguiendo los lineamientos de "Experimento ML 2026" y "Proyecto 4 Machine Learning".

## 📂 Estructura del Proyecto

La arquitectura ha sido diseñada para seguir el ciclo de vida de un proyecto de ML de forma secuencial:

```text
/
├── data/                       # Dataset (heart.csv)
├── shared/                     # Código compartido (data_loader.py)
├── experimento/                # Experimento (KNN & Decision Tree)
│   ├── 1_analisis_datos/       # EDA (eda.py)
│   ├── 2_limpieza_normalizacion/ # Preprocesamiento (data_cleaning.py)
│   ├── 3_entrenamiento/        
│   │   ├── knn/                # (train_knn.py)
│   │   └── decision_tree/      # (train_decision_tree.py)
│   ├── 4_matrices_confusion/   # Salidas de matrices de confusión
│   ├── 5_analisis_comparativo/ # Comparación (compare_results.py)
│   └── 6_resultados/           # Reportes y tablas finales
└── proyecto/                   # Proyecto 4 (SVM & MLP)
    ├── 1_analisis_datos/       
    ├── 2_limpieza_normalizacion/
    ├── 3_entrenamiento/        
    │   ├── svm/                # (train_svm.py)
    │   └── mlp/                # (train_mlp.py)
    ├── 4_matrices_confusion/   
    ├── 5_analisis_comparativo/ 
    └── 6_resultados/           
```

## 🚀 Cómo empezar

1. **Instalación de dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ejecución del flujo:**
   Cada etapa está numerada. Se recomienda seguir el orden:
   - **Análisis**: Ejecutar `python experimento/1_analisis_datos/eda.py`
   - **Limpieza**: Ejecutar `python experimento/2_limpieza_normalizacion/data_cleaning.py`
   - **Entrenamiento**: Ejecutar scripts en `3_entrenamiento` (ej. `python experimento/3_entrenamiento/knn/train_knn.py`)
   - **Comparación**: Ejecutar `python experimento/5_analisis_comparativo/compare_results.py`

3. **Resultados:**
   Los gráficos y tablas necesarios para la presentación PPT se guardarán automáticamente en las carpetas `4_matrices_confusion` y `6_resultados`.
