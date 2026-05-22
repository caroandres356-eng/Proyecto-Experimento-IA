import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Añadir el directorio raíz al path de Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from shared.data_loader import get_default_data_path

def perform_eda():
    """
    Realiza un Análisis Exploratorio de Datos (EDA) básico sobre el dataset.
    """
    print("--- Iniciando Análisis Exploratorio de Datos (EDA) ---")
    data_path = get_default_data_path()
    df = pd.read_csv(data_path)
    
    # 1. Información básica
    print("\n[INFO] Primeras filas del dataset:")
    print(df.head())
    
    print("\n[INFO] Descripción estadística:")
    print(df.describe())
    
    print("\n[INFO] Tipos de datos y valores nulos:")
    print(df.info())
    
    # 2. Distribución de la variable objetivo
    plt.figure(figsize=(8, 6))
    sns.countplot(x='target', data=df)
    plt.title('Distribución de la Variable Objetivo (Target)')
    plt.savefig(os.path.join(os.path.dirname(__file__), 'distribucion_target.png'))
    print("\n[OK] Gráfico de distribución guardado como 'distribucion_target.png'")
    
    # 3. Matriz de correlación
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Matriz de Correlación')
    plt.savefig(os.path.join(os.path.dirname(__file__), 'matriz_correlacion.png'))
    print("[OK] Matriz de correlación guardada como 'matriz_correlacion.png'")

if __name__ == '__main__':
    perform_eda()
