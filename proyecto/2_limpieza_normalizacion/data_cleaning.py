import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Añadir el directorio raíz al path de Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from shared.data_loader import get_default_data_path

def clean_and_normalize():
    """
    Ejemplo de script para limpieza y normalización manual.
    """
    print("--- Iniciando Limpieza y Normalización ---")
    data_path = get_default_data_path()
    df = pd.read_csv(data_path)
    
    # 1. Limpieza de duplicados
    initial_shape = df.shape
    df = df.drop_duplicates()
    print(f"\n[INFO] Filas eliminadas (duplicados): {initial_shape[0] - df.shape[0]}")
    
    # 2. Manejo de valores nulos (si los hubiera)
    if df.isnull().values.any():
        print("[INFO] Valores nulos detectados. Aplicando imputación...")
        df = df.fillna(df.median())
    else:
        print("[INFO] No se detectaron valores nulos.")
        
    # 3. Normalización (Escalado)
    scaler = StandardScaler()
    features = df.drop(columns=['target'])
    target = df['target']
    
    features_scaled = scaler.fit_transform(features)
    df_normalized = pd.DataFrame(features_scaled, columns=features.columns)
    df_normalized['target'] = target.values
    
    print("\n[OK] Datos normalizados correctamente.")
    print(df_normalized.head())
    
    return df_normalized

if __name__ == '__main__':
    clean_and_normalize()
