import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_default_data_path():
    """Retorna la ruta absoluta por defecto al archivo heart.csv."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # El archivo está en ../data/heart.csv relativo a shared/data_loader.py
    data_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'heart.csv'))
    return data_path

def load_and_preprocess_data(data_path=None, test_size=0.2, random_state=42, scale_features=True):
    """
    Carga el dataset de enfermedades cardíacas, realiza limpieza básica,
    divide en entrenamiento/prueba y opcionalmente escala los datos.
    
    Parámetros:
    -----------
    data_path : str, opcional
        Ruta al archivo CSV. Si es None, busca en la ruta relativa por defecto.
    test_size : float, por defecto 0.2
        Proporción del conjunto de prueba.
    random_state : int, por defecto 42
        Semilla para la división aleatoria para reproducibilidad.
    scale_features : bool, por defecto True
        Si es True, aplica StandardScaler a las variables explicativas.
        
    Retorna:
    --------
    tuple
        (X_train, X_test, y_train, y_test, scaler)
        scaler es None si scale_features es False.
    """
    if data_path is None:
        data_path = get_default_data_path()
        
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No se encontró el dataset en la ruta: {data_path}")
        
    # Carga de datos
    df = pd.read_csv(data_path)
    
    # Análisis básico y limpieza de duplicados (si los hay)
    df_clean = df.drop_duplicates()
    
    # Separación de características (X) y etiqueta objetivo (y)
    X = df_clean.drop(columns=['target'])
    y = df_clean['target']
    
    # División en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        # Ajustamos el escalador con los datos de entrenamiento para evitar fuga de información (data leakage)
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convertimos de nuevo a DataFrame para mantener los nombres de las columnas
        X_train = pd.DataFrame(X_train_scaled, columns=X.columns)
        X_test = pd.DataFrame(X_test_scaled, columns=X.columns)
        
    return X_train, X_test, y_train, y_test, scaler

if __name__ == '__main__':
    # Pequeña verificación cuando el script se ejecuta directamente
    print("Probando data_loader.py...")
    try:
        X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()
        print("[OK] Datos cargados y preprocesados correctamente.")
        print(f"Dimensiones de entrenamiento (X, y): {X_train.shape}, {y_train.shape}")
        print(f"Dimensiones de prueba (X, y): {X_test.shape}, {y_test.shape}")
        print("Distribución de clases en y_train:\n", y_train.value_counts(normalize=True))
    except Exception as e:
        print(f"[ERROR] Error al cargar los datos: {e}")
