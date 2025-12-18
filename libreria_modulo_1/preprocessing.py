"""
Módulo de preprocesamiento de datos
Funciones para limpieza, imputación y tratamiento de valores atípicos
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

def agrega_nan(datos: pd.DataFrame, min_frac: float = 0.0, max_frac: float = 0.3, seed: int = None) -> pd.DataFrame:
    """
    Agregar valores NaN aleatoriamente para simular datos faltantes.
    
    Parameters
    ----------
    datos : pd.DataFrame
        DataFrame de pandas al cual se le agregarán valores NaN.
    min_frac : float, default 0.0
        Fracción mínima de valores NaN por columna (entre 0 y 1).
    max_frac : float, default 0.3
        Fracción máxima de valores NaN por columna (entre 0 y 1).
    seed : int, optional
        Semilla para reproducibilidad de los valores aleatorios.
        
    Returns
    -------
    pd.DataFrame
        DataFrame con valores NaN agregados aleatoriamente.
        
    Raises
    ------
    ValueError
        Si las fracciones no están entre 0 y 1, o si min_frac > max_frac.
    """
    if seed is not None:
        np.random.seed(seed)
    
    if min_frac < 0 or max_frac > 1 or min_frac > max_frac:
        raise ValueError("Las fracciones deben estar entre 0 y 1, y min_frac <= max_frac")
    
    datos_nan = datos.copy()
    for col in datos_nan.columns:
        # Generar un porcentaje aleatorio diferente para cada columna
        random_frac = np.random.uniform(min_frac, max_frac)
        if random_frac > 0:  # Solo agregar NaN si la fracción es mayor a 0
            datos_nan.loc[datos_nan.sample(frac=random_frac).index, col] = np.nan
    return datos_nan

def delete_missing_values(df: pd.DataFrame, porcentage: float ) -> pd.DataFrame:
    """
    Elimina las COLUMNAS de un DataFrame que tienen un porcentaje de valores faltantes
    mayor al especificado.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de pandas a procesar.
    porcentage : float
        Porcentaje máximo de valores faltantes permitidos por columna (entre 0 y 1).
        Ejemplo: 0.2 significa que se eliminan columnas con más del 20% de NaN.
        
    Returns
    -------
    pd.DataFrame
        DataFrame filtrado sin las columnas que exceden el porcentaje de valores faltantes.
        
    Raises
    ------
    ValueError
        Si el porcentaje no está entre 0 y 1.
        
    Notes
    -----
    La función imprime información detallada sobre el proceso de filtrado,
    incluyendo el número de columnas eliminadas y las estadísticas finales.
    """
    if porcentage < 0 or porcentage > 1:
        raise ValueError("El porcentaje debe estar entre 0 y 1")
    
    df = df.copy()
    num_rows = df.shape[0]
    num_cols_original = df.shape[1]
    
    print(f"Análisis inicial:")
    print(f"- Filas: {num_rows}")
    print(f"- Columnas originales: {num_cols_original}")
    print(f"- Porcentaje máximo de NaN permitido por columna: {porcentage:.1%}")
    
    # Calcular el número máximo de NaN permitidos por columna
    max_nan_per_column = int(porcentage * num_rows)
    print(f"- Máximo de NaN permitidos por columna: {max_nan_per_column}/{num_rows}")
    
    # Calcular NaN por columna
    nan_per_column = df.isnull().sum()
    columns_to_drop = nan_per_column[nan_per_column > max_nan_per_column].index.tolist()
    
    print(f"\nAnálisis por columnas:")
    print(f"- Columnas con más de {porcentage:.1%} de NaN: {len(columns_to_drop)}")
    if len(columns_to_drop) > 0:
        print(f"- Columnas a eliminar: {columns_to_drop}")
    
    # Eliminar columnas que exceden el umbral
    df_filtrado = df.drop(columns=columns_to_drop)
    
    columnas_eliminadas = len(columns_to_drop)
    print(f"\nResultados:")
    print(f"- Columnas eliminadas: {columnas_eliminadas}")
    print(f"- Columnas restantes: {df_filtrado.shape[1]}")
    print(f"- Forma final del DataFrame: {df_filtrado.shape}")
    
    return df_filtrado


def impute_missing_values(df: pd.DataFrame, method: str = 'auto') -> pd.DataFrame:
    """
    Imputa valores faltantes usando diferentes métodos según el tipo de variable.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de pandas con valores faltantes a imputar.
    method : str, default 'auto'
        Método de imputación a utilizar. Opciones:
        - 'auto': media para numéricas, moda para categóricas
        - 'mean': media para variables numéricas
        - 'median': mediana para variables numéricas
        - 'knn': K-Nearest Neighbors para numéricas, moda para categóricas
        
    Returns
    -------
    pd.DataFrame
        DataFrame con valores faltantes imputados.
        
    Raises
    ------
    ValueError
        Si el método especificado no es reconocido.
        
    Notes
    -----
    La función imprime información sobre el proceso de imputación para cada columna,
    incluyendo el método utilizado y los valores de relleno aplicados.
    """
    df = df.copy()
    print(f"Valores faltantes antes de imputar:\n{df.isnull().sum().sum()} en total")
    
    if method == 'knn':
        # Separar columnas numéricas y categóricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        df_result = df.copy()
        
        # Primero imputar columnas categóricas con moda
        for col in categorical_cols:
            if df_result[col].isnull().sum() > 0:
                mode_value = df_result[col].mode()
                if not mode_value.empty:
                    fill_value = mode_value.iloc[0]
                else:
                    fill_value = 'Unknown'
                df_result[col].fillna(fill_value, inplace=True)
                print(f"Columna categórica '{col}': imputada con moda '{fill_value}'")
        
        # Luego aplicar KNN solo a columnas numéricas
        if len(numeric_cols) > 0 and df_result[numeric_cols].isnull().sum().sum() > 0:
            knn_imputer = KNNImputer(n_neighbors=5)
            numeric_data_imputed = knn_imputer.fit_transform(df_result[numeric_cols])
            df_result[numeric_cols] = pd.DataFrame(numeric_data_imputed, 
                                                  columns=numeric_cols, 
                                                  index=df_result.index)
            print(f"Columnas numéricas: imputadas con KNN")
        
        return df_result
    
    else:
        # Método automático o específico
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ['object', 'category']:
                    # Variables categóricas: usar moda
                    mode_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                    df[col].fillna(mode_value, inplace=True)
                    print(f"Columna '{col}' (categórica): imputada con moda '{mode_value}'")
                
                else:
                    # Variables numéricas
                    if method == 'mean' or method == 'auto':
                        fill_value = df[col].mean()
                        method_name = 'media'
                    elif method == 'median':
                        fill_value = df[col].median()
                        method_name = 'mediana'
                    else:
                        raise ValueError("Método no reconocido. Use 'auto', 'mean', 'median' o 'knn'.")
                    
                    df[col].fillna(fill_value, inplace=True)
                    print(f"Columna '{col}' (numérica): imputada con {method_name} ({fill_value:.2f})")
    
    print(f"\nValores faltantes después de imputar: {df.isnull().sum().sum()}")
    return df


def detect_outliers_iqr(df: pd.DataFrame, columns: list = None, factor: float = 1.5) -> pd.DataFrame:
    """
    Detecta outliers usando el método IQR (Rango Intercuartílico).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de pandas a analizar.
    columns : list, optional
        Lista de nombres de columnas a analizar. Si es None, analiza todas las columnas numéricas.
    factor : float, default 1.5
        Factor multiplicador para el IQR. El valor estándar es 1.5.
        
    Returns
    -------
    pd.DataFrame
        DataFrame con información detallada sobre outliers por columna, incluyendo:
        - Cuartiles Q1 y Q3
        - Rango intercuartílico (IQR)
        - Límites inferior y superior
        - Número y porcentaje de outliers
        
    Notes
    -----
    El método IQR define como outliers los valores que están fuera del rango
    [Q1 - factor*IQR, Q3 + factor*IQR], donde Q1 y Q3 son el primer y tercer
    cuartil respectivamente.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outlier_info = []
    
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        
        outlier_info.append({
            'columna': col,
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'limite_inferior': lower_bound,
            'limite_superior': upper_bound,
            'num_outliers': len(outliers),
            'porcentaje_outliers': (len(outliers) / len(df)) * 100
        })
    
    return pd.DataFrame(outlier_info)


def detect_outliers_zscore(df: pd.DataFrame, columns: list = None, threshold: float = 3.0) -> pd.DataFrame:
    """
    Detecta outliers usando el método Z-Score.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de pandas a analizar.
    columns : list, optional
        Lista de nombres de columnas a analizar. Si es None, analiza todas las columnas numéricas.
    threshold : float, default 3.0
        Umbral del z-score. Valores con |z-score| > threshold se consideran outliers.
        
    Returns
    -------
    pd.DataFrame
        DataFrame con información detallada sobre outliers por columna, incluyendo:
        - Media y desviación estándar
        - Umbral utilizado
        - Número y porcentaje de outliers
        
    Notes
    -----
    El método Z-Score identifica outliers basándose en cuántas desviaciones estándar
    se aleja un valor de la media. Se asume que los datos siguen una distribución normal.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outlier_info = []
    
    for col in columns:
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        outliers = df[z_scores > threshold]
        
        outlier_info.append({
            'columna': col,
            'media': df[col].mean(),
            'std': df[col].std(),
            'threshold': threshold,
            'num_outliers': len(outliers),
            'porcentaje_outliers': (len(outliers) / len(df)) * 100
        })
    
    return pd.DataFrame(outlier_info)


def remove_outliers(df: pd.DataFrame, method: str = 'iqr', columns: list = None, **kwargs) -> pd.DataFrame:
    """
    Elimina outliers del DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de pandas del cual eliminar outliers.
    method : str, default 'iqr'
        Método a usar para detectar outliers ('iqr' o 'zscore').
    columns : list, optional
        Lista de columnas a considerar para la detección de outliers.
        Si es None, considera todas las columnas numéricas.
    **kwargs : dict
        Argumentos adicionales para los métodos de detección:
        - Para 'iqr': 'factor' (default 1.5)
        - Para 'zscore': 'threshold' (default 3.0)
        
    Returns
    -------
    pd.DataFrame
        DataFrame sin los outliers detectados.
        
    Notes
    -----
    La función imprime estadísticas sobre el proceso de eliminación,
    incluyendo el número de filas eliminadas y el porcentaje del total.
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    initial_rows = len(df_clean)
    
    if method == 'iqr':
        factor = kwargs.get('factor', 1.5)
        for col in columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    elif method == 'zscore':
        print("Eliminando outliers usando el método Z-Score. threshold ")
        threshold = kwargs.get('threshold', 3.0)
        print(f"Eliminando outliers usando el método Z-Score. threshold {threshold}")   
        for col in columns:
            z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
            df_clean = df_clean[z_scores <= threshold]
    
    final_rows = len(df_clean)
    removed_rows = initial_rows - final_rows
    
    print(f"Filas originales: {initial_rows}")
    print(f"Filas eliminadas: {removed_rows}")
    print(f"Filas restantes: {final_rows}")
    print(f"Porcentaje eliminado: {(removed_rows/initial_rows)*100:.2f}%")
    
    return df_clean