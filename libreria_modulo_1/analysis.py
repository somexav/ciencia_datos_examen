"""
Módulo de análisis exploratorio de datos
Funciones para generar estadísticas descriptivas y análisis automático
"""

import pandas as pd
from typing import Dict


def completitud_datos(df: pd.DataFrame) -> pd.Series:    
    """
    Calcula el porcentaje de completitud de datos para DataFrames de Pandas.
    
    Esta función analiza cada columna del DataFrame y calcula el porcentaje de 
    valores nulos presentes, ordenados de mayor a menor porcentaje de nulos.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de pandas a analizar.
        
    Returns
    -------
    pd.Series
        Serie de pandas con los nombres de las columnas como índice
        y los porcentajes de valores nulos (entre 0.0 y 1.0) como valores,
        ordenados de mayor a menor porcentaje de nulos.
        
    """
    return round(df.isnull().sum().sort_values(ascending=False) / df.shape[0], 4)


def check_data_completeness_JavierMartinezReyes(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Análisis completo de completitud y características de los datos.
    
    Esta función realiza un análisis exhaustivo del DataFrame, clasificando
    automáticamente las columnas y proporcionando estadísticos detallados.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de pandas a analizar.
        
    Returns
    -------
    Dict[str, pd.DataFrame]
        Diccionario con los siguientes DataFrames:
        
        - 'resumen_general' : pd.DataFrame
            Información general de completitud, tipos de datos y clasificación
            de variables para cada columna.
        - 'estadisticos_dispersion' : pd.DataFrame
            Estadísticos descriptivos detallados para variables numéricas,
            incluyendo medidas de tendencia central, dispersión y posición.
        - 'clasificacion_variables' : pd.DataFrame
            Clasificación automática de variables según su tipo y características,
            con criterios de clasificación detallados.
        
    Notes
    -----
    La función clasifica automáticamente las variables en:
    
    - Variables numéricas:
        - 'Continua': más de 10 valores únicos
        - 'Discreta': 10 o menos valores únicos
    
    - Variables categóricas:
        - 'Categórica_Baja': 10 o menos valores únicos
        - 'Categórica_Media': entre 11 y 50 valores únicos
        - 'Categórica_Alta': más de 50 valores únicos
        
    La función también imprime un resumen completo en consola con estadísticas
    generales del DataFrame y clasificación de variables.
        
    """
    
    # 1. Información básica del DataFrame
    total_filas = df.shape[0]
    total_columnas = df.shape[1]
    
    # 2. Análisis de completitud por columna
    resumen_data = []
    estadisticos_data = []
    clasificacion_data = []
    
    for col in df.columns:
        # Conteo de nulos y completitud
        nulos = df[col].isnull().sum()
        no_nulos = total_filas - nulos
        porcentaje_completitud = round((no_nulos / total_filas) * 100, 2)
        porcentaje_nulos = round((nulos / total_filas) * 100, 2)
        
        # Tipo de dato
        tipo_dato = str(df[col].dtype)
        
        # Valores únicos
        valores_unicos = df[col].nunique()
        
        # Clasificación automática
        if pd.api.types.is_numeric_dtype(df[col]):
            if valores_unicos > 10:
                clasificacion = 'Continua'
            else:
                clasificacion = 'Discreta'
            
            # Estadísticos de dispersión para variables numéricas
            col_data = df[col].dropna()
            if len(col_data) > 0:
                estadisticos_data.append({
                    'columna': col,
                    'tipo': clasificacion,
                    'media': round(col_data.mean(), 4),
                    'mediana': round(col_data.median(), 4),
                    'desv_std': round(col_data.std(), 4),
                    'varianza': round(col_data.var(), 4),
                    'min': round(col_data.min(), 4),
                    'max': round(col_data.max(), 4),
                    'q25': round(col_data.quantile(0.25), 4),
                    'q75': round(col_data.quantile(0.75), 4),
                    'rango': round(col_data.max() - col_data.min(), 4),
                    'coef_variacion': round((col_data.std() / col_data.mean()) * 100, 4) if col_data.mean() != 0 else 0
                })
        else:
            if valores_unicos <= 10:
                clasificacion = 'Categórica_Baja'
            elif valores_unicos <= 50:
                clasificacion = 'Categórica_Media'
            else:
                clasificacion = 'Categórica_Alta'
        
        # Agregar al resumen general
        resumen_data.append({
            'columna': col,
            'tipo_dato': tipo_dato,
            'valores_totales': total_filas,
            'valores_no_nulos': no_nulos,
            'valores_nulos': nulos,
            'porcentaje_completitud': porcentaje_completitud,
            'porcentaje_nulos': porcentaje_nulos,
            'valores_unicos': valores_unicos,
            'clasificacion': clasificacion
        })
        
        # Clasificación detallada
        clasificacion_data.append({
            'columna': col,
            'clasificacion': clasificacion,
            'criterio': f"Valores únicos: {valores_unicos}",
            'tipo_original': tipo_dato,
            'es_numerica': pd.api.types.is_numeric_dtype(df[col]),
            'es_categorica': not pd.api.types.is_numeric_dtype(df[col])
        })
    
    # Crear DataFrames
    resumen_df = pd.DataFrame(resumen_data)
    estadisticos_df = pd.DataFrame(estadisticos_data)
    clasificacion_df = pd.DataFrame(clasificacion_data)
    
    # Ordenar por porcentaje de nulos (mayor a menor)
    resumen_df = resumen_df.sort_values('porcentaje_nulos', ascending=False).reset_index(drop=True)
    
    # Mostrar resumen en consola
    print("="*60)
    print("ANÁLISIS COMPLETO DE COMPLETITUD DE DATOS")
    print("="*60)
    print(f"Dimensiones del DataFrame: {total_filas} filas x {total_columnas} columnas")
    print(f"Total de valores: {total_filas * total_columnas:,}")
    print(f"Total de valores nulos: {df.isnull().sum().sum():,}")
    print(f"Porcentaje general de completitud: {((df.size - df.isnull().sum().sum()) / df.size * 100):.2f}%")
    
    print("\nCLASIFICACIÓN DE VARIABLES:")
    clasificacion_counts = clasificacion_df['clasificacion'].value_counts()
    for tipo, count in clasificacion_counts.items():
        print(f"- {tipo}: {count} columnas")
    
    print(f"\nColumnas con mayor porcentaje de nulos:")
    top_nulos = resumen_df.head(5)[['columna', 'porcentaje_nulos', 'clasificacion']]
    for _, row in top_nulos.iterrows():
        print(f"- {row['columna']}: {row['porcentaje_nulos']}% ({row['clasificacion']})")
    
    return {
        'resumen_general': resumen_df,
        'estadisticos_dispersion': estadisticos_df,
        'clasificacion_variables': clasificacion_df
    }
