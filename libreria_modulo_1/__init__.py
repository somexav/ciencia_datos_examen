"""
Librería de Análisis Exploratorio de Datos
==========================================

Esta librería proporciona herramientas completas para el análisis exploratorio de datos,
incluyendo preprocesamiento, visualización y análisis estadístico.

Módulos disponibles
-------------------
preprocessing : module
    Funciones para limpieza, imputación de valores faltantes, detección y 
    eliminación de outliers, y preprocesamiento general de datos.
    
visualization : module
    Funciones para crear visualizaciones interactivas y gráficos exploratorios
    utilizando Plotly, incluyendo histogramas, boxplots, heatmaps y más.
    
analysis : module
    Funciones para análisis estadístico descriptivo, evaluación de completitud
    de datos y reportes automáticos de calidad de datos.

Notes
-----
Todas las funciones están documentadas siguiendo el formato NumPy docstring
para máxima compatibilidad con herramientas de documentación automática.

Examples
--------
>>> import libreria_modulo_1 as lm1
>>> # Análisis de completitud
>>> resultado = lm1.completitud_datos(df)
>>> # Visualización interactiva
>>> fig = lm1.plot_interactive_histogram(df, 'columna')
>>> # Preprocesamiento
>>> df_clean = lm1.impute_missing_values(df, method='knn')

Autor: Estudiante de Data Science
Versión: 1.0.0
"""

# Importar funciones principales de cada módulo
from .preprocessing import (
    agrega_nan,
    delete_missing_values,
    impute_missing_values,
    detect_outliers_iqr,
    detect_outliers_zscore,
    remove_outliers
)

from .visualization import (
    # Funciones interactivas con Plotly
    plot_interactive_histogram,
    plot_interactive_boxplot,
    plot_interactive_bar_horizontal,
    plot_interactive_line_timeseries,
    plot_interactive_dot_comparison,
    plot_interactive_density_multiclass,
    plot_interactive_violin_swarm,
    plot_interactive_correlation_heatmap
)

from .analysis import (
    completitud_datos,
    check_data_completeness_JavierMartinezReyes
)

# Información de la librería
__version__ = "1.0.0"
__author__ = "Estudiante de Data Science"
__description__ = "Librería personalizada para análisis exploratorio de datos"

# Lista de todas las funciones disponibles
__all__ = [
    # Preprocessing
    'agrega_nan',
    'delete_missing_values',
    'impute_missing_values', 
    'detect_outliers_iqr',
    'detect_outliers_zscore',
    'remove_outliers',
    
    # Visualization - Funciones interactivas con Plotly
    'plot_interactive_histogram',
    'plot_interactive_boxplot', 
    'plot_interactive_bar_horizontal',
    'plot_interactive_line_timeseries',
    'plot_interactive_dot_comparison',
    'plot_interactive_density_multiclass',
    'plot_interactive_violin_swarm',
    'plot_interactive_correlation_heatmap',
    
    # Analysis
    'completitud_datos',
    'check_data_completeness_JavierMartinezReyes'
]
