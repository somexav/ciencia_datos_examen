"""
Módulo de visualización de datos
Funciones para crear gráficos y visualizaciones del análisis exploratorio
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from typing import Optional
import os
from datetime import datetime


def _ensure_plots_directory():
    """
    Asegurar que el directorio plots/ existe.
    
    Returns
    -------
    str
        Ruta del directorio plots/
    """
    plots_dir = "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        print(f"📁 Directorio '{plots_dir}' creado para guardar gráficos")
    return plots_dir


def _save_plot(fig, plot_name: str, save_plot: bool = True):
    """
    Guardar gráfico únicamente como JPG.
    
    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Figura de Plotly a guardar.
    plot_name : str
        Nombre base del archivo (sin extensión).
    save_plot : bool, default True
        Si guardar el gráfico automáticamente.
        
    Returns
    -------
    dict
        Diccionario con la ruta del archivo guardado.
    """
    if not save_plot:
        return {}
    
    plots_dir = _ensure_plots_directory()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{plot_name}_{timestamp}"
    
    saved_files = {}
    
    try:
        # Guardar únicamente como JPG
        jpg_path = os.path.join(plots_dir, f"{base_filename}.jpg")
        fig.write_image(jpg_path, format="jpeg", width=1200, height=800, scale=2)
        saved_files['jpg'] = jpg_path
        
        print(f"💾 Gráfico guardado como JPG: {jpg_path}")
        
    except Exception as e:
        print(f"⚠️  Error guardando gráfico: {str(e)}")
        print("💡 Instala kaleido para guardar imágenes: pip install kaleido")
    
    return saved_files


def plot_interactive_histogram(df: pd.DataFrame, column: str, group_by: Optional[str] = None, 
                             add_kde: bool = True, bins: int = 30, title: Optional[str] = None,
                             save_plot: bool = True):
    """
    Histograma interactivo con línea de densidad y KDE, customizable por grupo.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de pandas con los datos a graficar.
    column : str
        Nombre de la columna a graficar.
    group_by : Optional[str], default None
        Nombre de la columna para agrupar los datos (opcional).
    add_kde : bool, default True
        Si agregar línea de densidad KDE al histograma.
    bins : int, default 30
        Número de bins para el histograma.
    title : Optional[str], default None
        Título personalizado para el gráfico.
    save_plot : bool, default True
        Si guardar automáticamente el gráfico en la carpeta plots/.
        
    Returns
    -------
    plotly.graph_objects.Figure
        Figura interactiva de Plotly con el histograma.
    """
    if title is None:
        title = f'Distribución de la columna:{column}'
    
    if group_by is None:
        # Histograma simple
        fig = px.histogram(df, x=column, nbins=bins, title=title,
                          marginal="box", hover_data=df.columns)
        
        if add_kde:
            # Agregar línea KDE
            kde_x = np.linspace(df[column].min(), df[column].max(), 100)
            kde_y = stats.gaussian_kde(df[column].dropna())(kde_x)
            
            # Normalizar KDE para que se ajuste al histograma
            hist_data = np.histogram(df[column].dropna(), bins=bins)
            kde_y_scaled = kde_y * len(df[column].dropna()) * (hist_data[1][1] - hist_data[1][0])
            
            fig.add_trace(go.Scatter(x=kde_x, y=kde_y_scaled, mode='lines',
                                   name='KDE', line=dict(color='red', width=3)))
    else:
        # Histograma por grupos
        fig = px.histogram(df, x=column, color=group_by, nbins=bins, title=title,
                          marginal="box", barmode='overlay', opacity=0.7)
        
        if add_kde:
            # KDE por grupo
            for i, group in enumerate(df[group_by].unique()):
                group_data = df[df[group_by] == group][column].dropna()
                if len(group_data) > 1:
                    kde_x = np.linspace(group_data.min(), group_data.max(), 100)
                    kde_y = stats.gaussian_kde(group_data)(kde_x)
                    
                    # Normalizar KDE
                    hist_data = np.histogram(group_data, bins=bins)
                    kde_y_scaled = kde_y * len(group_data) * (hist_data[1][1] - hist_data[1][0])
                    
                    fig.add_trace(go.Scatter(x=kde_x, y=kde_y_scaled, mode='lines',
                                           name=f'KDE {group}', 
                                           line=dict(width=3, dash='dash')))
    
    fig.update_layout(bargap=0.1)
    
    # Guardar gráfico automáticamente
    if save_plot:
        plot_name = f"histogram_{column}"
        if group_by:
            plot_name += f"_by_{group_by}"
        _save_plot(fig, plot_name, save_plot)
    
    return fig


def plot_interactive_boxplot(df: pd.DataFrame, column: str, group_by: Optional[str] = None,
                           target_class: Optional[str] = None, title: Optional[str] = None,
                           save_plot: bool = True):
    """
    Boxplot interactivo con subgráficos por clase objetivo.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de pandas con los datos a graficar.
    column : str
        Nombre de la columna numérica a graficar.
    group_by : Optional[str], default None
        Nombre de la columna para agrupar los datos.
    target_class : Optional[str], default None
        Nombre de la columna de clase objetivo para crear subgráficos.
    title : Optional[str], default None
        Título personalizado para el gráfico.
    save_plot : bool, default True
        Si guardar automáticamente el gráfico en la carpeta plots/.
        
    Returns
    -------
    plotly.graph_objects.Figure
        Figura interactiva de Plotly con el boxplot.
    """
    if title is None:
        title = f'Boxplot de {column}'
    
    if target_class is not None:
        # Crear subgráficos por clase objetivo
        unique_classes = df[target_class].unique()
        fig = make_subplots(rows=1, cols=len(unique_classes),
                           subplot_titles=[f'{target_class}: {cls}' for cls in unique_classes])
        
        for i, cls in enumerate(unique_classes, 1):
            subset = df[df[target_class] == cls]
            
            if group_by is not None:
                for group in subset[group_by].unique():
                    group_data = subset[subset[group_by] == group][column]
                    fig.add_trace(go.Box(y=group_data, name=f'{group}', 
                                       boxpoints='outliers', showlegend=(i==1)),
                                 row=1, col=i)
            else:
                fig.add_trace(go.Box(y=subset[column], name=f'{cls}',
                                   boxpoints='outliers', showlegend=False),
                             row=1, col=i)
        
        fig.update_layout(title=title, height=500)
    else:
        # Boxplot simple o agrupado
        if group_by is not None:
            fig = px.box(df, y=column, x=group_by, title=title, points='outliers')
        else:
            fig = px.box(df, y=column, title=title, points='outliers')
    
    # Guardar gráfico automáticamente
    if save_plot:
        plot_name = f"boxplot_{column}"
        if group_by:
            plot_name += f"_by_{group_by}"
        if target_class:
            plot_name += f"_class_{target_class}"
        _save_plot(fig, plot_name, save_plot)
    
    return fig


def plot_interactive_bar_horizontal(df: pd.DataFrame, column: str, top_n: int = 20,
                                  title: Optional[str] = None, show_percentage: bool = True,
                                  color_scheme: str = 'viridis'):
    """
    Gráfico de barras horizontal ordenado por frecuencia descendente con porcentajes.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de pandas con los datos a graficar.
    column : str
        Nombre de la columna categórica a graficar.
    top_n : int, default 20
        Número de categorías más frecuentes a mostrar.
    title : Optional[str], default None
        Título personalizado para el gráfico.
    show_percentage : bool, default True
        Si mostrar porcentajes además de las frecuencias absolutas.
    color_scheme : str, default 'viridis'
        Esquema de colores a utilizar ('viridis', 'plasma', 'blues', 'reds').
        
    Returns
    -------
    plotly.graph_objects.Figure
        Figura interactiva de Plotly con el gráfico de barras horizontal.
    """
    if title is None:
        title = f'Top {top_n} categorías más frecuentes - {column}'
    
    # Contar frecuencias y ordenar
    value_counts = df[column].value_counts().head(top_n)
    total = len(df)
    percentages = (value_counts / total * 100).round(1)
    
    # Crear texto para las barras
    if show_percentage:
        text_labels = [f'{count} ({pct}%)' for count, pct in zip(value_counts.values, percentages.values)]
    else:
        text_labels = value_counts.values
    
    fig = px.bar(x=value_counts.values, y=value_counts.index, 
                 orientation='h', title=title,
                 labels={'x': 'Frecuencia', 'y': column},
                 text=text_labels,
                 color=value_counts.values,
                 color_continuous_scale=color_scheme)
    
    fig.update_traces(texttemplate='%{text}', textposition='outside', textfont_size=10)
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=max(400, len(value_counts) * 25),  # Altura dinámica
        showlegend=False,
        coloraxis_showscale=False  # Ocultar escala de color
    )
    
    # Agregar información estadística
    fig.add_annotation(
        text=f"Total de categorías únicas: {df[column].nunique()}<br>"
             f"Total de registros: {total:,}<br>"
             f"Categoría más frecuente: {value_counts.index[0]} ({value_counts.iloc[0]} casos)",
        showarrow=False,
        x=1.02, y=1,
        xref="paper", yref="paper",
        align="left",
        font=dict(size=10),
        bgcolor="rgba(240,240,240,0.8)",
        bordercolor="gray", borderwidth=1
    )
    
    return fig


def plot_interactive_line_timeseries(df: pd.DataFrame, y_column: str, 
                                    x_column: Optional[str] = None,
                                    simulate_time: bool = True, 
                                    title: Optional[str] = None):
    """
    Gráfico de líneas para serie temporal (real o simulada).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de pandas con los datos a graficar.
    y_column : str
        Nombre de la columna para el eje Y (variable dependiente).
    x_column : Optional[str], default None
        Nombre de la columna para el eje X. Si es None, usa el índice.
    simulate_time : bool, default True
        Si True, simula una variable de tiempo cuando x_column es None.
    title : Optional[str], default None
        Título personalizado para el gráfico.
        
    Returns
    -------
    plotly.graph_objects.Figure
        Figura interactiva de Plotly con el gráfico de líneas temporal.
    """
    df_copy = df.copy()
    
    if x_column is None:
        if simulate_time:
            # Simular fechas
            df_copy['fecha_simulada'] = pd.date_range(
                start='2023-01-01', periods=len(df_copy), freq='D'
            )
            x_col = 'fecha_simulada'
        else:
            # Usar índice
            df_copy = df_copy.reset_index()
            x_col = 'index'
    else:
        x_col = x_column
    
    # Ordenar por variable X
    df_copy = df_copy.sort_values(x_col)
    
    if title is None:
        title = f'Serie Temporal - {y_column}'
    
    fig = px.line(df_copy, x=x_col, y=y_column, title=title,
                  markers=True, hover_data=df_copy.columns)
    
    # Agregar línea de tendencia
    fig.add_trace(go.Scatter(x=df_copy[x_col], 
                           y=df_copy[y_column].rolling(window=min(10, len(df_copy)//10), center=True).mean(),
                           mode='lines', name='Tendencia (Media móvil)',
                           line=dict(color='red', width=2, dash='dash')))
    
    fig.update_layout(xaxis_title=x_col, yaxis_title=y_column)
    return fig


def plot_interactive_dot_comparison(df: pd.DataFrame, column: str, group1: str, group2: str,
                                  group_column: str, title: Optional[str] = None):
    """
    Dot plot para comparar dos grupos con overlay.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de pandas con los datos a graficar.
    column : str
        Nombre de la columna numérica a comparar entre grupos.
    group1 : str
        Nombre del primer grupo a comparar.
    group2 : str
        Nombre del segundo grupo a comparar.
    group_column : str
        Nombre de la columna que contiene los grupos.
    title : Optional[str], default None
        Título personalizado para el gráfico.
        
    Returns
    -------
    plotly.graph_objects.Figure
        Figura interactiva de Plotly con el dot plot de comparación.
        
    Raises
    ------
    ValueError
        Si la columna de agrupación o numérica no existe, o si los grupos
        especificados no se encuentran en los datos.
    """
    if title is None:
        title = f'Comparación Dot Plot: {group1} vs {group2}'
    
    # Verificar que la columna de agrupación existe
    if group_column not in df.columns:
        raise ValueError(f"La columna '{group_column}' no existe en el DataFrame")
    
    # Verificar que la columna numérica existe
    if column not in df.columns:
        raise ValueError(f"La columna '{column}' no existe en el DataFrame")
     
    # Verificar que los grupos existen
    unique_groups = df[group_column].unique()
    if group1 not in unique_groups:
        raise ValueError(f"El grupo '{group1}' no existe en la columna '{group_column}'. Grupos disponibles: {unique_groups}")
    if group2 not in unique_groups:
        raise ValueError(f"El grupo '{group2}' no existe en la columna '{group_column}'. Grupos disponibles: {unique_groups}")
    
    # Filtrar datos para los dos grupos
    df_group1 = df[df[group_column] == group1].dropna(subset=[column])
    df_group2 = df[df[group_column] == group2].dropna(subset=[column])
    
    # Verificar que hay datos después del filtro
    if len(df_group1) == 0:
        raise ValueError(f"No hay datos para el grupo '{group1}' en la columna '{column}'")
    if len(df_group2) == 0:
        raise ValueError(f"No hay datos para el grupo '{group2}' en la columna '{column}'")
    
    fig = go.Figure()
    
    # Agregar dots para grupo 1 con jitter más consistente
    np.random.seed(42)  # Para reproducibilidad
    jitter1 = np.random.normal(1, 0.05, len(df_group1))
    fig.add_trace(go.Scatter(x=df_group1[column], y=jitter1,
                           mode='markers', name=group1, opacity=0.7,
                           marker=dict(size=5, color='blue'),
                           text=df_group1.index, 
                           hovertemplate=f'<b>{group1}</b><br>{column}: %{{x}}<br>Índice: %{{text}}<extra></extra>'))
    
    # Agregar dots para grupo 2
    jitter2 = np.random.normal(2, 0.05, len(df_group2))
    fig.add_trace(go.Scatter(x=df_group2[column], y=jitter2,
                           mode='markers', name=group2, opacity=0.7,
                           marker=dict(size=5, color='red'),
                           text=df_group2.index,
                           hovertemplate=f'<b>{group2}</b><br>{column}: %{{x}}<br>Índice: %{{text}}<extra></extra>'))
    
    # Calcular y agregar líneas de media
    mean1 = df_group1[column].mean()
    mean2 = df_group2[column].mean()
    
    fig.add_vline(x=mean1, line_dash="dash", line_color="blue", line_width=2,
                  annotation_text=f"Media {group1}: {mean1:.2f}       ",
                  annotation_position="top left")
    fig.add_vline(x=mean2, line_dash="dash", line_color="red", line_width=2,
                  annotation_text=f"         Media {group2}: {mean2:.2f}",
                  annotation_position="top right")
    
    # Mejorar el layout
    fig.update_layout(
        title=title, 
        xaxis_title=column,
        yaxis_title="Grupos",
        yaxis=dict(
            tickmode='array', 
            tickvals=[1, 2], 
            ticktext=[f'{group1} (n={len(df_group1)})', f'{group2} (n={len(df_group2)})']
        ),
        height=500,
        showlegend=True
    )
    
    return fig


def plot_interactive_density_multiclass(df: pd.DataFrame, column: str, class_column: str,
                                       title: Optional[str] = None):
    """
    Gráfico de densidad con múltiples clases en diferentes colores.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de pandas con los datos a graficar.
    column : str
        Nombre de la columna numérica para calcular la densidad.
    class_column : str
        Nombre de la columna que contiene las clases.
    title : Optional[str], default None
        Título personalizado para el gráfico.
        
    Returns
    -------
    plotly.graph_objects.Figure
        Figura interactiva de Plotly con las curvas de densidad por clase.
    """
    if title is None:
        title = f'Distribución de Densidad por Clase - {column}'
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1
    
    for i, class_val in enumerate(df[class_column].unique()):
        class_data = df[df[class_column] == class_val][column].dropna()
        
        if len(class_data) > 1:
            # Calcular KDE
            kde_x = np.linspace(df[column].min(), df[column].max(), 100)
            kde_y = stats.gaussian_kde(class_data)(kde_x)
            
            fig.add_trace(go.Scatter(x=kde_x, y=kde_y, mode='lines',
                                   name=f'{class_column}: {class_val}',
                                   line=dict(color=colors[i % len(colors)], width=3),
                                   fill='tonexty' if i > 0 else None,
                                   fillcolor=colors[i % len(colors)],
                                   opacity=0.3))
    
    fig.update_layout(title=title, xaxis_title=column, yaxis_title='Densidad')
    return fig


def plot_interactive_violin_swarm(df: pd.DataFrame, column: str, group_by: str,
                                 max_groups: int = 15, show_points: bool = True,
                                 title: Optional[str] = None):
    """
    Gráfico de violín interactivo con distribución de densidad.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de pandas con los datos a graficar.
    column : str
        Nombre de la columna numérica a graficar.
    group_by : str
        Nombre de la columna de agrupación categórica.
    max_groups : int, default 15
        Número máximo de grupos a mostrar (selecciona los más frecuentes).
    show_points : bool, default True
        Si mostrar puntos individuales (outliers) dentro del violín.
    title : Optional[str], default None
        Título personalizado para el gráfico.
        
    Returns
    -------
    plotly.graph_objects.Figure
        Figura interactiva de Plotly con los gráficos de violín.
        
    Raises
    ------
    ValueError
        Si las columnas especificadas no existen o si la columna numérica
        no es de tipo numérico.
    """
    # Validaciones
    if column not in df.columns:
        raise ValueError(f"La columna '{column}' no existe en el DataFrame")
    if group_by not in df.columns:
        raise ValueError(f"La columna '{group_by}' no existe en el DataFrame")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"La columna '{column}' debe ser numérica")
    
    if title is None:
        title = f'Distribución Violin Plot - {column} por {group_by}'
    
    # Filtrar grupos más frecuentes si hay demasiados
    group_counts = df[group_by].value_counts()
    if len(group_counts) > max_groups:
        top_groups = group_counts.head(max_groups).index.tolist()
        df_filtered = df[df[group_by].isin(top_groups)].copy()
        title += f' (Top {max_groups} grupos)'
        print(f"⚠️  Mostrando solo los {max_groups} grupos más frecuentes de {len(group_counts)} totales")
    else:
        df_filtered = df.copy()
    
    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    
    groups_ordered = df_filtered[group_by].value_counts().index.tolist()
    
    for i, group in enumerate(groups_ordered):
        group_data = df_filtered[df_filtered[group_by] == group][column].dropna()
        
        if len(group_data) == 0:
            continue
            
        color = colors[i % len(colors)]
        
        # Violin plot principal
        fig.add_trace(go.Violin(
            y=group_data, 
            x=[str(group)] * len(group_data),
            name=f'{group} (n={len(group_data)})', 
            box_visible=True,           # Mostrar boxplot interno
            box=dict(                   # Configuración específica de la caja
                visible=True,
                fillcolor='rgba(255,255,255,0.8)',
                line=dict(color='black', width=2),
                width=0.3
            ),
            meanline_visible=True,      # Mostrar línea de media
            meanline=dict(              # Configuración de línea de media
                visible=True,
                color='red',
                width=2
            ),
            fillcolor=color, 
            opacity=0.7,
            line_color=color,
            line=dict(width=2),
            points='suspectedoutliers' if show_points else False,  # Mostrar outliers sospechosos
            pointpos=0,                 # Centrar puntos
            jitter=0.3,                # Dispersión horizontal de puntos
            scalemode='width',          # Escalar por ancho
            side='both',                # Violin completo (ambos lados)
            bandwidth=None,             # Ancho de banda automático para suavizado
            spanmode='hard',            # Límites duros en los extremos
            hoveron='points+kde',       # Hover en puntos y curva KDE
            hovertemplate=f'<b>{group}</b><br>{column}: %{{y}}<br>Grupo: {group}<extra></extra>'
        ))
    
    # Calcular estadísticas resumidas
    stats_summary = []
    for group in groups_ordered:
        group_data = df_filtered[df_filtered[group_by] == group][column].dropna()
        if len(group_data) > 0:
            stats_summary.append({
                'Grupo': str(group),
                'n': len(group_data),
                'Media': f"{group_data.mean():.2f}",
                'Mediana': f"{group_data.median():.2f}",
                'Std': f"{group_data.std():.2f}",
                'Min': f"{group_data.min():.2f}",
                'Max': f"{group_data.max():.2f}"
            })
    
    # Layout optimizado para violin plots
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font_size=16
        ),
        xaxis=dict(
            title=group_by,
            showgrid=False,
            showline=True,
            linewidth=2,
            linecolor='black'
        ),
        yaxis=dict(
            title=column,
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            showline=True,
            linewidth=2,
            linecolor='black'
        ),
        height=600,
        width=max(800, len(groups_ordered) * 120),  # Ancho dinámico
        hovermode='closest',
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        ),
        plot_bgcolor='white',
        font=dict(size=12)
    )
    
    # Agregar estadísticas como anotación si hay pocos grupos
    if len(stats_summary) <= 8:
        stats_text = "📊 Estadísticas por grupo:\n" + "\n".join([
            f"{s['Grupo']}: μ={s['Media']}, σ={s['Std']}, n={s['n']}"
            for s in stats_summary
        ])
        
        fig.add_annotation(
            text=stats_text,
            showarrow=False,
            x=1.15,
            y=0.5,
            xref="paper",
            yref="paper",
            align="left",
            font=dict(size=10, family="monospace"),
            bgcolor="rgba(240,240,240,0.8)",
            bordercolor="gray",
            borderwidth=1,
            borderpad=10
        )
    
    return fig


def plot_interactive_correlation_heatmap(df: pd.DataFrame, method: str = 'pearson',
                                       annot: bool = True, title: Optional[str] = None,
                                       max_variables: int = 20, show_only_significant: bool = False,
                                       threshold: float = 0.3, save_plot: bool = True):
    """
    Heatmap de correlación interactivo con anotaciones y selección de método.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de pandas con los datos para calcular correlaciones.
    method : str, default 'pearson'
        Método de correlación a utilizar ('pearson', 'spearman', 'kendall').
    annot : bool, default True
        Si mostrar anotaciones con los valores de correlación en el heatmap.
    title : Optional[str], default None
        Título personalizado para el gráfico.
    max_variables : int, default 20
        Número máximo de variables numéricas a incluir en el análisis.
    show_only_significant : bool, default False
        Si mostrar solo las correlaciones que superen el umbral de significancia.
    threshold : float, default 0.3
        Umbral mínimo de correlación absoluta para considerar significativa.
    save_plot : bool, default True
        Si guardar automáticamente el gráfico en la carpeta plots/.
        
    Returns
    -------
    plotly.graph_objects.Figure
        Figura interactiva de Plotly con el heatmap de correlación.
        
    Raises
    ------
    ValueError
        Si el método de correlación no es válido, si no hay suficientes
        variables numéricas, o si ocurre un error en el cálculo.
    """
    # Validaciones
    if method not in ['pearson', 'spearman', 'kendall']:
        raise ValueError("method debe ser 'pearson', 'spearman' o 'kendall'")
    
    if title is None:
        title = f'Matriz de Correlación ({method.capitalize()})'
    
    # Seleccionar solo columnas numéricas
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) == 0:
        raise ValueError("No hay columnas numéricas en el DataFrame para calcular correlaciones")
    
    if len(numeric_df.columns) == 1:
        raise ValueError("Se necesitan al menos 2 variables numéricas para calcular correlaciones")
    
    # Limitar número de variables si es necesario
    if len(numeric_df.columns) > max_variables:
        # Seleccionar variables con mayor varianza (más informativas)
        variances = numeric_df.var().sort_values(ascending=False)
        selected_columns = variances.head(max_variables).index.tolist()
        numeric_df = numeric_df[selected_columns]
        title += f' (Top {max_variables} variables)'
        print(f"⚠️  Mostrando {max_variables} de {len(df.select_dtypes(include=[np.number]).columns)} variables numéricas")
    
    # Calcular correlación
    try:
        corr_matrix = numeric_df.corr(method=method)
    except Exception as e:
        raise ValueError(f"Error calculando correlación {method}: {str(e)}")
    
    # Verificar que no hay solo NaN
    if corr_matrix.isna().all().all():
        raise ValueError("La matriz de correlación contiene solo valores NaN")
    
    # Filtrar correlaciones significativas si se solicita
    if show_only_significant:
        # Crear máscara para correlaciones no significativas
        significant_mask = np.abs(corr_matrix) >= threshold
        # Mantener la diagonal (correlación consigo mismo = 1)
        np.fill_diagonal(significant_mask.values, True)
        corr_matrix = corr_matrix.where(significant_mask)
        title += f' (|r| ≥ {threshold})'
    
    # Configurar escala de colores según método
    if method == 'pearson':
        color_scale = 'RdBu_r'
    elif method == 'spearman':
        color_scale = 'RdYlBu_r'
    else:  # kendall
        color_scale = 'Viridis'
    
    # Crear heatmap
    fig = px.imshow(
        corr_matrix, 
        title=title,
        color_continuous_scale=color_scale,
        range_color=[-1, 1],
        aspect='auto',
        text_auto=False,  # Siempre False para evitar duplicados
        labels=dict(color=f'Correlación {method}'),
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.columns.tolist()
    )
    
    # Agregar valores de texto si se solicita
    if annot:
        # Crear matriz de texto sin NaN
        text_matrix = np.round(corr_matrix.values, 2)
        text_matrix = np.where(np.isnan(text_matrix), '', text_matrix.astype(str))
        
        fig.update_traces(
            text=text_matrix,
            texttemplate='%{text}',
            textfont=dict(size=10, family="Arial")
        )
    
    # Calcular dimensiones dinámicas
    n_vars = len(corr_matrix.columns)
    fig_size = max(500, min(1200, n_vars * 50))
    
    # Personalizar layout
    fig.update_layout(
        width=fig_size,
        height=fig_size,
        xaxis=dict(
            title='Variables',
            tickangle=45,
            side='bottom',
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            title='Variables',
            tickfont=dict(size=10)
        ),
        coloraxis_colorbar=dict(
            title=f'Correlación<br>{method}',
            thickness=20,
            len=0.8,
            tickfont=dict(size=10)
        ),
        font=dict(size=12),
        margin=dict(l=100, r=100, t=100, b=100)
    )
    
    # Guardar gráfico automáticamente
    if save_plot:
        plot_name = f"correlation_heatmap_{method}"
        if show_only_significant:
            plot_name += f"_threshold_{threshold}"
        _save_plot(fig, plot_name, save_plot)
    
    return fig
