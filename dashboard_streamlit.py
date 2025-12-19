import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import radians, sin, cos, sqrt, asin

# Configuración de la página
st.set_page_config(
    page_title="Dashboard de Restaurantes",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🍽️ Dashboard de Análisis de Restaurantes")
st.markdown("---")

@st.cache_data
def cargar_datos():
    """Cargar y procesar todos los datos"""
    # Intentar cargar el dataset ya procesado
    try:
        df = pd.read_csv('datos/datos_limpios.csv')
        # Convertir rating_category a categórico
        df['rating_category'] = pd.Categorical(
            df['rating_category'], 
            categories=['Bajo', 'Medio', 'Alto'], 
            ordered=True
        )
        return df
    except FileNotFoundError:
        # Si no existe, procesar desde cero
        st.warning("⚠️ No se encontró dataset_final_limpio.csv, procesando datos desde cero...")
        
        # Cargar datasets
        users = pd.read_csv('datos/users.csv')
        usercuisine = pd.read_csv('datos/usercuisine.csv')
        userpayment = pd.read_csv('datos/userpayment.csv')
        ratings = pd.read_csv('datos/ratings.csv')
        restaurants = pd.read_csv('datos/restaurants.csv')
        cuisine = pd.read_csv('datos/cuisine.csv')
        payment_methods = pd.read_csv('datos/payment_methods.csv')
        
        # Reemplazar '?' con NaN
        dataframes = [users, usercuisine, userpayment, ratings, restaurants, cuisine, payment_methods]
        for df in dataframes:
            df.replace('?', np.nan, inplace=True)
        
        # Crear variables
        df = ratings.copy()
        
        # Merge con users y restaurants
        df = df.merge(
            users[['userID', 'latitude', 'longitude', 'smoker', 'ambience']],
            on='userID'
        )
        df = df.merge(
            restaurants[['placeID', 'latitude', 'longitude', 'smoking_area', 'Rambience', 'alcohol']],
            on='placeID',
            suffixes=('_user', '_rest')
        )
        
        # Calcular distancia
        def haversine(lat1, lon1, lat2, lon2):
            R = 6371
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            return R * 2 * asin(sqrt(a))
        
        df['distancia_km'] = df.apply(
            lambda row: haversine(row['latitude_user'], row['longitude_user'],
                                 row['latitude_rest'], row['longitude_rest']),
            axis=1
        )
        
        # Match cocina
        user_cuisine_dict = usercuisine.groupby('userID')['Rcuisine'].apply(set).to_dict()
        rest_cuisine_dict = cuisine.groupby('placeID')['Rcuisine'].apply(set).to_dict()
        
        def check_cocina_match(user_id, place_id):
            user_set = user_cuisine_dict.get(user_id, set())
            rest_set = rest_cuisine_dict.get(place_id, set())
            return 1 if len(user_set & rest_set) > 0 else 0
        
        df['match_cocina'] = df.apply(
            lambda row: check_cocina_match(row['userID'], row['placeID']), axis=1
        )
        
        # Compatibilidad fumador
        def compat_fumador(smoker, smoking_area):
            if smoker == 'true':
                return 2 if smoking_area != 'none' else 0
            else:
                return 1
        
        df['compat_fumador'] = df.apply(
            lambda row: compat_fumador(row['smoker'], row['smoking_area']), axis=1
        )
        
        # Payment match ratio
        user_payment_dict = userpayment.groupby('userID')['Upayment'].apply(set).to_dict()
        rest_payment_dict = payment_methods.groupby('placeID')['Rpayment'].apply(set).to_dict()
        
        def payment_match_ratio(user_id, place_id):
            user_set = user_payment_dict.get(user_id, set())
            rest_set = rest_payment_dict.get(place_id, set())
            if len(user_set) == 0:
                return 0
            return len(user_set & rest_set) / len(user_set)
        
        df['payment_match_ratio'] = df.apply(
            lambda row: payment_match_ratio(row['userID'], row['placeID']), axis=1
        )
        
        # Match ambiente
        df['match_ambiente'] = (df['ambience'] == df['Rambience']).astype(int)
        
        # Rating category
        df['rating_category'] = pd.cut(
            df['rating'],
            bins=[-float('inf'), 0, 1, float('inf')],
            labels=['Bajo', 'Medio', 'Alto']
        )
        
        return df

# Cargar datos
with st.spinner('Cargando datos...'):
    df = cargar_datos()

# Sidebar - Filtros
st.sidebar.header("🔍 Filtros")

# Sección 1: Filtros de Rating
st.sidebar.subheader("📊 Rating")
categorias = st.sidebar.multiselect(
    "Categoría de Rating",
    options=['Bajo', 'Medio', 'Alto'],
    default=['Bajo', 'Medio', 'Alto']
)

rating_range = st.sidebar.slider(
    "Rango de Rating",
    min_value=float(df['rating'].min()),
    max_value=float(df['rating'].max()),
    value=(float(df['rating'].min()), float(df['rating'].max())),
    step=0.1
)

st.sidebar.markdown("---")

# Sección 2: Filtros de Ubicación
st.sidebar.subheader("📍 Ubicación y Distancia")
distancia_max = st.sidebar.slider(
    "Distancia máxima (km)",
    min_value=0.0,
    max_value=float(df['distancia_km'].max()),
    value=float(df['distancia_km'].max()),
    step=0.5
)

st.sidebar.markdown("---")

# Sección 3: Filtros de Preferencias de Match
st.sidebar.subheader("🎯 Variables de Match")

# Match de cocina
match_cocina_filter = st.sidebar.radio(
    "Match de Cocina",
    options=["Todos", "Con Match", "Sin Match"]
)

# Match de ambiente
match_ambiente_filter = st.sidebar.radio(
    "Match de Ambiente",
    options=["Todos", "Con Match", "Sin Match"]
)

# Compatibilidad fumador
compat_fumador_filter = st.sidebar.multiselect(
    "Compatibilidad Fumador",
    options=[0, 1, 2],
    default=[0, 1, 2],
    help="0=Fumador sin área, 1=No fumador, 2=Fumador con área"
)

# Payment match ratio
payment_ratio_min = st.sidebar.slider(
    "Payment Match Ratio Mínimo",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.1,
    help="Proporción de métodos de pago coincidentes"
)

st.sidebar.markdown("---")

# Sección 4: Filtros adicionales
st.sidebar.subheader("🔧 Filtros Adicionales")

# Filtro por alcohol
if 'alcohol' in df.columns:
    alcohol_options = df['alcohol'].dropna().unique().tolist()
    if alcohol_options:
        alcohol_filter = st.sidebar.multiselect(
            "Disponibilidad de Alcohol",
            options=alcohol_options,
            default=alcohol_options
        )
    else:
        alcohol_filter = None
else:
    alcohol_filter = None

# Filtro por smoker
smoker_filter = st.sidebar.multiselect(
    "Tipo de Usuario",
    options=df['smoker'].dropna().unique().tolist(),
    default=df['smoker'].dropna().unique().tolist()
)

st.sidebar.markdown("---")

# Aplicar filtros
df_filtrado = df.copy()

# Filtros de rating
df_filtrado = df_filtrado[df_filtrado['rating_category'].isin(categorias)]
df_filtrado = df_filtrado[
    (df_filtrado['rating'] >= rating_range[0]) & 
    (df_filtrado['rating'] <= rating_range[1])
]

# Filtros de distancia
df_filtrado = df_filtrado[df_filtrado['distancia_km'] <= distancia_max]

# Filtros de match
if match_cocina_filter == "Con Match":
    df_filtrado = df_filtrado[df_filtrado['match_cocina'] == 1]
elif match_cocina_filter == "Sin Match":
    df_filtrado = df_filtrado[df_filtrado['match_cocina'] == 0]

if match_ambiente_filter == "Con Match":
    df_filtrado = df_filtrado[df_filtrado['match_ambiente'] == 1]
elif match_ambiente_filter == "Sin Match":
    df_filtrado = df_filtrado[df_filtrado['match_ambiente'] == 0]

df_filtrado = df_filtrado[df_filtrado['compat_fumador'].isin(compat_fumador_filter)]
df_filtrado = df_filtrado[df_filtrado['payment_match_ratio'] >= payment_ratio_min]

# Filtros adicionales
if alcohol_filter and 'alcohol' in df.columns:
    df_filtrado = df_filtrado[df_filtrado['alcohol'].isin(alcohol_filter)]

df_filtrado = df_filtrado[df_filtrado['smoker'].isin(smoker_filter)]

# Mostrar estadísticas de filtrado
st.sidebar.markdown("---")
st.sidebar.info(f"📊 Registros: {len(df_filtrado):,} de {len(df):,}")
porcentaje_mostrado = (len(df_filtrado) / len(df)) * 100
st.sidebar.success(f"Mostrando: {porcentaje_mostrado:.1f}%")

# KPIs principales
st.header("📈 KPIs Principales")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Rating Promedio",
        value=f"{df_filtrado['rating'].mean():.2f}",
        delta=f"{(df_filtrado['rating'].mean() - df['rating'].mean()):.2f}"
    )

with col2:
    pct_altos = (df_filtrado['rating_category'] == 'Alto').mean() * 100
    pct_altos_total = (df['rating_category'] == 'Alto').mean() * 100
    st.metric(
        label="% Ratings Altos",
        value=f"{pct_altos:.1f}%",
        delta=f"{(pct_altos - pct_altos_total):.1f}%"
    )

with col3:
    st.metric(
        label="Distancia Promedio",
        value=f"{df_filtrado['distancia_km'].mean():.1f} km",
        delta=f"{(df_filtrado['distancia_km'].mean() - df['distancia_km'].mean()):.1f} km",
        delta_color="inverse"
    )

with col4:
    pct_match = df_filtrado['match_cocina'].mean() * 100
    pct_match_total = df['match_cocina'].mean() * 100
    st.metric(
        label="% Match Cocina",
        value=f"{pct_match:.1f}%",
        delta=f"{(pct_match - pct_match_total):.1f}%"
    )

st.markdown("---")

# Tabs para diferentes visualizaciones
tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Mapa", "📊 Análisis de Distancia", "🎯 Variables de Match", "📋 Datos"])

with tab1:
    st.subheader("Mapa de Calor: Rating Promedio por Ubicación")
    
    # Agrupar datos
    df_mapa = df_filtrado.groupby(['latitude_rest', 'longitude_rest']).agg({
        'rating': 'mean',
        'placeID': 'count'
    }).reset_index()
    df_mapa.columns = ['latitude_rest', 'longitude_rest', 'rating_promedio', 'num_calificaciones']
    
    # Crear mapa
    fig_mapa = px.density_map(
        df_mapa,
        lat='latitude_rest',
        lon='longitude_rest',
        z='rating_promedio',
        radius=15,
        center=dict(lat=df_mapa['latitude_rest'].mean(), 
                   lon=df_mapa['longitude_rest'].mean()),
        zoom=5,
        map_style="carto-positron",
        title="Rating Promedio por Ubicación de Restaurante",
        height=600
    )
    
    st.plotly_chart(fig_mapa, use_container_width=True)
    
    # Mostrar top restaurantes
    st.subheader("📍 Top 10 Ubicaciones con Mejor Rating")
    top_lugares = df_mapa.nlargest(10, 'rating_promedio')[
        ['latitude_rest', 'longitude_rest', 'rating_promedio', 'num_calificaciones']
    ]
    st.dataframe(top_lugares, use_container_width=True)

with tab2:
    st.subheader("Análisis de Distancia vs Rating")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Boxplot
        fig_box = px.box(
            df_filtrado,
            x='rating_category',
            y='distancia_km',
            color='rating_category',
            title="Distancia vs Categoría de Rating",
            labels={'distancia_km': 'Distancia (km)', 'rating_category': 'Categoría Rating'}
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    with col2:
        # Histograma
        fig_hist = px.histogram(
            df_filtrado,
            x='distancia_km',
            color='rating_category',
            nbins=30,
            title="Distribución de Distancia por Categoría",
            labels={'distancia_km': 'Distancia (km)'},
            barmode='overlay',
            opacity=0.7
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Scatter plot
    fig_scatter = px.scatter(
        df_filtrado,
        x='distancia_km',
        y='rating',
        color='rating_category',
        size='match_cocina',
        title="Rating vs Distancia (tamaño = match cocina)",
        labels={'distancia_km': 'Distancia (km)', 'rating': 'Rating'},
        height=400
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with tab3:
    st.subheader("Variables de Match vs Rating")
    
    # Crear subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Match Cocina', 'Compatibilidad Fumador', 'Match Ambiente')
    )
    
    # Match cocina
    match_cocina_data = df_filtrado.groupby(['match_cocina', 'rating_category'], observed=False).size().unstack(fill_value=0)
    for categoria in ['Bajo', 'Medio', 'Alto']:
        if categoria in match_cocina_data.columns:
            fig.add_trace(
                go.Bar(x=match_cocina_data.index, y=match_cocina_data[categoria], name=categoria),
                row=1, col=1
            )
    
    # Compatibilidad fumador
    compat_data = df_filtrado.groupby(['compat_fumador', 'rating_category'], observed=False).size().unstack(fill_value=0)
    for categoria in ['Bajo', 'Medio', 'Alto']:
        if categoria in compat_data.columns:
            fig.add_trace(
                go.Bar(x=compat_data.index, y=compat_data[categoria], name=categoria, showlegend=False),
                row=1, col=2
            )
    
    # Match ambiente
    ambiente_data = df_filtrado.groupby(['match_ambiente', 'rating_category'], observed=False).size().unstack(fill_value=0)
    for categoria in ['Bajo', 'Medio', 'Alto']:
        if categoria in ambiente_data.columns:
            fig.add_trace(
                go.Bar(x=ambiente_data.index, y=ambiente_data[categoria], name=categoria, showlegend=False),
                row=1, col=3
            )
    
    fig.update_layout(height=500, title_text="Variables de Match vs Rating")
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabla de estadísticas
    st.subheader("📊 Estadísticas de Match")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Match Cocina Promedio", f"{df_filtrado['match_cocina'].mean()*100:.1f}%")
    with col2:
        st.metric("Compat. Fumador Promedio", f"{df_filtrado['compat_fumador'].mean():.2f}")
    with col3:
        st.metric("Match Ambiente Promedio", f"{df_filtrado['match_ambiente'].mean()*100:.1f}%")

with tab4:
    st.subheader("📋 Vista de Datos")
    
    # Selector de columnas
    columnas_mostrar = st.multiselect(
        "Selecciona las columnas a mostrar",
        options=df_filtrado.columns.tolist(),
        default=['userID', 'placeID', 'rating', 'rating_category', 'distancia_km', 
                'match_cocina', 'compat_fumador', 'match_ambiente']
    )
    
    # Mostrar datos
    st.dataframe(
        df_filtrado[columnas_mostrar],
        use_container_width=True,
        height=400
    )
    
    # Botón de descarga
    csv = df_filtrado[columnas_mostrar].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descargar datos filtrados (CSV)",
        data=csv,
        file_name='datos_filtrados.csv',
        mime='text/csv',
    )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Dashboard de Análisis de Restaurantes | Módulo 1 - Ciencia de Datos</p>
    </div>
    """,
    unsafe_allow_html=True
)
