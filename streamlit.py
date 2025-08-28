# Cargue de librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import kagglehub
import os
import altair as alt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from matplotlib import pyplot
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import mca
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import streamlit as st
from graphviz import Digraph

import warnings
warnings.filterwarnings("ignore")


st.set_page_config(page_title="Cirrosis Hepatica Streamlit App", layout="wide")
st.title("Clasificación de los estadios de la cirrosis hepática con métodos de Machine Learning")

# ----------------------------
# Sección de Metodología
# ----------------------------
st.title("🧪 Metodología del Proyecto")

st.markdown("""
Este proyecto sigue una **metodología de Machine Learning** para la clasificación de la cirrosis hepática.  
A continuación, se presentan los pasos de manera interactiva:
""")

# Paso 1
with st.expander("📌 Paso 1: Carga y Análisis Exploratorio de Datos"):
    st.write("""
    - Se utilizó un dataset con información clínica de pacientes.  
    - El archivo fue almacenado en GitHub y cargado en streamlit.  
    - Se revisó la calidad de los datos para identificar valores nulos
    - Se crean dos secciones con filtros para revisar las variables categóricas y numéricas.
    """)

# Paso 2
with st.expander("📌 Paso 2: Preprocesamiento"):
    st.write("""
    - Limpieza de datos: imputación de valores faltantes.  
    - Codificación de variables categóricas (One-Hot Encoding).  
    - Estandarización de las variables numéricas.  
    """)

# Paso 3
with st.expander("📌 Paso 3: Selección de características"):
    st.write("""
    - Se utilizaron técnicas de filtrado de variables como: 
        - Variables categóricas: $\chi^2$ e información mutua
        - Variables numéricas: ANOVA e información mutua
    - MCA y PCA
    - RFE (Recursive Feature Elimination) con validación cruzada (selección por envoltura)
    - Esto permite quedarnos solo con las variables más relevantes para el modelo.  
    """)

# Paso 4
with st.expander("📌 Paso 4: Entrenamiento del modelo"):
    st.write("""
    - Se probaron algoritmos como **Decission tree**, **Regresión Logística**, **Random forest**, **KNN (K-Nearest Neighbors)** y **SVM (Support Vector Machine)**.  
    -   
    """)

# Paso 5
with st.expander("📌 Paso 5: Evaluación"):
    st.write("""
    - Se calcularon métricas como **Accuracy, Precision, Recall y F1-Score**.  
    - También se aplicó validación cruzada para obtener una estimación más robusta.  
    """)

st.subheader("🔎 Flujo Metodológico")

dot = Digraph()

dot.node("A", "Carga de Datos", shape="box")
dot.node("B", "Preprocesamiento", shape="box")
dot.node("C", "Selección de características", shape="box")
dot.node("D", "Entrenamiento del modelo\n(Logistic Regression, SVM)", shape="box")
dot.node("E", "Evaluación del modelo\n(Accuracy, Recall, F1-Score)", shape="box")

dot.edges(["AB", "BC", "CD", "DE"])

st.graphviz_chart(dot)

st.caption("Estudio clínico de cirrosis hepática — ficha de variables")

texto = """
### **Variables:**

* **N_Days**: Número de días transcurridos entre el registro y la fecha más temprana entre fallecimiento, trasplante o análisis del estudio en 1986.  
* **Status**: estado del paciente C (censurado), CL (censurado por tratamiento hepático) o D (fallecimiento).  
* **Drug**: tipo de fármaco: D-penicilamina o placebo.  
* **Age**: edad en días.  
* **Sex**: M (hombre) o F (mujer).  
* **Ascites**: presencia de ascitis N (No) o Y (Sí).  
* **Hepatomegaly**: presencia de hepatomegalia N (No) o Y (Sí).  
* **Spiders**: presencia de aracnosis N (No) o Y (Sí).  
* **Edema**: presencia de edema N (sin edema ni tratamiento diurético), S (edema presente sin diuréticos o resuelto con diuréticos) o Y (edema a pesar del tratamiento diurético).  
* **Bilirubin**: bilirrubina sérica en mg/dl.  
* **Cholesterol**: colesterol sérico en mg/dl.  
* **Albumin**: albúmina en g/dl.  
* **Copper**: cobre en orina en µg/día.  
* **Alk_Phos**: fosfatasa alcalina en U/litro.  
* **SGOT**: SGOT en U/ml.  
* **Tryglicerides**: triglicéridos en mg/dl.  
* **Platelets**: plaquetas por metro cúbico [ml/1000].  
* **Prothrombin**: tiempo de protrombina en segundos.  
* **Stage**: estadio histológico de la enfermedad (1, 2 o 3).  

---

### **Dimensiones del dataset**
- **Tamaño:** 25 000 filas, 19 columnas  
- **Faltantes:** 0% en todas las columnas  

---
"""

st.markdown(texto)

# --- Cargar dataset ---
# URL del CSV en GitHub (raw)
url = "https://raw.githubusercontent.com/DiegoNaranjo84/cirrosis_hepatica/main/liver_cirrhosis.csv"

# Cargar el dataset
df = pd.read_csv(url)

# Filtrar solo columnas categóricas (tipo "object" o "category")
df['Stage'] = pd.to_numeric(df['Stage'], errors='coerce')
df['Stage'] = pd.Categorical(df['Stage'], ordered=True)
cat_cols = df.select_dtypes(include=['object', 'category'])

st.subheader("Primeras 10 filas del dataset")
st.dataframe(df.head(10), use_container_width=True)

# ------- Helpers -------
def format_uniques(series, max_items=20):
    """Convierte valores únicos a una cadena legible, acota a max_items."""
    uniques = pd.Series(series.dropna().unique())
    head = uniques.head(max_items).astype(str).tolist()
    txt = ", ".join(head)
    if uniques.size > max_items:
        txt += f" … (+{uniques.size - max_items} más)"
    return txt

# ------- Detectar tipos -------
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = df.select_dtypes(include=["number"]).columns.tolist()

# ------- Resumen variables categóricas -------
cat_summary = pd.DataFrame({
    "Variable": cat_cols,
    "Tipo de dato": [df[c].dtype for c in cat_cols],
    "Nº de categorías únicas": [df[c].nunique(dropna=True) for c in cat_cols],
    "Nº de datos no nulos": [df[c].notna().sum() for c in cat_cols],
    "Categorías": [format_uniques(df[c], max_items=20) for c in cat_cols],
})

# ------- Resumen variables numéricas -------
num_summary = pd.DataFrame({
    "Variable": num_cols,
    "Tipo de dato": [df[c].dtype for c in num_cols],
    "Nº de datos no nulos": [df[c].notna().sum() for c in num_cols],
    "Mínimo": [df[c].min(skipna=True) for c in num_cols],
    "Máximo": [df[c].max(skipna=True) for c in num_cols],
    "Media":  [df[c].mean(skipna=True) for c in num_cols],
    "Desviación estándar": [df[c].std(skipna=True) for c in num_cols],
}).round(2)

# ------- Mostrar en dos columnas iguales con separación uniforme -------
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("Resumen variables categóricas")
    st.dataframe(cat_summary, use_container_width=True)

with col2:
    st.subheader("Resumen variables numéricas")
    st.dataframe(num_summary, use_container_width=True)

#####--------------------------------------------------------------------------------------#########

st.markdown("""### Análisis de variables categóricas""")
st.caption("Selecciona una variable para ver su distribución en tabla y gráfico de torta.")

variables_categoricas = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

if not variables_categoricas:
    st.warning("No se detectaron variables categóricas (object/category/bool) en df.")
    st.stop()

# =========================
# Controles (En la sección)
# =========================
st.markdown("*Controles*")
with st.container():
    c1, c2 = st.columns([1.6, 1.1])
    with c1:
        var = st.selectbox(
            "Variable categórica",
            options=variables_categoricas,
            index=0,
            key="cat_var_local"
        )
        top_n = st.slider(
            "Top N (agrupa el resto en 'Otros')",
            min_value=3, max_value=30, value=10, step=1,
            help="Agrupa las categorías menos frecuentes en 'Otros'",
            key="cat_topn_local"
        )
    with c2:
        orden_alfabetico = st.checkbox("Ordenar alfabéticamente (solo tabla)", value=False, key="cat_orden_local")

# =========================
# Preparar datos (siempre incluye NaN)
# =========================
serie = df[var].copy()

vc = serie.value_counts(dropna=True)  # Se mantienen los NaN para contarlos

# Etiqueta amigable para NaN
labels = vc.index.to_list()
labels = ["(NaN)" if (isinstance(x, float) and np.isnan(x)) else str(x) for x in labels]
counts = vc.values

data = pd.DataFrame({"Categoría": labels, "Conteo": counts})
data["Porcentaje"] = (data["Conteo"] / data["Conteo"].sum() * 100).round(2)

# Agrupar en "Otros" si supera Top N
if len(data) > top_n:
    top = data.nlargest(top_n, "Conteo").copy()
    otros = data.drop(top.index)
    fila_otros = pd.DataFrame({
        "Categoría": ["Otros"],
        "Conteo": [int(otros["Conteo"].sum())],
        "Porcentaje": [round(float(otros["Porcentaje"].sum()), 2)]
    })
    data_plot = pd.concat([top, fila_otros], ignore_index=True)
else:
    data_plot = data.copy()

# Orden por Conteo (siempre)
data_plot = data_plot.sort_values("Conteo", ascending=False).reset_index(drop=True)

# Orden opcional alfabético en la tabla (no afecta el gráfico)
data_table = data_plot.copy()
if orden_alfabetico:
    data_table = data_table.sort_values("Categoría").reset_index(drop=True)

# =========================
# Mostrar tabla y gráfico
# =========================
tcol, gcol = st.columns([1.1, 1.3], gap="large")

with tcol:
    st.subheader(f"Distribución de {var}")
    st.dataframe(
        data_table.assign(Porcentaje=data_table["Porcentaje"].round(2)),
        use_container_width=True
    )

with gcol:
    st.subheader("Gráfico de torta")
    chart = (
        alt.Chart(data_plot)
        .mark_arc(outerRadius=110)
        .encode(
            theta=alt.Theta(field="Conteo", type="quantitative"),
            color=alt.Color("Categoría:N", legend=alt.Legend(title="Categoría")),
            tooltip=[
                alt.Tooltip("Categoría:N"),
                alt.Tooltip("Conteo:Q", format=","),
                alt.Tooltip("Porcentaje:Q", format=".2f")
            ],
        )
        .properties(height=380)
    )
    st.altair_chart(chart, use_container_width=True)

# =========================
# Extras informativos
# =========================
st.divider()
c1, c2 = st.columns(2)
with c1:
    st.metric("Categorías mostradas", f"{len(data_plot)}")
with c2:
    st.metric("Total registros (variable seleccionada)", f"{int(serie.shape[0]):,}".replace(",", "."))

st.caption("Consejo: usa *Top N* para simplificar la lectura y agrupar categorías poco frecuentes en 'Otros'.")

#####--------------------------------------------------------------------------------------#########

# =========================
# Análisis de variables numéricas
# =========================
st.markdown("""### Análisis de variables numéricas""")
st.caption("Selecciona una variable para ver su distribución en tabla, boxplot e histograma.")

# Detectar variables numéricas
variables_numericas = df.select_dtypes(include=["number"]).columns.tolist()

if not variables_numericas:
    st.warning("No se detectaron variables numéricas en df.")
    st.stop()

# =========================
# Controles dentro de la sección
# =========================
st.markdown("*Controles*")
with st.container():
    c1, c2 = st.columns([1.6, 1.4])
    with c1:
        var_num = st.selectbox(
            "Variable numérica",
            options=variables_numericas,
            index=0,
            key="num_var_local"
        )
    with c2:
        bins = st.slider(
            "Número de bins (histograma)",
            min_value=5, max_value=100, value=30, step=5,
            key="num_bins_local"
        )

# =========================
# Preparar serie
# =========================
serie_num = df[var_num].dropna()

# =========================
# Métricas descriptivas
# =========================
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Nº datos no nulos", f"{serie_num.shape[0]:,}".replace(",", "."))
with c2:
    st.metric("Mínimo", f"{serie_num.min():.2f}")
with c3:
    st.metric("Máximo", f"{serie_num.max():.2f}")
with c4:
    st.metric("Media", f"{serie_num.mean():.2f}")
with c5:
    st.metric("Desv. Estándar", f"{serie_num.std():.2f}")

# =========================
# Gráficos
# =========================
g1, g2 = st.columns(2, gap="large")

# --- Boxplot vertical y ancho ---
with g1:
    st.subheader(f"Boxplot de {var_num} ")
    box_data = pd.DataFrame({var_num: serie_num})
    box_data["_grupo_"] = "Distribución"  # ancla un grupo único en X

    box_chart = (
        alt.Chart(box_data)
        .mark_boxplot(size=140, extent=1.5)  # size = ancho de la caja; extent=1.5 => whiskers tipo Tukey
        .encode(
            x=alt.X("_grupo_:N", axis=None, title=""),
            y=alt.Y(f"{var_num}:Q", title=var_num)
        )
        .properties(height=350)
    )
    st.altair_chart(box_chart, use_container_width=True)

# --- Histograma ---
with g2:
    st.subheader(f"Histograma de {var_num}")
    hist_data = pd.DataFrame({var_num: serie_num})
    hist_chart = (
        alt.Chart(hist_data)
        .mark_bar()
        .encode(
            alt.X(var_num, bin=alt.Bin(maxbins=bins)),
            y='count()',
            tooltip=[
                alt.Tooltip(var_num, bin=alt.Bin(maxbins=bins)),
                alt.Tooltip('count()', title="Frecuencia")
            ]
        )
        .properties(height=350)
    )
    st.altair_chart(hist_chart, use_container_width=True)

# =========================
# Matriz de Correlación
# =========================
st.markdown("### Matriz de Correlación")
correlacion = df.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlacion, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
ax.set_title("Matriz de Correlación")
st.pyplot(fig)
#________________________________________________________________________________________________________________________________________________________________




# ________________________________________________________________________________________________________________________________________________________________
st.markdown("""# 1. Selección de carácteristicas""")
# ________________________________________________________________________________________________________________________________________________________________
st.markdown("""## 1.1. Selección de carácteristicas categóricas""")
# ________________________________________________________________________________________________________________________________________________________________
st.markdown("""## 1.2. Selección de carácteristicas numéricas""")
# ________________________________________________________________________________________________________________________________________________________________
st.markdown("""## 1.3. Unión de variables categóricas y númericas""")



# ________________________________________________________________________________________________________________________________________________________________
st.markdown("""# 2. MCA Y PCA""")
# ________________________________________________________________________________________________________________________________________________________________
# ________________________________________________________________________________________________________________________________________________________________
st.markdown("""## 2.1. MCA""")

# split into train and test sets
df_cat = df.select_dtypes(include=['object', 'category'])
df_cat.info()

X = df_cat
y = df['Stage']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.33, random_state=1
)

# Codificación del conjunto de entrenamiento
X_train_encoded = pd.get_dummies(X_train)

# Codificación del conjunto de prueba
X_test_encoded = pd.get_dummies(X_test)

# Aplicar MCA
mca_cirrosis = mca.MCA(X_train_encoded, benzecri=True)

# Valores singulares y autovalores
sv = mca_cirrosis.s
eigvals = sv ** 2
explained_var = eigvals / eigvals.sum()
cum_explained_var = np.cumsum(explained_var)

# Graficar varianza acumulada
fig1, ax1 = plt.subplots(figsize=(8,5))
ax1.plot(range(1, len(cum_explained_var)+1), cum_explained_var, marker='o', linestyle='--')
ax1.axhline(y=0.8, color='r', linestyle='-')
ax1.set_xlabel('Dimensiones MCA')
ax1.set_ylabel('Varianza acumulada explicada')
ax1.set_title('Varianza acumulada explicada por MCA')
ax1.grid(True)
st.pyplot(fig1)

n_dims_90 = np.argmax(cum_explained_var >= 0.8) + 1  # +1 porque los índices empiezan en 0
st.write(f'Se necesitan {n_dims_90} dimensiones para explicar al menos el 80% de la varianza.')

# Coordenadas individuos (2 primeras dimensiones)
coords = mca_cirrosis.fs_r(N=3)

fig2, ax2 = plt.subplots(figsize=(8,6))
sns.scatterplot(x=coords[:,0], y=coords[:,1], hue=y_train, palette='Set1', alpha=0.7, ax=ax2)
ax2.set_xlabel('Dimensión 1')
ax2.set_ylabel('Dimensión 2')
ax2.set_title('Scatterplot MCA Dim 1 vs Dim 2')
ax2.legend(title='Clase', labels=['Estadio 1', 'Estadio 2','Estadio 3'])
st.pyplot(fig2)

# Cargas variables categóricas (loadings) primeras 2 dimensiones
loadings_cat = pd.DataFrame(mca_cirrosis.fs_c()[:, :2], index=X_train_encoded.columns)

# Calcular contribución de cada variable (cuadrado / suma por dimensión)
loadings_sq = loadings_cat ** 2
contrib_cat = loadings_sq.div(loadings_sq.sum(axis=0), axis=1)

# Sumar contribuciones por variable
contrib_var = contrib_cat.sum(axis=1).sort_values(ascending=False)

fig3, ax3 = plt.subplots(figsize=(12,6))
contrib_var.plot(kind='bar', color='teal', ax=ax3)
ax3.set_ylabel('Contribución total a Dim 1 y 2')
ax3.set_title('Contribución de variables a las primeras 2 dimensiones MCA')
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
fig3.tight_layout()
st.pyplot(fig3)

# ________________________________________________________________________________________________________________________________________________________________
st.markdown("""## 2.2. PCA""")

df_num = df.select_dtypes(include=['int64', 'float64'])
df_num.info()

X = df_num
y = df['Stage']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.33, random_state=1
)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# PCA con todos los componentes
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Varianza acumulada
explained_var = np.cumsum(pca.explained_variance_ratio_)

fig, ax = plt.subplots(figsize=(8,5))
ax.plot(range(1, len(explained_var) + 1), explained_var, marker='o', linestyle='--')
ax.axhline(y=0.8, color='r', linestyle='-')
ax.set_xlabel('Número de componentes principales')
ax.set_ylabel('Varianza acumulada explicada')
ax.set_title('Varianza acumulada explicada por PCA')
ax.grid(True)
st.pyplot(fig)

n_dims_90 = np.argmax(explained_var >= 0.8) + 1
st.write(f'Se necesitan {n_dims_90} dimensiones para explicar al menos el 80% de la varianza.')

# Scatterplot PC1 vs PC2
fig2, ax2 = plt.subplots(figsize=(8,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y_train, palette='Set1', alpha=0.7, ax=ax2)
ax2.set_xlabel('PC1')
ax2.set_ylabel('PC2')
ax2.set_title('Scatterplot PC1 vs PC2')
ax2.legend(title='Clase', labels=['Estadio 1', 'Estadio 2', 'Estadio 3'])
st.pyplot(fig2)

loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(pca.n_components_)],
    index=X_train.columns
)

#ráfico en 3D PCA

# Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# PCA con 3 componentes
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Crear DataFrame PCA
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])

# Asegurar que y_train está alineado y convertir a DataFrame
df_pca['Clase'] = y_train.values

# Mapeo correcto de clases 1, 2, 3
df_pca['Clase'] = df_pca['Clase'].astype(int).map({
    1: 'Estadio 1',
    2: 'Estadio 2',
    3: 'Estadio 3'
})

# Agrega hover con el índice si no tienes otra variable informativa
df_pca['ID'] = df_pca.index.astype(str)

# Mostrar el DataFrame por si acaso
# st.write(df_pca.head())

# Crear gráfico interactivo
fig = px.scatter_3d(
    df_pca,
    x='PC1',
    y='PC2',
    z='PC3',
    color='Clase',
    hover_name='ID',
    color_discrete_sequence=px.colors.qualitative.Set1,
    title='PCA 3D - Componentes Principales',
    labels={'Clase': 'Estadio'},
    opacity=0.7
)

st.plotly_chart(fig)

fig3, ax3 = plt.subplots(figsize=(12,8))
sns.heatmap(loadings.iloc[:, :9], annot=True, cmap='coolwarm', center=0, ax=ax3)
ax3.set_title('Heatmap de loadings (primeras 9 PCs)')
st.pyplot(fig3)

# PCA con componentes que explican al menos 80% varianza
pca_80 = PCA(n_components=0.80)
X_pca_80 = pca_80.fit_transform(X_scaled)

st.write(f"Número de componentes principales para explicar 80% varianza: {pca_80.n_components_}")
st.write(f"Varianza explicada acumulada por estas componentes: {sum(pca_80.explained_variance_ratio_)*100:.4f}%")



# ________________________________________________________________________________________________________________________________________________________________


st.markdown("""# 3. RFE""")

# Convertir Stage a categórica
df["Stage"] = df["Stage"].astype("category")

# Definir variables categóricas y numéricas
categorical = df.select_dtypes(include=["object","category"])
categorical_features = categorical.columns.drop("Stage").tolist()
numerical_features = df.select_dtypes(include=["int64","float64"]).columns.tolist()

# Separar X e y
X = df[categorical_features + numerical_features]
y = df["Stage"]

# Partición train-test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42, stratify=y
)

# OneHotEncoder compatible con distintas versiones de sklearn
try:
    OHE = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    OHE = OneHotEncoder(handle_unknown="ignore", sparse=False)

# Preprocesador
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numerical_features),
        ("cat", OHE, categorical_features),
    ]
)

# Modelos disponibles
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
}

# ---- Control: seleccionar 1 modelo (por defecto Random Forest) ----
model_names = list(models.keys())
default_index = model_names.index("Decision Tree") if "Random Forest" in model_names else 0
modelo_elegido = st.selectbox("Modelo a ejecutar", options=model_names, index=default_index, key="rfe_modelo")
model = models[modelo_elegido]

st.title("Resultados de Selección de Características con RFE-CV")
st.subheader(f"Modelo: {modelo_elegido}")

# RFECV
rfe = RFECV(
    estimator=model,
    step=1,
    cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42),
    scoring="accuracy",
    n_jobs=-1,
)

# Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("feature_selection", rfe),
    ("model", model),
])

# Entrenar
pipeline.fit(X_train, y_train)

# Evaluar
accuracy_test = pipeline.score(X_test, y_test)
mask = pipeline.named_steps["feature_selection"].support_
feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
selected_names = feature_names[mask]

# Mostrar en la app
st.write(f"*Accuracy en test set:* {accuracy_test:.3f}")
st.write(f"*Variables seleccionadas:* {len(selected_names)}")
st.write(f"*Nombres:* {list(selected_names)}")

# Resumen final (solo el modelo elegido)
st.header("Resumen Final")
st.markdown(f"""
*Modelo:* {modelo_elegido}  
- Accuracy: {accuracy_test:.3f}  
- Variables seleccionadas: {len(selected_names)}  
- Nombres: {list(selected_names)}  
""")

