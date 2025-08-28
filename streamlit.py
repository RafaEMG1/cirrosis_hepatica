# Cargue de librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import kagglehub
import os
import altair as alt
import plotly.express as px
import prince

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
from scipy.stats import uniform
from scipy.stats import randint

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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold

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
# Matriz de Correlación (figura y texto reducidos)
# =========================
st.markdown("### Matriz de Correlación")
correlacion = df.corr(numeric_only=True)

fig, ax = plt.subplots(figsize=(3, 2))  # tamaño del gráfico más pequeño
sns.heatmap(
    correlacion,
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    ax=ax,
    annot_kws={"size": 4},   # texto dentro de las celdas
    cbar_kws={"shrink": 0.5} # barra de color más pequeña
)
ax.set_title("Matriz de Correlación", fontsize=7)   # título más pequeño
ax.tick_params(axis='x', labelsize=4)                # etiquetas X más pequeñas
ax.tick_params(axis='y', labelsize=4)                # etiquetas Y más pequeñas
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


# === INICIO SECCIÓN 2 (filtros en cada subsección) ===
st.markdown("# 2. MCA Y PCA")

# -----------------------------------------------------
# 2.1. MCA  (filtros dentro de la subsección)
# -----------------------------------------------------
st.markdown("## 2.1. MCA")

df_cat_all = df.select_dtypes(include=["object", "category", "bool"]).copy()
cat_cols_all = [c for c in df_cat_all.columns if c != "Stage"]

# Controles MCA (en línea, no sidebar)
mca_ctrl = st.container()
with mca_ctrl:
    c_m1, c_m2, c_m3 = st.columns([2, 1, 1])
    cat_sel = c_m1.multiselect(
        "Variables categóricas para MCA",
        options=cat_cols_all,
        default=cat_cols_all,
        key="mca_vars_sel_section"
    )
    var_target_mca = c_m2.slider(
        "Varianza objetivo (%)",
        min_value=80, max_value=99, value=80, step=1, key="mca_var_pct"
    ) / 100.0
    top_k_mca = c_m3.slider(
        "Top-K variables",
        min_value=5, max_value=50, value=15, step=1, key="mca_topk"
    )

if cat_sel:
    df_cat = df_cat_all[cat_sel]
    y = df["Stage"]
    X_train, X_test, y_train, y_test = train_test_split(
        df_cat, y, stratify=y, test_size=0.33, random_state=1
    )

    X_train_encoded = pd.get_dummies(X_train, drop_first=False)
    if X_train_encoded.shape[1] == 0:
        st.info("Selecciona al menos una variable categórica para ejecutar MCA.")
    else:
        mca_cirrosis = mca.MCA(X_train_encoded, benzecri=True)

        # Varianza acumulada
        sv = mca_cirrosis.s
        eigvals = sv**2
        explained_var = eigvals / eigvals.sum()
        cum_explained_var = np.cumsum(explained_var)
        n_dims_target = int(np.argmax(cum_explained_var >= var_target_mca) + 1)

        c1, c2 = st.columns(2)
        with c1:
            fig_mca_var, ax = plt.subplots(figsize=(6, 4))
            ax.plot(range(1, len(cum_explained_var) + 1), cum_explained_var, marker="o", linestyle="--")
            ax.axhline(y=var_target_mca)
            ax.set_xlabel("Dimensiones MCA")
            ax.set_ylabel("Varianza acumulada explicada")
            ax.set_title("MCA - Varianza acumulada")
            ax.grid(True)
            st.pyplot(fig_mca_var)
            st.write(f"Dimensiones para ≥ {var_target_mca*100:.0f}%: **{n_dims_target}**")

        with c2:
            coords = mca_cirrosis.fs_r(N=3)
            y_train_align = y_train.iloc[:coords.shape[0]]
            fig_mca_sc, ax2 = plt.subplots(figsize=(6, 4))
            sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=y_train_align, alpha=0.7, ax=ax2)
            ax2.set_xlabel("Dimensión 1")
            ax2.set_ylabel("Dimensión 2")
            ax2.set_title("MCA: Dim 1 vs Dim 2")
            ax2.legend(title="Clase")
            st.pyplot(fig_mca_sc)

        # Contribución de variables (Top-K)
        loadings_cat = pd.DataFrame(mca_cirrosis.fs_c()[:, :n_dims_target], index=X_train_encoded.columns)
        contrib = (loadings_cat**2).div((loadings_cat**2).sum(axis=0), axis=1)
        contrib_total = contrib.sum(axis=1).sort_values(ascending=False)
        top_contrib = contrib_total.head(top_k_mca)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown(f"**Top-{top_k_mca} aportes (MCA)** — sobre {n_dims_target} dim")
            fig_mca_bar, ax3 = plt.subplots(figsize=(8, 4))
            top_contrib.plot(kind="bar", ax=ax3)
            ax3.set_ylabel("Contribución total")
            ax3.set_title("Aporte de variables/dummies (MCA)")
            ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha="right")
            fig_mca_bar.tight_layout()
            st.pyplot(fig_mca_bar)

        with c4:
            st.markdown("**Resumen (MCA)**")
            st.markdown(
                f"- Varianza objetivo: **{var_target_mca*100:.0f}%**  \n"
                f"- Dimensiones usadas: **{n_dims_target}**  \n"
                f"- Variables categóricas seleccionadas: **{len(cat_sel)}**"
            )
else:
    st.info("Selecciona variables categóricas para ejecutar MCA.")

# -----------------------------------------------------
# 2.2. PCA  (filtros dentro de la subsección)
# -----------------------------------------------------
st.markdown("## 2.2. PCA")

df_num = df.select_dtypes(include=["int64", "float64"]).copy()

# Controles PCA (en línea, no sidebar)
pca_ctrl = st.container()
with pca_ctrl:
    c_p1, c_p2 = st.columns([1, 1])
    var_target_pca = c_p1.slider(
        "Varianza objetivo (%)",
        min_value=80, max_value=99, value=80, step=1, key="pca_var_pct"
    ) / 100.0
    top_k_pca = c_p2.slider(
        "Top-K variables",
        min_value=5, max_value=50, value=15, step=1, key="pca_topk"
    )

if df_num.empty:
    st.warning("No hay variables numéricas para PCA.")
else:
    y = df["Stage"]
    X_train, X_test, y_train, y_test = train_test_split(
        df_num, y, stratify=y, test_size=0.33, random_state=1
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # PCA completo (para varianza acumulada y PC1 vs PC2)
    pca_full = PCA()
    X_pca_full = pca_full.fit_transform(X_scaled)
    explained_cum = np.cumsum(pca_full.explained_variance_ratio_)
    n_pc_target = int(np.argmax(explained_cum >= var_target_pca) + 1)

    c5, c6 = st.columns(2)
    with c5:
        fig_pca_var, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(1, len(explained_cum) + 1), explained_cum, marker="o", linestyle="--")
        ax.axhline(y=var_target_pca)
        ax.set_xlabel("Número de componentes principales")
        ax.set_ylabel("Varianza acumulada explicada")
        ax.set_title("PCA - Varianza acumulada")
        ax.grid(True)
        st.pyplot(fig_pca_var)
        st.write(f"Componentes para ≥ {var_target_pca*100:.0f}%: **{n_pc_target}**")

    with c6:
        fig_pca_sc, ax2 = plt.subplots(figsize=(6, 4))
        y_train_align = y_train.iloc[:X_pca_full.shape[0]]
        sns.scatterplot(x=X_pca_full[:, 0], y=X_pca_full[:, 1], hue=y_train_align, alpha=0.7, ax=ax2)
        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")
        ax2.set_title("PCA: PC1 vs PC2")
        ax2.legend(title="Clase")
        st.pyplot(fig_pca_sc)

    # Loadings y Top-K por PCs seleccionadas
    loadings = pd.DataFrame(
        pca_full.components_.T,
        columns=[f"PC{i+1}" for i in range(pca_full.n_components_)],
        index=X_train.columns
    )
    var_importance = (loadings.iloc[:, :n_pc_target] ** 2).sum(axis=1).sort_values(ascending=False)
    top_vars_pca = var_importance.head(top_k_pca)

    c7, c8 = st.columns(2)
    with c7:
        st.markdown(f"**Top-{top_k_pca} variables PCA** — sobre {n_pc_target} PCs")
        fig_pca_bar, ax3 = plt.subplots(figsize=(8, 4))
        top_vars_pca.plot(kind="bar", ax=ax3)
        ax3.set_ylabel("Aporte total (suma de cuadrados de loadings)")
        ax3.set_title("Aporte por variable a PCs seleccionadas")
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha="right")
        fig_pca_bar.tight_layout()
        st.pyplot(fig_pca_bar)

    with c8:
        # Resumen con PCA a varianza objetivo (para confirmar números)
        pca_target = PCA(n_components=var_target_pca)
        _ = pca_target.fit_transform(X_scaled)
        st.markdown("**Resumen (PCA)**")
        st.markdown(
            f"- Varianza objetivo: **{var_target_pca*100:.0f}%**  \n"
            f"- Componentes usadas: **{pca_target.n_components_}**  \n"
            f"- Varianza acumulada lograda: **{pca_target.explained_variance_ratio_.sum()*100:.2f}%**"
        )

# ______________________________________________________________________________________________________
# ______________________________________________________________________________________________________
st.markdown("## 2.3. Concatenar las dos matrices")

# ========= 0) Preparación: columnas y split común por índices =========
y_full = df["Stage"]
num_cols = df.select_dtypes(include=["number"]).columns.tolist()
cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

# Asegura que 'Stage' no esté en cat_cols
cat_cols = [c for c in cat_cols if c != "Stage"]

# Split por índices para alinear filas entre numéricas y categóricas
idx_train, idx_test = train_test_split(
    df.index, stratify=y_full, test_size=0.33, random_state=1
)
y_train = y_full.loc[idx_train]
y_test  = y_full.loc[idx_test]

X_num_train = df.loc[idx_train, num_cols]
X_num_test  = df.loc[idx_test,  num_cols]

X_cat_train = df.loc[idx_train, cat_cols]
X_cat_test  = df.loc[idx_test,  cat_cols]

# ========= 1) PCA sobre numéricas =========
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_num_train)  # fit solo con train
X_test_scaled  = scaler.transform(X_num_test)       # transform en test

pca = PCA(n_components=8)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)

# ========= 2) MCA sobre categóricas con columnas alineadas =========
# Dummies en train y test, y alineación de columnas
X_train_encoded = pd.get_dummies(X_cat_train, drop_first=False)
X_test_encoded  = pd.get_dummies(X_cat_test,  drop_first=False)

# Alinear columnas de test a las de train (rellenar faltantes con 0)
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

# Modelo prince.MCA (evita sombrear el import `mca`)
mca_pr = prince.MCA(n_components=6, random_state=42)
mca_pr = mca_pr.fit(X_train_encoded)

X_train_mca = mca_pr.transform(X_train_encoded)
X_test_mca  = mca_pr.transform(X_test_encoded)

# ========= 3) DataFrames con índices y nombres de columnas =========
X_train_pca_df = pd.DataFrame(
    X_train_pca, index=idx_train, columns=[f"PCA_{i+1}" for i in range(X_train_pca.shape[1])]
)
X_test_pca_df = pd.DataFrame(
    X_test_pca, index=idx_test, columns=[f"PCA_{i+1}" for i in range(X_test_pca.shape[1])]
)

# prince devuelve DataFrame; asegurar índices correctos y nombres
X_train_mca_df = pd.DataFrame(
    X_train_mca.values, index=idx_train, columns=[f"MCA_{i+1}" for i in range(X_train_mca.shape[1])]
)
X_test_mca_df = pd.DataFrame(
    X_test_mca.values, index=idx_test, columns=[f"MCA_{i+1}" for i in range(X_test_mca.shape[1])]
)

# ========= 4) Concatenación final =========
X_train_final = pd.concat([X_train_pca_df, X_train_mca_df], axis=1)
X_test_final  = pd.concat([X_test_pca_df,  X_test_mca_df],  axis=1)

# ========= 5) Vista rápida =========
cA, cB = st.columns(2)
with cA:
    st.subheader("Train: PCA+MCA (shape)")
    st.write(X_train_final.shape)
    st.dataframe(X_train_final.head(10))
with cB:
    st.subheader("Test: PCA+MCA (shape)")
    st.write(X_test_final.shape)
    st.dataframe(X_test_final.head(10))
# ______________________________________________________________________________________________________

# __________________________________________________________________________________________________
st.markdown("""## 2.4. Modelado""")


# --- Filtro único de la subsección (por defecto: Logistic Regression)
model_name_24 = st.selectbox(
    "Elige el modelo a evaluar (CV 5-fold)",
    options=["Logistic Regression", "KNN", "SVC", "Decision Tree", "Random Forest"],
    index=0,  # Logistic Regression por defecto
    key="model_sel_24"
)

# --- Construcción del modelo según selección
def build_model(name: str):
    if name == "Logistic Regression":
        return LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=2000, class_weight="balanced", random_state=42)
    if name == "KNN":
        return KNeighborsClassifier()
    if name == "SVC":
        return SVC()  # por defecto rbf; podrías envolver en Pipeline con StandardScaler si lo deseas
    if name == "Decision Tree":
        return DecisionTreeClassifier(random_state=42)
    if name == "Random Forest":
        return RandomForestClassifier(random_state=42)
    raise ValueError("Modelo no soportado")

modelo_24 = build_model(model_name_24)

# --- CV estratificado para mayor estabilidad
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(modelo_24, X_train_final, y_train, cv=cv5, scoring="accuracy", n_jobs=-1)

st.subheader("Resultados de validación cruzada")
st.write(f"**Modelo:** {model_name_24}")
st.write(f"**Accuracy (media CV):** {scores.mean():.4f}  |  **Std:** {scores.std():.4f}")

# __________________________________________________________________________________________________
st.markdown("""## 2.5. Ajuste de hiperparámetros""")

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform, loguniform

# --- Filtro único de la subsección (por defecto: Logistic Regression)
model_name_25 = st.selectbox(
    "Elige el modelo a ajustar (RandomizedSearchCV)",
    options=["Logistic Regression", "KNN", "SVC", "Decision Tree", "Random Forest"],
    index=0,
    key="model_sel_25"
)

# --- Espacios de búsqueda por modelo (evitando combinaciones inválidas)
def get_model_and_searchspace(name: str):
    if name == "Logistic Regression":
        model = LogisticRegression(multi_class="multinomial", solver="lbfgs", penalty="l2", max_iter=5000, class_weight="balanced", random_state=42)
        # solo C para evitar incompatibilidades
        param_dist = {"C": loguniform(1e-3, 1e2)}
        return model, param_dist, "accuracy"
    if name == "KNN":
        model = KNeighborsClassifier()
        param_dist = {
            "n_neighbors": randint(3, 30),
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan", "minkowski"],
        }
        return model, param_dist, "accuracy"
    if name == "SVC":
        model = SVC(probability=False, random_state=42)
        # espacio mixto simple; si kernel='linear', gamma se ignora; no pasa nada en SVC
        param_dist = {
            "C": loguniform(1e-2, 1e2),
            "kernel": ["linear", "rbf", "poly"],
            "gamma": ["scale", "auto"],
            "degree": randint(2, 5),  # aplica si poly
        }
        return model, param_dist, "accuracy"
    if name == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
        param_dist = {
            "max_depth": randint(3, 20),
            "min_samples_split": randint(2, 10),
            #"min_samples_leaf": randint(1, 20),
            "criterion": ["gini", "entropy"],
        }
        return model, param_dist, "accuracy"
    if name == "Random Forest":
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
        param_dist = {
            "n_estimators": randint(100, 600),
            "max_depth": randint(3, 40),
            "min_samples_split": randint(2, 20),
            "min_samples_leaf": randint(1, 20),
            "max_features": ["sqrt", "log2", None],
        }
        return model, param_dist, "accuracy"
    raise ValueError("Modelo no soportado")

estimator_25, searchspace_25, metric_25 = get_model_and_searchspace(model_name_25)

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
random_search = RandomizedSearchCV(
    estimator=estimator_25,
    param_distributions=searchspace_25,
    n_iter=25,
    cv=cv5,
    scoring=metric_25,
    n_jobs=-1,
    verbose=1,
    random_state=42
)
random_search.fit(X_train_final, y_train)

st.subheader("Mejores hiperparámetros")
st.write(f"**Modelo:** {model_name_25}")
st.write("**Best params:**", random_search.best_params_)
st.write(f"**Mejor {metric_25} (CV):** {random_search.best_score_:.4f}")

# Guardar el mejor estimador en session_state para reusarlo en 2.6
if "best_estimators" not in st.session_state:
    st.session_state.best_estimators = {}
st.session_state.best_estimators[model_name_25] = random_search.best_estimator_

# __________________________________________________________________________________________________
st.markdown("""## 2.6. Comparación de modelos optimizados""")

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Filtro único de la subsección (por defecto: Logistic Regression)
model_name_26 = st.selectbox(
    "Elige el modelo a evaluar en Test",
    options=["Logistic Regression", "KNN", "SVC", "Decision Tree", "Random Forest"],
    index=0,
    key="model_sel_26"
)

# Recuperar el mejor estimador si ya fue ajustado en 2.5; si no, construir y entrenar rápido
if "best_estimators" in st.session_state and model_name_26 in st.session_state.best_estimators:
    modelo_26 = st.session_state.best_estimators[model_name_26]
else:
    # fallback: usa el modelo por defecto sin tuning
    modelo_26 = build_model(model_name_26)
    # (opcional) podrías entrenar sobre todo el train antes de evaluar
    # pero la comparación se verá mejor si haces CV o tuning. Aquí entrenamos simple:
    modelo_26.fit(X_train_final, y_train)

# CV del modelo final (opcional para mostrar referencia)
scores_cv = cross_val_score(modelo_26, X_train_final, y_train, cv=cv5, scoring="accuracy", n_jobs=-1)
mean_cv = scores_cv.mean()
std_cv = scores_cv.std()

# Entrenamiento en train y evaluación en test
modelo_26.fit(X_train_final, y_train)
y_pred = modelo_26.predict(X_test_final)
acc_test = accuracy_score(y_test, y_pred)

st.markdown(f"### 📌 Modelo: {model_name_26}")
st.markdown(f"**Accuracy CV (media ± std):** {mean_cv:.4f} ± {std_cv:.4f}")
st.markdown(f"**Accuracy Test:** {acc_test:.4f}")
st.text("📋 Classification Report (Test):")
st.text(classification_report(y_test, y_pred))
st.text("🧩 Matriz de Confusión (Test):")
st.write(pd.DataFrame(confusion_matrix(y_test, y_pred), index=sorted(y_test.unique()), columns=sorted(y_test.unique())))



# === FIN SECCIÓN 2 ===





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

