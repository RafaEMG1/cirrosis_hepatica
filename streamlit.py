# -------------------- Carga y manipulación de datos --------------------
import numpy as np
import pandas as pd
import os
import warnings

# -------------------- Visualización --------------------
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import streamlit as st

# -------------------- Kaggle dataset --------------------
import kagglehub

# -------------------- Scikit-learn: Preprocesamiento y Modelos --------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# -------------------- Selección de características --------------------
from sklearn.feature_selection import (
    SelectKBest, chi2, mutual_info_classif, mutual_info_regression, f_classif, f_regression, RFE
)

# -------------------- Modelos --------------------
from sklearn.linear_model import LogisticRegression, LinearRegression, Perceptron
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# -------------------- Métricas --------------------
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, mean_absolute_error

# -------------------- Validación y Búsqueda de Hiperparámetros --------------------
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_val_score

# -------------------- Configuración --------------------
warnings.filterwarnings("ignore")



import warnings
warnings.filterwarnings("ignore")


st.set_page_config(page_title="Cirrosis Hepatica Streamlit App", layout="wide")
st.title("Clasificación de los estadios de la cirrosis hepática con métodos de Machine Learning")

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


# Descargar el dataset
path = kagglehub.dataset_download("aadarshvelu/liver-cirrhosis-stage-classification")
print("Ruta local del dataset:", path)

# Ver los archivos del dataset cargado
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        print(os.path.join(dirname, filename))

file_path = os.path.join(path, "liver_cirrhosis.csv")
df = pd.read_csv(file_path)

# Filtrar solo columnas categóricas (tipo "object" o "category")
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

##################### Categóricas #############################################
st.markdown("""---""")
st.markdown("""### Análisis de variables categóricas""")
st.caption("Selecciona una variable para ver su distribución en tabla y gráfico de torta.")

# =========================
# Detectar variables categóricas
# =========================
variables_categoricas = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
if not variables_categoricas:
    st.warning("No se detectaron variables categóricas (object/category/bool) en `df`.")
    st.stop()

# =========================
# Controles con fondo gris claro
# =========================
st.markdown("""
<div style="background-color:#f5f5f5; padding: 12px; border-radius: 8px; margin-bottom: 15px;">
<b>Controles de visualización</b>
</div>
""", unsafe_allow_html=True)

# =========================
# Controles superiores
# =========================
with st.container():
    c1, c2 = st.columns([1.5, 1])  # solo dos columnas ahora
    with c1:
        var = st.selectbox(
            "Variable categórica",
            options=variables_categoricas,
            index=0,
            key="cat_var"
        )
    with c2:
        incluir_na = st.checkbox("Incluir NaN", value=True, key="cat_incluir_na")
        orden_alfabetico = st.checkbox("Orden alfabético", value=False, key="cat_orden")

# =========================
# Preparar datos
# =========================
serie = df[var].copy()
if not incluir_na:
    serie = serie.dropna()

vc = serie.value_counts(dropna=incluir_na)
labels = vc.index.to_list()
labels = ["(NaN)" if (isinstance(x, float) and np.isnan(x)) else str(x) for x in labels]
counts = vc.values

data = pd.DataFrame({"Categoría": labels, "Conteo": counts})
data["Porcentaje"] = (data["Conteo"] / data["Conteo"].sum() * 100).round(2)

# Usamos Porcentaje como métrica por defecto
data_plot = data.sort_values("Porcentaje", ascending=False).reset_index(drop=True)

# Orden alfabético en tabla si se selecciona
data_table = data_plot.copy()
if orden_alfabetico:
    data_table = data_table.sort_values("Categoría").reset_index(drop=True)

# =========================
# Mostrar tabla y gráfico
# =========================
tcol, gcol = st.columns([1.1, 1.3], gap="large")

with tcol:
    st.subheader(f"Distribución de `{var}`")
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
            theta=alt.Theta(field="Porcentaje", type="quantitative"),  # usamos Porcentaje fijo
            color=alt.Color("Categoría:N", legend=alt.Legend(title="Categoría")),
            tooltip=[
                alt.Tooltip("Categoría:N"),
                alt.Tooltip("Conteo:Q", format=","),
                alt.Tooltip("Porcentaje:Q", format=".2f")
            ],
        )
        .properties(width="container", height=380)
    )
    st.altair_chart(chart, use_container_width=True)


##################### Numéricas #############################################

st.markdown("""---""")
# =========================
# Análisis de variables numéricas
# =========================
st.markdown("""### Análisis de variables numéricas""")
st.caption("Selecciona una variable para ver su distribución en tabla, boxplot e histograma.")

# Detectar variables numéricas
variables_numericas = df.select_dtypes(include=["number"]).columns.tolist()
if not variables_numericas:
    st.warning("No se detectaron variables numéricas en `df`.")
    st.stop()

# =========================
# Controles con fondo gris claro
# =========================
st.markdown("""
<div style="background-color:#f5f5f5; padding: 12px; border-radius: 8px; margin-bottom: 15px;">
<b>Controles de visualización - Numéricas</b>
</div>
""", unsafe_allow_html=True)

with st.container():
    c1, c2 = st.columns([2, 1])

    with c1:
        var_num = st.selectbox(
            "Variable numérica",
            options=variables_numericas,
            index=0,
            key="num_var_top"
        )
    with c2:
        bins = st.slider(
            "Número de bins (histograma)",
            min_value=5, max_value=100, value=30, step=5,
            key="num_bins_top"
        )

# Preparar serie
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
g1, g2 = st.columns([1.1, 1.3], gap="large")  # más espacio para histograma si es necesario

with g1:
    st.subheader(f"Boxplot de `{var_num}`")
    box_data = pd.DataFrame({var_num: serie_num})
    box_chart = (
        alt.Chart(box_data)
        .mark_boxplot(size=100)  # grosor de la caja
        .encode(
            y=alt.Y(var_num, type="quantitative")  # vertical
        )
        .properties(width=200, height=400)  # más ancho y alto
    )
    st.altair_chart(box_chart, use_container_width=True)

with g2:
    st.subheader(f"Histograma de `{var_num}`")
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
        .properties(height=400)
    )
    st.altair_chart(hist_chart, use_container_width=True)



# ________________________________________________________________________________________________________________________________________________________________
st.markdown("""# 1. Selección de carácteristicas""")




# ==============================
# 2.1 Selección Categóricas (Clasificación con Stage)
# ==============================
try:
    st.markdown("---")
    st.markdown("## 2.1. Selección de características categóricas (Clasificación)")

    y_raw = df[TARGET_COL]
    cat_cols = df.select_dtypes(include=["object","category","bool"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c != TARGET_COL]

    if not cat_cols:
        st.info("No hay variables categóricas disponibles (excluyendo la variable objetivo).")
    else:
        st.markdown("""
        <div style="background-color:#f5f5f5; padding: 10px; border-radius: 8px; margin-bottom: 10px;">
        <b>Controles</b>
        </div>
        """, unsafe_allow_html=True)
        c1, c2, c3 = st.columns([2.2, 1, 1])
        with c1:
            cats_sel = st.multiselect("Categóricas a evaluar", options=cat_cols,
                                      default=cat_cols[:min(10, len(cat_cols))],
                                      key="cat21_sel")
        with c2:
            metodo = st.radio("Método", ["Chi²", "Mutual Info"], index=0, horizontal=True, key="cat21_m")
        with c3:
            topk = st.slider("Top K", 3, 50, 10, 1, key="cat21_topk")

        if cats_sel:
            from sklearn.pipeline import Pipeline
            from sklearn.feature_selection import chi2, mutual_info_classif

            X_cat = df[cats_sel].copy()
            y_codes, _ = pd.factorize(y_raw)

            cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                 ("oh", OH_ENCODER)])
            X_enc = cat_pipe.fit_transform(X_cat)
            feat_names = cat_pipe.named_steps["oh"].get_feature_names_out(cats_sel)

            scores = chi2(X_enc, y_codes)[0] if metodo == "Chi²" else \
                     mutual_info_classif(X_enc, y_codes, discrete_features=True, random_state=42)

            sc_df = pd.DataFrame({"feature_dummy": feat_names, "score": scores}).sort_values("score", ascending=False)
            sc_df["variable"] = sc_df["feature_dummy"].str.split("_", n=1).str[0]
            agg_df = sc_df.groupby("variable", as_index=False)["score"].sum().sort_values("score", ascending=False)

            t1, t2 = st.tabs(["Detalle (dummy)", "Agregado (variable)"])
            with t1:
                st.dataframe(arrow_safe(sc_df.head(topk)), use_container_width=True)
                st.altair_chart(
                    alt.Chart(sc_df.head(topk)).mark_bar().encode(
                        x=alt.X("score:Q", title="Score"),
                        y=alt.Y("feature_dummy:N", sort="-x", title="Dummy (col=valor)"),
                        tooltip=["feature_dummy","score"]
                    ).properties(height=min(34*topk, 480)),
                    use_container_width=True
                )
            with t2:
                st.dataframe(arrow_safe(agg_df.head(topk)), use_container_width=True)
                st.altair_chart(
                    alt.Chart(agg_df.head(topk)).mark_bar().encode(
                        x=alt.X("score:Q", title="Score (sumado por variable)"),
                        y=alt.Y("variable:N", sort="-x", title="Variable"),
                        tooltip=["variable","score"]
                    ).properties(height=min(34*topk, 480)),
                    use_container_width=True
                )
            st.caption(f"Objetivo: **{TARGET_COL}** · Clases: {dict(pd.Series(y_raw).value_counts().sort_index())}")
        else:
            st.warning("Selecciona al menos una variable categórica para evaluar.")
except Exception as e:
    st.error("💥 Se produjo un error en 2.1 (Categóricas). Revisa el detalle abajo.")
    st.exception(e)






# ________________________________________________________________________________________________________________________________________________________________
st.markdown("""## 1.1. Selección de carácteristicas categóricas""")
# ________________________________________________________________________________________________________________________________________________________________
st.markdown("""## 1.2. Selección de carácteristicas numéricas""")
# ________________________________________________________________________________________________________________________________________________________________
st.markdown("""## 1.3. Unión de variables categóricas y númericas""")



# ________________________________________________________________________________________________________________________________________________________________
st.markdown("""# 2. MCA Y PCA""")
# ________________________________________________________________________________________________________________________________________________________________
st.markdown("""## 2.1. MCA""")
# ________________________________________________________________________________________________________________________________________________________________
st.markdown("""## 2.2. PCA""")
# ________________________________________________________________________________________________________________________________________________________________
st.markdown("""# 3. RFE""")
# ________________________________________________________________________________________________________________________________________________________________

