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

# ====== UTILIDADES GLOBALES ======

# 1) Variable objetivo fija
TARGET_COL = "Stage"

# 2) OneHotEncoder compatible (según versión de scikit-learn)
try:
    OH_ENCODER = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    OH_ENCODER = OneHotEncoder(handle_unknown="ignore", sparse=False)

# 3) Helper para evitar problemas con Arrow en st.dataframe
def arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            out[c] = s.astype(str)
        elif pd.api.types.is_integer_dtype(s):
            out[c] = s.astype(float) if s.isna().any() else s.astype("int64")
        elif pd.api.types.is_bool_dtype(s):
            out[c] = s.fillna(False).astype(bool)
    return out




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
    cat_summary["Tipo de dato"] = cat_summary["Tipo de dato"].astype(str)
    st.dataframe(arrow_safe(cat_summary), use_container_width=True)

with col2:
    st.subheader("Resumen variables numéricas")
    num_summary["Tipo de dato"] = num_summary["Tipo de dato"].astype(str)
    st.dataframe(arrow_safe(num_summary), use_container_width=True)

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
labels = ["(NaN)" if pd.isna(x) else str(x) for x in labels]
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


# ________________________________________________________________________________________________________________________________________________________________
st.markdown("""## 1.1. Selección de carácteristicas categóricas""")


# ==============================
# 2.1 Selección Categóricas (Clasificación con Stage)
# ==============================
try:
    st.markdown("---")
    st.markdown("## 2.1. Selección de características categóricas (Clasificación)")


    # Validar que Stage exista
    if TARGET_COL not in df.columns:
        st.error("❌ No se encontró la columna objetivo 'Stage' en el DataFrame.")
        st.stop()


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
st.markdown("""## 1.2. Selección de carácteristicas numéricas""")

# =========================
# 2.2. Selección de características NUMÉRICAS (Clasificación, y = Stage)
# =========================
try:
    st.markdown("---")
    st.markdown("## 2.2. Selección de características numéricas (Clasificación)")

    # Validar y preparar objetivo
    if TARGET_COL not in df.columns:
        st.error("❌ No se encontró la columna objetivo 'Stage' en el DataFrame.")
        st.stop()
    y_raw = df[TARGET_COL]
    y_codes, _ = pd.factorize(y_raw)

    # Detectar numéricas (excluye la y)
    num_cols_22 = df.select_dtypes(include=["number"]).columns.tolist()
    num_cols_22 = [c for c in num_cols_22 if c != TARGET_COL]

    if not num_cols_22:
        st.info("No hay variables numéricas disponibles (excluyendo la variable objetivo).")
    else:
        # Controles
        st.markdown("""
        <div style="background-color:#f5f5f5; padding: 10px; border-radius: 8px; margin-bottom: 10px;">
        <b>Controles</b>
        </div>
        """, unsafe_allow_html=True)

        n1, n2, n3 = st.columns([2.2, 1, 1])
        with n1:
            nums_sel_22 = st.multiselect(
                "Numéricas a evaluar",
                options=num_cols_22,
                default=num_cols_22[:min(10, len(num_cols_22))],
                key="num22_sel"
            )
        with n2:
            metodo_num_22 = st.radio(
                "Método", ["ANOVA F", "Mutual Info"],
                index=0, horizontal=True, key="num22_m"
            )
        with n3:
            topk_22 = st.slider("Top K", 3, 50, 10, 1, key="num22_topk")

        if nums_sel_22:
            # Imputación + escalado
            num_pipe = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler())
            ])
            Xn = num_pipe.fit_transform(df[nums_sel_22])

            # Scores
            if metodo_num_22 == "ANOVA F":
                scores = f_classif(Xn, y_codes)[0]  # (F, pval) -> tomamos F
            else:
                scores = mutual_info_classif(Xn, y_codes, random_state=42)

            sc2_df = pd.DataFrame({"feature": nums_sel_22, "score": scores}) \
                        .sort_values("score", ascending=False)

            # Tabla + gráfico
            st.dataframe(arrow_safe(sc2_df.head(topk_22)), use_container_width=True)
            st.altair_chart(
                alt.Chart(sc2_df.head(topk_22)).mark_bar().encode(
                    x=alt.X("score:Q", title="Score"),
                    y=alt.Y("feature:N", sort="-x", title="Variable"),
                    tooltip=["feature", "score"]
                ).properties(height=min(34*topk_22, 480)),
                use_container_width=True
            )

            st.caption(f"Objetivo: **{TARGET_COL}** · Clases: {dict(pd.Series(y_raw).value_counts().sort_index())}")
        else:
            st.warning("Selecciona al menos una variable numérica para evaluar.")

except Exception as e:
    st.error("💥 Se produjo un error en 2.2 (Numéricas). Detalle:")
    st.exception(e)

# ________________________________________________________________________________________________________________________________________________________________
st.markdown("""## 1.3. Unión de variables categóricas y númericas""")

# =========================
# 2.3. Unión de variables Categóricas + Numéricas (Clasificación, y = Stage)
# =========================
try:
    st.markdown("---")
    st.markdown("## 2.3. Unión de variables categóricas y numéricas")

    # Validar objetivo
    if TARGET_COL not in df.columns:
        st.error("❌ No se encontró la columna objetivo 'Stage' en el DataFrame.")
        st.stop()
    y_union = df[TARGET_COL]

    # Detectar columnas por tipo (excluye y)
    cat_cols_23 = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols_23 = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols_23 = [c for c in cat_cols_23 if c != TARGET_COL]
    num_cols_23 = [c for c in num_cols_23 if c != TARGET_COL]

    if len(cat_cols_23) + len(num_cols_23) == 0:
        st.info("No hay variables disponibles para unir.")
    else:
        # Controles
        st.markdown("""
        <div style="background-color:#f5f5f5; padding: 10px; border-radius: 8px; margin-bottom: 10px;">
        <b>Controles</b>
        </div>
        """, unsafe_allow_html=True)

        u1, u2 = st.columns([2, 1])
        with u1:
            cats_u = st.multiselect(
                "Categóricas a incluir",
                options=cat_cols_23,
                default=cat_cols_23[:min(5, len(cat_cols_23))],
                key="union23_cats"
            )
            nums_u = st.multiselect(
                "Numéricas a incluir",
                options=num_cols_23,
                default=num_cols_23[:min(5, len(num_cols_23))],
                key="union23_nums"
            )
        with u2:
            show_feat = st.checkbox("Ver nombres de features", True, key="union23_show")

        if len(cats_u) + len(nums_u) == 0:
            st.warning("Selecciona al menos una variable (categórica o numérica).")
        else:
            # Pipelines
            num_pipe = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler())
            ])
            cat_pipe = Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OH_ENCODER)  # usa el encoder compatible global
            ])

            pre = ColumnTransformer(
                transformers=[
                    ("num", num_pipe, nums_u),
                    ("cat", cat_pipe, cats_u)
                ],
                remainder="drop"
            )

            X_raw = df[nums_u + cats_u]
            X_all = pre.fit_transform(X_raw, y_union)

            st.success(f"X transformada: **{X_all.shape[0]} filas × {X_all.shape[1]} columnas**")

            if show_feat:
                try:
                    names = pre.get_feature_names_out()
                except Exception:
                    names = [f"f{i}" for i in range(X_all.shape[1])]
                st.caption("Vista rápida de nombres de características generadas:")
                st.dataframe(
                    arrow_safe(pd.DataFrame({"feature": names}).head(60)),
                    use_container_width=True
                )

            # Persistir selección para 2.4 (si la vas a usar)
            st.session_state["union23_cats"] = cats_u
            st.session_state["union23_nums"] = nums_u

except Exception as e:
    st.error("💥 Se produjo un error en 2.3 (Unión). Detalle:")
    st.exception(e)




# ________________________________________________________________________________________________________________________________________________________________
st.markdown("""# 2. MCA Y PCA""")
# ________________________________________________________________________________________________________________________________________________________________
st.markdown("""## 2.1. MCA""")
# ________________________________________________________________________________________________________________________________________________________________
st.markdown("""## 2.2. PCA""")
# ________________________________________________________________________________________________________________________________________________________________
st.markdown("""# 3. RFE""")
# ________________________________________________________________________________________________________________________________________________________________

