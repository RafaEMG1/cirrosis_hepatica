# =========================
# Imports y configuración
# =========================
import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.feature_selection import chi2, mutual_info_classif, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

warnings.filterwarnings("ignore")
alt.data_transformers.disable_max_rows()

# XGBoost opcional
try:
    from xgboost import XGBClassifier
    XGB_OK = True
except Exception:
    XGB_OK = False

# =========================
# Constantes y helpers
# =========================
TARGET_COL = "Stage"

# OneHotEncoder compatible con varias versiones de sklearn
try:
    OH_ENCODER = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    OH_ENCODER = OneHotEncoder(handle_unknown="ignore", sparse=False)

def arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte tipos conflictivos para que st.dataframe no falle con Arrow."""
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

def section_header(title: str, caption: str | None = None):
    st.markdown("---")
    st.markdown(f"### {title}")
    if caption:
        st.caption(caption)

def card_controls(title: str):
    st.markdown(
        f"""
        <div style="background-color:#f5f5f5; padding: 10px; border-radius: 8px; margin-bottom: 10px;">
            <b>{title}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

def safe_run(name: str, fn):
    try:
        fn()
    except Exception as e:
        st.error(f"💥 Error en la sección: {name}")
        st.exception(e)

@st.cache_data(show_spinner=False)
def load_kaggle_dataset() -> pd.DataFrame:
    path = kagglehub.dataset_download("aadarshvelu/liver-cirrhosis-stage-classification")
    file_path = os.path.join(path, "liver_cirrhosis.csv")
    return pd.read_csv(file_path)

def format_uniques(series: pd.Series, max_items=20) -> str:
    uniques = pd.Series(series.dropna().unique())
    head = uniques.head(max_items).astype(str).tolist()
    txt = ", ".join(head)
    if uniques.size > max_items:
        txt += f" … (+{uniques.size - max_items} más)"
    return txt

# =========================
# Página
# =========================
st.set_page_config(page_title="Cirrosis Hepatica Streamlit App", layout="wide")
st.title("Clasificación de los estadios de la cirrosis hepática con métodos de Machine Learning")
st.caption("Estudio clínico de cirrosis hepática — ficha de variables")

st.sidebar.markdown("### ⚙️ Opciones")
DEBUG = st.sidebar.checkbox("🪲 Modo debug", value=False)

INTRO = """
### **Variables:**
* **N_Days**: Número de días transcurridos entre el registro y la fecha más temprana entre fallecimiento, trasplante o análisis del estudio en 1986. 
* **Status**: estado del paciente C (censurado), CL (censurado por tratamiento hepático) o D (fallecimiento). 
* **Drug**: tipo de fármaco: D-penicilamina o placebo. 
* **Age**: edad en días. * **Sex**: M (hombre) o F (mujer). 
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
"""
st.markdown(INTRO)

# =========================
# Carga de datos
# =========================
try:
    df = load_kaggle_dataset()
except Exception as e:
    st.error("❌ No se pudo descargar/cargar el dataset de Kaggle.")
    st.exception(e)
    st.stop()

# =========================
# Sección: Resúmenes cat/num
# =========================
def sec_resumen():
    st.subheader("Primeras 10 filas del dataset")
    st.dataframe(df.head(10), use_container_width=True)

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()

    cat_summary = pd.DataFrame({
        "Variable": cat_cols,
        "Tipo de dato": [str(df[c].dtype) for c in cat_cols],
        "Nº de categorías únicas": [df[c].nunique(dropna=True) for c in cat_cols],
        "Nº de datos no nulos": [df[c].notna().sum() for c in cat_cols],
        "Categorías": [format_uniques(df[c], max_items=20) for c in cat_cols],
    })
    num_summary = pd.DataFrame({
        "Variable": num_cols,
        "Tipo de dato": [str(df[c].dtype) for c in num_cols],
        "Nº de datos no nulos": [df[c].notna().sum() for c in num_cols],
        "Mínimo": [df[c].min(skipna=True) for c in num_cols],
        "Máximo": [df[c].max(skipna=True) for c in num_cols],
        "Media":  [df[c].mean(skipna=True) for c in num_cols],
        "Desviación estándar": [df[c].std(skipna=True) for c in num_cols],
    }).round(2)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.subheader("Resumen variables categóricas")
        st.dataframe(arrow_safe(cat_summary), use_container_width=True)
    with c2:
        st.subheader("Resumen variables numéricas")
        st.dataframe(arrow_safe(num_summary), use_container_width=True)

safe_run("Resumen de datos", sec_resumen)

# =========================
# Sección: Análisis Categóricas
# =========================
# =========================
# 2.1. Selección de características CATEGÓRICAS (Clasificación)
# =========================
def sec_21_cat_selection():
    import numpy as np
    import pandas as pd
    import altair as alt
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.feature_selection import chi2, mutual_info_classif

    # ---------- helpers locales ----------
    def section_header(txt: str):
        st.markdown("---")
        st.markdown(f"## {txt}")

    def card_controls(title: str):
        st.markdown(
            f"""
            <div style="background-color:#f5f5f5;padding:10px;border-radius:8px;margin-bottom:10px;">
            <b>{title}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def add_percent_cols(df_scores: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
        out = df_scores.copy()
        total = out[score_col].sum()
        out["pct"] = np.where(total > 0, 100.0 * out[score_col] / total, 0.0)
        out["pct_acum"] = out["pct"].cumsum()
        return out

    def vline_90():
        return alt.Chart(pd.DataFrame({"x": [90.0]})).mark_rule(
            strokeDash=[6, 6], color="red"
        ).encode(x="x:Q")

    # ---------- sección ----------
    section_header("2.1. Selección de características categóricas (Clasificación)")

    if TARGET_COL not in df.columns:
        st.error("❌ No se encontró la columna objetivo 'Stage'.")
        return

    # columnas categóricas (excluye y)
    y_raw = df[TARGET_COL]
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c != TARGET_COL]
    if not cat_cols:
        st.info("No hay variables categóricas (excluyendo la objetivo).")
        return

    # ---- Controles (claves únicas con prefijo) ----
    prefix = "s21"  # <- prefijo único para esta sección
    card_controls("Controles")
    c1, c2, c3 = st.columns([2.2, 1, 1])
    with c1:
        cats_sel = st.multiselect(
            "Categóricas a evaluar",
            options=cat_cols,
            default=cat_cols[: min(10, len(cat_cols))],
            key=f"{prefix}_sel",
        )
    with c2:
        metodo = st.radio(
            "Método", ["Chi²", "Mutual Info"], 0, horizontal=True, key=f"{prefix}_m"
        )
    with c3:
        topk = st.slider("Top K", 3, 50, 10, 1, key=f"{prefix}_k")

    if not cats_sel:
        st.warning("Selecciona al menos una variable categórica para evaluar.")
        return

    # ---- OneHot + scoring ----
    X_cat = df[cats_sel].copy()
    y_codes, _ = pd.factorize(y_raw)  # y en códigos 0..n-1

    cat_pipe = Pipeline(
        [("imp", SimpleImputer(strategy="most_frequent")), ("oh", OH_ENCODER)]
    )
    X_enc = cat_pipe.fit_transform(X_cat)
    feat_names = cat_pipe.named_steps["oh"].get_feature_names_out(cats_sel)

    scores = (
        chi2(X_enc, y_codes)[0]
        if metodo == "Chi²"
        else mutual_info_classif(
            X_enc, y_codes, discrete_features=True, random_state=42
        )
    )

    # ---- Tablas: detalle por dummy y agregado por variable ----
    sc_df = (
        pd.DataFrame({"feature_dummy": feat_names, "score": scores})
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
    # % y % acumulado (dummy)
    sc_df = add_percent_cols(sc_df, "score")

    # Agregado por variable original
    sc_df["variable"] = sc_df["feature_dummy"].str.split("_", n=1).str[0]
    agg_df = (
        sc_df.groupby("variable", as_index=False)["score"].sum()
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
    # % y % acumulado (agregado)
    agg_df = add_percent_cols(agg_df, "score")

    # ---- Pestañas: Detalle (dummy) / Agregado (variable) ----
    t1, t2 = st.tabs(["Detalle (dummy)", "Agregado (variable)"])

    # Detalle por dummy
    with t1:
        st.dataframe(
            sc_df.loc[:, ["feature_dummy", "score", "pct", "pct_acum"]]
            .head(topk)
            .rename(
                columns={
                    "feature_dummy": "Dummy (col=valor)",
                    "score": "Score",
                    "pct": "%",  # porcentaje
                    "pct_acum": "% acumulado",
                }
            ),
            use_container_width=True,
        )

        bars = (
            alt.Chart(sc_df.head(topk))
            .mark_bar()
            .encode(
                x=alt.X("score:Q", title="Score"),
                y=alt.Y("feature_dummy:N", sort="-x", title="Dummy (col=valor)"),
                tooltip=[
                    alt.Tooltip("feature_dummy:N", title="Dummy"),
                    alt.Tooltip("score:Q", format=".4f", title="Score"),
                    alt.Tooltip("pct:Q", format=".2f", title="%"),
                    alt.Tooltip("pct_acum:Q", format=".2f", title="% acumulado"),
                ],
            )
            .properties(height=min(34 * topk, 480))
        )

        # línea vertical en 90% (en eje X de score no aplica; añadimos una línea auxiliar en % con un gráfico secundario)
        st.altair_chart(bars, use_container_width=True)

    # Agregado por variable
    with t2:
        st.dataframe(
            agg_df.loc[:, ["variable", "score", "pct", "pct_acum"]]
            .head(topk)
            .rename(
                columns={
                    "variable": "Variable",
                    "score": "Score",
                    "pct": "%",
                    "pct_acum": "% acumulado",
                }
            ),
            use_container_width=True,
        )

        bars2 = (
            alt.Chart(agg_df.head(topk))
            .mark_bar()
            .encode(
                x=alt.X("pct:Q", title="%"),
                y=alt.Y("variable:N", sort="-x", title="Variable"),
                tooltip=[
                    alt.Tooltip("variable:N", title="Variable"),
                    alt.Tooltip("score:Q", format=".4f", title="Score"),
                    alt.Tooltip("pct:Q", format=".2f", title="%"),
                    alt.Tooltip("pct_acum:Q", format=".2f", title="% acumulado"),
                ],
            )
            .properties(height=min(34 * topk, 480))
        )

        st.altair_chart(bars2 + vline_90(), use_container_width=True)

    st.caption(
        f"Objetivo: **{TARGET_COL}** · Clases: "
        f"{dict(pd.Series(y_raw).value_counts().sort_index())}. "
        f"Las tablas incluyen **%** y **% acumulado** sobre el total del score."
    )

# Llamada protegida (si usas tu envoltorio, mantenlo; si no, llama directo)
# safe_run("2.1 Selección categóricas", sec_21_cat_selection)
sec_21_cat_selection()



# =========================
# Sección: Análisis Numéricas
# =========================
def sec_analisis_num():
    section_header("Análisis de variables numéricas", "Boxplot e histograma.")
    variables_numericas = df.select_dtypes(include=["number"]).columns.tolist()
    if not variables_numericas:
        st.warning("No se detectaron variables numéricas en `df`.")
        return

    card_controls("Controles de visualización - Numéricas")
    with st.container():
        c1, c2 = st.columns([2, 1])
        with c1:
            var_num = st.selectbox("Variable numérica", options=variables_numericas, index=0, key="num_var_top")
        with c2:
            bins = st.slider("Número de bins (histograma)", 5, 100, 30, 5, key="num_bins_top")

    serie_num = df[var_num].dropna()

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1: st.metric("Nº datos no nulos", f"{serie_num.shape[0]:,}".replace(",", "."))
    with m2: st.metric("Mínimo", f"{serie_num.min():.2f}")
    with m3: st.metric("Máximo", f"{serie_num.max():.2f}")
    with m4: st.metric("Media", f"{serie_num.mean():.2f}")
    with m5: st.metric("Desv. Estándar", f"{serie_num.std():.2f}")

    g1, g2 = st.columns([1.1, 1.3], gap="large")
    with g1:
        st.subheader(f"Boxplot de `{var_num}`")
        box_data = pd.DataFrame({var_num: serie_num})
        box_chart = (
            alt.Chart(box_data)
            .mark_boxplot(size=100)
            .encode(y=alt.Y(var_num, type="quantitative"))
            .properties(width=200, height=400)
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
                y="count()",
                tooltip=[alt.Tooltip(var_num, bin=alt.Bin(maxbins=bins)), alt.Tooltip("count()", title="Frecuencia")],
            )
            .properties(height=400)
        )
        st.altair_chart(hist_chart, use_container_width=True)

safe_run("Análisis numéricas", sec_analisis_num)

# =========================
# 2.1 Selección de categóricas (Clasificación: y = Stage)
# =========================
def sec_21_cat_selection():
    section_header("2.1. Selección de características categóricas (Clasificación)")
    if TARGET_COL not in df.columns:
        st.error("❌ No se encontró la columna objetivo 'Stage'.")
        return

    y_raw = df[TARGET_COL]
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c != TARGET_COL]
    if not cat_cols:
        st.info("No hay variables categóricas (excluyendo la objetivo).")
        return

    card_controls("Controles")
    c1, c2, c3 = st.columns([2.2, 1, 1])
    with c1:
        cats_sel = st.multiselect("Categóricas a evaluar", options=cat_cols,
                                  default=cat_cols[:min(10, len(cat_cols))], key="cat21_sel")
    with c2:
        metodo = st.radio("Método", ["Chi²", "Mutual Info"], 0, horizontal=True, key="cat21_m")
    with c3:
        topk = st.slider("Top K", 3, 50, 10, 1, key="cat21_topk")

    if not cats_sel:
        st.warning("Selecciona al menos una variable categórica para evaluar.")
        return

    X_cat = df[cats_sel].copy()
    y_codes, _ = pd.factorize(y_raw)

    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OH_ENCODER)])
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
                tooltip=["feature_dummy", "score"],
            ).properties(height=min(34 * topk, 480)),
            use_container_width=True,
        )
    with t2:
        st.dataframe(arrow_safe(agg_df.head(topk)), use_container_width=True)
        st.altair_chart(
            alt.Chart(agg_df.head(topk)).mark_bar().encode(
                x=alt.X("score:Q", title="Score (sumado por variable)"),
                y=alt.Y("variable:N", sort="-x", title="Variable"),
                tooltip=["variable", "score"],
            ).properties(height=min(34 * topk, 480)),
            use_container_width=True,
        )
    st.caption(f"Objetivo: **{TARGET_COL}** · Clases: {dict(pd.Series(y_raw).value_counts().sort_index())}")

safe_run("2.1 Selección categóricas", sec_21_cat_selection)

# =========================
# 2.2 Selección de numéricas (Clasificación: y = Stage)
# =========================
def sec_22_num_selection():
    section_header("2.2. Selección de características numéricas (Clasificación)")
    if TARGET_COL not in df.columns:
        st.error("❌ No se encontró la columna objetivo 'Stage'.")
        return

    y_raw = df[TARGET_COL]
    y_codes, _ = pd.factorize(y_raw)
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    num_cols = [c for c in num_cols if c != TARGET_COL]
    if not num_cols:
        st.info("No hay variables numéricas (excluyendo la objetivo).")
        return

    card_controls("Controles")
    n1, n2, n3 = st.columns([2.2, 1, 1])
    with n1:
        nums_sel = st.multiselect("Numéricas a evaluar", options=num_cols,
                                  default=num_cols[:min(10, len(num_cols))], key="num22_sel")
    with n2:
        metodo_num = st.radio("Método", ["ANOVA F", "Mutual Info"], 0, horizontal=True, key="num22_m")
    with n3:
        topk_22 = st.slider("Top K", 3, 50, 10, 1, key="num22_topk")

    if not nums_sel:
        st.warning("Selecciona al menos una variable numérica para evaluar.")
        return

    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    Xn = num_pipe.fit_transform(df[nums_sel])

    scores = f_classif(Xn, y_codes)[0] if metodo_num == "ANOVA F" else \
             mutual_info_classif(Xn, y_codes, random_state=42)

    sc2_df = pd.DataFrame({"feature": nums_sel, "score": scores}).sort_values("score", ascending=False)

    st.dataframe(arrow_safe(sc2_df.head(topk_22)), use_container_width=True)
    st.altair_chart(
        alt.Chart(sc2_df.head(topk_22)).mark_bar().encode(
            x=alt.X("score:Q", title="Score"),
            y=alt.Y("feature:N", sort="-x", title="Variable"),
            tooltip=["feature", "score"],
        ).properties(height=min(34 * topk_22, 480)),
        use_container_width=True,
    )
    st.caption(f"Objetivo: **{TARGET_COL}** · Clases: {dict(pd.Series(y_raw).value_counts().sort_index())}")

safe_run("2.2 Selección numéricas", sec_22_num_selection)

# =========================
# 2.3 Unión Cat + Num (Clasificación: y = Stage)
# =========================
def sec_23_union():
    section_header("2.3. Unión de variables categóricas y numéricas")
    if TARGET_COL not in df.columns:
        st.error("❌ No se encontró la columna objetivo 'Stage'.")
        return

    y_union = df[TARGET_COL]
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c != TARGET_COL]
    num_cols = [c for c in num_cols if c != TARGET_COL]

    if len(cat_cols) + len(num_cols) == 0:
        st.info("No hay variables disponibles para unir.")
        return

    card_controls("Controles")
    u1, u2 = st.columns([2, 1])
    with u1:
        cats_u = st.multiselect("Categóricas a incluir", options=cat_cols,
                                default=cat_cols[:min(5, len(cat_cols))], key="union23_cats")
        nums_u = st.multiselect("Numéricas a incluir", options=num_cols,
                                default=num_cols[:min(5, len(num_cols))], key="union23_nums")
    with u2:
        show_feat = st.checkbox("Ver nombres de features", True, key="union23_show")

    if len(cats_u) + len(nums_u) == 0:
        st.warning("Selecciona al menos una variable (categórica o numérica).")
        return

    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OH_ENCODER)])
    pre = ColumnTransformer([("num", num_pipe, nums_u), ("cat", cat_pipe, cats_u)], remainder="drop")

    X_raw = df[nums_u + cats_u]
    X_all = pre.fit_transform(X_raw, y_union)

    st.success(f"X transformada: **{X_all.shape[0]} filas × {X_all.shape[1]} columnas**")

    if show_feat:
        try:
            names = pre.get_feature_names_out()
        except Exception:
            names = [f"f{i}" for i in range(X_all.shape[1])]
        st.caption("Vista rápida de nombres de características generadas:")
        st.dataframe(arrow_safe(pd.DataFrame({"feature": names}).head(60)), use_container_width=True)

    # Persistir selección sin chocar con keys de widgets
    st.session_state["union23_cats_saved"] = list(cats_u)
    st.session_state["union23_nums_saved"] = list(nums_u)

safe_run("2.3 Unión", sec_23_union)


# =========================
# Modo debug (opcional)
# =========================
if DEBUG:
    st.markdown("---")
    st.markdown("#### ℹ️ Debug info")
    try:
        import sklearn, streamlit
        st.json({
            "python": os.sys.version,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "streamlit": streamlit.__version__,
            "altair": alt.__version__,
            "xgboost": __import__("xgboost").__version__ if XGB_OK else "no importado"
        })
    except Exception:
        pass


# ________________________________________________________________________________________________________________________________________________________________
st.markdown("""# 2. MCA Y PCA""")
# ________________________________________________________________________________________________________________________________________________________________
st.markdown("""## 2.1. MCA""")
# ________________________________________________________________________________________________________________________________________________________________
st.markdown("""## 2.2. PCA""")
# ________________________________________________________________________________________________________________________________________________________________
st.markdown("""# 3. RFE""")
# ________________________________________________________________________________________________________________________________________________________________

