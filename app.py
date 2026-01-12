import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import plotly.express as px

st.set_page_config(page_title="AMASBA | Ingresos & Salidas", layout="wide")

# ----------------------------
# DB
# ----------------------------
@st.cache_resource
def get_engine():
    cfg = st.secrets["mysql"]
    url = (
        f"mysql+pymysql://{cfg['user']}:{cfg['password']}"
        f"@{cfg['host']}:{int(cfg.get('port', 3306))}/{cfg['database']}?charset=utf8mb4"
    )
    return create_engine(url, pool_pre_ping=True, pool_recycle=3600)

@st.cache_data(ttl=300)
def load_ingresos(d1, d2):
    sql = """
    SELECT id_ingreso, fecha, monto_ing, concepto, tipo_ingreso, tipo_pago, Tipo_Caja, id_registro_salida
    FROM tbl_ingreso
    WHERE fecha BETWEEN :d1 AND :d2
    """
    with get_engine().connect() as con:
        df = pd.read_sql(text(sql), con, params={"d1": d1, "d2": d2})
    return df

@st.cache_data(ttl=300)
def load_salidas(d1, d2):
    sql = """
    SELECT id_registro_salida, fecha, tipo_pago, monto_esperado, monto_sal, monto_pendiente, deuda, observacion, id_certificado
    FROM tbl_registro_salida
    WHERE fecha BETWEEN :d1 AND :d2
    """
    with get_engine().connect() as con:
        df = pd.read_sql(text(sql), con, params={"d1": d1, "d2": d2})
    return df

# ----------------------------
# Utils
# ----------------------------
def to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def add_time_features(df, date_col="fecha"):
    if df.empty or date_col not in df.columns:
        return df
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    # Semana como inicio de semana (lunes) usando Period W (start_time)
    df["semana"] = df[date_col].dt.to_period("W").apply(lambda r: r.start_time)
    df["mes"] = df[date_col].dt.to_period("M").apply(lambda r: r.start_time)
    df["dia_semana"] = df[date_col].dt.day_name()
    df["dia"] = df[date_col].dt.date
    return df

def mode_or_na(s: pd.Series):
    if s is None or s.empty:
        return "N/A"
    m = s.dropna().mode()
    return m.iloc[0] if not m.empty else "N/A"

def pct_table(s: pd.Series, label="Categoría"):
    if s is None or s.empty:
        return pd.DataFrame(columns=[label, "%", "n"])
    vc = s.dropna().value_counts(dropna=True)
    df = pd.DataFrame({label: vc.index, "n": vc.values})
    df["%"] = (df["n"] / df["n"].sum() * 100).round(2)
    return df[[label, "%", "n"]]

def corr_value(df, x, y, method="pearson"):
    if df.empty or x not in df.columns or y not in df.columns:
        return np.nan, 0
    a = pd.to_numeric(df[x], errors="coerce")
    b = pd.to_numeric(df[y], errors="coerce")
    mask = a.notna() & b.notna()
    if mask.sum() < 3:
        return np.nan, int(mask.sum())
    return a[mask].corr(b[mask], method=method), int(mask.sum())

# ----------------------------
# UI
# ----------------------------
st.title("AMASBA | Analítica de Ingresos y Salidas de Mineral")

today = pd.Timestamp.today().normalize()
d1, d2 = st.sidebar.date_input(
    "Rango de fechas",
    value=((today - pd.Timedelta(days=180)).date(), today.date())
)
if isinstance(d1, (tuple, list)):
    d1, d2 = d1

# Load data
df_ing = load_ingresos(d1, d2)
df_sal = load_salidas(d1, d2)

# Normalize
if not df_ing.empty:
    df_ing["fecha"] = pd.to_datetime(df_ing["fecha"], errors="coerce")
    df_ing = df_ing.dropna(subset=["fecha"])
    df_ing = to_num(df_ing, ["monto_ing"])
    df_ing["monto_ing"] = df_ing["monto_ing"].fillna(0)

if not df_sal.empty:
    df_sal["fecha"] = pd.to_datetime(df_sal["fecha"], errors="coerce")
    df_sal = df_sal.dropna(subset=["fecha"])
    df_sal = to_num(df_sal, ["monto_esperado", "monto_sal", "monto_pendiente"])
    for c in ["monto_esperado", "monto_sal", "monto_pendiente"]:
        if c in df_sal.columns:
            df_sal[c] = df_sal[c].fillna(0)

# Add time features
ing = add_time_features(df_ing, "fecha")
sal = add_time_features(df_sal, "fecha")

# Sidebar filters
st.sidebar.subheader("Filtros")
if not sal.empty and "tipo_pago" in sal.columns:
    f_sal_pago = st.sidebar.multiselect(
        "Salidas: tipo_pago",
        sorted(sal["tipo_pago"].dropna().unique().tolist()),
        default=sorted(sal["tipo_pago"].dropna().unique().tolist())
    )
else:
    f_sal_pago = []

if not ing.empty and "tipo_pago" in ing.columns:
    f_ing_pago = st.sidebar.multiselect(
        "Ingresos: tipo_pago",
        sorted(ing["tipo_pago"].dropna().unique().tolist()),
        default=sorted(ing["tipo_pago"].dropna().unique().tolist())
    )
else:
    f_ing_pago = []

if not ing.empty and "tipo_ingreso" in ing.columns:
    f_tipo_ing = st.sidebar.multiselect(
        "Ingresos: tipo_ingreso",
        sorted(ing["tipo_ingreso"].dropna().unique().tolist()),
        default=sorted(ing["tipo_ingreso"].dropna().unique().tolist())
    )
else:
    f_tipo_ing = []

# Apply filters
if f_sal_pago:
    sal = sal[sal["tipo_pago"].isin(f_sal_pago)]
if f_ing_pago:
    ing = ing[ing["tipo_pago"].isin(f_ing_pago)]
if f_tipo_ing:
    ing = ing[ing["tipo_ingreso"].isin(f_tipo_ing)]

# KPIs base
total_ing = float(ing["monto_ing"].sum()) if not ing.empty else 0.0
total_sal = float(sal["monto_sal"].sum()) if not sal.empty else 0.0
total_pend = float(sal["monto_pendiente"].sum()) if not sal.empty else 0.0
neto = total_ing - total_sal
ratio = (total_ing / total_sal) if total_sal > 0 else np.nan

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Ingresos", f"S/ {total_ing:,.2f}")
k2.metric("Total Salidas", f"S/ {total_sal:,.2f}")
k3.metric("Total Pendiente", f"S/ {total_pend:,.2f}")
k4.metric("Neto (Ing - Sal)", f"S/ {neto:,.2f}")
k5.metric("Ratio Ing/Sal", f"{ratio:.2f}" if np.isfinite(ratio) else "N/A")

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📈 Estadística", "🔗 Correlación"])

# ----------------------------
# TAB 1: DASHBOARD
# ----------------------------
with tab1:
    colA, colB = st.columns(2)

    with colA:
        st.subheader("Ingresos por mes")
        if ing.empty:
            st.info("Sin datos de ingresos en el rango/filtrado.")
        else:
            ing_m = (ing.groupby("mes", as_index=False)
                       .agg(ingresos=("monto_ing", "sum")))
            fig = px.line(ing_m, x="mes", y="ingresos", markers=True)
            st.plotly_chart(fig, use_container_width=True)

    with colB:
        st.subheader("Salidas y Pendiente por mes")
        if sal.empty:
            st.info("Sin datos de salidas en el rango/filtrado.")
        else:
            sal_m = (sal.groupby("mes", as_index=False)
                       .agg(salidas=("monto_sal", "sum"), pendiente=("monto_pendiente", "sum")))
            fig = px.line(sal_m, x="mes", y=["salidas", "pendiente"], markers=True)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Composición")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.write("Salidas por tipo_pago")
        if sal.empty:
            st.info("Sin datos.")
        else:
            tmp = sal.groupby("tipo_pago", as_index=False).agg(total=("monto_sal", "sum"))
            st.plotly_chart(px.pie(tmp, names="tipo_pago", values="total"), use_container_width=True)

    with c2:
        st.write("Ingresos por tipo_pago")
        if ing.empty:
            st.info("Sin datos.")
        else:
            tmp = ing.groupby("tipo_pago", as_index=False).agg(total=("monto_ing", "sum"))
            st.plotly_chart(px.pie(tmp, names="tipo_pago", values="total"), use_container_width=True)

    with c3:
        st.write("Salidas: deuda (resumen)")
        if sal.empty:
            st.info("Sin datos.")
        else:
            tmp = sal.groupby("deuda", as_index=False).agg(
                total_sal=("monto_sal", "sum"),
                total_pend=("monto_pendiente", "sum"),
                n=("id_registro_salida", "count")
            )
            st.dataframe(tmp, use_container_width=True, hide_index=True)

    st.subheader("Detalle (muestra)")
    t1, t2 = st.columns(2)
    with t1:
        st.write("tbl_ingreso (Top 200)")
        st.dataframe(ing.sort_values("fecha", ascending=False).head(200), use_container_width=True, hide_index=True)
    with t2:
        st.write("tbl_registro_salida (Top 200)")
        st.dataframe(sal.sort_values("fecha", ascending=False).head(200), use_container_width=True, hide_index=True)

# ----------------------------
# TAB 2: ESTADÍSTICA
# ----------------------------
with tab2:
    st.subheader("Estadística descriptiva y patrones operativos")

    s1, s2 = st.columns(2)

    with s1:
        st.markdown("### Salidas (monto_sal)")
        if sal.empty:
            st.info("Sin datos de salidas.")
        else:
            # Resumen general
            mean_d = sal.groupby("dia", as_index=False).agg(total=("monto_sal", "sum"))
            mean_w = sal.groupby("semana", as_index=False).agg(total=("monto_sal", "sum"))
            mean_m = sal.groupby("mes", as_index=False).agg(total=("monto_sal", "sum"))

            media_d = mean_d["total"].mean()
            media_w = mean_w["total"].mean()
            media_m = mean_m["total"].mean()

            mediana_w = mean_w["total"].median()
            std_w = mean_w["total"].std()

            moda_pago_sal = mode_or_na(sal["tipo_pago"])
            dia_top_sal = mode_or_na(sal["dia_semana"])  # modo del día de semana

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Media diaria", f"S/ {media_d:,.2f}")
            c2.metric("Media semanal", f"S/ {media_w:,.2f}")
            c3.metric("Mediana semanal", f"S/ {mediana_w:,.2f}")
            c4.metric("Desv. est. semanal", f"S/ {std_w:,.2f}" if np.isfinite(std_w) else "N/A")

            c5, c6 = st.columns(2)
            c5.metric("Moda tipo_pago (Salidas)", moda_pago_sal)
            c6.metric("Día más frecuente (Salidas)", dia_top_sal)

            # Media móvil 4 semanas
            mean_w = mean_w.sort_values("semana")
            mean_w["media_movil_4"] = mean_w["total"].rolling(window=4).mean()

            fig = px.line(mean_w, x="semana", y=["total", "media_movil_4"], markers=True,
                          title="Salidas semanales vs Media móvil (4 semanas)")
            st.plotly_chart(fig, use_container_width=True)

            st.write("Distribución % tipo_pago (Salidas)")
            st.dataframe(pct_table(sal["tipo_pago"], "tipo_pago"), use_container_width=True, hide_index=True)

    with s2:
        st.markdown("### Ingresos (monto_ing)")
        if ing.empty:
            st.info("Sin datos de ingresos.")
        else:
            mean_d = ing.groupby("dia", as_index=False).agg(total=("monto_ing", "sum"))
            mean_w = ing.groupby("semana", as_index=False).agg(total=("monto_ing", "sum"))
            mean_m = ing.groupby("mes", as_index=False).agg(total=("monto_ing", "sum"))

            media_d = mean_d["total"].mean()
            media_w = mean_w["total"].mean()
            media_m = mean_m["total"].mean()

            mediana_w = mean_w["total"].median()
            std_w = mean_w["total"].std()

            moda_pago_ing = mode_or_na(ing["tipo_pago"])
            moda_tipo_ing = mode_or_na(ing["tipo_ingreso"])
            dia_top_ing = mode_or_na(ing["dia_semana"])

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Media diaria", f"S/ {media_d:,.2f}")
            c2.metric("Media semanal", f"S/ {media_w:,.2f}")
            c3.metric("Mediana semanal", f"S/ {mediana_w:,.2f}")
            c4.metric("Desv. est. semanal", f"S/ {std_w:,.2f}" if np.isfinite(std_w) else "N/A")

            c5, c6, c7 = st.columns(3)
            c5.metric("Moda tipo_pago (Ingresos)", moda_pago_ing)
            c6.metric("Moda tipo_ingreso", moda_tipo_ing)
            c7.metric("Día más frecuente (Ingresos)", dia_top_ing)

            mean_w = mean_w.sort_values("semana")
            mean_w["media_movil_4"] = mean_w["total"].rolling(window=4).mean()

            fig = px.line(mean_w, x="semana", y=["total", "media_movil_4"], markers=True,
                          title="Ingresos semanales vs Media móvil (4 semanas)")
            st.plotly_chart(fig, use_container_width=True)

            st.write("Distribución % tipo_pago (Ingresos)")
            st.dataframe(pct_table(ing["tipo_pago"], "tipo_pago"), use_container_width=True, hide_index=True)

            st.write("Distribución % tipo_ingreso")
            st.dataframe(pct_table(ing["tipo_ingreso"], "tipo_ingreso"), use_container_width=True, hide_index=True)

# ----------------------------
# TAB 3: CORRELACIÓN
# ----------------------------
with tab3:
    st.subheader("Correlación y relación Ingresos vs Salidas")

    method = st.selectbox("Método", ["pearson", "spearman", "kendall"], index=0)

    mode = st.radio(
        "Modo de análisis",
        ["Agregado semanal", "Agregado mensual", "Por registro (join id_registro_salida)"],
        index=0
    )

    if mode == "Agregado semanal":
        if ing.empty and sal.empty:
            st.info("No hay datos para correlación.")
        else:
            ing_w = (ing.groupby("semana", as_index=False).agg(ingresos=("monto_ing", "sum"))
                     if not ing.empty else pd.DataFrame(columns=["semana", "ingresos"]))
            sal_w = (sal.groupby("semana", as_index=False).agg(salidas=("monto_sal", "sum"), pendiente=("monto_pendiente", "sum"))
                     if not sal.empty else pd.DataFrame(columns=["semana", "salidas", "pendiente"]))

            agg = pd.merge(ing_w, sal_w, on="semana", how="outer").fillna(0).sort_values("semana")

            x = st.selectbox("Variable X", ["ingresos", "salidas", "pendiente"], index=0)
            y = st.selectbox("Variable Y", ["ingresos", "salidas", "pendiente"], index=1)

            corr, n = corr_value(agg, x, y, method=method)
            st.metric("Correlación", f"{corr:.4f}" if np.isfinite(corr) else "N/A")
            st.caption(f"Pares válidos: {n}")

            st.plotly_chart(px.scatter(agg, x=x, y=y, title=f"Scatter semanal: {x} vs {y}"), use_container_width=True)
            st.dataframe(agg, use_container_width=True, hide_index=True)

    elif mode == "Agregado mensual":
        if ing.empty and sal.empty:
            st.info("No hay datos para correlación.")
        else:
            ing_m = (ing.groupby("mes", as_index=False).agg(ingresos=("monto_ing", "sum"))
                     if not ing.empty else pd.DataFrame(columns=["mes", "ingresos"]))
            sal_m = (sal.groupby("mes", as_index=False).agg(salidas=("monto_sal", "sum"), pendiente=("monto_pendiente", "sum"))
                     if not sal.empty else pd.DataFrame(columns=["mes", "salidas", "pendiente"]))

            agg = pd.merge(ing_m, sal_m, on="mes", how="outer").fillna(0).sort_values("mes")

            x = st.selectbox("Variable X", ["ingresos", "salidas", "pendiente"], index=0, key="mx")
            y = st.selectbox("Variable Y", ["ingresos", "salidas", "pendiente"], index=1, key="my")

            corr, n = corr_value(agg, x, y, method=method)
            st.metric("Correlación", f"{corr:.4f}" if np.isfinite(corr) else "N/A")
            st.caption(f"Pares válidos: {n}")

            st.plotly_chart(px.scatter(agg, x=x, y=y, title=f"Scatter mensual: {x} vs {y}"), use_container_width=True)
            st.dataframe(agg, use_container_width=True, hide_index=True)

    else:
        # Por registro (join)
        if ing.empty or sal.empty:
            st.info("Necesitas datos de ingresos y salidas para hacer join.")
        else:
            # Requiere que id_registro_salida esté poblado en ingresos
            ing_join = ing.dropna(subset=["id_registro_salida"]).copy()
            ing_join["id_registro_salida"] = pd.to_numeric(ing_join["id_registro_salida"], errors="coerce")
            ing_join = ing_join.dropna(subset=["id_registro_salida"])

            if ing_join.empty:
                st.warning("No hay ingresos vinculados a salidas (id_registro_salida) en el rango/filtrado.")
            else:
                sal_base = sal[["id_registro_salida", "monto_sal", "monto_esperado", "monto_pendiente"]].copy()
                joined = pd.merge(
                    ing_join[["id_ingreso", "monto_ing", "id_registro_salida"]],
                    sal_base,
                    on="id_registro_salida",
                    how="inner"
                )

                st.write(f"Registros correlacionables: {len(joined):,}")

                vars_join = ["monto_ing", "monto_sal", "monto_esperado", "monto_pendiente"]
                x = st.selectbox("Variable X", vars_join, index=0, key="jx")
                y = st.selectbox("Variable Y", vars_join, index=1, key="jy")

                corr, n = corr_value(joined, x, y, method=method)
                st.metric("Correlación", f"{corr:.4f}" if np.isfinite(corr) else "N/A")
                st.caption(f"Pares válidos: {n}")

                st.plotly_chart(px.scatter(joined, x=x, y=y, title=f"Scatter por registro: {x} vs {y}"), use_container_width=True)
                st.dataframe(joined.sort_values("id_ingreso", ascending=False).head(500), use_container_width=True, hide_index=True)
