import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import plotly.express as px

st.set_page_config(page_title="AMASBA | Ingresos & Salidas", layout="wide")

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
        return pd.read_sql(text(sql), con, params={"d1": d1, "d2": d2})

@st.cache_data(ttl=300)
def load_salidas(d1, d2):
    sql = """
    SELECT id_registro_salida, fecha, tipo_pago, monto_esperado, monto_sal, monto_pendiente, deuda, observacion, id_certificado
    FROM tbl_registro_salida
    WHERE fecha BETWEEN :d1 AND :d2
    """
    with get_engine().connect() as con:
        return pd.read_sql(text(sql), con, params={"d1": d1, "d2": d2})

st.title("AMASBA | Dashboard de Ingresos y Salidas de Mineral")

today = pd.Timestamp.today().normalize()
d1, d2 = st.sidebar.date_input(
    "Rango de fechas",
    value=((today - pd.Timedelta(days=180)).date(), today.date())
)
if isinstance(d1, (tuple, list)):
    d1, d2 = d1

df_ing = load_ingresos(d1, d2)
df_sal = load_salidas(d1, d2)

# Tipado
if not df_ing.empty:
    df_ing["fecha"] = pd.to_datetime(df_ing["fecha"], errors="coerce")
    df_ing["monto_ing"] = pd.to_numeric(df_ing["monto_ing"], errors="coerce").fillna(0)

if not df_sal.empty:
    df_sal["fecha"] = pd.to_datetime(df_sal["fecha"], errors="coerce")
    for c in ["monto_esperado", "monto_sal", "monto_pendiente"]:
        df_sal[c] = pd.to_numeric(df_sal[c], errors="coerce").fillna(0)

# Filtros
st.sidebar.subheader("Filtros")
if not df_sal.empty:
    f_sal_pago = st.sidebar.multiselect(
        "Salidas: tipo_pago",
        sorted(df_sal["tipo_pago"].dropna().unique().tolist()),
        default=sorted(df_sal["tipo_pago"].dropna().unique().tolist())
    )
else:
    f_sal_pago = []

if not df_ing.empty:
    f_ing_tipo = st.sidebar.multiselect(
        "Ingresos: tipo_ingreso",
        sorted(df_ing["tipo_ingreso"].dropna().unique().tolist()),
        default=sorted(df_ing["tipo_ingreso"].dropna().unique().tolist())
    )
else:
    f_ing_tipo = []

sal = df_sal.copy()
ing = df_ing.copy()
if f_sal_pago and "tipo_pago" in sal.columns:
    sal = sal[sal["tipo_pago"].isin(f_sal_pago)]
if f_ing_tipo and "tipo_ingreso" in ing.columns:
    ing = ing[ing["tipo_ingreso"].isin(f_ing_tipo)]

# KPIs
total_ing = float(ing["monto_ing"].sum()) if not ing.empty else 0.0
total_sal = float(sal["monto_sal"].sum()) if not sal.empty else 0.0
total_pend = float(sal["monto_pendiente"].sum()) if not sal.empty else 0.0
neto = total_ing - total_sal

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Ingresos", f"S/ {total_ing:,.2f}")
c2.metric("Total Salidas", f"S/ {total_sal:,.2f}")
c3.metric("Total Pendiente", f"S/ {total_pend:,.2f}")
c4.metric("Neto (Ing - Sal)", f"S/ {neto:,.2f}")

# Tendencia mensual
colA, colB = st.columns(2)

with colA:
    st.subheader("Ingresos por mes")
    if ing.empty:
        st.info("Sin datos.")
    else:
        ing_m = (ing.assign(mes=ing["fecha"].dt.to_period("M").dt.to_timestamp())
                   .groupby("mes", as_index=False)
                   .agg(ingresos=("monto_ing", "sum")))
        st.plotly_chart(px.line(ing_m, x="mes", y="ingresos", markers=True), use_container_width=True)

with colB:
    st.subheader("Salidas por mes")
    if sal.empty:
        st.info("Sin datos.")
    else:
        sal_m = (sal.assign(mes=sal["fecha"].dt.to_period("M").dt.to_timestamp())
                   .groupby("mes", as_index=False)
                   .agg(salidas=("monto_sal", "sum"), pendiente=("monto_pendiente", "sum")))
        fig = px.line(sal_m, x="mes", y=["salidas", "pendiente"], markers=True)
        st.plotly_chart(fig, use_container_width=True)

# Composición salidas
st.subheader("Composición de Salidas")
c5, c6 = st.columns(2)
with c5:
    if sal.empty:
        st.info("Sin datos.")
    else:
        tmp = sal.groupby("tipo_pago", as_index=False).agg(total=("monto_sal", "sum"))
        st.plotly_chart(px.pie(tmp, names="tipo_pago", values="total"), use_container_width=True)

with c6:
    if sal.empty:
        st.info("Sin datos.")
    else:
        tmp = sal.groupby("deuda", as_index=False).agg(total_sal=("monto_sal", "sum"), total_pend=("monto_pendiente", "sum"), n=("id_registro_salida", "count"))
        st.dataframe(tmp, use_container_width=True, hide_index=True)

# Tablas
st.subheader("Detalle (muestra)")
t1, t2 = st.columns(2)
with t1:
    st.write("tbl_ingreso (Top 200)")
    st.dataframe(ing.sort_values("fecha", ascending=False).head(200), use_container_width=True, hide_index=True)
with t2:
    st.write("tbl_registro_salida (Top 200)")
    st.dataframe(sal.sort_values("fecha", ascending=False).head(200), use_container_width=True, hide_index=True)
