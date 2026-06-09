import streamlit as st
import psycopg2
import pandas as pd
import plotly.express as px
from datetime import date, timedelta

# =========================================================
# KONFIGURASI
# =========================================================

st.set_page_config(
    page_title="Dashboard Harga Pangan Jawa Tengah",
    page_icon="🛒",
    layout="wide",
)

try:
    DATABASE_URL = st.secrets["DATABASE_URL"]
except Exception:
    st.error(
        "DATABASE_URL belum dikonfigurasi. "
        "Buka Streamlit Cloud → App Settings → Secrets."
    )
    st.stop()

# =========================================================
# HELPER QUERY
# =========================================================

@st.cache_data(ttl=300)
def fetch_wilayah():
    with psycopg2.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT wilayah_key, provinsi, kabupaten_kota
                FROM dim_wilayah
                ORDER BY kabupaten_kota
            """)
            rows = cur.fetchall()

    return [
        {
            "wilayah_key": r[0],
            "provinsi": r[1],
            "kabupaten_kota": r[2]
        }
        for r in rows
    ]


@st.cache_data(ttl=300)
def fetch_komoditas():
    with psycopg2.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT komoditas_key, komoditas, unit
                FROM dim_komoditas
                ORDER BY komoditas
            """)
            rows = cur.fetchall()

    return [
        {
            "komoditas_key": r[0],
            "komoditas": r[1],
            "unit": r[2]
        }
        for r in rows
    ]


@st.cache_data(ttl=60)
def fetch_harga(
    wilayah_keys: tuple,
    komoditas_key: int,
    tanggal_mulai: str,
    tanggal_akhir: str
):
    placeholders = ", ".join(["%s"] * len(wilayah_keys))

    query = f"""
        SELECT
            f.tanggal,
            w.kabupaten_kota,
            k.komoditas,
            k.unit,
            f.harga
        FROM fact_harga_harian f
        JOIN dim_wilayah w
            ON w.wilayah_key = f.wilayah_key
        JOIN dim_komoditas k
            ON k.komoditas_key = f.komoditas_key
        WHERE f.wilayah_key IN ({placeholders})
          AND f.komoditas_key = %s
          AND f.tanggal BETWEEN %s AND %s
        ORDER BY f.tanggal, w.kabupaten_kota
        LIMIT 5000
    """

    params = (
        list(wilayah_keys)
        + [komoditas_key, tanggal_mulai, tanggal_akhir]
    )

    with psycopg2.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        rows,
        columns=[
            "tanggal",
            "kabupaten_kota",
            "komoditas",
            "unit",
            "harga",
        ],
    )

    df["tanggal"] = pd.to_datetime(df["tanggal"])
    df["harga"] = pd.to_numeric(df["harga"], errors="coerce")

    return df


# =========================================================
# HEADER
# =========================================================

st.title("🛒 Dashboard Harga Pangan")
st.caption("Jawa Tengah · Data Bank Indonesia")
st.divider()

# =========================================================
# LOAD DIMENSI
# =========================================================

try:
    wilayah_list = fetch_wilayah()
    komoditas_list = fetch_komoditas()
except Exception as e:
    st.error(f"Gagal terhubung ke database: {e}")
    st.stop()

wilayah_map = {
    w["kabupaten_kota"]: w["wilayah_key"]
    for w in wilayah_list
}

komoditas_map = {
    k["komoditas"]: k["komoditas_key"]
    for k in komoditas_list
}

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.header("Filter")

    selected_wilayah = st.multiselect(
        "Wilayah",
        options=list(wilayah_map.keys()),
        default=list(wilayah_map.keys())[:3],
    )

    selected_komoditas = st.selectbox(
        "Komoditas",
        options=list(komoditas_map.keys()),
    )

    col1, col2 = st.columns(2)

    with col1:
        tgl_mulai = st.date_input(
            "Dari",
            value=date.today() - timedelta(days=90),
        )

    with col2:
        tgl_akhir = st.date_input(
            "Sampai",
            value=date.today(),
        )

# =========================================================
# VALIDASI
# =========================================================

if not selected_wilayah:
    st.warning("Pilih minimal satu wilayah.")
    st.stop()

if tgl_mulai > tgl_akhir:
    st.error("Tanggal tidak valid.")
    st.stop()

# =========================================================
# DATA
# =========================================================

wilayah_keys = tuple(
    wilayah_map[w]
    for w in selected_wilayah
)

komoditas_key = komoditas_map[selected_komoditas]

komoditas_unit = next(
    k["unit"]
    for k in komoditas_list
    if k["komoditas"] == selected_komoditas
)

with st.spinner("Mengambil data..."):
    df = fetch_harga(
        wilayah_keys,
        komoditas_key,
        tgl_mulai.isoformat(),
        tgl_akhir.isoformat(),
    )

if df.empty:
    st.info("Tidak ada data.")
    st.stop()

# =========================================================
# METRIK
# =========================================================

st.subheader(
    f"{selected_komoditas} · {tgl_mulai} - {tgl_akhir}"
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Rata-rata",
    f"Rp {df['harga'].mean():,.0f}"
)

c2.metric(
    "Terendah",
    f"Rp {df['harga'].min():,.0f}"
)

c3.metric(
    "Tertinggi",
    f"Rp {df['harga'].max():,.0f}"
)

c4.metric(
    "Jumlah Data",
    f"{len(df):,}"
)

st.divider()

# =========================================================
# TAB
# =========================================================

tab1, tab2, tab3 = st.tabs(
    [
        "📈 Tren Harga",
        "📊 Perbandingan",
        "🗂 Data Tabular",
    ]
)

with tab1:
    trend = (
        df.groupby(
            ["tanggal", "kabupaten_kota"]
        )["harga"]
        .mean()
        .reset_index()
    )

    fig = px.line(
        trend,
        x="tanggal",
        y="harga",
        color="kabupaten_kota",
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with tab2:
    bar = (
        df.groupby("kabupaten_kota")["harga"]
        .mean()
        .reset_index()
    )

    fig2 = px.bar(
        bar,
        x="kabupaten_kota",
        y="harga",
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

with tab3:
    st.dataframe(
        df,
        use_container_width=True
    )
