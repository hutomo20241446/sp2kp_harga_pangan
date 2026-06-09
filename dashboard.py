import streamlit as st
import psycopg2
import pandas as pd
import plotly.express as px
from datetime import date, timedelta
from dotenv import load_dotenv
import os

# =========================================================
# KONFIGURASI
# =========================================================

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

st.set_page_config(
    page_title="Dashboard Harga Pangan Jawa Tengah",
    page_icon="🛒",
    layout="wide",
)

# =========================================================
# HELPER QUERY
# =========================================================

@st.cache_data(ttl=300)
def fetch_wilayah():
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT wilayah_key, provinsi, kabupaten_kota
                FROM dim_wilayah
                ORDER BY kabupaten_kota
            """)
            rows = cur.fetchall()
    return [{"wilayah_key": r[0], "provinsi": r[1], "kabupaten_kota": r[2]} for r in rows]


@st.cache_data(ttl=300)
def fetch_komoditas():
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT komoditas_key, komoditas, unit
                FROM dim_komoditas
                ORDER BY komoditas
            """)
            rows = cur.fetchall()
    return [{"komoditas_key": r[0], "komoditas": r[1], "unit": r[2]} for r in rows]


@st.cache_data(ttl=60)
def fetch_harga(wilayah_keys: tuple, komoditas_key: int, tanggal_mulai: str, tanggal_akhir: str):
    placeholders = ", ".join(["%s"] * len(wilayah_keys))
    query = f"""
        SELECT
            f.tanggal,
            w.kabupaten_kota,
            k.komoditas,
            k.unit,
            f.harga
        FROM fact_harga_harian f
        JOIN dim_wilayah   w ON w.wilayah_key   = f.wilayah_key
        JOIN dim_komoditas k ON k.komoditas_key = f.komoditas_key
        WHERE f.wilayah_key   IN ({placeholders})
          AND f.komoditas_key = %s
          AND f.tanggal BETWEEN %s AND %s
        ORDER BY f.tanggal, w.kabupaten_kota
        LIMIT 5000
    """
    params = list(wilayah_keys) + [komoditas_key, tanggal_mulai, tanggal_akhir]

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["tanggal", "kabupaten_kota", "komoditas", "unit", "harga"])
    df["tanggal"] = pd.to_datetime(df["tanggal"])
    df["harga"]   = pd.to_numeric(df["harga"], errors="coerce")
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
    wilayah_list   = fetch_wilayah()
    komoditas_list = fetch_komoditas()
except Exception as e:
    st.error(f"Gagal terhubung ke database: {e}")
    st.stop()

wilayah_map   = {w["kabupaten_kota"]: w["wilayah_key"] for w in wilayah_list}
komoditas_map = {k["komoditas"]: k["komoditas_key"]    for k in komoditas_list}

# =========================================================
# SIDEBAR FILTER
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
            min_value=date(2021, 1, 1),
            max_value=date.today(),
        )
    with col2:
        tgl_akhir = st.date_input(
            "Sampai",
            value=date.today(),
            min_value=date(2021, 1, 1),
            max_value=date.today(),
        )

    st.divider()
    st.caption("Cache diperbarui setiap 5 menit")

# =========================================================
# VALIDASI
# =========================================================

if not selected_wilayah:
    st.warning("Pilih minimal satu wilayah.")
    st.stop()

if tgl_mulai > tgl_akhir:
    st.error("Tanggal 'Dari' tidak boleh lebih besar dari 'Sampai'.")
    st.stop()

# =========================================================
# FETCH DATA
# =========================================================

wilayah_keys   = tuple(wilayah_map[w] for w in selected_wilayah)
komoditas_key  = komoditas_map[selected_komoditas]
komoditas_unit = next(k["unit"] for k in komoditas_list if k["komoditas"] == selected_komoditas)

with st.spinner("Mengambil data..."):
    df = fetch_harga(
        wilayah_keys,
        komoditas_key,
        tgl_mulai.isoformat(),
        tgl_akhir.isoformat(),
    )

if df.empty:
    st.info("Tidak ada data untuk filter yang dipilih.")
    st.stop()

# =========================================================
# METRIK RINGKASAN
# =========================================================

st.subheader(f"{selected_komoditas}  ·  {tgl_mulai} – {tgl_akhir}")

col1, col2, col3, col4 = st.columns(4)

harga_avg = df["harga"].mean()
harga_min = df["harga"].min()
harga_max = df["harga"].max()

cutoff    = df["tanggal"].max() - timedelta(days=7)
avg_baru  = df[df["tanggal"] >= cutoff]["harga"].mean()
avg_lama  = df[df["tanggal"] <  cutoff]["harga"].mean()
delta_pct = ((avg_baru - avg_lama) / avg_lama * 100) if avg_lama else 0

col1.metric("Rata-rata",   f"Rp {harga_avg:,.0f}", f"{delta_pct:+.1f}% vs minggu lalu")
col2.metric("Terendah",    f"Rp {harga_min:,.0f}")
col3.metric("Tertinggi",   f"Rp {harga_max:,.0f}")
col4.metric("Jumlah data", f"{len(df):,} baris")

st.divider()

# =========================================================
# GRAFIK
# =========================================================

tab1, tab2, tab3 = st.tabs(["📈 Tren Harga", "📊 Perbandingan Wilayah", "🗂 Data Mentah"])

with tab1:
    df_trend = (
        df.groupby(["tanggal", "kabupaten_kota"])["harga"]
        .mean()
        .reset_index()
    )
    fig = px.line(
        df_trend,
        x="tanggal",
        y="harga",
        color="kabupaten_kota",
        labels={
            "tanggal":        "Tanggal",
            "harga":          f"Harga (Rp/{komoditas_unit})",
            "kabupaten_kota": "Wilayah",
        },
        title=f"Tren Harga {selected_komoditas}",
    )
    fig.update_layout(
        legend_title_text="Wilayah",
        hovermode="x unified",
        yaxis_tickformat=",.0f",
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    df_bar = (
        df.groupby("kabupaten_kota")["harga"]
        .mean()
        .reset_index()
        .sort_values("harga", ascending=False)
    )
    fig2 = px.bar(
        df_bar,
        x="kabupaten_kota",
        y="harga",
        labels={
            "kabupaten_kota": "Wilayah",
            "harga":          f"Rata-rata Harga (Rp/{komoditas_unit})",
        },
        title=f"Rata-rata Harga {selected_komoditas} per Wilayah",
        color="harga",
        color_continuous_scale="Blues",
    )
    fig2.update_layout(
        yaxis_tickformat=",.0f",
        coloraxis_showscale=False,
        xaxis_tickangle=-30,
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.dataframe(
        df[["tanggal", "kabupaten_kota", "komoditas", "unit", "harga"]]
        .sort_values(["tanggal", "kabupaten_kota"])
        .assign(harga=df["harga"].map("Rp {:,.0f}".format))
        .reset_index(drop=True),
        use_container_width=True,
        height=400,
    )
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download CSV",
        data=csv,
        file_name=f"harga_{selected_komoditas}_{tgl_mulai}_{tgl_akhir}.csv",
        mime="text/csv",
    )
