import streamlit as st
import psycopg2
import pandas as pd
import plotly.express as px
from datetime import date, timedelta
import google.generativeai as genai
import json
import re

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

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
except Exception:
    GEMINI_API_KEY = None

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
        {"wilayah_key": r[0], "provinsi": r[1], "kabupaten_kota": r[2]}
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
        {"komoditas_key": r[0], "komoditas": r[1], "unit": r[2]}
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
        JOIN dim_wilayah w ON w.wilayah_key = f.wilayah_key
        JOIN dim_komoditas k ON k.komoditas_key = f.komoditas_key
        WHERE f.wilayah_key IN ({placeholders})
          AND f.komoditas_key = %s
          AND f.tanggal BETWEEN %s AND %s
        ORDER BY f.tanggal, w.kabupaten_kota
        LIMIT 5000
    """
    params = list(wilayah_keys) + [komoditas_key, tanggal_mulai, tanggal_akhir]
    with psycopg2.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(
        rows,
        columns=["tanggal", "kabupaten_kota", "komoditas", "unit", "harga"]
    )
    df["tanggal"] = pd.to_datetime(df["tanggal"])
    df["harga"] = pd.to_numeric(df["harga"], errors="coerce")
    return df


def run_ai_query(sql: str) -> pd.DataFrame:
    sql_clean = sql.strip().rstrip(";")
    if not re.match(r"^\s*SELECT\b", sql_clean, re.IGNORECASE):
        raise ValueError("Hanya query SELECT yang diizinkan.")
    with psycopg2.connect(DATABASE_URL) as conn:
        return pd.read_sql_query(sql_clean, conn)


def build_system_prompt(wilayah_list, komoditas_list) -> str:
    wilayah_names = ", ".join(w["kabupaten_kota"] for w in wilayah_list)
    komoditas_names = ", ".join(
        f"{k['komoditas']} ({k['unit']})" for k in komoditas_list
    )
    return f"""Kamu adalah asisten analisis harga pangan Jawa Tengah.
Kamu memiliki akses ke database PostgreSQL dengan skema berikut:

TABLE dim_wilayah (wilayah_key INTEGER PK, provinsi VARCHAR, kabupaten_kota VARCHAR)
TABLE dim_komoditas (komoditas_key INTEGER PK, komoditas VARCHAR, unit VARCHAR)
TABLE fact_harga_harian (tanggal DATE, wilayah_key INT FK, komoditas_key INT FK, harga NUMERIC(12,2))

Wilayah tersedia (35 kabupaten/kota di Jawa Tengah): {wilayah_names}
Komoditas tersedia: {komoditas_names}

INSTRUKSI:
1. Buat SQL query PostgreSQL valid, hanya SELECT, LIMIT 200.
2. Selalu JOIN ke dim_wilayah dan dim_komoditas agar nama readable.
3. Kembalikan JSON PERSIS seperti ini tanpa markdown atau backtick:
{{"sql": "SELECT ...", "penjelasan_query": "...", "catatan": null}}
4. Jika pertanyaan tidak bisa dijawab dengan data yang ada, kembalikan sql: null.
"""


def ask_gemini(pertanyaan: str, wilayah_list, komoditas_list) -> dict:
    model = genai.GenerativeModel("gemini-2.5-flash-lite")
    prompt = build_system_prompt(wilayah_list, komoditas_list)
    full_prompt = f"{prompt}\n\nPertanyaan pengguna: {pertanyaan}"
    response = model.generate_content(full_prompt)
    raw = response.text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Respons AI tidak valid JSON:\n{raw}")


def summarize_gemini(pertanyaan: str, df_result: pd.DataFrame) -> str:
    model = genai.GenerativeModel("gemini-2.5-flash-lite")
    data_preview = df_result.head(20).to_string(index=False)
    prompt = f"""Berdasarkan pertanyaan: "{pertanyaan}"

Data hasil query (maks 20 baris pertama):
{data_preview}

Total baris: {len(df_result)}

Berikan ringkasan analisis singkat dan insightful dalam 2-4 kalimat bahasa Indonesia.
Fokus pada temuan utama, angka penting, dan kesimpulan yang actionable."""
    response = model.generate_content(prompt)
    return response.text.strip()


# =========================================================
# HEADER
# =========================================================

st.title("🛒 Dashboard Harga Pangan")
st.caption("Jawa Tengah · Data SP2KP Kementerian Perdagangan Republik Indonesia")
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

wilayah_map = {w["kabupaten_kota"]: w["wilayah_key"] for w in wilayah_list}
komoditas_map = {k["komoditas"]: k["komoditas_key"] for k in komoditas_list}

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

wilayah_keys = tuple(wilayah_map[w] for w in selected_wilayah)
komoditas_key = komoditas_map[selected_komoditas]
komoditas_unit = next(
    k["unit"] for k in komoditas_list if k["komoditas"] == selected_komoditas
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

st.subheader(f"{selected_komoditas} · {tgl_mulai} - {tgl_akhir}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rata-rata", f"Rp {df['harga'].mean():,.0f}")
c2.metric("Terendah", f"Rp {df['harga'].min():,.0f}")
c3.metric("Tertinggi", f"Rp {df['harga'].max():,.0f}")
c4.metric("Jumlah Data", f"{len(df):,}")

st.divider()

# =========================================================
# TAB
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Tren Harga",
    "📊 Perbandingan",
    "🗂 Data Tabular",
    "🤖 Tanya AI",
])

with tab1:
    trend = (
        df.groupby(["tanggal", "kabupaten_kota"])["harga"]
        .mean()
        .reset_index()
    )
    fig = px.line(trend, x="tanggal", y="harga", color="kabupaten_kota")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    bar = (
        df.groupby("kabupaten_kota")["harga"]
        .mean()
        .reset_index()
    )
    fig2 = px.bar(bar, x="kabupaten_kota", y="harga")
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.dataframe(df, use_container_width=True)

# =========================================================
# TAB 4 — TANYA AI
# =========================================================

with tab4:
    if not GEMINI_API_KEY:
        st.error(
            "GEMINI_API_KEY belum dikonfigurasi. "
            "Tambahkan di Streamlit Cloud → App Settings → Secrets."
        )
        st.stop()

    st.markdown("### 🤖 Tanya AI tentang Harga Pangan")
    st.caption(
        "Ajukan pertanyaan bebas. AI akan membuat query SQL, "
        "menjalankannya ke database, lalu merangkum hasilnya."
    )

    with st.expander("💡 Contoh pertanyaan"):
        for contoh in [
            "Wilayah mana yang punya harga beras paling mahal bulan ini?",
            "Bandingkan rata-rata harga cabai merah di Semarang dan Solo minggu lalu",
            "Tren harga minyak goreng di seluruh Jawa Tengah 30 hari terakhir",
            "5 komoditas dengan harga tertinggi hari ini",
            "Wilayah dengan harga bawang merah paling murah",
        ]:
            st.markdown(f"- *{contoh}*")

    if "ai_chat_history" not in st.session_state:
        st.session_state.ai_chat_history = []

    for entry in st.session_state.ai_chat_history:
        with st.chat_message("user"):
            st.write(entry["pertanyaan"])
        with st.chat_message("assistant"):
            if entry.get("error"):
                st.error(entry["error"])
            else:
                if entry.get("penjelasan_query"):
                    st.caption(f"🔍 {entry['penjelasan_query']}")
                if entry.get("df_result") is not None and not entry["df_result"].empty:
                    st.dataframe(entry["df_result"], use_container_width=True)
                if entry.get("ringkasan"):
                    st.markdown(entry["ringkasan"])
                if entry.get("catatan"):
                    st.info(entry["catatan"])
                if entry.get("sql"):
                    with st.expander("🔎 Lihat SQL yang dijalankan"):
                        st.code(entry["sql"], language="sql")

    pertanyaan = st.chat_input("Ketik pertanyaan tentang harga pangan...")

    if pertanyaan:
        with st.spinner("AI sedang memproses..."):
            entry = {"pertanyaan": pertanyaan}
            try:
                # Step 1: generate SQL
                ai_response = ask_gemini(pertanyaan, wilayah_list, komoditas_list)
                sql = ai_response.get("sql")
                entry["penjelasan_query"] = ai_response.get("penjelasan_query", "")
                entry["catatan"] = ai_response.get("catatan")
                entry["sql"] = sql

                if not sql:
                    entry["ringkasan"] = (
                        "Maaf, pertanyaan ini tidak dapat dijawab "
                        "dengan data yang tersedia di database."
                    )
                else:
                    # Step 2: eksekusi SQL
                    df_result = run_ai_query(sql)
                    entry["df_result"] = df_result

                    # Step 3: ringkas hasil
                    if df_result.empty:
                        entry["ringkasan"] = (
                            "Query berhasil dijalankan namun tidak ada data ditemukan."
                        )
                    else:
                        entry["ringkasan"] = summarize_gemini(pertanyaan, df_result)

            except ValueError as ve:
                entry["error"] = f"⚠️ {ve}"
            except Exception as e:
                entry["error"] = f"Terjadi kesalahan: {e}"

            st.session_state.ai_chat_history.append(entry)
            st.rerun()

    if st.session_state.ai_chat_history:
        if st.button("🗑️ Hapus Riwayat Chat", type="secondary"):
            st.session_state.ai_chat_history = []
            st.rerun()

# import streamlit as st
# import psycopg2
# import pandas as pd
# import plotly.express as px
# from datetime import date, timedelta

# # =========================================================
# # KONFIGURASI
# # =========================================================

# st.set_page_config(
#     page_title="Dashboard Harga Pangan Jawa Tengah",
#     page_icon="🛒",
#     layout="wide",
# )

# try:
#     DATABASE_URL = st.secrets["DATABASE_URL"]
# except Exception:
#     st.error(
#         "DATABASE_URL belum dikonfigurasi. "
#         "Buka Streamlit Cloud → App Settings → Secrets."
#     )
#     st.stop()

# # =========================================================
# # HELPER QUERY
# # =========================================================

# @st.cache_data(ttl=300)
# def fetch_wilayah():
#     with psycopg2.connect(DATABASE_URL) as conn:
#         with conn.cursor() as cur:
#             cur.execute("""
#                 SELECT wilayah_key, provinsi, kabupaten_kota
#                 FROM dim_wilayah
#                 ORDER BY kabupaten_kota
#             """)
#             rows = cur.fetchall()

#     return [
#         {
#             "wilayah_key": r[0],
#             "provinsi": r[1],
#             "kabupaten_kota": r[2]
#         }
#         for r in rows
#     ]


# @st.cache_data(ttl=300)
# def fetch_komoditas():
#     with psycopg2.connect(DATABASE_URL) as conn:
#         with conn.cursor() as cur:
#             cur.execute("""
#                 SELECT komoditas_key, komoditas, unit
#                 FROM dim_komoditas
#                 ORDER BY komoditas
#             """)
#             rows = cur.fetchall()

#     return [
#         {
#             "komoditas_key": r[0],
#             "komoditas": r[1],
#             "unit": r[2]
#         }
#         for r in rows
#     ]


# @st.cache_data(ttl=60)
# def fetch_harga(
#     wilayah_keys: tuple,
#     komoditas_key: int,
#     tanggal_mulai: str,
#     tanggal_akhir: str
# ):
#     placeholders = ", ".join(["%s"] * len(wilayah_keys))

#     query = f"""
#         SELECT
#             f.tanggal,
#             w.kabupaten_kota,
#             k.komoditas,
#             k.unit,
#             f.harga
#         FROM fact_harga_harian f
#         JOIN dim_wilayah w
#             ON w.wilayah_key = f.wilayah_key
#         JOIN dim_komoditas k
#             ON k.komoditas_key = f.komoditas_key
#         WHERE f.wilayah_key IN ({placeholders})
#           AND f.komoditas_key = %s
#           AND f.tanggal BETWEEN %s AND %s
#         ORDER BY f.tanggal, w.kabupaten_kota
#         LIMIT 5000
#     """

#     params = (
#         list(wilayah_keys)
#         + [komoditas_key, tanggal_mulai, tanggal_akhir]
#     )

#     with psycopg2.connect(DATABASE_URL) as conn:
#         with conn.cursor() as cur:
#             cur.execute(query, params)
#             rows = cur.fetchall()

#     if not rows:
#         return pd.DataFrame()

#     df = pd.DataFrame(
#         rows,
#         columns=[
#             "tanggal",
#             "kabupaten_kota",
#             "komoditas",
#             "unit",
#             "harga",
#         ],
#     )

#     df["tanggal"] = pd.to_datetime(df["tanggal"])
#     df["harga"] = pd.to_numeric(df["harga"], errors="coerce")

#     return df


# # =========================================================
# # HEADER
# # =========================================================

# st.title("🛒 Dashboard Harga Pangan")
# st.caption("Jawa Tengah · Data Bank Indonesia")
# st.divider()

# # =========================================================
# # LOAD DIMENSI
# # =========================================================

# try:
#     wilayah_list = fetch_wilayah()
#     komoditas_list = fetch_komoditas()
# except Exception as e:
#     st.error(f"Gagal terhubung ke database: {e}")
#     st.stop()

# wilayah_map = {
#     w["kabupaten_kota"]: w["wilayah_key"]
#     for w in wilayah_list
# }

# komoditas_map = {
#     k["komoditas"]: k["komoditas_key"]
#     for k in komoditas_list
# }

# # =========================================================
# # SIDEBAR
# # =========================================================

# with st.sidebar:
#     st.header("Filter")

#     selected_wilayah = st.multiselect(
#         "Wilayah",
#         options=list(wilayah_map.keys()),
#         default=list(wilayah_map.keys())[:3],
#     )

#     selected_komoditas = st.selectbox(
#         "Komoditas",
#         options=list(komoditas_map.keys()),
#     )

#     col1, col2 = st.columns(2)

#     with col1:
#         tgl_mulai = st.date_input(
#             "Dari",
#             value=date.today() - timedelta(days=90),
#         )

#     with col2:
#         tgl_akhir = st.date_input(
#             "Sampai",
#             value=date.today(),
#         )

# # =========================================================
# # VALIDASI
# # =========================================================

# if not selected_wilayah:
#     st.warning("Pilih minimal satu wilayah.")
#     st.stop()

# if tgl_mulai > tgl_akhir:
#     st.error("Tanggal tidak valid.")
#     st.stop()

# # =========================================================
# # DATA
# # =========================================================

# wilayah_keys = tuple(
#     wilayah_map[w]
#     for w in selected_wilayah
# )

# komoditas_key = komoditas_map[selected_komoditas]

# komoditas_unit = next(
#     k["unit"]
#     for k in komoditas_list
#     if k["komoditas"] == selected_komoditas
# )

# with st.spinner("Mengambil data..."):
#     df = fetch_harga(
#         wilayah_keys,
#         komoditas_key,
#         tgl_mulai.isoformat(),
#         tgl_akhir.isoformat(),
#     )

# if df.empty:
#     st.info("Tidak ada data.")
#     st.stop()

# # =========================================================
# # METRIK
# # =========================================================

# st.subheader(
#     f"{selected_komoditas} · {tgl_mulai} - {tgl_akhir}"
# )

# c1, c2, c3, c4 = st.columns(4)

# c1.metric(
#     "Rata-rata",
#     f"Rp {df['harga'].mean():,.0f}"
# )

# c2.metric(
#     "Terendah",
#     f"Rp {df['harga'].min():,.0f}"
# )

# c3.metric(
#     "Tertinggi",
#     f"Rp {df['harga'].max():,.0f}"
# )

# c4.metric(
#     "Jumlah Data",
#     f"{len(df):,}"
# )

# st.divider()

# # =========================================================
# # TAB
# # =========================================================

# tab1, tab2, tab3 = st.tabs(
#     [
#         "📈 Tren Harga",
#         "📊 Perbandingan",
#         "🗂 Data Tabular",
#     ]
# )

# with tab1:
#     trend = (
#         df.groupby(
#             ["tanggal", "kabupaten_kota"]
#         )["harga"]
#         .mean()
#         .reset_index()
#     )

#     fig = px.line(
#         trend,
#         x="tanggal",
#         y="harga",
#         color="kabupaten_kota",
#     )

#     st.plotly_chart(
#         fig,
#         use_container_width=True
#     )

# with tab2:
#     bar = (
#         df.groupby("kabupaten_kota")["harga"]
#         .mean()
#         .reset_index()
#     )

#     fig2 = px.bar(
#         bar,
#         x="kabupaten_kota",
#         y="harga",
#     )

#     st.plotly_chart(
#         fig2,
#         use_container_width=True
#     )

# with tab3:
#     st.dataframe(
#         df,
#         use_container_width=True
#     )
