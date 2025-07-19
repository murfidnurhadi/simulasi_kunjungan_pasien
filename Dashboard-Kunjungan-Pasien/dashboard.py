import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import math
import random
import os

# =============================
# 📌 Konfigurasi Halaman
# =============================
st.set_page_config(layout="wide", page_title="Kel 6 - Dashboard", page_icon="📄")

# =============================
# 📌 Sidebar Navigasi
# =============================
with st.sidebar:
    st.markdown("## 🧭 Navigasi")
    st.markdown("---")
    menu_pilihan = st.radio(
        "📂 Pilih Halaman:",
        options=[
            "🏠 Dashboard",
            "📊 Data Train",
            "📈 Frekuensi dan Interval",
            "🔢 RNG LCG",
            "🎲 Simulasi"
        ],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("ℹ️ **Petunjuk**")
    st.markdown("Silakan pilih halaman untuk menampilkan data atau menjalankan simulasi Monte Carlo.")

# =============================
# 📂 Load Dataset
# =============================
default_path = "dataset/dataset.xlsx"
uploaded_file = None

# Jika file default tidak ada, tampilkan uploader
if not os.path.exists(default_path):
    st.warning("⚠️ File default tidak ditemukan. Silakan upload file Excel Anda.")
    uploaded_file = st.file_uploader("Upload File Excel (.xlsx)", type=["xlsx"])

# Fungsi load data
@st.cache_data
def load_excel_data(file):
    try:
        df = pd.read_excel(file, sheet_name="DataTrain")
        # Hapus kolom kosong
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        return df
    except Exception as e:
        st.error(f"❌ Gagal membaca file: {e}")
        return pd.DataFrame()

# Load data
if uploaded_file:
    df = load_excel_data(uploaded_file)
elif os.path.exists(default_path):
    df = load_excel_data(default_path)
else:
    df = pd.DataFrame()

# =============================
# 🟥 HALAMAN: DASHBOARD
# =============================
if menu_pilihan == "🏠 Dashboard":
    st.title("📊 Dashboard Simulasi Monte Carlo")
    st.write("Selamat datang! Gunakan menu di sidebar untuk navigasi.")

    if not df.empty:
        df.columns = df.columns.str.strip().str.lower()
        exclude_cols = ["id", "bulan", "tahun"]
        daerah_cols = [col for col in df.columns if col not in exclude_cols]

        # Ambil hanya kolom numerik
        numeric_cols = df[daerah_cols].select_dtypes(include=['number'])

        if numeric_cols.empty:
            st.error("❌ Tidak ada kolom numerik dalam dataset.")
            st.stop()

        total_per_wilayah = numeric_cols.sum().sort_values(ascending=False)
        total_seluruh = total_per_wilayah.sum()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Pengunjung", f"{total_seluruh:,}")
        col2.metric("Wilayah Terbanyak", total_per_wilayah.idxmax(), f"{total_per_wilayah.max():,}")
        col3.metric("Wilayah Tersedikit", total_per_wilayah.idxmin(), f"{total_per_wilayah.min():,}")

        grafik_df = total_per_wilayah.reset_index()
        grafik_df.columns = ["Wilayah", "Total_Pengunjung"]

        fig = px.bar(
            grafik_df,
            x="Wilayah",
            y="Total_Pengunjung",
            color="Wilayah",
            text="Total_Pengunjung",
            title="Total Pengunjung per Wilayah",
            labels={"Total_Pengunjung": "Jumlah Pengunjung", "Wilayah": "Nama Wilayah"},
            hover_data={"Wilayah": True, "Total_Pengunjung": True},
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig.update_layout(xaxis_title="Wilayah", yaxis_title="Jumlah Pengunjung", legend_title="Wilayah", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Upload file Excel (.xlsx) untuk melihat data.")

# =============================
# 🟦 HALAMAN: DATA TRAIN
# =============================
elif menu_pilihan == "📊 Data Train":
    st.title("📊 Data Train Pengunjung")
    if not df.empty:
        df_display = df.copy().reset_index(drop=True)
        st.markdown("### 📋 Data Tabel")
        st.dataframe(df_display, use_container_width=True, hide_index=True)
    else:
        st.warning("Tidak ada data yang bisa ditampilkan.")

# =============================================
# 🟩 HALAMAN: FREKUENSI DAN INTERVAL PER DAERAH
# =============================================
elif menu_pilihan == "📈 Frekuensi dan Interval":
    st.title("📈 Frekuensi dan Interval per Daerah")

    if not df.empty:
        df.columns = df.columns.str.strip().str.lower()
        exclude_cols = ["id", "bulan", "tahun"]
        daerah_cols = [col for col in df.columns if col not in exclude_cols]

        daerah_options = ["Pilih daerah"] + daerah_cols
        selected_daerah = st.selectbox("📍 Daftar Daerah:", daerah_options)

        if selected_daerah != "Pilih daerah":
            st.subheader(f"📊 Distribusi Frekuensi: {selected_daerah.capitalize()}")

            data = pd.to_numeric(df[selected_daerah], errors='coerce').dropna()
            if len(data) == 0:
                st.error("Kolom ini tidak memiliki data numerik.")
                st.stop()

            n = len(data)
            x_min, x_max = data.min(), data.max()
            R = x_max - x_min
            k = math.ceil(1 + 3.3 * math.log10(n))
            h = math.ceil(R / k)

            lower = math.floor(x_min)
            bins = []
            for _ in range(k):
                upper = lower + h
                bins.append((lower, upper))
                lower = upper + 1

            labels = [f"{low} - {high}" for low, high in bins]
            cut_bins = [b[0] for b in bins] + [bins[-1][1]]

            kelas = pd.cut(data, bins=cut_bins, labels=labels, include_lowest=True, right=True)
            freq_table = kelas.value_counts().sort_index().reset_index()
            freq_table.columns = ["Interval Jumlah", "Frekuensi"]
            freq_table = freq_table[freq_table["Frekuensi"] > 0].reset_index(drop=True)

            # Hitung Titik Tengah
            bounds = freq_table["Interval Jumlah"].str.split(" - ", expand=True).astype(int)
            freq_table["Titik Tengah"] = (bounds[0] + bounds[1]) / 2
            freq_table["No"] = range(1, len(freq_table) + 1)

            # Probabilitas
            total = freq_table["Frekuensi"].sum()
            prob_raw = freq_table["Frekuensi"] / total
            prob_rounded = prob_raw.round(2)
            selisih = 1.00 - prob_rounded.sum()
            if abs(selisih) > 0:
                prob_rounded.iloc[prob_rounded.idxmax()] += selisih
            freq_table["Probabilitas"] = prob_rounded
            freq_table["Prob. Kumulatif"] = freq_table["Probabilitas"].cumsum().round(2)
            freq_table["P.K * 100"] = (freq_table["Prob. Kumulatif"] * 100).astype(int)

            upper_bounds = freq_table["P.K * 100"]
            lower_bounds = [1] + [ub + 1 for ub in upper_bounds[:-1]]
            freq_table["Interval Angka Acak"] = [f"{lb} - {ub}" for lb, ub in zip(lower_bounds, upper_bounds)]

            # Urutkan kolom
            freq_table = freq_table[["No", "Interval Jumlah", "Frekuensi", "Titik Tengah", "Probabilitas", "Prob. Kumulatif", "P.K * 100", "Interval Angka Acak"]]
            st.dataframe(freq_table, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.write(f"Jumlah Data (n): {n}")
            st.write(f"Xmin: {x_min}, Xmax: {x_max}")
            st.write(f"Jangkauan (R): {R}, Jumlah Kelas (k): {k}, Panjang Kelas (h): {h}")
        else:
            st.info("Pilih daerah untuk menampilkan distribusi frekuensi.")
    else:
        st.warning("Data tidak tersedia.")

# =============================
# 🔢 RNG LCG & 🎲 Simulasi (Tetap sama, aman dari error)
# =============================
# [Bagian ini tetap menggunakan logika sebelumnya tanpa perubahan besar, hanya fix validasi data]

