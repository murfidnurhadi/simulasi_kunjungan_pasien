import streamlit as st
import pandas as pd
import numpy as np
import math
import os
import plotly.express as px
import gdown

# -----------------------
# KONFIGURASI HALAMAN
# -----------------------
st.set_page_config(page_title="Dashboard Simulasi Monte Carlo", layout="wide", page_icon="📄")
st.title("📊 Dashboard Simulasi Monte Carlo & Distribusi")

# -----------------------
# SIDEBAR NAVIGASI
# -----------------------
st.sidebar.header("📌 Menu Navigasi")
menu = st.sidebar.radio("Pilih Halaman:", [
    "Data Kunjungan Pasien",
    "Data Train",
    "Frekuensi dan Interval",
    "RNG LCG",
    "Simulasi"
])

source_option = st.sidebar.radio("Ambil Data Dari:", ["Lokal", "Google Drive"])

# -----------------------
# PATH LOKAL
# -----------------------
excel_path = "dataset/dataset.xlsx"
csv_kunjungan = "Data_Kunjungan_Pasien.csv"
csv_datatrain = "DataTrain.csv"

# -----------------------
# LINK GOOGLE DRIVE
# -----------------------
gdrive_links = {
    "Data Kunjungan Pasien": "https://drive.google.com/file/d/1ZHKsj50sICmlYDrDVhVo9u1871pIXiLm/view?usp=sharing",
    "Data Train": "https://drive.google.com/file/d/1ty1Ob9NBbpOt3cT7Mw2s78tjwjZw_9da/view?usp=sharing"
}

def convert_gdrive_link(link):
    if "/d/" in link:
        file_id = link.split("/d/")[1].split("/")[0]
        return f"https://drive.google.com/uc?id={file_id}"
    return link

# -----------------------
# FUNGSI LOAD DATA
# -----------------------
@st.cache_data
def load_excel(sheet_name):
    try:
        return pd.read_excel(excel_path, sheet_name=sheet_name)
    except Exception as e:
        st.error(f"Gagal load sheet {sheet_name}: {e}")
        return pd.DataFrame()

@st.cache_data
def load_csv(path):
    try:
        return pd.read_csv(path)
    except:
        return pd.DataFrame()

@st.cache_data
def load_csv_gdrive(link, local_name):
    try:
        url = convert_gdrive_link(link)
        output = local_name
        if not os.path.exists(output):
            gdown.download(url, output, quiet=False)
        return pd.read_csv(output)
    except Exception as e:
        st.error(f"Gagal ambil dari Google Drive: {e}")
        return pd.DataFrame()

def tampilkan_tabel(df, filename):
    if not df.empty:
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", csv, file_name=filename, mime="text/csv")
    else:
        st.warning("⚠ Data tidak tersedia.")

# ========================
# ✅ DATA KUNJUNGAN PASIEN
# ========================
if menu == "Data Kunjungan Pasien":
    st.subheader("📋 Data Kunjungan Pasien")
    if source_option == "Lokal":
        df = load_csv(csv_kunjungan)
    else:
        df = load_csv_gdrive(gdrive_links["Data Kunjungan Pasien"], "Data_Kunjungan_Pasien.csv")
    tampilkan_tabel(df, "Data_Kunjungan_Pasien.csv")

# ========================
# ✅ DATA TRAIN
# ========================
elif menu == "Data Train":
    st.subheader("📊 Data Train")
    if source_option == "Lokal":
        df = load_excel("DataTrain")
        if df.empty:
            df = load_csv(csv_datatrain)
    else:
        df = load_csv_gdrive(gdrive_links["Data Train"], "DataTrain.csv")

    if not df.empty:
        tampilkan_tabel(df, "DataTrain.csv")

        with st.expander("🔍 Filter Data"):
            df_filtered = df.copy()
            for col in df.columns:
                if df[col].dtype == "object":
                    opt = st.multiselect(f"Filter {col}", df[col].dropna().unique())
                    if opt:
                        df_filtered = df_filtered[df_filtered[col].isin(opt)]
            st.dataframe(df_filtered)

        if len(df.columns) >= 2:
            st.subheader("📈 Visualisasi")
            chart = st.radio("Jenis Grafik", ["Bar", "Line", "Pie"], horizontal=True)
            col_x = st.selectbox("Kolom X", df.columns)
            col_y = st.selectbox("Kolom Y", df.columns)
            try:
                if chart == "Bar":
                    fig = px.bar(df, x=col_x, y=col_y)
                elif chart == "Line":
                    fig = px.line(df, x=col_x, y=col_y)
                else:
                    fig = px.pie(df, names=col_x, values=col_y)
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.warning("Kolom Y harus numerik untuk Bar/Line.")

# ========================
# ✅ FREKUENSI DAN INTERVAL
# ========================
elif menu == "Frekuensi dan Interval":
    st.subheader("📈 Distribusi Frekuensi dan Interval")
    df = load_excel("DataTrain")
    if not df.empty:
        df.columns = df.columns.str.strip().str.lower()
        exclude = ["id", "bulan", "tahun"]
        daerah_cols = [c for c in df.columns if c not in exclude]

        pilih = st.selectbox("📍 Pilih Daerah:", ["Pilih"] + daerah_cols)
        if pilih != "Pilih":
            data = df[pilih].dropna()
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

            labels = [f"{low}-{high}" for low, high in bins]
            cut_bins = [b[0] for b in bins] + [bins[-1][1]]

            kelas = pd.cut(data, bins=cut_bins, labels=labels, include_lowest=True)
            freq_table = kelas.value_counts().sort_index().reset_index()
            freq_table.columns = ["Interval", "Frekuensi"]
            freq_table["No"] = range(1, len(freq_table)+1)
            total = freq_table["Frekuensi"].sum()
            prob = (freq_table["Frekuensi"]/total).round(2)
            selisih = 1 - prob.sum()
            if abs(selisih) > 0:
                prob.iloc[prob.idxmax()] += selisih
            freq_table["Probabilitas"] = prob
            freq_table["Prob. Kumulatif"] = prob.cumsum().round(2)
            upper_bounds = (freq_table["Prob. Kumulatif"] * 100).astype(int)
            lower_bounds = [1] + [u+1 for u in upper_bounds[:-1]]
            freq_table["Interval Angka Acak"] = [f"{lb}-{ub}" for lb, ub in zip(lower_bounds, upper_bounds)]
            st.dataframe(freq_table, use_container_width=True)

# ========================
# ✅ RNG & SIMULASI
# ========================
elif menu == "RNG LCG":
    st.subheader("🎲 RNG LCG")
    df = load_excel("RNG_LCG")
    tampilkan_tabel(df, "RNG_LCG.csv")

elif menu == "Simulasi":
    st.subheader("🧮 Simulasi Monte Carlo")
    df = load_excel("Simulasi")
    tampilkan_tabel(df, "Simulasi.csv")
