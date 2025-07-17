import streamlit as st
import pandas as pd
import numpy as np
import math
import os
import plotly.express as px

# -----------------------
# KONFIGURASI HALAMAN
# -----------------------
st.set_page_config(page_title="Dashboard Simulasi Monte Carlo", layout="wide", page_icon="📄")
st.title("📊 Dashboard Simulasi Monte Carlo & Distribusi")

# -----------------------
# SIDEBAR MENU
# -----------------------
st.sidebar.header("📌 Menu Navigasi")
menu = st.sidebar.radio("Pilih Halaman:", [
    "Data Kunjungan Pasien",
    "Data Train",
    "Frekuensi dan Interval",
    "RNG LCG",
    "Simulasi"
])

# -----------------------
# PATH FILE
# -----------------------
excel_path = "dataset/dataset.xlsx"  # Excel di folder dataset
csv_kunjungan = "Data_Kunjungan_Pasien.csv"  # CSV di root
csv_datatrain = "DataTrain.csv"  # CSV cadangan di root

# -----------------------
# FUNGSI LOAD DATA
# -----------------------
@st.cache_data
def load_excel_sheet(sheet_name):
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        return df
    except Exception as e:
        st.error(f"❌ Gagal memuat sheet '{sheet_name}': {e}")
        return pd.DataFrame()

def load_csv(path):
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        st.error(f"❌ Gagal memuat CSV '{path}': {e}")
        return pd.DataFrame()

def tampilkan_tabel(df, download_name):
    if not df.empty:
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", csv, file_name=download_name, mime="text/csv")
    else:
        st.warning("⚠ Data tidak tersedia.")

# ========================
# ✅ HALAMAN: DATA KUNJUNGAN PASIEN
# ========================
if menu == "Data Kunjungan Pasien":
    st.subheader("📋 Data Kunjungan Pasien")
    df = load_csv(csv_kunjungan)
    tampilkan_tabel(df, "Data_Kunjungan_Pasien.csv")

# ========================
# ✅ HALAMAN: DATA TRAIN
# ========================
elif menu == "Data Train":
    st.subheader("📊 Data Train")
    df = load_excel_sheet("DataTrain")
    if df.empty:
        st.info("Menggunakan backup CSV...")
        df = load_csv(csv_datatrain)

    if not df.empty:
        tampilkan_tabel(df, "DataTrain.csv")

        # Filter dinamis
        with st.expander("🔍 Filter Data"):
            df_filtered = df.copy()
            for col in df.columns:
                if df[col].dtype == "object":
                    options = st.multiselect(f"Filter {col}", df[col].dropna().unique())
                    if options:
                        df_filtered = df_filtered[df_filtered[col].isin(options)]
            st.dataframe(df_filtered)

        # Visualisasi
        if len(df.columns) >= 2:
            st.subheader("📈 Visualisasi Data")
            chart_type = st.radio("Pilih Jenis Grafik", ["Bar", "Line", "Pie"], horizontal=True)
            col_x = st.selectbox("Kolom X", df.columns)
            col_y = st.selectbox("Kolom Y", df.columns)

            try:
                if chart_type == "Bar":
                    fig = px.bar(df, x=col_x, y=col_y, title="Bar Chart")
                elif chart_type == "Line":
                    fig = px.line(df, x=col_x, y=col_y, title="Line Chart")
                else:
                    fig = px.pie(df, names=col_x, values=col_y, title="Pie Chart")
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.warning("Kolom Y harus numerik untuk Bar/Line, dan Pie butuh kolom nama & nilai.")

# ========================
# ✅ HALAMAN: FREKUENSI DAN INTERVAL
# ========================
elif menu == "Frekuensi dan Interval":
    st.subheader("📈 Distribusi Frekuensi dan Interval")
    df = load_excel_sheet("DataTrain")

    if not df.empty:
        df.columns = df.columns.str.strip().str.lower()
        exclude_cols = ["id", "bulan", "tahun"]
        daerah_cols = [col for col in df.columns if col not in exclude_cols]

        daerah_options = ["Pilih daerah"] + daerah_cols
        selected_daerah = st.selectbox("📍 Pilih Daerah:", daerah_options)

        if selected_daerah != "Pilih daerah":
            data = df[selected_daerah].dropna()
            n = len(data)
            x_min, x_max = data.min(), data.max()
            R = x_max - x_min
            k = math.ceil(1 + 3.3 * math.log10(n))
            h = math.ceil(R / k)

            # Interval kelas
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
            freq_table.columns = ["Interval", "Frekuensi"]

            freq_table["No"] = range(1, len(freq_table) + 1)
            total = freq_table["Frekuensi"].sum()
            prob_raw = freq_table["Frekuensi"] / total
            prob_rounded = prob_raw.round(2)
            selisih = 1.00 - prob_rounded.sum()
            if abs(selisih) > 0:
                idx_max = prob_rounded.idxmax()
                prob_rounded.iloc[idx_max] += selisih
                prob_rounded = prob_rounded.round(2)

            freq_table["Probabilitas"] = prob_rounded
            freq_table["Prob. Kumulatif"] = freq_table["Probabilitas"].cumsum().round(2)

            upper_bounds = (freq_table["Prob. Kumulatif"] * 100).astype(int)
            lower_bounds = [1] + [ub + 1 for ub in upper_bounds[:-1]]
            freq_table["Interval Angka Acak"] = [f"{lb} - {ub}" for lb, ub in zip(lower_bounds, upper_bounds)]

            freq_table = freq_table[["No", "Interval", "Frekuensi", "Probabilitas", "Prob. Kumulatif", "Interval Angka Acak"]]
            st.dataframe(freq_table, use_container_width=True)

            # Info tambahan
            st.markdown(f"**Jumlah Data (n):** {n}")
            st.markdown(f"x_min: {x_min}, x_max: {x_max}, R: {R}, k: {k}, h: {h}")

# ========================
# ✅ HALAMAN: RNG & SIMULASI
# ========================
elif menu == "RNG LCG":
    st.subheader("🎲 Data RNG LCG")
    df = load_excel_sheet("RNG_LCG")
    tampilkan_tabel(df, "RNG_LCG.csv")

elif menu == "Simulasi":
    st.subheader("🧮 Data Simulasi Monte Carlo")
    df = load_excel_sheet("Simulasi")
    tampilkan_tabel(df, "Simulasi.csv")
