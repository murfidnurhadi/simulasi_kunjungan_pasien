import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.express as px
import os
import random

# ========================
# 🔧 Konfigurasi Halaman
# ========================
logo_path = "images/unikom.png"
if os.path.exists(logo_path):
    st.set_page_config(page_title="Dashboard Simulasi Monte Carlo", layout="wide", page_icon=logo_path)
else:
    st.set_page_config(page_title="Dashboard Simulasi Monte Carlo", layout="wide", page_icon="📊")

# ========================
# Sidebar Logo & Menu
# ========================
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, use_container_width=True)
else:
    st.sidebar.markdown("### 📊 Dashboard Kelompok 6")

st.sidebar.header("📌 Menu Navigasi")
menu = st.sidebar.radio("Pilih Halaman:", [
    "Dashboard", "Data Train", "Frekuensi & Interval", "RNG LCG", "Simulasi Monte Carlo"
])

# ========================
# Path File Dataset
# ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
excel_path = os.path.join(BASE_DIR, "dataset", "dataset.xlsx")
csv_path = os.path.join(BASE_DIR, "dataset", "DataTrain.csv")

@st.cache_data
def load_data():
    if os.path.exists(excel_path):
        try:
            return pd.read_excel(excel_path, sheet_name="DataTrain")
        except Exception as e:
            st.error(f"❌ Gagal membaca Excel: {e}")
            return pd.DataFrame()
    elif os.path.exists(csv_path):
        try:
            return pd.read_csv(csv_path)
        except Exception as e:
            st.error(f"❌ Gagal membaca CSV: {e}")
            return pd.DataFrame()
    else:
        st.warning("⚠ Dataset tidak ditemukan. Silakan upload file.")
        uploaded_file = st.file_uploader("Upload file CSV atau Excel", type=["csv", "xlsx"])
        if uploaded_file:
            try:
                if uploaded_file.name.endswith(".csv"):
                    return pd.read_csv(uploaded_file)
                else:
                    return pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"❌ Gagal membaca file upload: {e}")
        return pd.DataFrame()

# Load Data
df = load_data()

# ========================
# 🟥 HALAMAN: DASHBOARD
# ========================
if menu == "Dashboard":
    st.title("📊 Dashboard Simulasi Monte Carlo - Kelompok 6")
    if not df.empty:
        # Normalisasi nama kolom
        df.columns = df.columns.str.strip().str.lower()
        exclude_cols = ["id", "bulan", "tahun"]
        daerah_cols = [col for col in df.columns if col not in exclude_cols]

        # Hitung total
        total_per_wilayah = df[daerah_cols].sum().sort_values(ascending=False)
        total_seluruh = total_per_wilayah.sum()
        wilayah_terbanyak = total_per_wilayah.idxmax()
        nilai_terbanyak = total_per_wilayah.max()
        wilayah_tersedikit = total_per_wilayah.idxmin()
        nilai_tersedikit = total_per_wilayah.min()

        # Tampilkan informasi
        st.markdown(f"**Total seluruh pengunjung:** {total_seluruh}")
        st.markdown(f"**Wilayah terbanyak:** {wilayah_terbanyak.capitalize()} ({nilai_terbanyak})")
        st.markdown(f"**Wilayah tersedikit:** {wilayah_tersedikit.capitalize()} ({nilai_tersedikit})")

        # Grafik
        fig = px.bar(total_per_wilayah, x=total_per_wilayah.index, y=total_per_wilayah.values,
                     labels={'x': 'Wilayah', 'y': 'Total Pengunjung'},
                     title="Total Pengunjung per Wilayah", text=total_per_wilayah.values)
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠ Data tidak tersedia.")

# ========================
# 🟦 HALAMAN: DATA TRAIN
# ========================
elif menu == "Data Train":
    st.title("📋 Data Train Pengunjung")
    if not df.empty:
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("⚠ Data tidak tersedia.")

# ================================
# 🟩 HALAMAN: FREKUENSI & INTERVAL
# ================================
elif menu == "Frekuensi & Interval":
    st.title("📈 Distribusi Frekuensi per Daerah")
    if not df.empty:
        df.columns = df.columns.str.strip().str.lower()
        exclude_cols = ["id", "bulan", "tahun"]
        daerah_cols = [col for col in df.columns if col not in exclude_cols]

        selected_daerah = st.selectbox("📍 Pilih Daerah:", ["Pilih daerah"] + daerah_cols)
        if selected_daerah == "Pilih daerah":
            st.info("Pilih daerah untuk melihat distribusi frekuensi.")
        else:
            st.subheader(f"Distribusi Frekuensi: {selected_daerah.capitalize()}")
            data = df[selected_daerah].dropna()
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

            freq_table["No"] = range(1, len(freq_table) + 1)
            total = freq_table["Frekuensi"].sum()
            prob_raw = freq_table["Frekuensi"] / total
            prob_rounded = prob_raw.round(2)
            selisih = 1.00 - prob_rounded.sum()
            if abs(selisih) > 0:
                idx_max = prob_rounded.idxmax()
                prob_rounded.iloc[idx_max] += selisih

            freq_table["Probabilitas"] = prob_rounded
            freq_table["Prob. Kumulatif"] = freq_table["Probabilitas"].cumsum().round(2)

            upper_bounds = (freq_table["Prob. Kumulatif"] * 100).astype(int)
            lower_bounds = [1] + [ub + 1 for ub in upper_bounds[:-1]]
            freq_table["Interval Angka Acak"] = [f"{lb}-{ub}" for lb, ub in zip(lower_bounds, upper_bounds)]

            st.dataframe(freq_table, use_container_width=True)
            st.markdown(f"Jumlah Data: {n}, R: {R}, k: {k}, h: {h}")
    else:
        st.warning("⚠ Data tidak tersedia.")

# ========================
# 🟨 HALAMAN: RNG LCG
# ========================
elif menu == "RNG LCG":
    st.title("🔢 Linear Congruential Generator (LCG)")
    m = st.number_input("Modulus (m)", min_value=1, value=100, step=1)
    a = st.number_input("Multiplier (a)", min_value=1, value=5, step=1)
    c = st.number_input("Increment (c)", min_value=0, value=1, step=1)
    x0 = st.number_input("Seed (x0)", min_value=0, value=1, step=1)
    n_gen = st.number_input("Jumlah Bilangan Acak", min_value=1, value=10, step=1)

    if st.button("Generate"):
        rng_values = []
        xi = x0
        for _ in range(n_gen):
            xi = (a * xi + c) % m
            rng_values.append(xi)

        rng_df = pd.DataFrame({
            "i": range(1, n_gen + 1),
            "X_i": rng_values,
            "R_i = X_i/m": [round(val / m, 4) for val in rng_values]
        })
        st.dataframe(rng_df, use_container_width=True)

# ================================
# 🟧 HALAMAN: SIMULASI MONTE CARLO
# ================================
elif menu == "Simulasi Monte Carlo":
    st.title("🎲 Simulasi Monte Carlo")
    n_simulasi = st.number_input("Jumlah Simulasi", min_value=1, value=10, step=1)
    if st.button("Mulai Simulasi"):
        hasil = [random.randint(1, 100) for _ in range(n_simulasi)]
        sim_df = pd.DataFrame({"Percobaan": range(1, n_simulasi + 1), "Angka Acak": hasil})
        st.dataframe(sim_df, use_container_width=True)
