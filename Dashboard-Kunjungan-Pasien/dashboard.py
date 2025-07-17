import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.express as px

# ========================
# 🔧 Konfigurasi Halaman
# ========================
st.set_page_config(page_title="Dashboard Simulasi Monte Carlo", layout="wide")
st.title("📊 Dashboard Simulasi Monte Carlo - Kelompok 6")

# ========================
# Sidebar Navigasi
# ========================
st.sidebar.header("📌 Menu Navigasi")
menu = st.sidebar.radio("Pilih Halaman:", [
    "Dashboard", "Data Train", "Frekuensi & Interval", "RNG LCG"
])

# ========================
# Path File Excel
# ========================
excel_path = "Tubes_Mosi.xlsx"  # Pastikan file ini ada di repo GitHub kamu

@st.cache_data
def load_excel(sheet_name="DataTrain"):
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        return df
    except FileNotFoundError:
        st.error(f"❌ File {excel_path} tidak ditemukan.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return pd.DataFrame()

# ========================
# Load Data
# ========================
df = load_excel("DataTrain")

# ========================
# 🟥 HALAMAN: DASHBOARD
# ========================
if menu == "Dashboard":
    st.subheader("📊 Ringkasan Data Pengunjung")

    if not df.empty:
        df.columns = df.columns.str.strip().str.lower()
        exclude_cols = ["id", "bulan", "tahun"]
        daerah_cols = [col for col in df.columns if col not in exclude_cols]

        total_per_wilayah = df[daerah_cols].sum().sort_values(ascending=False)
        total_seluruh = total_per_wilayah.sum()
        wilayah_terbanyak = total_per_wilayah.idxmax()
        nilai_terbanyak = total_per_wilayah.max()
        wilayah_tersedikit = total_per_wilayah.idxmin()
        nilai_tersedikit = total_per_wilayah.min()

        st.markdown(f"**Total seluruh pengunjung:** {total_seluruh}")
        st.markdown(f"**Wilayah terbanyak:** {wilayah_terbanyak.capitalize()} ({nilai_terbanyak})")
        st.markdown(f"**Wilayah tersedikit:** {wilayah_tersedikit.capitalize()} ({nilai_tersedikit})")

        # Visualisasi dengan Plotly
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
    st.subheader("📋 Data Train Pengunjung")
    if not df.empty:
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("⚠ Data tidak tersedia.")

# ================================
# 🟩 HALAMAN: FREKUENSI & INTERVAL
# ================================
elif menu == "Frekuensi & Interval":
    st.subheader("📈 Distribusi Frekuensi per Daerah")

    if not df.empty:
        df.columns = df.columns.str.strip().str.lower()
        exclude_cols = ["id", "bulan", "tahun"]
        daerah_cols = [col for col in df.columns if col not in exclude_cols]

        daerah_options = ["Pilih daerah"] + daerah_cols
        selected_daerah = st.selectbox("📍 Pilih Daerah:", daerah_options)

        if selected_daerah == "Pilih daerah":
            st.info("Silakan pilih daerah untuk melihat distribusi frekuensi.")
        else:
            st.write(f"### Distribusi Frekuensi: **{selected_daerah.capitalize()}**")

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
                prob_rounded = prob_rounded.round(2)

            freq_table["Probabilitas"] = prob_rounded
            freq_table["Prob. Kumulatif"] = freq_table["Probabilitas"].cumsum().round(2)

            upper_bounds = (freq_table["Prob. Kumulatif"] * 100).astype(int)
            lower_bounds = [1] + [ub + 1 for ub in upper_bounds[:-1]]
            freq_table["Interval Angka Acak"] = [
                f"{lb} - {ub}" for lb, ub in zip(lower_bounds, upper_bounds)
            ]

            freq_table = freq_table[[
                "No", "Interval Jumlah", "Frekuensi", "Probabilitas", "Prob. Kumulatif", "Interval Angka Acak"
            ]]

            st.dataframe(freq_table, use_container_width=True)

            st.markdown("#### ℹ️ Info:")
            st.write(f"Jumlah Data (n): {n}")
            st.write(f"x_min: {x_min}, x_max: {x_max}")
            st.write(f"Jangkauan (R): {R}")
            st.write(f"Jumlah Kelas (k): {k}")
            st.write(f"Panjang Kelas (h): {h}")
    else:
        st.warning("⚠ Data tidak tersedia.")

# ========================
# 🟨 HALAMAN: RNG LCG
# ========================
elif menu == "RNG LCG":
    st.subheader("🔢 Linear Congruential Generator (LCG)")

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
