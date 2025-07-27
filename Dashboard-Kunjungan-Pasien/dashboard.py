import streamlit as st
import pandas as pd
import numpy as np
import math
import os
import random
import plotly.express as px

# ========================
# 🎨 Konfigurasi Halaman
# ========================
st.set_page_config(layout="wide", page_title="Simulasi Monte Carlo", page_icon="🎲")

# ========================
# 📂 Sidebar Navigasi
# ========================
with st.sidebar:
    st.title("🧭 Navigasi")
    menu = st.radio(
        "Pilih Halaman:",
        ["🏠 Dashboard", "📊 Data Train", "📈 Frekuensi dan Interval", "🔢 RNG LCG", "🎲 Simulasi"]
    )

# ========================
# 📂 Load Data
# ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
excel_path = os.path.join(BASE_DIR, "dataset", "dataset.xlsx")

@st.cache_data
def load_excel():
    if os.path.exists(excel_path):
        df = pd.read_excel(excel_path, sheet_name="DataTrain")
    else:
        st.warning("⚠ File Excel tidak ditemukan. Upload file .xlsx.")
        uploaded_file = st.file_uploader("Upload file Excel (.xlsx)", type=["xlsx"])
        if uploaded_file:
            df = pd.read_excel(uploaded_file, sheet_name="DataTrain")
        else:
            return pd.DataFrame()
    
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df

df = load_excel()

# ========================
# RNG Function
# ========================
def generate_rng(a=21, c=17, m=100, n_gen=10):
    z0 = random.randint(1, 99999999)  # Seed otomatis random
    zi = z0
    rng_data = []

    for i in range(1, n_gen + 1):
        zi_minus_1 = zi
        zi = (a * zi_minus_1 + c) % m
        ui = zi / m
        angka_acak = int(ui * 100)
        rng_data.append((i, zi_minus_1, zi, round(ui, 4), angka_acak))

    rng_df = pd.DataFrame(rng_data, columns=["i", "Zᵢ₋₁", "Zᵢ", "Uᵢ", "Angka Acak"])
    return rng_df, z0

# ========================
# 🏠 Dashboard
# ========================
if menu == "🏠 Dashboard":
    st.title("📊 Dashboard Simulasi Monte Carlo")
    if not df.empty:
        df.columns = df.columns.str.strip().str.lower()
        exclude_cols = ["id", "bulan", "tahun"]
        daerah_cols = [col for col in df.columns if col not in exclude_cols]

        total_per_wilayah = df[daerah_cols].sum().sort_values(ascending=False)
        total_seluruh = total_per_wilayah.sum()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Pengunjung", f"{total_seluruh:,}".replace(",", "."))
        col2.metric("Wilayah Terbanyak", total_per_wilayah.idxmax(), f"{total_per_wilayah.max():,}".replace(",", "."))
        col3.metric("Wilayah Tersedikit", total_per_wilayah.idxmin(), f"{total_per_wilayah.min():,}".replace(",", "."))

        fig = px.bar(
            total_per_wilayah.reset_index(),
            x="index",
            y=0,
            color="index",
            text=0,
            title="Total Pengunjung per Wilayah",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Upload file Excel terlebih dahulu.")

# ========================
# 📊 Data Train
# ========================
elif menu == "📊 Data Train":
    st.title("📊 Data Train Pengunjung")
    if not df.empty:
        st.dataframe(df.reset_index(drop=True), use_container_width=True, hide_index=True)
    else:
        st.warning("Data tidak tersedia.")

# ========================
# 📈 Frekuensi Semua Wilayah
# ========================
elif menu == "📈 Frekuensi dan Interval":
    st.title("📈 Distribusi Frekuensi dan Interval (Semua Wilayah)")
    if not df.empty:
        df.columns = df.columns.str.strip().str.lower()
        exclude_cols = ["id", "bulan", "tahun"]
        daerah_cols = [col for col in df.columns if col not in exclude_cols]

        for col in daerah_cols:
            st.subheader(f"📍 Wilayah: {col.capitalize()}")
            data = df[col].dropna()
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

            bounds = freq_table["Interval Jumlah"].str.split(" - ", expand=True).astype(int)
            freq_table["Titik Tengah"] = ((bounds[0] + bounds[1]) / 2).round(0).astype(int)
            freq_table["Probabilitas"] = (freq_table["Frekuensi"] / n).round(2)
            freq_table["Prob. Kumulatif"] = freq_table["Probabilitas"].cumsum().round(2)
            freq_table["P.K * 100"] = (freq_table["Prob. Kumulatif"] * 100).astype(int)

            upper_bounds = freq_table["P.K * 100"]
            lower_bounds = [1] + [ub + 1 for ub in upper_bounds[:-1]]
            freq_table["Interval Angka Acak"] = [f"{lb} - {ub}" for lb, ub in zip(lower_bounds, upper_bounds)]

            st.dataframe(freq_table, use_container_width=True)
    else:
        st.warning("Data tidak tersedia.")

# ========================
# 🔢 RNG LCG
# ========================
elif menu == "🔢 RNG LCG":
    st.title("🔢 RNG LCG (Linear Congruential Generator)")
    st.markdown("""
    **Apa itu RNG LCG?**
    RNG LCG adalah metode untuk menghasilkan bilangan acak menggunakan rumus:
    \n`Zi = (a * Zi-1 + c) mod m`
    \n- **a** = multiplier
    \n- **c** = increment
    \n- **m** = modulus
    \n- **Zi** = bilangan acak ke-i
    """)

    if st.button("🎲 Generate Ulang RNG"):
        rng_df, seed = generate_rng()
        st.session_state["rng_df"] = rng_df
        st.success(f"Bilangan acak berhasil digenerate dengan seed: {seed}")

    if "rng_df" in st.session_state:
        st.subheader("📊 Hasil RNG")
        st.dataframe(st.session_state["rng_df"], use_container_width=True)
    else:
        st.info("Klik tombol di atas untuk generate RNG.")

# ========================
# 🎲 Simulasi Monte Carlo
# ========================
elif menu == "🎲 Simulasi":
    st.title("🎲 Hasil Simulasi Semua Wilayah")
    if not df.empty and "rng_df" in st.session_state:
        rng_df = st.session_state["rng_df"]
        df.columns = df.columns.str.strip().str.lower()
        exclude_cols = ["id", "bulan", "tahun"]
        daerah_cols = [col for col in df.columns if col not in exclude_cols]

        sim_data = {"i": rng_df["i"]}
        for daerah in daerah_cols:
            data = df[daerah].dropna()
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
            bounds = freq_table["Interval Jumlah"].str.split(" - ", expand=True).astype(int)
            freq_table["Titik Tengah"] = ((bounds[0] + bounds[1]) / 2).round(0).astype(int)

            prob = (freq_table["Frekuensi"] / n).round(2)
            freq_table["Probabilitas"] = prob
            freq_table["Prob. Kumulatif"] = prob.cumsum().round(2)
            freq_table["P.K * 100"] = (freq_table["Prob. Kumulatif"] * 100).astype(int)

            upper_bounds = freq_table["P.K * 100"]
            lower_bounds = [1] + [ub + 1 for ub in upper_bounds[:-1]]
            freq_table["Interval Angka Acak"] = [f"{lb} - {ub}" for lb, ub in zip(lower_bounds, upper_bounds)]

            def get_value(acak):
                for _, row in freq_table.iterrows():
                    low, high = map(int, row["Interval Angka Acak"].split(' - '))
                    if low <= acak <= high:
                        return row["Titik Tengah"]
                return 0

            sim_data[daerah] = [get_value(a) for a in rng_df["Angka Acak"]]

        sim_df = pd.DataFrame(sim_data)
        st.dataframe(sim_df, use_container_width=True)

        st.subheader("📊 Visualisasi")
        avg_per_wilayah = sim_df.drop(columns=["i"]).mean()
        total_per_wilayah = sim_df.drop(columns=["i"]).sum()

        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.bar(x=avg_per_wilayah.index, y=avg_per_wilayah.values,
                          title="Rata-rata Simulasi Pengunjung per Wilayah",
                          color=avg_per_wilayah.index)
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            fig2 = px.pie(values=total_per_wilayah.values, names=total_per_wilayah.index,
                          title="Distribusi Total Pengunjung per Wilayah")
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Generate RNG dulu di menu RNG LCG.")
