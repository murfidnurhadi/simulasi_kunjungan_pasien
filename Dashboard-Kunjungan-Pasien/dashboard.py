import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import math
import random
import os

# ========================
# 🎨 Konfigurasi Halaman
# ========================
st.set_page_config(layout="wide", page_title="Simulasi Monte Carlo - Kelompok 6", page_icon="🎲")

# ========================
# 📂 Navigasi Sidebar
# ========================
with st.sidebar:
    st.markdown("## 🧭 Navigasi")
    st.markdown("---")
    menu = st.radio(
        "📂 Pilih Halaman:",
        options=["🏠 Dashboard", "📊 Data Train", "📈 Frekuensi dan Interval", "🔢 RNG LCG", "🎲 Simulasi"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("ℹ️ Pilih halaman untuk menampilkan data atau menjalankan simulasi.")

# ========================
# 📂 Load Data (DIKUNCI)
# ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
excel_path = os.path.join(BASE_DIR, "dataset", "data-awal.xlsx")

@st.cache_data
def load_excel():
    if os.path.exists(excel_path):
        try:
            return pd.read_excel(excel_path, sheet_name="DataTrain")
        except Exception as e:
            st.error(f"❌ Gagal membaca Excel: {e}")
            return pd.DataFrame()
    else:
        st.warning("⚠ File Excel tidak ditemukan. Upload file .xlsx.")
        uploaded_file = st.file_uploader("Upload file Excel (.xlsx)", type=["xlsx"])
        if uploaded_file:
            try:
                return pd.read_excel(uploaded_file, sheet_name="DataTrain")
            except Exception as e:
                st.error(f"❌ Gagal membaca file upload: {e}")
        return pd.DataFrame()

df = load_excel()  # 🔒 Kode ini dikunci, tidak diubah

# ========================
# 🏠 Dashboard
# ========================
if menu == "🏠 Dashboard":
    st.title("📊 Dashboard Simulasi Monte Carlo")
    st.write("Selamat datang! Gunakan menu di sidebar untuk navigasi.")

    if not df.empty:
        df.columns = df.columns.str.strip().str.lower()
        exclude_cols = ["id", "bulan", "tahun"]
        daerah_cols = [col for col in df.columns if col not in exclude_cols]

        total_per_wilayah = df[daerah_cols].sum().sort_values(ascending=False)
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
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Upload file Excel (.xlsx) untuk melihat data.")

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
# 📈 Frekuensi & Interval
# ========================
elif menu == "📈 Frekuensi dan Interval":
    st.title("📈 Frekuensi dan Interval")
    if not df.empty:
        df.columns = df.columns.str.strip().str.lower()
        exclude_cols = ["id", "bulan", "tahun"]
        daerah_cols = [col for col in df.columns if col not in exclude_cols]

        selected_daerah = st.selectbox("📍 Pilih Daerah:", ["Pilih daerah"] + daerah_cols)
        if selected_daerah != "Pilih daerah":
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

            total = freq_table["Frekuensi"].sum()
            prob_raw = freq_table["Frekuensi"] / total
            prob_rounded = prob_raw.round(2)
            selisih = 1.00 - prob_rounded.sum()
            if abs(selisih) > 0:
                idx_max = prob_rounded.idxmax()
                prob_rounded.iloc[idx_max] += selisih

            freq_table["Probabilitas"] = prob_rounded
            freq_table["Prob. Kumulatif"] = freq_table["Probabilitas"].cumsum().round(2)
            freq_table["P.K * 100"] = (freq_table["Prob. Kumulatif"] * 100).astype(int)
            upper_bounds = freq_table["P.K * 100"]
            lower_bounds = [1] + [ub + 1 for ub in upper_bounds[:-1]]
            freq_table["Interval Angka Acak"] = [f"{lb} - {ub}" for lb, ub in zip(lower_bounds, upper_bounds)]

            st.dataframe(freq_table, use_container_width=True)
            st.markdown(f"Jumlah Data: {n} | R: {R} | k: {k} | h: {h}")
    else:
        st.warning("Data tidak tersedia.")

# ========================
# 🔢 RNG LCG
# ========================
elif menu == "🔢 RNG LCG":
    st.title("🔢 Linear Congruential Generator (LCG)")
    m = st.number_input("Modulus (m)", min_value=1, value=100)
    a = st.number_input("Multiplier (a)", min_value=1, value=5)
    c = st.number_input("Increment (c)", min_value=0, value=1)
    x0 = st.number_input("Seed (x₀)", min_value=0, value=1)
    n_gen = st.number_input("Jumlah Bilangan Acak", min_value=1, value=10)

    if st.button("Generate"):
        rng_values = []
        xi = x0
        for _ in range(n_gen):
            xi = (a * xi + c) % m
            rng_values.append(xi)

        rng_df = pd.DataFrame({
            "i": range(1, n_gen + 1),
            "Xᵢ": rng_values,
            "Uᵢ": [round(val / m, 4) for val in rng_values]
        })
        st.session_state['rng_df'] = rng_df
        st.dataframe(rng_df, use_container_width=True)

# ========================
# 🎲 Simulasi Monte Carlo
# ========================
elif menu == "🎲 Simulasi":
    st.title("🎲 Simulasi Monte Carlo")
    if 'rng_df' not in st.session_state:
        st.warning("Generate bilangan acak dulu di menu RNG LCG.")
    else:
        rng_df = st.session_state['rng_df']
        st.subheader("Bilangan Acak:")
        st.dataframe(rng_df, use_container_width=True)

        if not df.empty:
            df.columns = df.columns.str.strip().str.lower()
            exclude_cols = ["id", "bulan", "tahun"]
            daerah_cols = [col for col in df.columns if col not in exclude_cols]
            selected_daerah = st.selectbox("Pilih Daerah:", ["Pilih daerah"] + daerah_cols)

            if selected_daerah != "Pilih daerah":
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

                total = freq_table["Frekuensi"].sum()
                prob_raw = freq_table["Frekuensi"] / total
                prob_rounded = prob_raw.round(2)
                selisih = 1.00 - prob_rounded.sum()
                if abs(selisih) > 0:
                    idx_max = prob_rounded.idxmax()
                    prob_rounded.iloc[idx_max] += selisih

                freq_table["Probabilitas"] = prob_rounded
                freq_table["Prob. Kumulatif"] = freq_table["Probabilitas"].cumsum().round(2)
                freq_table["P.K * 100"] = (freq_table["Prob. Kumulatif"] * 100).astype(int)
                upper_bounds = freq_table["P.K * 100"]
                lower_bounds = [1] + [ub + 1 for ub in upper_bounds[:-1]]
                freq_table["Interval Angka Acak"] = [f"{lb} - {ub}" for lb, ub in zip(lower_bounds, upper_bounds)]

                st.subheader("Tabel Distribusi")
                st.dataframe(freq_table, use_container_width=True)

                # Simulasi
                def get_simulated_value(rand, freq_table):
                    angka_acak = int(rand * 100)
                    if angka_acak == 0: angka_acak = 1
                    for _, row in freq_table.iterrows():
                        low, high = map(int, row["Interval Angka Acak"].split(' - '))
                        if low <= angka_acak <= high:
                            jumlah_low, jumlah_high = map(int, row["Interval Jumlah"].split(' - '))
                            return random.randint(jumlah_low, jumlah_high), angka_acak
                    return None, angka_acak

                sim_results = []
                for _, row in rng_df.iterrows():
                    val, acak = get_simulated_value(row["Uᵢ"], freq_table)
                    sim_results.append({"Percobaan": row["i"], "Bilangan Acak": acak, "Jumlah Pengunjung": val})

                st.subheader("Hasil Simulasi")
                sim_df = pd.DataFrame(sim_results)
                st.dataframe(sim_df, use_container_width=True)

                st.markdown(f"**Total Simulasi:** {sim_df['Jumlah Pengunjung'].sum()}")
                st.markdown(f"**Rata-rata:** {sim_df['Jumlah Pengunjung'].mean():.2f}")
