import streamlit as st
import pandas as pd
import numpy as np
import math
import random
import os
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
        return pd.read_excel(excel_path, sheet_name="DataTrain")
    else:
        st.warning("⚠ File Excel tidak ditemukan. Upload file .xlsx.")
        uploaded_file = st.file_uploader("Upload file Excel (.xlsx)", type=["xlsx"])
        if uploaded_file:
            return pd.read_excel(uploaded_file, sheet_name="DataTrain")
    return pd.DataFrame()

df = load_excel()

# ========================
# 🏠 Dashboard
# ========================
if menu == "🏠 Dashboard":
    st.title("📊 Dashboard Simulasi Monte Carlo")
    if not df.empty:
        st.write("Selamat datang! Data sudah dimuat.")
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
# 📈 Frekuensi & Interval
# ========================
elif menu == "📈 Frekuensi dan Interval":
    st.title("📈 Distribusi Frekuensi dan Interval")
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

            # Buat interval kelas
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

            # Probabilitas & kumulatif
            prob_raw = freq_table["Frekuensi"] / n
            prob_rounded = prob_raw.round(2)
            selisih = 1.00 - prob_rounded.sum()
            if abs(selisih) > 0:
                idx_max = prob_rounded.idxmax()
                prob_rounded.iloc[idx_max] += selisih

            freq_table["Probabilitas"] = prob_rounded
            freq_table["Prob. Kumulatif"] = freq_table["Probabilitas"].cumsum().round(2)
            freq_table["P.K * 100"] = (freq_table["Prob. Kumulatif"] * 100).astype(int)

            # Interval Angka Acak
            upper_bounds = freq_table["P.K * 100"]
            lower_bounds = [1] + [ub + 1 for ub in upper_bounds[:-1]]
            freq_table["Interval Angka Acak"] = [f"{lb} - {ub}" for lb, ub in zip(lower_bounds, upper_bounds)]

            # Tampilkan tabel
            st.dataframe(freq_table, use_container_width=True)
            st.session_state['freq_table'] = freq_table

            # Info tambahan
            st.markdown("---")
            st.subheader("ℹ️ Informasi Tambahan")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Xmin", x_min)
            col2.metric("Xmax", x_max)
            col3.metric("Range (R)", R)
            col4.metric("Kelas (k)", k)
            col5.metric("Panjang (h)", h)
            col6.metric("Jumlah Data (n)", n)
    else:
        st.warning("Data tidak tersedia.")

# ========================
# 🔢 RNG LCG
# ========================
elif menu == "🔢 RNG LCG":
    st.title("🔢 RNG LCG (Linear Congruential Generator)")

    # Input parameter
    a = st.number_input("Multiplier (a)", min_value=1, value=21)
    c = st.number_input("Increment (c)", min_value=0, value=17)
    m = st.number_input("Modulus (m)", min_value=1, value=100)
    z0 = st.number_input("Seed (Z₀)", min_value=0, value=42)
    n_gen = st.number_input("Jumlah Bilangan Acak", min_value=1, value=20)

    if st.button("🎲 Generate"):
        zi = z0
        rng_data = []
        all_zi = []
        duplicate_flag = False

        for i in range(1, n_gen + 1):
            zi_minus_1 = zi
            zi = (a * zi_minus_1 + c) % m
            ui = zi / m
            angka_acak = int(ui * 100)

            # Simpan ke tabel dengan Zᵢ₋₁ dikurangi 1
            zi_minus_1_display = zi_minus_1 - 1

            # Cek duplikat
            if zi in all_zi:
                duplicate_flag = True
            all_zi.append(zi)

            rng_data.append((i, zi_minus_1_display, zi, round(ui, 4), angka_acak))

        # Buat DataFrame
        rng_df = pd.DataFrame(
            rng_data,
            columns=["i", "Zᵢ₋₁ (display)", "Zᵢ", "Uᵢ", "Angka Acak (Uᵢ×100)"]
        )

        st.session_state['rng_df'] = rng_df

        # Tampilkan tabel
        st.subheader("📊 Hasil RNG LCG")
        st.dataframe(rng_df, use_container_width=True)

        # Info duplikat
        if duplicate_flag:
            st.warning("⚠️ Terdapat nilai Zᵢ yang duplikat.")
        else:
            st.success("✅ Tidak ada duplikat.")

        # Visualisasi
        st.subheader("📉 Visualisasi Zᵢ")
        fig = px.line(rng_df, x="i", y="Zᵢ", title="Perkembangan Nilai Zᵢ", markers=True)
        st.plotly_chart(fig, use_container_width=True)

# ========================
# 🎲 Simulasi Monte Carlo
# ========================
elif menu == "🎲 Simulasi":
    st.title("🎲 Simulasi Monte Carlo")
    if 'rng_df' not in st.session_state or 'freq_table' not in st.session_state:
        st.warning("Generate bilangan acak & buat tabel distribusi dulu.")
    else:
        rng_df = st.session_state['rng_df']
        freq_table = st.session_state['freq_table']

        st.subheader("Bilangan Acak:")
        st.dataframe(rng_df, use_container_width=True)

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

        sim_df = pd.DataFrame(sim_results)

        st.subheader("Hasil Simulasi")
        st.dataframe(sim_df, use_container_width=True)

        total = sim_df['Jumlah Pengunjung'].sum()
        avg = sim_df['Jumlah Pengunjung'].mean()

        st.markdown(f"**Total Simulasi:** {total}")
        st.markdown(f"**Rata-rata:** {avg:.2f}")

        # Visualisasi hasil simulasi
        fig2 = px.bar(sim_df, x="Percobaan", y="Jumlah Pengunjung", text="Jumlah Pengunjung",
                      title="Hasil Simulasi Monte Carlo", color="Jumlah Pengunjung", color_continuous_scale="Blues")
        fig2.update_traces(textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)
