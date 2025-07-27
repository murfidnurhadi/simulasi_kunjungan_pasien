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
# 🔢 RNG LCG AUTO (Default)
# ========================
def generate_rng_auto():
    a = random.randint(10, 50)
    c = random.randint(1, 30)
    m = random.randint(80, 200)
    z0 = random.randint(1, 50)
    n_gen = 48

    zi = z0
    rng_data = []
    for i in range(1, n_gen + 1):
        zi_minus_1 = zi
        zi = (a * zi_minus_1 + c) % m
        ui = zi / m
        angka_acak = int(ui * 100)
        if angka_acak == 0:
            angka_acak = 1
        rng_data.append((i, zi_minus_1, zi, round(ui, 4), angka_acak))

    rng_df = pd.DataFrame(rng_data, columns=["i", "Zᵢ₋₁", "Zᵢ", "Uᵢ", "Angka Acak (Uᵢ×100)"])
    return rng_df

if 'rng_df' not in st.session_state:
    st.session_state['rng_df'] = generate_rng_auto()

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
        col1.metric("Total Pengunjung", f"{total_seluruh:,}".replace(",", "."))
        col2.metric("Wilayah Terbanyak", total_per_wilayah.idxmax(), f"{total_per_wilayah.max():,}".replace(",", "."))
        col3.metric("Wilayah Tersedikit", total_per_wilayah.idxmin(), f"{total_per_wilayah.min():,}".replace(",", "."))

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
# 📈 Frekuensi & Interval (Semua Wilayah)
# ========================
elif menu == "📈 Frekuensi dan Interval":
    st.title("📈 Distribusi Frekuensi dan Interval (Semua Wilayah)")
    if not df.empty:
        df.columns = df.columns.str.strip().str.lower()
        exclude_cols = ["id", "bulan", "tahun"]
        daerah_cols = [col for col in df.columns if col not in exclude_cols]

        for daerah in daerah_cols:
            st.subheader(f"📍 Wilayah: {daerah.capitalize()}")
            data = df[daerah].dropna()
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

            bounds = freq_table["Interval Jumlah"].str.split(" - ", expand=True).astype(int)
            freq_table["Titik Tengah"] = ((bounds[0] + bounds[1]) / 2).round(0).astype(int)

            prob_raw = freq_table["Frekuensi"] / n
            prob_rounded = prob_raw.round(2)
            selisih = 1.00 - prob_rounded.sum()
            if abs(selisih) > 0:
                idx_max = prob_rounded.idxmax()
                prob_rounded.iloc[idx_max] += selisih

            freq_table["Probabilitas"] = prob_rounded
            freq_table["Prob. Kumulatif"] = freq_table["Probabilitas"].cumsum().round(2)
            freq_table["P.K * 100"] = (freq_table["Prob. Kumulatif"] * 100).astype(int)
            freq_table["Interval Angka Acak"] = [
                f"{lb} - {ub}" for lb, ub in zip([1] + [x+1 for x in freq_table["P.K * 100"][:-1]], freq_table["P.K * 100"])
            ]

            st.dataframe(freq_table, use_container_width=True)

            fig_bar = px.bar(freq_table, x="Interval Jumlah", y="Frekuensi", text="Frekuensi",
                             title=f"Distribusi Frekuensi - {daerah.capitalize()}",
                             color="Interval Jumlah")
            fig_bar.update_traces(textposition="outside")
            st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown(f"""
            **Xmin:** {x_min} | **Xmax:** {x_max} | **Range (R):** {R}  
            **Kelas (k):** {k} | **Panjang (h):** {h} | **Jumlah Data (n):** {n}
            """)

# ========================
# 🔢 RNG LCG
# ========================
elif menu == "🔢 RNG LCG":
    st.title("🔢 RNG LCG (Linear Congruential Generator)")
    st.markdown("""
    **Apa itu RNG LCG?**  
    RNG (Random Number Generator) adalah metode untuk menghasilkan bilangan acak.  
    **LCG (Linear Congruential Generator)** menggunakan rumus:  
    `Zᵢ = (a * Zᵢ₋₁ + c) mod m`  
    - **a (Multiplier):** Pengali  
    - **c (Increment):** Penambah  
    - **m (Modulus):** Batas bilangan  
    - **Z₀ (Seed):** Nilai awal  
    - **n:** Jumlah bilangan acak  
    """)

    a = st.number_input("Multiplier (a)", min_value=1, value=random.randint(10, 50))
    c = st.number_input("Increment (c)", min_value=0, value=random.randint(1, 30))
    m = st.number_input("Modulus (m)", min_value=10, value=random.randint(80, 200))
    z0 = st.number_input("Seed (Z₀)", min_value=0, value=random.randint(1, 50))
    n_gen = st.number_input("Jumlah Bilangan Acak", min_value=1, value=48)

    if st.button("🎲 Create Random Number"):
        zi = z0
        rng_data = []
        all_zi = []

        for i in range(1, n_gen + 1):
            zi_minus_1 = zi
            zi = (a * zi_minus_1 + c) % m
            ui = zi / m
            angka_acak = int(ui * 100)
            if angka_acak == 0: angka_acak = 1
            rng_data.append((i, zi_minus_1, zi, round(ui, 4), angka_acak))
            all_zi.append(zi)

        rng_df = pd.DataFrame(rng_data, columns=["i", "Zᵢ₋₁", "Zᵢ", "Uᵢ", "Angka Acak (Uᵢ×100)"])
        st.session_state['rng_df'] = rng_df

        st.subheader("📊 Hasil RNG LCG")
        st.dataframe(rng_df, use_container_width=True)

        duplicate_flag = len(all_zi) != len(set(all_zi))
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Bilangan", n_gen)
        col2.metric("Nilai Unik", len(set(all_zi)))
        col3.metric("Duplikat", n_gen - len(set(all_zi)))

        if duplicate_flag:
            st.warning("⚠️ Terdapat nilai Zᵢ yang duplikat.")
        else:
            st.success("✅ Tidak ada duplikat.")
