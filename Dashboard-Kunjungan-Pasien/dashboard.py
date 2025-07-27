import streamlit as st
import pandas as pd
import numpy as np
import math
import random
import os
import plotly.express as px
from io import BytesIO

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
if not df.empty:
    df.columns = df.columns.str.strip().str.lower()
    exclude_cols = ["id", "bulan", "tahun"]
    daerah_cols = [col for col in df.columns if col not in exclude_cols]

# ========================
# 🔢 RNG LCG Auto
# ========================
def generate_rng_auto(n_gen=48):
    a = random.randint(10, 50)
    c = random.randint(1, 30)
    m = random.randint(80, 200)
    z0 = random.randint(1, 99999999)  # Random Seed

    rng_data = []
    zi = z0
    for i in range(1, n_gen + 1):
        zi_minus_1 = zi
        zi = (a * zi_minus_1 + c) % m
        ui = zi / m
        angka_acak = int(ui * 100)
        if angka_acak == 0:
            angka_acak = 1
        rng_data.append((i, zi_minus_1, zi, round(ui, 4), angka_acak))

    rng_df = pd.DataFrame(rng_data, columns=["i", "Zᵢ₋₁", "Zᵢ", "Uᵢ", "Angka Acak"])
    return rng_df, a, c, m, z0

if "rng_df" not in st.session_state:
    st.session_state["rng_df"], st.session_state["a"], st.session_state["c"], st.session_state["m"], st.session_state["z0"] = generate_rng_auto()

# ========================
# 🏠 Dashboard
# ========================
if menu == "🏠 Dashboard":
    st.title("📊 Dashboard Simulasi Monte Carlo")
    if not df.empty:
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

        fig = px.bar(grafik_df, x="Wilayah", y="Total_Pengunjung", color="Wilayah",
                     text="Total_Pengunjung", title="Total Pengunjung per Wilayah",
                     color_discrete_sequence=px.colors.qualitative.Set3)
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
# 📈 Frekuensi & Interval Semua Wilayah
# ========================
elif menu == "📈 Frekuensi dan Interval":
    st.title("📈 Distribusi Frekuensi dan Interval (Semua Wilayah)")
    if not df.empty:
        for daerah in daerah_cols:
            st.subheader(f"📍 Wilayah: {daerah.capitalize()}")
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

            kelas = pd.cut(data, bins=cut_bins, labels=labels, include_lowest=True)
            freq_table = kelas.value_counts().sort_index().reset_index()
            freq_table.columns = ["Interval Jumlah", "Frekuensi"]
            st.dataframe(freq_table, use_container_width=True)
            st.markdown("---")

# ========================
# 🔢 RNG LCG Menu
# ========================
elif menu == "🔢 RNG LCG":
    st.title("🔢 RNG LCG (Linear Congruential Generator)")
    st.markdown("""
    **Apa itu RNG LCG?**  
    RNG (Random Number Generator) digunakan untuk menghasilkan bilangan acak.  
    LCG adalah salah satu metode RNG dengan formula:  
    **Zᵢ = (a × Zᵢ₋₁ + c) mod m**  
    - `a`: Multiplier  
    - `c`: Increment  
    - `m`: Modulus  
    - `Z₀`: Seed (nilai awal)  
    """)

    if st.button("🔄 Generate Ulang RNG"):
        st.session_state["rng_df"], st.session_state["a"], st.session_state["c"], st.session_state["m"], st.session_state["z0"] = generate_rng_auto()

    rng_df = st.session_state["rng_df"]
    st.subheader("📊 Hasil RNG LCG")
    st.write(f"a = {st.session_state['a']}, c = {st.session_state['c']}, m = {st.session_state['m']}, Z₀ = {st.session_state['z0']}")
    st.dataframe(rng_df, use_container_width=True)

# ========================
# 🎲 Simulasi Monte Carlo
# ========================
elif menu == "🎲 Simulasi":
    st.title("🎲 Hasil Simulasi Semua Wilayah")
    if not df.empty and "rng_df" in st.session_state:
        rng_df = st.session_state["rng_df"]
        sim_results = []

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
            freq_series = pd.cut(data, bins=cut_bins, labels=labels, include_lowest=True)
            freq_table = freq_series.value_counts().sort_index().reset_index()
            freq_table.columns = ["Interval Jumlah", "Frekuensi"]

            bounds = freq_table["Interval Jumlah"].str.split(" - ", expand=True).astype(int)
            freq_table["Titik Tengah"] = ((bounds[0] + bounds[1]) / 2).round(0).astype(int)
            total = freq_table["Frekuensi"].sum()
            prob = freq_table["Frekuensi"] / total
            freq_table["Probabilitas"] = prob.round(2)
            freq_table["Prob. Kumulatif"] = freq_table["Probabilitas"].cumsum().round(2)
            freq_table["P.K * 100"] = (freq_table["Prob. Kumulatif"] * 100).astype(int)

            upper_bounds = freq_table["P.K * 100"]
            lower_bounds = [1] + [ub + 1 for ub in upper_bounds[:-1]]
            freq_table["Interval Angka Acak"] = [f"{lb} - {ub}" for lb, ub in zip(lower_bounds, upper_bounds)]

            def get_value(rand):
                for _, row in freq_table.iterrows():
                    low, high = map(int, row["Interval Angka Acak"].split(' - '))
                    if low <= rand <= high:
                        return row["Titik Tengah"]
                return 0

            values = [get_value(acak) for acak in rng_df["Angka Acak"]]
            sim_results.append(values)

        sim_df = pd.DataFrame(sim_results).T
        sim_df.columns = daerah_cols
        sim_df.insert(0, "i", range(1, len(sim_df) + 1))

        st.dataframe(sim_df.head(10), use_container_width=True)

        st.subheader("📊 Visualisasi")
        avg_vals = sim_df[daerah_cols].mean()
        total_vals = sim_df[daerah_cols].sum()

        col1, col2 = st.columns(2)
        fig_bar = px.bar(avg_vals.reset_index(), x="index", y=0, text=0,
                         title="Rata-rata Simulasi Pengunjung per Wilayah",
                         color="index", color_discrete_sequence=px.colors.qualitative.Set3)
        fig_bar.update_traces(texttemplate='%{text:.0f}', textposition='outside')
        col1.plotly_chart(fig_bar, use_container_width=True)

        fig_pie = px.pie(total_vals.reset_index(), names="index", values=0,
                         title="Distribusi Total Pengunjung per Wilayah",
                         color_discrete_sequence=px.colors.qualitative.Set3)
        col2.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("### 📌 Kesimpulan")
        st.markdown(f"""
        - **Wilayah dengan pengunjung terbanyak:** {total_vals.idxmax()} ({int(total_vals.max()):,})
        - **Wilayah dengan pengunjung tersedikit:** {total_vals.idxmin()} ({int(total_vals.min()):,})
        - **Rata-rata seluruh wilayah:** {int(avg_vals.mean()):,} pengunjung
        """.replace(",", "."))

        # Tombol Download
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            sim_df.to_excel(writer, sheet_name="Simulasi", index=False)
        st.download_button("⬇ Download Hasil Simulasi (Excel)", buffer.getvalue(),
                           file_name="hasil_simulasi.xlsx", mime="application/vnd.ms-excel")
