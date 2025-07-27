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
st.markdown("""
<style>
    body { background-color: #f9f9f9; }
    .css-18e3th9 { background-color: white; }
    h1, h2, h3, h4 { color: #333; }
</style>
""", unsafe_allow_html=True)

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
# 🔢 RNG LCG AUTO
# ========================
def generate_rng_auto(a=None, c=None, m=None, z0=None, n_gen=48):
    if a is None: a = random.randint(10, 50)
    if c is None: c = random.randint(1, 30)
    if m is None: m = random.randint(80, 200)
    if z0 is None: z0 = random.randint(1, 99999999)


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
    return rng_df, a, c, m, z0

# ✅ Inisialisasi Session State
if 'a' not in st.session_state or 'rng_df' not in st.session_state:
    rng_df, a, c, m, z0 = generate_rng_auto()
    st.session_state['rng_df'] = rng_df
    st.session_state['a'] = a
    st.session_state['c'] = c
    st.session_state['m'] = m
    st.session_state['z0'] = z0

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
# 📈 Frekuensi & Interval
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
    - **a (Multiplier):** Pengali (acak otomatis)  
    - **c (Increment):** Penambah (acak otomatis)  
    - **m (Modulus):** Batas bilangan (acak otomatis)  
    - **Z₀ (Seed):** Limit = 99999999  
    - **n:** Jumlah bilangan acak = 48  
    """)

    st.info(f"""
    **Parameter Saat Ini:**  
    - a = {st.session_state['a']}  
    - c = {st.session_state['c']}  
    - m = {st.session_state['m']}  
    - z₀ = {st.session_state['z0']}  
    """)

    if st.button("🎲 Generate Ulang RNG"):
        st.session_state['rng_df'], st.session_state['a'], st.session_state['c'], st.session_state['m'], st.session_state['z0'] = generate_rng_auto()

    rng_df = st.session_state['rng_df']
    st.subheader("📊 Hasil RNG LCG")
    st.dataframe(rng_df, use_container_width=True)

# ========================
# 🎲 Simulasi Monte Carlo
# ========================
elif menu == "🎲 Simulasi":
    st.title("🎲 Simulasi Monte Carlo (Semua Wilayah)")
    if not df.empty and 'rng_df' in st.session_state:
        df.columns = df.columns.str.strip().str.lower()
        exclude_cols = ["id", "bulan", "tahun"]
        daerah_cols = [col for col in df.columns if col not in exclude_cols]
        rng_df = st.session_state['rng_df']

        results_summary = []
        st.subheader("📌 Hasil Simulasi per Wilayah")
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

            freq_series = pd.cut(data, bins=cut_bins, labels=labels, include_lowest=True, right=True)
            freq_table = freq_series.value_counts().sort_index().reset_index()
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

            def get_simulated_value(rand, freq_table):
                for _, row in freq_table.iterrows():
                    low, high = map(int, row["Interval Angka Acak"].split(' - '))
                    if low <= rand <= high:
                        return row["Titik Tengah"]
                return freq_table["Titik Tengah"].iloc[-1]

            sim_results = []
            for _, row in rng_df.iterrows():
                val = get_simulated_value(row["Angka Acak (Uᵢ×100)"], freq_table)
                sim_results.append(val)

            sim_df = pd.DataFrame({"Bulan": range(1, len(sim_results)+1), "Jumlah Pengunjung": sim_results})
            total_sim = int(round(sim_df['Jumlah Pengunjung'].sum()))
            avg_sim = int(round(sim_df['Jumlah Pengunjung'].mean()))

            results_summary.append({"Wilayah": daerah, "Total": total_sim, "Rata-rata": avg_sim})

            st.write(f"### 📍 Wilayah: {daerah.capitalize()}")
            st.dataframe(sim_df, use_container_width=True)

            fig_line = px.line(sim_df, x="Bulan", y="Jumlah Pengunjung", markers=True,
                                title=f"Grafik Tren - {daerah.capitalize()}")
            st.plotly_chart(fig_line, use_container_width=True)

        st.markdown("---")
        st.subheader("📌 Ringkasan Semua Wilayah")
        summary_df = pd.DataFrame(results_summary)
        st.dataframe(summary_df, use_container_width=True)

        # Bar Chart
        fig_bar = px.bar(summary_df, x="Wilayah", y="Total", text="Total", color="Wilayah",
                         title="Total Simulasi Pengunjung per Wilayah")
        fig_bar.update_traces(textposition="outside")
        st.plotly_chart(fig_bar, use_container_width=True)

        # Pie Chart
        fig_pie = px.pie(summary_df, names="Wilayah", values="Total", title="Proporsi Kunjungan per Wilayah",
                         color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig_pie, use_container_width=True)

        # Kesimpulan
        st.markdown("### ✅ Kesimpulan:")
        st.write(f"- **Wilayah tertinggi:** {summary_df.loc[summary_df['Total'].idxmax(), 'Wilayah'].capitalize()} dengan {summary_df['Total'].max():,} pengunjung.")
        st.write(f"- **Wilayah terendah:** {summary_df.loc[summary_df['Total'].idxmin(), 'Wilayah'].capitalize()} dengan {summary_df['Total'].min():,} pengunjung.")
        st.write(f"- **Rata-rata seluruh wilayah:** {int(summary_df['Rata-rata'].mean()):,} pengunjung.")
