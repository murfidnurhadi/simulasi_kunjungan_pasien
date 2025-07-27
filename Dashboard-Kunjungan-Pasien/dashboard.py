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
        ["🏠 Dashboard", "📊 Data Train", "📈 Frekuensi dan Interval", "🎲 Simulasi"]
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
# 🔢 RNG LCG Otomatis
# ========================
def generate_rng_auto():
    a = random.randint(10, 50)      # multiplier
    c = random.randint(1, 30)       # increment
    m = random.randint(80, 200)     # modulus
    z0 = random.randint(1, 50)      # seed
    n_gen = 48                      # jumlah bilangan

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

# Simpan RNG otomatis sekali
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
# 📈 Frekuensi & Interval
# ========================
elif menu == "📈 Frekuensi dan Interval":
    st.title("📈 Distribusi Frekuensi dan Interval")
    st.info("Pilih salah satu daerah untuk melihat distribusi frekuensi.")
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

            # ✅ Tambah Titik Tengah
            bounds = freq_table["Interval Jumlah"].str.split(" - ", expand=True).astype(int)
            freq_table["Titik Tengah"] = ((bounds[0] + bounds[1]) / 2).round(0).astype(int)

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

            freq_table["Interval Angka Acak"] = [
                f"{lb} - {ub}" for lb, ub in zip([1] + [x+1 for x in freq_table["P.K * 100"][:-1]], freq_table["P.K * 100"])
            ]

            st.dataframe(freq_table, use_container_width=True)
    else:
        st.warning("Data tidak tersedia.")

# ========================
# 🎲 Simulasi Monte Carlo (All Wilayah)
# ========================
elif menu == "🎲 Simulasi":
    st.title("🎲 Simulasi Monte Carlo - Semua Wilayah")
    rng_df = st.session_state['rng_df']
    if not df.empty:
        df.columns = df.columns.str.strip().str.lower()
        exclude_cols = ["id", "bulan", "tahun"]
        daerah_cols = [col for col in df.columns if col not in exclude_cols]

        hasil_simulasi = {}
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
                prob_rounded.iloc[prob_rounded.idxmax()] += selisih

            freq_table["Probabilitas"] = prob_rounded
            freq_table["Prob. Kumulatif"] = freq_table["Probabilitas"].cumsum().round(2)
            freq_table["P.K * 100"] = (freq_table["Prob. Kumulatif"] * 100).astype(int)
            freq_table["Interval Angka Acak"] = [
                f"{lb} - {ub}" for lb, ub in zip([1] + [x+1 for x in freq_table["P.K * 100"][:-1]], freq_table["P.K * 100"])
            ]

            def get_simulated_value(rand):
                angka_acak = int(rand * 100)
                if angka_acak == 0: angka_acak = 1
                for _, row in freq_table.iterrows():
                    low, high = map(int, row["Interval Angka Acak"].split(' - '))
                    if low <= angka_acak <= high:
                        return row["Titik Tengah"]
                return 0

            sim_results = [get_simulated_value(row["Uᵢ"]) for _, row in rng_df.iterrows()]
            hasil_simulasi[daerah] = sim_results

        sim_df = pd.DataFrame(hasil_simulasi)
        sim_df.index = rng_df["i"]
        st.subheader("📊 Hasil Simulasi Semua Wilayah")
        st.dataframe(sim_df, use_container_width=True)

        # Ringkasan rata-rata per daerah
        avg_per_daerah = sim_df.mean().sort_values(ascending=False)
        total_per_daerah = sim_df.sum().sort_values(ascending=False)

        st.subheader("📊 Visualisasi")
        col1, col2 = st.columns(2)
        with col1:
            fig_bar = px.bar(x=avg_per_daerah.index, y=avg_per_daerah.values, color=avg_per_daerah.index,
                             text=avg_per_daerah.values.round(0),
                             title="Rata-rata Simulasi Pengunjung per Wilayah")
            st.plotly_chart(fig_bar, use_container_width=True)
        with col2:
            fig_pie = px.pie(names=total_per_daerah.index, values=total_per_daerah.values,
                             title="Distribusi Total Pengunjung per Wilayah")
            st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("📈 Tren Simulasi (Line Chart)")
        fig_line = px.line(sim_df, markers=True,
                           title="Tren Simulasi Monte Carlo Semua Wilayah")
        fig_line.update_traces(line=dict(width=2))
        st.plotly_chart(fig_line, use_container_width=True)

        # ========================
        # 🔍 Kesimpulan Rapi
        # ========================
        st.subheader("📌 Kesimpulan Analisis")
        st.markdown(f"""
        **1. Rata-rata kunjungan tertinggi:** `{avg_per_daerah.idxmax()}` dengan **{avg_per_daerah.max():,.0f}**.
        **2. Rata-rata kunjungan terendah:** `{avg_per_daerah.idxmin()}` dengan **{avg_per_daerah.min():,.0f}**.
        **3. Total kunjungan simulasi:** {int(total_per_daerah.sum()):,}.
        **4. Wilayah dengan porsi terbesar:** {total_per_daerah.idxmax()} ({round((total_per_daerah.max()/total_per_daerah.sum())*100,2)}% dari total).
        **5. Rekomendasi:**
           - Fokus pada persediaan untuk wilayah `{avg_per_daerah.idxmax()}`.
           - Jika variasi antar bulan tinggi (fluktuasi besar), siapkan kapasitas tambahan.
        """.replace(",", "."))
    else:
        st.warning("Data tidak tersedia.")
        
