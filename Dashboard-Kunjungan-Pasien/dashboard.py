import streamlit as st
import pandas as pd
import numpy as np
import math
import random
import plotly.express as px
import os

# =========================
# Konfigurasi Halaman
# =========================
st.set_page_config(layout="wide", page_title="Simulasi Monte Carlo", page_icon="🎲")

# =========================
# Sidebar Navigasi
# =========================
with st.sidebar:
    st.markdown("## 🧭 Navigasi")
    menu = st.radio(
        "📂 Pilih Halaman:",
        options=[
            "🏠 Dashboard",
            "📊 Data Train",
            "📈 Frekuensi dan Interval",
            "🔢 RNG LCG",
            "🎲 Simulasi"
        ]
    )
    st.markdown("---")
    st.info("Pilih halaman dari menu di atas untuk menampilkan data atau menjalankan simulasi.")

# =========================
# 📂 Load Data
# ========================
@st.cache_data
def load_data(path):
    try:
        return pd.read_excel(path)
    except:
        st.error("❌ Gagal memuat file Excel. Pastikan file 'dataset.xlsx' ada di folder dataset.")
        return pd.DataFrame()

excel_path = os.path.join("dataset", "dataset.xlsx")
df = load_data(excel_path)

# =========================
# 🏠 Dashboard
# =========================
if menu == "🏠 Dashboard":
    st.title("📊 Dashboard Simulasi Monte Carlo")
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

        fig = px.bar(total_per_wilayah.reset_index(), x="index", y=0, color="index",
                     title="Total Pengunjung per Wilayah",
                     labels={"index": "Wilayah", 0: "Total Pengunjung"},
                     color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Upload file dataset.xlsx untuk menampilkan dashboard.")

# =========================
# 📊 Data Train
# =========================
elif menu == "📊 Data Train":
    st.title("📋 Data Train Pengunjung")
    if not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)
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

            # ✅ Titik Tengah
            bounds = freq_table["Interval Jumlah"].str.split(" - ", expand=True).astype(int)
            freq_table["Titik Tengah"] = ((bounds[0] + bounds[1]) / 2).round(0).astype(int)

            # ✅ Probabilitas & kumulatif
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

            st.dataframe(freq_table, use_container_width=True, hide_index=True)
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

# =========================
# 🔢 RNG LCG
# =========================
elif menu == "🔢 RNG LCG":
    st.title("🔢 RNG - Linear Congruential Generator")
    if "rng_data" not in st.session_state:
        st.session_state.rng_data = None

    col1, col2 = st.columns(2)
    with col1:
        m = st.number_input("Modulus (m)", 1, value=100)
        a = st.number_input("Multiplier (a)", 1, value=5)
    with col2:
        c = st.number_input("Increment (c)", 0, value=1)
        z0 = st.number_input("Seed (z₀)", 0, value=1)
    n_gen = st.number_input("Jumlah Bilangan Acak", 1, value=10)

    col_gen, col_reset = st.columns(2)
    with col_gen:
        gen_btn = st.button("🎲 Generate", use_container_width=True)
    with col_reset:
        reset_btn = st.button("❌ Reset", use_container_width=True)

    if gen_btn:
        zi = z0
        rng = []
        for i in range(1, n_gen + 1):
            zi_next = (a * zi + c) % m
            ui = round(zi_next / m, 4)
            rng.append((i, zi, zi_next, ui))
            zi = zi_next
        st.session_state.rng_data = pd.DataFrame(rng, columns=["i", "Zᵢ₋₁", "Zᵢ", "Uᵢ"])
        st.success("Bilangan acak berhasil dibuat!")

    if reset_btn:
        st.session_state.rng_data = None
        st.info("Data RNG direset.")

    if st.session_state.rng_data is not None:
        st.dataframe(st.session_state.rng_data, use_container_width=True, hide_index=True)

# =========================
# 🎲 Simulasi Monte Carlo
# =========================
elif menu == "🎲 Simulasi":
    st.title("🎲 Simulasi Monte Carlo")
    if st.session_state.rng_data is None:
        st.warning("Generate bilangan acak di halaman RNG LCG terlebih dahulu.")
        st.stop()
    if df.empty:
        st.warning("Data tidak tersedia.")
        st.stop()

    df.columns = df.columns.str.strip().str.lower()
    exclude_cols = ["id", "bulan", "tahun"]
    daerah_cols = [col for col in df.columns if col not in exclude_cols]
    daerah = st.selectbox("Pilih Daerah:", ["Pilih"] + daerah_cols)
    if daerah == "Pilih":
        st.stop()

    data = df[daerah].dropna()
    n = len(data)
    R = data.max() - data.min()
    k = math.ceil(1 + 3.3 * math.log10(n))
    h = math.ceil(R / k)

    lower = math.floor(data.min())
    bins = []
    for _ in range(k):
        upper = lower + h
        bins.append((lower, upper))
        lower = upper + 1

    labels = [f"{low}-{high}" for low, high in bins]
    cut_bins = [b[0] for b in bins] + [bins[-1][1]]
    kelas = pd.cut(data, bins=cut_bins, labels=labels, include_lowest=True)
    freq = kelas.value_counts().sort_index().reset_index()
    freq.columns = ["Interval", "Frekuensi"]
    total = freq["Frekuensi"].sum()
    prob = freq["Frekuensi"] / total
    prob_kum = prob.cumsum().round(2)
    pk = (prob_kum * 100).astype(int)
    low_bounds = [1] + [x + 1 for x in pk[:-1]]
    interval_acak = [f"{l}-{u}" for l, u in zip(low_bounds, pk)]
    freq["Interval Angka Acak"] = interval_acak

    rng_df = st.session_state.rng_data
    sim_result = []
    for _, row in rng_df.iterrows():
        acak = int(row["Uᵢ"] * 100)
        if acak == 0:
            acak = 1
        for idx, r in freq.iterrows():
            low, high = map(int, r["Interval Angka Acak"].split("-"))
            if low <= acak <= high:
                low_val, high_val = map(int, r["Interval"].split("-"))
                sim_result.append((row["i"], acak, random.randint(low_val, high_val)))
                break

    hasil_df = pd.DataFrame(sim_result, columns=["Bulan Ke-", "Bilangan Acak", "Jumlah Pengunjung"])
    st.subheader("📋 Hasil Simulasi")
    st.dataframe(hasil_df, use_container_width=True, hide_index=True)
    st.markdown(f"**Total:** {hasil_df['Jumlah Pengunjung'].sum():,}")
    st.markdown(f"**Rata-rata:** {hasil_df['Jumlah Pengunjung'].mean():.2f}")
