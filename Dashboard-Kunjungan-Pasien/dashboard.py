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
excel_path = os.path.join(BASE_DIR, "dataset", "dataset.xlsx")

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
            st.title("📊 Distribusi Frekuensi: Kota " + selected_daerah.capitalize())  # dipindahkan ke sini

            data = df[selected_daerah].dropna()
            n = len(data)

            # Hitung nilai dasar
            x_min, x_max = data.min(), data.max()
            R = x_max - x_min
            k = math.ceil(1 + 3.3 * math.log10(n))
            h = int(R / k)

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
            freq_table["Titik Tengah"] = ((bounds[0] + bounds[1]) / 2).astype(int)

            # Probabilitas dan kumulatif
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

            # Tambahkan nomor
            freq_table.insert(0, "No", range(1, len(freq_table) + 1))

            # Tampilkan tabel
            st.dataframe(freq_table, use_container_width=True, hide_index=True)

            # Informasi tambahan
            st.markdown("---")
            st.markdown("### ℹ️ Informasi Tambahan")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Terkecil (Xmin)", f"{x_min}")
            col2.metric("Terbesar (Xmax)", f"{x_max}")
            col3.metric("Jangkauan (R)", f"{R}")
            col4.metric("Jumlah Kelas (k)", f"{k}")
            col5.metric("Panjang Kelas (h)", f"{h}")
            col6.metric("Jumlah Data (n)", f"{n}")
    else:
        st.warning("Data tidak tersedia.")

import streamlit as st
import pandas as pd

# ========================
# 🔢 RNG LCG Generator
# ========================
elif menu == "🔢 RNG LCG":
    st.title("🔢 Linear Congruential Generator (LCG)")

    # Input Parameter
    m = st.number_input("Modulus (m)", min_value=2, value=10140000)
    a = st.number_input("Multiplier (a)", min_value=1, value=21)
    c = st.number_input("Increment (c)", min_value=0, value=7)
    x0 = st.number_input("Seed (Z₀)", min_value=0, value=10123020)
    n_gen = st.number_input("Jumlah Bilangan Acak", min_value=1, value=48)
    presisi = st.number_input("Presisi Uᵢ", min_value=4, max_value=12, value=6)

    if st.button("Generate"):
        zi_values = []
        ui_values = []

        zi = x0
        seen_zi = set()
        attempts = 0
        max_attempts = m * 2  # Hindari infinite loop

        while len(zi_values) < n_gen and attempts < max_attempts:
            zi = (a * zi + c) % m
            attempts += 1
            if zi not in seen_zi:  # Pastikan Zi unik
                seen_zi.add(zi)
                zi_values.append(zi)
                ui_values.append(round(zi / m, presisi))

        # Buat DataFrame
        rng_df = pd.DataFrame({
            "i": range(1, len(zi_values) + 1),
            "Zi": zi_values,
            "Ui": ui_values
        })

        st.session_state['rng_df'] = rng_df
        st.subheader("Hasil Bilangan Acak:")
        st.dataframe(rng_df, use_container_width=True, hide_index=True)

        # Info tambahan
        if len(zi_values) < n_gen:
            st.warning(f"⚠ Hanya {len(zi_values)} bilangan unik yang dihasilkan (karena modulus terbatas).")

        # Tombol Download Excel
        excel_data = rng_df.to_excel(index=False)
        st.download_button(
            label="📥 Download Hasil ke Excel",
            data=excel_data,
            file_name="hasil_rng_lcg.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

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
        st.dataframe(rng_df, use_container_width=True, hide_index=True)

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
                st.dataframe(freq_table, use_container_width=True, hide_index=True)

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
                st.dataframe(sim_df, use_container_width=True, hide_index=True)

                st.markdown(f"**Total Simulasi:** {sim_df['Jumlah Pengunjung'].sum()}")
                st.markdown(f"**Rata-rata:** {sim_df['Jumlah Pengunjung'].mean():.2f}")
