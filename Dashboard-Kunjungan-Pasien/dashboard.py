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

# ==========================================
# 🟨 HALAMAN: RNG LCG
# ==========================================
elif menu_pilihan == "🔢 RNG LCG":
    st.title("🔢 RNG - Linear Congruential Generator (LCG)")

    # Inisialisasi state
    if "rng_data" not in st.session_state:
        st.session_state.rng_data = None
    if "show_rng" not in st.session_state:
        st.session_state.show_rng = False

    st.markdown("Masukkan parameter untuk metode LCG:")
    col1, col2 = st.columns(2)
    with col1:
        m = st.number_input("Modulus (m)", min_value=1, value=100, step=1)
        a = st.number_input("Multiplier (a)", min_value=1, value=5, step=1)
    with col2:
        c = st.number_input("Increment (c)", min_value=0, value=1, step=1)
        z0 = st.number_input("Seed (z₀)", min_value=0, value=1, step=1)
    
    n_gen = st.number_input("Jumlah Bilangan Acak", min_value=1, value=10, step=1)

    # Tombol sejajar
    col_gen, col_reset = st.columns([1, 1])
    with col_gen:
        generate_clicked = st.button("🎲 Generate", use_container_width=True)
    with col_reset:
        reset_clicked = st.button("❌ Reset", use_container_width=True)

    # Generate bilangan acak
    if generate_clicked:
        zi = z0
        rng_data = []
        all_zi = []  # Untuk menyimpan semua nilai zi untuk pengecekan duplikat

        for i in range(1, n_gen + 1):
            zi_minus_1 = zi  # Simpan nilai sebelumnya untuk ditampilkan
            zi_next = (a * zi_minus_1 + c) % m  # Hitung zi baru
            ui = zi_next / m
            angka_acak = int(ui * 100)  # Hitung ui * 100 sebagai integer
            
            # Simpan data
            rng_data.append((i, zi_minus_1, zi_next, ui, angka_acak))
            all_zi.append(zi_next)
            zi = zi_next  # Set zi untuk iterasi berikutnya

        rng_df = pd.DataFrame(rng_data, columns=["i", "Zᵢ₋₁", "Zᵢ", "Uᵢ", "Uᵢ*100"])
        st.session_state.rng_data = rng_df
        st.session_state.show_rng = True
        st.session_state.all_zi = all_zi  # Simpan untuk pengecekan duplikat
        st.success("Bilangan acak berhasil dibuat!")

    # Reset
    if reset_clicked:
        st.session_state.rng_data = None
        st.session_state.show_rng = False
        st.session_state.all_zi = []
        st.info("RNG berhasil direset.")

    # Tampilkan hasil jika tersedia
    if st.session_state.show_rng and st.session_state.rng_data is not None:
        st.markdown("### 📋 Hasil LCG")
        st.dataframe(st.session_state.rng_data, use_container_width=True, hide_index=True)
        
        # Cek duplikat
        st.markdown("### 🔍 Pengecekan Duplikat Zᵢ")
        all_zi = st.session_state.all_zi
        duplicates = {}
        
        # Temukan nilai yang duplikat
        for i, zi_val in enumerate(all_zi, 1):
            if zi_val in duplicates:
                duplicates[zi_val].append(i)
            else:
                duplicates[zi_val] = [i]
        
        # Filter hanya yang memiliki duplikat
        real_duplicates = {k: v for k, v in duplicates.items() if len(v) > 1}
        
        if real_duplicates:
            st.warning("⚠️ Ditemukan duplikat nilai Zᵢ:")
            for zi_val, indices in real_duplicates.items():
                st.write(f"Nilai Zᵢ = {zi_val} muncul pada iterasi ke: {', '.join(map(str, indices))}")
        else:
            st.success("✅ Tidak ditemukan nilai Zᵢ yang duplikat")

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

                # ✅ SIMULASI TANPA DUPLIKAT ANGKA ACAK
                used_randoms = set()
                sim_results = []

                for _, row in rng_df.iterrows():
                    rand = row["Uᵢ"]
                    angka_acak = int(rand * 100)
                    if angka_acak == 0:
                        angka_acak = 1

                    if angka_acak in used_randoms:
                        continue  # Lewati duplikat
                    used_randoms.add(angka_acak)

                    jumlah_pengunjung = None
                    for _, freq_row in freq_table.iterrows():
                        low, high = map(int, freq_row["Interval Angka Acak"].split(' - '))
                        if low <= angka_acak <= high:
                            jumlah_low, jumlah_high = map(int, freq_row["Interval Jumlah"].split(' - '))
                            jumlah_pengunjung = random.randint(jumlah_low, jumlah_high)
                            break

                    if jumlah_pengunjung is not None:
                        sim_results.append({
                            "Percobaan": row["i"],
                            "Bilangan Acak": angka_acak,
                            "Jumlah Pengunjung": jumlah_pengunjung
                        })

                st.subheader("Hasil Simulasi")
                sim_df = pd.DataFrame(sim_results)
                st.dataframe(sim_df, use_container_width=True, hide_index=True)

                st.markdown(f"**Total Simulasi:** {sim_df['Jumlah Pengunjung'].sum()}")
                st.markdown(f"**Rata-rata:** {sim_df['Jumlah Pengunjung'].mean():.2f}")
