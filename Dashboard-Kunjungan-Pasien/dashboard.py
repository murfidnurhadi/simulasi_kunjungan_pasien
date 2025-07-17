import streamlit as st
import pandas as pd
import numpy as np
import math
import os

# -----------------------
# Konfigurasi Halaman
# -----------------------
st.set_page_config(page_title="Dashboard Simulasi Monte Carlo", layout="wide", page_icon="📄")
st.title("📊 Dashboard Simulasi Monte Carlo & Distribusi")

# -----------------------
# Sidebar Navigasi
# -----------------------
st.sidebar.header("📌 Menu Navigasi")
menu = st.sidebar.radio("Pilih Halaman:", [
    "Data Kunjungan Pasien",
    "Data Train",
    "Frekuensi dan Interval",
    "RNG LCG",
    "Simulasi"
])

# -----------------------
# Path Excel File
# -----------------------
excel_path = r"D:\Tubes_Mosi\Tubes_Mosi.xlsx"

@st.cache_data
def load_excel_sheet(sheet_name):
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        return df
    except FileNotFoundError:
        st.error(f"❌ File tidak ditemukan: {excel_path}")
        return pd.DataFrame()
    except ValueError as e:
        st.error(f"❌ Gagal membaca sheet: {e}")
        return pd.DataFrame()

# -----------------------
# Fungsi tampilkan tabel
# -----------------------
def tampilkan_tabel(df):
    if not df.empty:
        df_display = df.reset_index(drop=True)
        df_display.index = [''] * len(df_display)
        st.dataframe(df_display, use_container_width=True)
    else:
        st.warning("⚠ Data tidak tersedia.")

# ========================
# ✅ HALAMAN: DATA KUNJUNGAN PASIEN
# ========================
if menu == "Data Kunjungan Pasien":
    st.subheader("📋 Data Kunjungan Pasien")
    # Contoh CSV masih pakai cara lama
    csv_path = os.path.join(os.getcwd(), "Data_Kunjungan_Pasien.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        tampilkan_tabel(df)
        st.download_button(
            label="📥 Download Data",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="Data_Kunjungan_Pasien.csv",
            mime="text/csv"
        )
    else:
        st.warning("File CSV tidak ditemukan.")

# ========================
# ✅ HALAMAN: DATA TRAIN
# ========================
elif menu == "Data Train":
    st.subheader("📊 Data Train Pengunjung")
    df = load_excel_sheet("DataTrain")
    tampilkan_tabel(df)

# ========================
# ✅ HALAMAN: FREKUENSI DAN INTERVAL
# ========================
elif menu == "Frekuensi dan Interval":
    st.subheader("📈 Distribusi Frekuensi dan Interval")
    df = load_excel_sheet("DataTrain")

    if not df.empty:
        # Normalisasi nama kolom
        df.columns = df.columns.str.strip().str.lower()

        # Kolom yang tidak termasuk daerah
        exclude_cols = ["id", "bulan", "tahun"]
        daerah_cols = [col for col in df.columns if col not in exclude_cols]

        daerah_options = ["Pilih daerah"] + daerah_cols
        selected_daerah = st.selectbox("📍 Pilih Daerah:", daerah_options)

        if selected_daerah == "Pilih daerah":
            st.info("Pilih daerah terlebih dahulu untuk menampilkan distribusi.")
        else:
            st.write(f"**Distribusi untuk:** {selected_daerah.upper()}")

            data = df[selected_daerah].dropna()
            n = len(data)
            x_min, x_max = data.min(), data.max()
            R = x_max - x_min
            k = math.ceil(1 + 3.3 * math.log10(n))
            h = math.ceil(R / k)

            # Interval
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
            freq_table.columns = ["Interval", "Frekuensi"]
            freq_table = freq_table[freq_table["Frekuensi"] > 0].reset_index(drop=True)

            # Tambah kolom probabilitas
            freq_table["No"] = range(1, len(freq_table) + 1)
            total = freq_table["Frekuensi"].sum()
            prob_raw = freq_table["Frekuensi"] / total
            prob_rounded = prob_raw.round(2)

            # Koreksi selisih probabilitas
            selisih = 1.00 - prob_rounded.sum()
            if abs(selisih) > 0:
                idx_max = prob_rounded.idxmax()
                prob_rounded.iloc[idx_max] += selisih
                prob_rounded = prob_rounded.round(2)

            freq_table["Probabilitas"] = prob_rounded
            freq_table["Prob. Kumulatif"] = freq_table["Probabilitas"].cumsum().round(2)

            # Interval angka acak
            upper_bounds = (freq_table["Prob. Kumulatif"] * 100).astype(int)
            lower_bounds = [1] + [ub + 1 for ub in upper_bounds[:-1]]
            freq_table["Interval Angka Acak"] = [
                f"{lb} - {ub}" for lb, ub in zip(lower_bounds, upper_bounds)
            ]

            # Tampilkan tabel
            freq_table = freq_table[[
                "No", "Interval", "Frekuensi", "Probabilitas", "Prob. Kumulatif", "Interval Angka Acak"
            ]]
            st.dataframe(freq_table, use_container_width=True)

            # Info tambahan
            st.markdown("---")
            st.markdown(f"Jumlah Data (n): {n}")
            st.markdown(f"x_min: {x_min}")
            st.markdown(f"x_max: {x_max}")
            st.markdown(f"Jangkauan (R): {R}")
            st.markdown(f"Jumlah Kelas (k): {k}")
            st.markdown(f"Panjang Kelas (h): {h}")
    else:
        st.warning("Data tidak tersedia.")

# ========================
# ✅ HALAMAN: RNG & SIMULASI
# ========================
elif menu == "RNG LCG":
    st.subheader("🎲 Data RNG LCG")
    df = load_excel_sheet("RNG_LCG")
    tampilkan_tabel(df)

elif menu == "Simulasi":
    st.subheader("🧮 Data Simulasi Monte Carlo")
    df = load_excel_sheet("Simulasi")
    tampilkan_tabel(df)