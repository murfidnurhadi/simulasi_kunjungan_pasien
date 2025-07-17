import streamlit as st
import pandas as pd
import numpy as np
import math
import os
import plotly.express as px

# ========================
# 🎨 Konfigurasi Halaman
# ========================
st.set_page_config(
    page_title="Simulasi Monte Carlo - Kelompok 6",
    page_icon="🎲",
    layout="wide"
)

# CSS Custom untuk UI lebih menarik
st.markdown("""
    <style>
    .main { background-color: #f9f9f9; }
    .block-container { padding-top: 1rem; }
    h1, h2, h3, h4 { color: #333333; }
    .sidebar .sidebar-content { background-color: #f0f2f6; }
    .stButton>button {
        background-color: #4CAF50; color: white; border-radius: 8px; padding: 10px;
    }
    .stButton>button:hover { background-color: #45a049; }
    .stSelectbox, .stNumberInput { font-size: 16px; }
    </style>
""", unsafe_allow_html=True)

# ========================
# Sidebar Menu
# ========================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/a/a7/UNIKOM.png", use_container_width=True)
st.sidebar.title("📂 Menu Navigasi")
menu = st.sidebar.radio("Pilih Halaman:", [
    "🏠 Dashboard", "📋 Data Train", "📈 Frekuensi & Interval", "🔢 RNG LCG", "🎲 Simulasi Monte Carlo"
])

# ========================
# Path File Dataset
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

df = load_excel()

# ========================
# Fungsi bantu
# ========================
def hitung_frekuensi(data):
    n = len(data)
    x_min, x_max = int(data.min()), int(data.max())
    R = x_max - x_min
    k = math.ceil(1 + 3.3 * math.log10(n))
    h = math.ceil(R / k)

    bins = range(x_min, x_max + h, h)
    labels = [f"{bins[i]} - {bins[i+1]-1}" for i in range(len(bins)-1)]
    
    if pd.cut([x_max], bins=bins, right=False).isnull().all() and len(labels) > 0:
        last_low = int(labels[-1].split(' - ')[0])
        labels[-1] = f"{last_low} - {x_max}"

    freq_table = pd.cut(data, bins=bins, labels=labels, right=False, include_lowest=True).value_counts().sort_index().reset_index()
    freq_table.columns = ["Interval Jumlah", "Frekuensi"]
    
    freq_table["No"] = range(1, len(freq_table) + 1)
    total = freq_table["Frekuensi"].sum()
    prob_raw = freq_table["Frekuensi"] / total
    prob_rounded = prob_raw.round(2)
    selisih = 1.00 - prob_rounded.sum()
    if selisih != 0:
        diff_indices = (prob_raw - prob_rounded).nlargest(int(abs(selisih) * 100)).index
        prob_rounded.loc[diff_indices] += 0.01 * np.sign(selisih)

    freq_table["Probabilitas"] = prob_rounded
    freq_table["Prob. Kumulatif"] = freq_table["Probabilitas"].cumsum().round(2)

    upper_bounds = (freq_table["Prob. Kumulatif"] * 100).astype(int)
    lower_bounds = [1] + [ub + 1 for ub in upper_bounds[:-1]]
    freq_table["Interval Angka Acak"] = [f"{lb}-{ub}" for lb, ub in zip(lower_bounds, upper_bounds)]
    
    return freq_table, n, R, k, h

# ========================
# 🟥 Dashboard
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
        col1.metric("Total Pengunjung", f"{total_seluruh}")
        col2.metric("Wilayah Terbanyak", total_per_wilayah.idxmax(), f"{total_per_wilayah.max()}")
        col3.metric("Wilayah Tersedikit", total_per_wilayah.idxmin(), f"{total_per_wilayah.min()}")

        fig = px.bar(total_per_wilayah, x=total_per_wilayah.index, y=total_per_wilayah.values,
                     title="Total Pengunjung per Wilayah", text=total_per_wilayah.values,
                     color=total_per_wilayah.values, color_continuous_scale="Blues")
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Upload file Excel (.xlsx) untuk melihat data.")

# ========================
# 🟦 Data Train
# ========================
elif menu == "📋 Data Train":
    st.title("📋 Data Train Pengunjung")
    if not df.empty:
        st.dataframe(df.reset_index(drop=True), use_container_width=True)
    else:
        st.warning("Data tidak tersedia.")

# ========================
# 🟩 Frekuensi & Interval
# ========================
elif menu == "📈 Frekuensi & Interval":
    st.title("📈 Distribusi Frekuensi")
    if not df.empty:
        df.columns = df.columns.str.strip().str.lower()
        exclude_cols = ["id", "bulan", "tahun"]
        daerah_cols = [col for col in df.columns if col not in exclude_cols]

        selected_daerah = st.selectbox("Pilih Daerah:", ["pilih daerah"] + daerah_cols)
        if selected_daerah != "pilih daerah":
            freq_table, n, R, k, h = hitung_frekuensi(df[selected_daerah].dropna())
            st.dataframe(freq_table, use_container_width=True)
            st.markdown(f"**Jumlah Data (n):** {n} | **R:** {R} | **k:** {k} | **h:** {h}")
    else:
        st.warning("Data tidak tersedia.")

# ========================
# 🟨 RNG LCG
# ========================
elif menu == "🔢 RNG LCG":
    st.title("🔢 Linear Congruential Generator (LCG)")
    m = st.number_input("Modulus (m)", min_value=1, value=100, step=1)
    a = st.number_input("Multiplier (a)", min_value=1, value=5, step=1)
    c = st.number_input("Increment (c)", min_value=0, value=1, step=1)
    x0 = st.number_input("Seed (x0)", min_value=0, value=1, step=1)
    n_gen = st.number_input("Jumlah Bilangan Acak", min_value=1, value=10, step=1)

    if st.button("Generate"):
        rng_values = []
        xi = x0
        for _ in range(n_gen):
            xi = (a * xi + c) % m
            rng_values.append(xi)

        rng_df = pd.DataFrame({
            "i": range(1, n_gen + 1),
            "X_i": rng_values,
            "R_i = X_i/m": [round(val / m, 4) for val in rng_values]
        })
        st.session_state['rng_df'] = rng_df
        st.dataframe(rng_df, use_container_width=True)
        st.success("Bilangan acak berhasil dibuat!")

# ========================
# 🟧 Simulasi Monte Carlo
# ========================
elif menu == "🎲 Simulasi Monte Carlo":
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
            selected_daerah = st.selectbox("Pilih Daerah untuk Simulasi:", ["pilih daerah"] + daerah_cols)

            if selected_daerah != "pilih daerah":
                freq_table, _, _, _, _ = hitung_frekuensi(df[selected_daerah].dropna())

                def get_interval(rand, freq_table):
                    angka_acak = int(rand * 100)
                    if angka_acak == 0: angka_acak = 1
                    for _, row in freq_table.iterrows():
                        low, high = map(int, row["Interval Angka Acak"].split('-'))
                        if low <= angka_acak <= high:
                            return row["Interval Jumlah"]
                    return "-"

                sim_results = []
                for _, row in rng_df.iterrows():
                    sim_results.append({
                        "Percobaan": row["i"],
                        "Bilangan Acak": row["R_i = X_i/m"],
                        "Jumlah Pengunjung (Simulasi)": get_interval(row["R_i = X_i/m"], freq_table)
                    })

                st.subheader("Hasil Simulasi")
                st.dataframe(pd.DataFrame(sim_results), use_container_width=True)
