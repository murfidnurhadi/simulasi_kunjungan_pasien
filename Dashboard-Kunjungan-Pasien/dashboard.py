import streamlit as st
import pandas as pd
import gdown
import os
import plotly.express as px

# -----------------------
# Konfigurasi Halaman
# -----------------------
st.set_page_config(page_title="Dashboard Simulasi Monte Carlo", layout="wide")
st.title("📊 Dashboard Simulasi Monte Carlo")

# -----------------------
# Sidebar Navigasi
# -----------------------
st.sidebar.header("📌 Menu Navigasi")
menu = st.sidebar.selectbox("Pilih Dataset:", [
    "Data Kunjungan Pasien",
    "Data Train",
    "Kota Cirebon",
    "Kab Cirebon",
    "Kuningan",
    "Indramayu",
    "Majalengka",
    "Lain2",
    "RNG LCG",
    "Simulasi"
])

source_option = st.sidebar.radio("Ambil data dari:", ["Otomatis (Lokal/Drive)", "Upload Manual"])

# -----------------------
# Data Sources Lengkap
# -----------------------
data_sources = {
    "Data Kunjungan Pasien": [
        {"file": "Data_Kunjungan_Pasien.csv", "gdrive": "https://drive.google.com/file/d/1GUHcM1xVSjU4aEIuH2QmRYFGYo0cCDEH/view?usp=sharing"}
    ],
    "Data Train": [
        {"file": "DataTrain.csv", "gdrive": "https://drive.google.com/file/d/11tgNQ2GqMSIM97DH2U9HNTbIh4CceuiF/view?usp=sharing"}
    ],
    "Kota Cirebon": [
        {"file": "DataTrainKotaCirebon.csv", "gdrive": "https://drive.google.com/file/d/1ptLOuscYMjzGw_v6qOI_Gvmc6t-kW9OS/view?usp=sharing"},
        {"file": "DataIntervalKotaCirebon.csv", "gdrive": "https://drive.google.com/file/d/1jbyntLDGUegAL-9n-uJWwVsH-V1UrxDp/view?usp=sharing"}
    ],
    "Kab Cirebon": [
        {"file": "DataTrainKabCirebon.csv", "gdrive": "https://drive.google.com/file/d/12-5l_9EFB_VRARq6CIOMTlaWH2VOx9wO/view?usp=sharing"},
        {"file": "DataIntervalKabCirebon.csv", "gdrive": "https://drive.google.com/file/d/1uM_5BcU1Zbl998W4yE5XsDcfiyj_itdC/view?usp=sharing"}
    ],
    "Kuningan": [
        {"file": "DataTrainKuningan.csv", "gdrive": "https://drive.google.com/file/d/113p_LARFjkQthew9S3t0dXdBq4sqME9H/view?usp=sharing"},
        {"file": "DataIntervalKuningan.csv", "gdrive": "https://drive.google.com/file/d/1yYhh5m3YIkLaCNP6hERswTAQ8pv6kyfW/view?usp=sharing"}
    ],
    "Indramayu": [
        {"file": "DataTrainIndramayu.csv", "gdrive": "https://drive.google.com/file/d/118Hl_6dvhYUgeE6tQG-aG_Amq1OneWlL/view?usp=sharing"},
        {"file": "DataIntervalIndramayu.csv", "gdrive": "https://drive.google.com/file/d/1n94Wtw5RYS1zwABz0xosM4N1JMob1rXW/view?usp=sharing"}
    ],
    "Majalengka": [
        {"file": "DataTrainMajalengka.csv", "gdrive": "https://drive.google.com/file/d/12-5l_9EFB_VRARq6CIOMTlaWH2VOx9wO/view?usp=sharing"},
        {"file": "DataIntervalMajalengka.csv", "gdrive": "https://drive.google.com/file/d/1QydFBUgwsrsV1kz9djJ06oz0KfHBZpIw/view?usp=sharing"}
    ],
    "Lain2": [
        {"file": "DataTrainLain2.csv", "gdrive": "https://drive.google.com/file/d/1BuDy0YlPazm7eoCabXHUq-BzsiKVDG67/view?usp=sharing"},
        {"file": "DataIntervalLain2.csv", "gdrive": "https://drive.google.com/file/d/1VinCHBclblPMivT_Xndg6-DAjC3wf9_w/view?usp=sharing"}
    ],
    "RNG LCG": [
        {"file": "RNG.csv", "gdrive": "https://drive.google.com/file/d/1fBZgfx9rYpBv29trUKkR1h9KT36g7udD/view?usp=sharing"},
        {"file": "LCG.csv", "gdrive": "https://drive.google.com/file/d/1MJMALE9054J2F6c1w-HYNRe9sftk2JGP/view?usp=sharing"}
    ],
    "Simulasi": [
        {"file": "Simulasi_Monte_Carlo.csv", "gdrive": "https://drive.google.com/file/d/1sbZlQXUjU7Km5pQrCcM4e1gmOL0mAzzX/view?usp=sharing"}
    ]
}

# -----------------------
# Fungsi Konversi Link
# -----------------------
def convert_gdrive_link(link):
    if "/d/" in link:
        file_id = link.split("/d/")[1].split("/")[0]
        return f"https://drive.google.com/uc?id={file_id}"
    return link

# -----------------------
# Fungsi Load Data dengan Fallback Encoding
# -----------------------
@st.cache_data
def load_data(local_path, gdrive_url):
    try:
        if not os.path.exists(local_path):
            st.info(f"📥 Mengunduh {os.path.basename(local_path)} dari Google Drive...")
            gdown.download(convert_gdrive_link(gdrive_url), local_path, quiet=False)

        try:
            df = pd.read_csv(local_path, low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(local_path, encoding="ISO-8859-1", low_memory=False)

        if df.shape[1] == 1:  # delimiter salah
            try:
                df = pd.read_csv(local_path, delimiter=";", encoding="utf-8", low_memory=False)
            except UnicodeDecodeError:
                df = pd.read_csv(local_path, delimiter=";", encoding="ISO-8859-1", low_memory=False)

        return df
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
        return pd.DataFrame()

# -----------------------
# Upload Manual
# -----------------------
uploaded_file = None
if source_option == "Upload Manual":
    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

# -----------------------
# Tampilkan Dataset
# -----------------------
datasets = data_sources.get(menu, [])
if not datasets:
    st.warning("⚠ Dataset belum tersedia.")
else:
    for idx, info in enumerate(datasets):
        st.subheader(f"📄 Dataset {idx+1} - {os.path.basename(info['file'])}")
        file_path = os.path.join(os.getcwd(), info["file"])

        # Load data
        if source_option == "Otomatis (Lokal/Drive)":
            df = load_data(file_path, info["gdrive"])
        elif uploaded_file:
            try:
                df = pd.read_csv(uploaded_file, encoding="utf-8", low_memory=False)
            except UnicodeDecodeError:
                df = pd.read_csv(uploaded_file, encoding="ISO-8859-1", low_memory=False)
        else:
            df = pd.DataFrame()

        # Jika kosong
        if df.empty:
            st.warning("⚠ Tidak ada data untuk ditampilkan.")
            continue

        # ✅ Tampilkan Data
        st.dataframe(df, use_container_width=True)
        st.markdown(f"**Jumlah Data:** {len(df)} baris")

        # ✅ Jika Data Kunjungan Pasien → TIDAK ada filter & grafik
        if "Data_Kunjungan_Pasien.csv" in info["file"]:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Data (CSV)",
                data=csv,
                file_name=f"{info['file']}",
                mime="text/csv",
                key=f"download_{idx}"
            )
            continue

        # ✅ Filter Dinamis
        with st.expander("🔍 Filter Data"):
            df_filtered = df.copy()
            for col in df.columns:
                if df[col].dtype == "object":
                    options = st.multiselect(f"Filter {col}", df[col].dropna().unique(), key=f"filter_{col}_{idx}")
                    if options:
                        df_filtered = df_filtered[df_filtered[col].isin(options)]
                else:
                    try:
                        min_val = float(df[col].min())
                        max_val = float(df[col].max())
                        if min_val != max_val:
                            min_slider, max_slider = st.slider(f"Rentang {col}", min_val, max_val, (min_val, max_val), key=f"slider_{col}_{idx}")
                            df_filtered = df_filtered[(df_filtered[col] >= min_slider) & (df_filtered[col] <= max_slider)]
                    except:
                        pass

            st.dataframe(df_filtered)

        # ✅ Visualisasi
        if len(df_filtered.columns) >= 2:
            st.subheader("📊 Visualisasi Data")
            chart_type = st.radio("Pilih Grafik", ["Bar", "Line", "Pie"], horizontal=True, key=f"chart_{idx}")
            col_x = st.selectbox("Kolom X", df_filtered.columns, key=f"x_{idx}")
            col_y = st.selectbox("Kolom Y", df_filtered.columns, key=f"y_{idx}")

            try:
                if chart_type == "Bar":
                    fig = px.bar(df_filtered, x=col_x, y=col_y, title="Bar Chart")
                elif chart_type == "Line":
                    fig = px.line(df_filtered, x=col_x, y=col_y, title="Line Chart")
                else:
                    fig = px.pie(df_filtered, names=col_x, values=col_y, title="Pie Chart")
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.warning("⚠ Tidak bisa membuat grafik. Pastikan kolom numerik untuk Y.")

        # ✅ Download Button
        csv = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Data (CSV)",
            data=csv,
            file_name=f"filtered_data_{idx}.csv",
            mime="text/csv",
            key=f"download_{idx}"
        )
