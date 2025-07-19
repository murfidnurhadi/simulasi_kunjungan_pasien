import streamlit as st
import pandas as pd
import numpy as np
import math
import random
import plotly.express as px

# =============================
# Konfigurasi Halaman
# =============================
st.set_page_config(layout="wide", page_title="Simulasi Monte Carlo", page_icon="🎲")

# =============================
# Sidebar Navigasi
# =============================
with st.sidebar:
    st.markdown("## 🧭 Navigasi")
    st.markdown("---")
    menu = st.radio(
        "📂 Pilih Halaman:",
        ["🏠 Dashboard", "📊 Data Train", "📈 Frekuensi dan Interval", "🔢 RNG LCG", "🎲 Simulasi"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.info("Pilih menu untuk menampilkan data atau simulasi.")

# =============================
# Dataset Handling
# =============================
default_path = "dataset/dataset.xlsx"

@st.cache_data
def load_excel(path):
    return pd.read_excel(path, sheet_name="DataTrain")

# Simpan dataset di session_state agar persist
if "df" not in st.session_state:
    try:
        st.session_state.df = load_excel(default_path)
    except:
        st.session_state.df = pd.DataFrame()

# Upload jika file default tidak tersedia
if st.session_state.df.empty:
    uploaded_file = st.file_uploader("📂 Upload File Excel", type=["xlsx"])
    if uploaded_file:
        st.session_state.df = pd.read_excel(uploaded_file, sheet_name="DataTrain")
        st.success("✅ File berhasil dimuat!")

df = st.session_state.df

# =============================
# 🏠 Dashboard
# =============================
if menu == "🏠 Dashboard":
    st.title("📊 Dashboard Simulasi Monte Carlo")

    if df.empty:
        st.warning("Dataset tidak tersedia. Upload file terlebih dahulu.")
    else:
        df.columns = df.columns.str.strip().str.lower()
        exclude = ["id", "bulan", "tahun"]
        daerah_cols = [c for c in df.columns if c not in exclude]

        total_per_wilayah = df[daerah_cols].sum().sort_values(ascending=False)
        total_all = total_per_wilayah.sum()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Pengunjung", f"{total_all:,}")
        col2.metric("Wilayah Terbanyak", total_per_wilayah.idxmax(), f"{total_per_wilayah.max():,}")
        col3.metric("Wilayah Tersedikit", total_per_wilayah.idxmin(), f"{total_per_wilayah.min():,}")

        grafik_df = total_per_wilayah.reset_index()
        grafik_df.columns = ["Wilayah", "Total"]

        fig = px.bar(
            grafik_df,
            x="Wilayah", y="Total", color="Wilayah",
            text="Total", title="Total Pengunjung per Wilayah",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

# =============================
# 📊 Data Train
# =============================
elif menu == "📊 Data Train":
    st.title("📊 Data Train Pengunjung")
    if df.empty:
        st.warning("Data tidak tersedia.")
    else:
        st.dataframe(df.reset_index(drop=True), use_container_width=True, hide_index=True)

# =============================
# 📈 Frekuensi & Interval
# =============================
elif menu == "📈 Frekuensi dan Interval":
    st.title("📈 Frekuensi dan Interval per Daerah")
    if df.empty:
        st.warning("Data tidak tersedia.")
    else:
        df.columns = df.columns.str.strip().str.lower()
        exclude = ["id", "bulan", "tahun"]
        daerah_cols = [c for c in df.columns if c not in exclude]

        pilih = st.selectbox("📍 Pilih Daerah", ["Pilih"] + daerah_cols)
        if pilih == "Pilih":
            st.info("Pilih daerah terlebih dahulu.")
        else:
            data = df[pilih].dropna()
            n = len(data)
            x_min, x_max = data.min(), data.max()
            R = x_max - x_min
            k = math.ceil(1 + 3.3 * math.log10(n))
            h = math.ceil(R / k)

            # Buat interval
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

            # Hitung titik tengah
            bounds = freq_table["Interval Jumlah"].str.split(" - ", expand=True).astype(int)
            freq_table["Titik Tengah"] = (bounds[0] + bounds[1]) / 2
            freq_table["No"] = range(1, len(freq_table)+1)

            total = freq_table["Frekuensi"].sum()
            prob = (freq_table["Frekuensi"]/total).round(2)
            diff = 1.0 - prob.sum()
            if abs(diff) > 0: prob.iloc[prob.idxmax()] += diff
            freq_table["Probabilitas"] = prob
            freq_table["Prob. Kumulatif"] = prob.cumsum().round(2)
            freq_table["P.K * 100"] = (freq_table["Prob. Kumulatif"]*100).astype(int)

            upper_bounds = freq_table["P.K * 100"]
            lower_bounds = [1]+[u+1 for u in upper_bounds[:-1]]
            freq_table["Interval Angka Acak"] = [f"{l}-{u}" for l,u in zip(lower_bounds, upper_bounds)]

            freq_table = freq_table[["No","Interval Jumlah","Frekuensi","Titik Tengah","Probabilitas","Prob. Kumulatif","P.K * 100","Interval Angka Acak"]]
            st.dataframe(freq_table, use_container_width=True, hide_index=True)

# =============================
# 🔢 RNG LCG
# =============================
elif menu == "🔢 RNG LCG":
    st.title("🔢 RNG - Linear Congruential Generator")
    if "rng" not in st.session_state: st.session_state.rng = None

    m = st.number_input("Modulus (m)", 1, value=100)
    a = st.number_input("Multiplier (a)", 1, value=5)
    c = st.number_input("Increment (c)", 0, value=1)
    z0 = st.number_input("Seed (z₀)", 0, value=1)
    n_gen = st.number_input("Jumlah Bilangan Acak", 1, value=10)

    col_gen, col_reset = st.columns(2)
    if col_gen.button("🎲 Generate"):
        zi = z0
        rng_data = []
        for i in range(1, n_gen+1):
            zi_next = (a*zi + c) % m
            ui = zi_next/m
            rng_data.append((i, zi, zi_next, round(ui,4), int(ui*100)))
            zi = zi_next
        st.session_state.rng = pd.DataFrame(rng_data, columns=["i","Zᵢ₋₁","Zᵢ","Uᵢ","Uᵢ*100"])
        st.success("Bilangan acak berhasil dibuat!")
    if col_reset.button("❌ Reset"):
        st.session_state.rng = None
        st.info("Reset berhasil.")

    if st.session_state.rng is not None:
        st.dataframe(st.session_state.rng, use_container_width=True, hide_index=True)

# =============================
# 🎲 Simulasi Monte Carlo
# =============================
elif menu == "🎲 Simulasi":
    st.title("🎲 Simulasi Monte Carlo")
    if st.session_state.rng is None:
        st.warning("Generate bilangan acak dulu di halaman RNG.")
    elif df.empty:
        st.warning("Data train tidak tersedia.")
    else:
        rng_df = st.session_state.rng
        st.subheader("Bilangan Acak")
        st.dataframe(rng_df, use_container_width=True, hide_index=True)

        df.columns = df.columns.str.strip().str.lower()
        daerah_cols = [c for c in df.columns if c not in ["id","bulan","tahun"]]
        pilih = st.selectbox("📍 Pilih Daerah", ["Pilih"]+daerah_cols)
        if pilih != "Pilih":
            data = df[pilih].dropna()
            n = len(data)
            R = data.max()-data.min()
            k = math.ceil(1+3.3*math.log10(n))
            h = math.ceil(R/k)
            bins = []
            low = math.floor(data.min())
            for _ in range(k):
                up = low+h
                bins.append((low,up))
                low = up+1
            labels = [f"{l}-{u}" for l,u in bins]
            cut_bins = [b[0] for b in bins]+[bins[-1][1]]
            kelas = pd.cut(data,bins=cut_bins,labels=labels,include_lowest=True)
            freq = kelas.value_counts().sort_index().reset_index()
            freq.columns=["Interval Jumlah","Frekuensi"]
            total=freq["Frekuensi"].sum()
            prob=(freq["Frekuensi"]/total).round(2)
            diff=1.0-prob.sum()
            if abs(diff)>0: prob.iloc[prob.idxmax()]+=diff
            freq["Probabilitas"]=prob
            freq["Prob. Kumulatif"]=prob.cumsum().round(2)
            freq["P.K*100"]=(freq["Prob. Kumulatif"]*100).astype(int)
            upb=freq["P.K*100"];lowb=[1]+[u+1 for u in upb[:-1]]
            freq["Interval Angka Acak"]=[f"{l}-{u}" for l,u in zip(lowb,upb)]
            st.subheader("Tabel Interval")
            st.dataframe(freq,use_container_width=True,hide_index=True)

            def get_value(u):
                val=int(u*100)
                if val==0: val=1
                for _,r in freq.iterrows():
                    l,h=map(int,r["Interval Angka Acak"].split("-"))
                    if l<=val<=h:
                        low,high=map(int,r["Interval Jumlah"].split("-"))
                        return random.randint(low,high),val
                return None,val

            hasil=[]
            for _,r in rng_df.iterrows():
                val,num=get_value(r["Uᵢ"])
                hasil.append({"Bulan":r["i"],"Bilangan Acak":num,"Jumlah Pengunjung":val})
            sim=pd.DataFrame(hasil)
            st.subheader("Hasil Simulasi")
            st.dataframe(sim,use_container_width=True,hide_index=True)
            st.markdown(f"**Total:** {sim['Jumlah Pengunjung'].sum():,}")
            st.markdown(f"**Rata-rata:** {sim['Jumlah Pengunjung'].mean():,.2f}")
