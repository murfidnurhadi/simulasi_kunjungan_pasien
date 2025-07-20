import streamlit as st
import pandas as pd
import math
import random
import os
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import plotly.express as px

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
# 📂 Load Data
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
# 🏠 Dashboard
# ========================
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
            st.header(f"📊 Distribusi Frekuensi: Kota {selected_daerah.capitalize()}")
            data = df[selected_daerah].dropna()
            n = len(data)

            x_min, x_max = data.min(), data.max()
            R = x_max - x_min
            k = math.ceil(1 + 3.3 * math.log10(n))
            h = int(R / k)

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

            prob_raw = freq_table["Frekuensi"] / n
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

            freq_table.insert(0, "No", range(1, len(freq_table) + 1))
            st.dataframe(freq_table, use_container_width=True, hide_index=True)
    else:
        st.warning("Data tidak tersedia.")

# ========================
# 🔢 RNG LCG (Lengkap + PDF)
# ========================
elif menu == "🔢 RNG LCG":
    st.title("🔢 Linear Congruential Generator (LCG) - PDF Export")

    a = st.number_input("Multiplier (a)", min_value=1, value=21)
    c = st.number_input("Increment (c)", min_value=0, value=17)
    m = st.number_input("Modulus (m)", min_value=1, value=100)
    z0 = st.number_input("Seed (Z₀)", min_value=0, value=42)
    n_gen = st.number_input("Jumlah Bilangan Acak", min_value=1, value=20)

    if st.button("🎲 Generate"):
        zi = z0
        rng_data = []
        all_zi = []
        duplicate_flag = False

        for i in range(1, n_gen + 1):
            zi_minus_1 = zi
            zi = (a * zi_minus_1 + c) % m
            ui = zi / m
            angka_acak = int(ui * 100)

            if zi in all_zi:
                duplicate_flag = True
            all_zi.append(zi)

            rng_data.append((i, zi_minus_1, zi, round(ui, 4), angka_acak))

        rng_df = pd.DataFrame(rng_data, columns=["i", "Zᵢ₋₁", "Zᵢ", "Uᵢ", "Angka Acak (Uᵢ×100)"])
        st.dataframe(rng_df, use_container_width=True)
        st.session_state['rng_df'] = rng_df

        if duplicate_flag:
            st.warning("⚠️ Ada nilai Zᵢ yang duplikat!")
        else:
            st.success("✅ Tidak ada duplikat.")

        st.markdown("### 📉 Visualisasi")
        fig = px.line(rng_df, x="i", y="Zᵢ", title="Perkembangan Nilai Zᵢ", markers=True)
        st.plotly_chart(fig, use_container_width=True)

        # Export PDF
        def create_pdf():
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            elements = []

            elements.append(Paragraph("Hasil RNG LCG", styles["Title"]))
            elements.append(Spacer(1, 12))
            elements.append(Paragraph(f"Parameter: a={a}, c={c}, m={m}, Z₀={z0}", styles["Normal"]))
            elements.append(Spacer(1, 12))

            data_table = [["i", "Zᵢ₋₁", "Zᵢ", "Uᵢ", "Angka Acak"]]
            for row in rng_data:
                data_table.append(list(row))

            table = Table(data_table)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(table)

            if duplicate_flag:
                elements.append(Paragraph("⚠️ Terdapat nilai Zᵢ yang duplikat.", styles["Normal"]))
            else:
                elements.append(Paragraph("✅ Tidak ada nilai Zᵢ yang duplikat.", styles["Normal"]))

            doc.build(elements)
            pdf = buffer.getvalue()
            buffer.close()
            return pdf

        pdf_data = create_pdf()
        st.download_button("📥 Download PDF", data=pdf_data, file_name="hasil_rng_lcg.pdf", mime="application/pdf")

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
        st.dataframe(rng_df, use_container_width=True)
