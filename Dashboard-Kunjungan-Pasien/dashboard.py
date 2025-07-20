# ========================
# 📈 Frekuensi & Interval
# ========================
elif menu == "📈 Frekuensi dan Interval":
    st.title("📈 Distribusi Frekuensi dan Interval")
    if not df.empty:
        # Normalisasi nama kolom
        df.columns = df.columns.str.strip().str.lower()
        exclude_cols = ["id", "bulan", "tahun"]
        daerah_cols = [col for col in df.columns if col not in exclude_cols]

        # Pilih daerah
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

            # Interval Angka Acak
            upper_bounds = freq_table["P.K * 100"]
            lower_bounds = [1] + [ub + 1 for ub in upper_bounds[:-1]]
            freq_table["Interval Angka Acak"] = [f"{lb} - {ub}" for lb, ub in zip(lower_bounds, upper_bounds)]

            # ✅ Format angka Titik Tengah
            freq_table["Titik Tengah"] = freq_table["Titik Tengah"].apply(lambda x: f"{x:,}".replace(",", "."))

            # ✅ Hapus kolom kosong otomatis Streamlit
            st.dataframe(freq_table.reset_index(drop=True), use_container_width=True)

            # Simpan ke session_state untuk dipakai di simulasi
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
