import numpy as np
import streamlit as st

import charts as ch
from wine_core import (FEATURES, FEATURE_COLS, THRESHOLD_SKOR, fmt_int, fmt_pct, load_clean,
                       load_raw, outlier_table)

raw = load_raw()
clean, info = load_clean()

st.title("Tentang Dataset")
st.caption("Wine Quality (WineQT): hasil uji fisikokimia wine merah Vinho Verde dari Portugal.")

t1, t2, t3, t4 = st.tabs(["Pengantar", "Kamus Fitur", "Lihat Data", "Kualitas Data"])

with t1:
    c = st.columns(4)
    c[0].metric("Baris", fmt_int(len(raw)), border=True)
    c[1].metric("Kolom", raw.shape[1], border=True)
    c[2].metric("Fitur kimia", len(FEATURE_COLS), border=True)
    c[3].metric("Rasio biasa : baik", f"{(raw.quality < THRESHOLD_SKOR).sum() / (raw.quality >= THRESHOLD_SKOR).sum():.2f} : 1", border=True)

    st.markdown(
        f"- **Jenis masalah:** klasifikasi (skor kualitas diubah menjadi biner).\n"
        f"- **Target:** `quality`, skor penilaian sensorik 3 sampai 8. "
        f"Skor ≥ {THRESHOLD_SKOR} = **Baik (1)**, skor ≤ {THRESHOLD_SKOR - 1} = **Biasa (0)**.\n"
        f"- **Fitur:** 11 hasil uji fisikokimia, semuanya numerik.\n"
        f"- **Catatan:** kolom `Id` hanya penomor baris dan tidak dipakai sebagai fitur."
    )

    skor = raw["quality"].value_counts().sort_index()
    baik = int((raw.quality >= THRESHOLD_SKOR).sum())
    kiri, kanan = st.columns(2)
    with kiri:
        fig, ax = ch.fig_baru(5.2, 3.4)
        warna = [ch.BAIK if s >= THRESHOLD_SKOR else ch.BIASA for s in skor.index]
        bars = ax.bar(skor.index.astype(str), skor.values, color=warna)
        ax.set_ylim(0, skor.max() * 1.15)
        ch.label_batang(ax, bars)
        ax.set_title("Jumlah wine per skor kualitas")
        ax.set_xlabel("Skor quality")
        ax.set_ylabel("Jumlah wine")
        st.pyplot(fig)
    with kanan:
        fig, ax = ch.fig_baru(4.2, 3.4, grid=None)
        ch.donat(ax, [len(raw) - baik, baik], ["Biasa (≤ 6)", "Baik (≥ 7)"], [ch.BIASA, ch.BAIK], "Proporsi kelas")
        st.pyplot(fig)

with t2:
    st.markdown("Kamus fitur: arti dan satuan setiap kolom pada dataset.")
    kamus = [{"Kolom": k, "Nama": v[0], "Tipe": "Numerik", "Satuan": v[1], "Keterangan": v[2]}
             for k, v in FEATURES.items()]
    kamus.append({"Kolom": "quality", "Nama": "Skor kualitas", "Tipe": "Target (ordinal)", "Satuan": "skor 3-8",
                  "Keterangan": "Hasil penilaian sensorik; diubah menjadi label Baik / Biasa."})
    kamus.append({"Kolom": "Id", "Nama": "Nomor baris", "Tipe": "Identitas", "Satuan": "-",
                  "Keterangan": "Penanda baris; tidak dipakai sebagai fitur."})
    st.dataframe(kamus, width="stretch", hide_index=True)

with t3:
    a, b = st.columns([3, 1])
    kolom = a.multiselect("Pilih kolom", list(raw.columns), default=list(raw.columns))
    mode = b.radio("Mode", ["Awal", "Acak", "Akhir"], horizontal=True)
    n = st.slider("Jumlah baris ditampilkan", 5, 200, 20)
    if not kolom:
        st.warning("Pilih minimal satu kolom.")
    else:
        if mode == "Awal":
            tampil = raw[kolom].head(n)
        elif mode == "Akhir":
            tampil = raw[kolom].tail(n)
        else:
            tampil = raw[kolom].sample(min(n, len(raw)), random_state=42)
        st.caption(f"Menampilkan {len(tampil)} dari {fmt_int(len(raw))} baris.")
        st.dataframe(tampil, width="stretch")
        st.download_button("Unduh tampilan (CSV)", tampil.to_csv(index=False),
                           file_name="wine_tampilan.csv", mime="text/csv")

with t4:
    c = st.columns(3)
    c[0].metric("Nilai hilang (missing)", int(raw.isna().sum().sum()), border=True)
    c[1].metric("Baris duplikat (tanpa Id)", info["n_dup"], border=True)
    c[2].metric("Baris setelah dibersihkan", fmt_int(info["n_clean"]), border=True)

    st.subheader("Ringkasan per fitur")
    ot = outlier_table(raw)
    ringkas = ot[["Fitur", "Min", "Median", "Maks", "Skewness", "Outlier"]].copy()
    ringkas.insert(1, "Tipe data", [str(raw[f].dtype) for f in ot["Fitur"]])
    st.dataframe(ringkas, width="stretch", hide_index=True,
                 column_config={"Skewness": st.column_config.NumberColumn(format="%.2f"),
                                "Min": st.column_config.NumberColumn(format="%.4g"),
                                "Median": st.column_config.NumberColumn(format="%.4g"),
                                "Maks": st.column_config.NumberColumn(format="%.4g")})
    st.warning(
        f"Tidak ada nilai kosong, tetapi ada **{info['n_dup']} baris** yang nilai fiturnya identik dengan "
        f"baris lain bila `Id` diabaikan. Baris kembar yang masuk ke data latih dan data uji sekaligus bisa "
        f"membuat skor model tampak lebih baik dari kenyataan, jadi dihapus pada tahap preprocessing."
    )
    top = ot.sort_values("Outlier", ascending=False).head(3)
    st.info("Fitur dengan outlier terbanyak (aturan IQR): " +
            ", ".join(f"**{r.Fitur}** ({r.Outlier})" for r in top.itertuples()) + ".")
