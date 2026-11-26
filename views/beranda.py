import streamlit as st

from wine_core import (ANGGOTA, DOSEN, PRODI, KELOMPOK, THRESHOLD_SKOR, fmt_int, fmt_pct,
                       load_clean, load_raw, pca_info, train_all)

raw = load_raw()
clean, info = load_clean()
res = train_all()
pc = pca_info()
best = res["best"]
m = res["metrics"].loc[best]

st.title("Prediksi Kualitas Wine Merah")
st.caption("Implementasi Machine Learning berbasis web dengan Streamlit")
st.markdown(
    "Website ini memaparkan seluruh tahapan proyek **prediksi wine berkualitas baik** pada dataset "
    "*Wine Quality (WineQT)*: mulai dari pengenalan data, eksplorasi, preprocessing, reduksi dimensi "
    "(PCA), perbandingan lima algoritma klasifikasi, hingga **prediksi langsung** yang bisa dicoba pengguna."
)

c = st.columns(4)
c[0].metric("Jumlah sampel wine", fmt_int(len(raw)), border=True)
c[1].metric(f"Wine baik (skor ≥ {THRESHOLD_SKOR})", fmt_pct(res["y"].mean()), border=True)
c[2].metric("Model terbaik (F1)", best, border=True)
c[3].metric("ROC-AUC model terbaik", f"{m['ROC-AUC']:.3f}", border=True)

st.divider()
st.subheader("Alur pengerjaan")
alur = [
    ("1. Dataset", f"Karakteristik data: {fmt_int(len(raw))} baris, {raw.shape[1]} kolom, 11 fitur kimia, "
                   f"target skor kualitas 3 sampai 8."),
    ("2. EDA & Preprocessing", f"Membuang Id, menghapus {info['n_dup']} baris duplikat, membuat label biner, "
                               "dan menangani ketidakseimbangan kelas."),
    ("3. Reduksi PCA", f"11 fitur diringkas menjadi {pc['k90']} sampai {pc['k95']} komponen "
                       "(90% sampai 95% varians)."),
    ("4. Modelling", "Lima algoritma dibandingkan, dievaluasi dengan cross-validation, "
                     "di-tuning, lalu dipakai untuk prediksi."),
]
for col, (judul, teks) in zip(st.columns(4), alur):
    with col.container(border=True):
        st.markdown(f"**{judul}**")
        st.write(teks)

st.divider()
st.subheader(f"Anggota Kelompok {KELOMPOK}")
for col, (nama, nim) in zip(st.columns(len(ANGGOTA)), ANGGOTA):
    with col.container(border=True):
        st.markdown(f"**{nama}**")
        st.caption(nim)
st.markdown(f"Dosen pengampu: **{DOSEN}**")
st.caption(PRODI)

with st.expander("Mengapa memprediksi kualitas wine penting?"):
    st.markdown(
        f"- Penilaian kualitas wine dilakukan lewat uji sensorik oleh panelis, sehingga mahal dan subjektif.\n"
        f"- Dengan model yang membaca hasil uji laboratorium, produsen bisa menyaring sampel yang "
        f"berpotensi baik lebih awal.\n"
        f"- Wine berkualitas baik hanya sekitar {fmt_pct(res['y'].mean())} dari data, sehingga model dinilai "
        f"dengan **Recall, F1, dan ROC-AUC**, bukan akurasi saja."
    )
