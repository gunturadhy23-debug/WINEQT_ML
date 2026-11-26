import streamlit as st

from wine_core import (THRESHOLD_SKOR, cv_results, fmt_int, fmt_pct, importance, load_clean, load_raw,
                       pca_info, train_all)

raw = load_raw()
clean, info = load_clean()
res = train_all()
met = res["metrics"]
best = res["best"]
cv = cv_results()
pc = pca_info()
imp = importance(best)
b = met.loc[best]
top3 = imp["Fitur"].head(3).tolist()
cv_best = cv["F1 (rata-rata)"].idxmax()
bn = pc["bench"]
rf_bench = bn[bn["Model"] == "Random Forest"]["F1"].tolist()
rf_asli, rf_pca = rf_bench[0], rf_bench[1]   # urutan: tanpa PCA, lalu dengan PCA

st.title("Kesimpulan & Saran")

st.subheader("Kesimpulan")
st.markdown(
    f"1. **Model terbaik** berdasarkan F1 adalah **{best}** (F1 {b['F1']:.3f}, Recall {b['Recall']:.3f}, "
    f"Precision {b['Precision']:.3f}, ROC-AUC {b['ROC-AUC']:.3f}).\n"
    f"2. Kelas baik (skor ≥ {THRESHOLD_SKOR}) hanya {fmt_pct(res['y'].mean())} dari data, sehingga akurasi "
    f"saja menyesatkan; penilaian memakai Recall, F1, dan ROC-AUC.\n"
    f"3. Setelah membuang Id dan {info['n_dup']} duplikat, tersisa {fmt_int(info['n_clean'])} wine.\n"
    f"4. PCA meringkas 11 fitur menjadi {pc['k90']} komponen (90% varians), tetapi tidak menaikkan kinerja: "
    f"F1 Random Forest {rf_pca:.3f} dengan PCA vs {rf_asli:.3f} tanpa PCA. Pemodelan utama karena itu memakai "
    f"fitur asli yang juga lebih mudah ditafsirkan.\n"
    f"5. Fitur terpenting bagi {best}: **{top3[0]}**, **{top3[1]}**, dan **{top3[2]}**.\n"
    f"6. Pada cross-validation 5-fold, F1 rata-rata tertinggi dimiliki **{cv_best}** "
    f"({cv.loc[cv_best, 'F1 (rata-rata)']:.3f}), sedangkan {best} memperoleh {cv.loc[best, 'F1 (rata-rata)']:.3f}. "
    f"Data uji hanya {fmt_int(len(res['y_te']))} wine dengan {int(res['y_te'].sum())} wine baik, sehingga "
    f"perbedaan antar model dibaca sebagai indikasi, bukan kepastian."
)

st.subheader("Perbandingan ringkas lima model")
ringkas = met[["Precision", "Recall", "F1", "ROC-AUC"]].copy()
ringkas["F1 (CV 5-fold)"] = cv["F1 (rata-rata)"]
st.dataframe(ringkas.style.highlight_max(axis=0, color="#E8CDD3"), width="stretch",
             column_config={k: st.column_config.NumberColumn(format="%.3f") for k in ringkas.columns})

st.subheader("Saran bisnis")
st.markdown(
    "- Gunakan model sebagai **penyaring awal**: sampel dengan probabilitas tinggi diprioritaskan untuk uji "
    "sensorik, bukan menggantikan panelis.\n"
    f"- Pantau fitur yang paling berpengaruh ({', '.join(top3)}) selama proses produksi.\n"
    "- Sesuaikan threshold dengan tujuan: turunkan untuk menjaring lebih banyak wine baik (Recall naik), "
    "naikkan bila ingin lebih yakin (Precision naik).\n"
    "- Perbarui model secara berkala saat data panen dan penilaian baru tersedia."
)

st.subheader("Keterbatasan")
st.markdown(
    f"- Data kecil ({fmt_int(info['n_clean'])} wine) dan hanya satu jenis: wine merah Vinho Verde.\n"
    "- Skor kualitas berasal dari penilaian sensorik yang subjektif.\n"
    f"- Kelas baik hanya {fmt_int(res['y'].sum())} sampel, sehingga metrik pada data uji bisa berubah cukup "
    "besar bila pembagian data diganti.\n"
    "- Tidak ada informasi varietas anggur, tahun panen, atau harga.\n"
    "- Peta pada halaman EDA hanya ilustrasi karena dataset tidak memiliki koordinat."
)
