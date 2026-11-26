import pandas as pd
import streamlit as st

from wine_core import (FEATURE_COLS, FEATURES, THRESHOLD_SKOR, fmt_pct, load_clean, load_raw,
                       prediksi_proba, train_all)

raw = load_raw()
clean, _ = load_clean()
res = train_all()
modelnya = list(res["metrics"].index)
baik = clean["quality"] >= THRESHOLD_SKOR

st.title("Prediksi Satu Wine")
st.caption("Masukkan hasil uji laboratorium sebuah wine, lalu lihat prediksi kualitasnya.")

a, b, c = st.columns([2, 2, 2])
nama = a.selectbox("Model yang dipakai", modelnya, index=modelnya.index(res["best"]))
thr = b.slider("Threshold prediksi", 0.05, 0.95, 0.50, 0.05,
               help="Wine diprediksi baik bila probabilitas ≥ threshold.")
profil_nama = c.selectbox("Isi awal form dengan", ["Median seluruh data", "Profil wine baik (median)",
                                                   "Profil wine biasa (median)"])
profil = {
    "Median seluruh data": clean[FEATURE_COLS].median(),
    "Profil wine baik (median)": clean.loc[baik, FEATURE_COLS].median(),
    "Profil wine biasa (median)": clean.loc[~baik, FEATURE_COLS].median(),
}[profil_nama]
slug = profil_nama.split()[0] + str(len(profil_nama))

with st.form("form_prediksi"):
    nilai = {}
    kolom = st.columns(3)
    for i, f in enumerate(FEATURE_COLS):
        label, satuan, ket, step, fmt = FEATURES[f]
        mn, mx = float(raw[f].min()), float(raw[f].max())
        rentang = mx - mn
        lo = max(0.0, mn - 0.1 * rentang)
        hi = mx + 0.1 * rentang
        awal = float(min(max(profil[f], lo), hi))
        nilai[f] = kolom[i % 3].number_input(
            f"{f} ({satuan})" if satuan != "-" else f, min_value=lo, max_value=hi, value=awal,
            step=step, format=fmt, key=f"in_{f}_{slug}", help=ket)
    kirim = st.form_submit_button("Prediksi", type="primary")

if kirim:
    x = pd.DataFrame([nilai])[FEATURE_COLS]
    p = float(prediksi_proba(nama, x)[0])
    if p >= thr:
        st.success(f"Prediksi: wine berkualitas **BAIK** (skor ≥ {THRESHOLD_SKOR}).")
    else:
        st.info(f"Prediksi: wine **BIASA** (skor ≤ {THRESHOLD_SKOR - 1}).")
    kiri, kanan = st.columns([1, 2])
    kiri.metric("Probabilitas wine baik", fmt_pct(p), border=True)
    kanan.progress(min(max(p, 0.0), 1.0), text=f"Probabilitas {fmt_pct(p)} dengan threshold {thr:.2f}")

    st.subheader("Perbandingan dengan profil rata-rata")
    banding = pd.DataFrame({
        "Fitur": FEATURE_COLS,
        "Input": [nilai[f] for f in FEATURE_COLS],
        "Median wine baik": [clean.loc[baik, f].median() for f in FEATURE_COLS],
        "Median wine biasa": [clean.loc[~baik, f].median() for f in FEATURE_COLS],
    })
    st.dataframe(banding, width="stretch", hide_index=True,
                 column_config={k: st.column_config.NumberColumn(format="%.4g") for k in
                                ("Input", "Median wine baik", "Median wine biasa")})
    st.caption("Model dilatih pada sekitar seribu wine merah Vinho Verde, sehingga hanya cocok untuk "
               "wine sejenis. Hasilnya probabilitas, bukan kepastian.")
