import pandas as pd
import streamlit as st

from wine_core import (FEATURE_COLS, THRESHOLD_SKOR, fmt_int, fmt_pct, hitung_metrik, load_raw,
                       prediksi_proba, train_all)

raw = load_raw()
res = train_all()
modelnya = list(res["metrics"].index)

st.title("Prediksi Massal (CSV)")
st.caption("Unggah banyak wine sekaligus, lalu unduh hasil prediksinya.")

a, b = st.columns(2)
nama = a.selectbox("Model yang dipakai", modelnya, index=modelnya.index(res["best"]))
thr = b.slider("Threshold prediksi", 0.05, 0.95, 0.50, 0.05)

with st.expander("Format kolom yang dibutuhkan"):
    st.markdown("File CSV harus memiliki 11 kolom fitur berikut (nama persis sama). Kolom lain "
                "(misalnya `Id`) diabaikan. Jika ada kolom `quality`, hasil prediksi dibandingkan dengan "
                f"skor aslinya (baik = skor ≥ {THRESHOLD_SKOR}).")
    st.code(", ".join(FEATURE_COLS))
    st.download_button("Unduh contoh template (CSV)", raw[FEATURE_COLS].head(5).to_csv(index=False),
                       file_name="template_wine.csv", mime="text/csv")

file = st.file_uploader("Unggah file CSV", type="csv")
if file is None:
    st.info("Belum ada file. Unggah CSV untuk memulai, atau unduh template di atas.")
    st.stop()

try:
    data = pd.read_csv(file, sep=None, engine="python")
except Exception as e:
    st.error(f"File tidak bisa dibaca: {e}")
    st.stop()

hilang = [c for c in FEATURE_COLS if c not in data.columns]
if hilang:
    st.error("Kolom berikut tidak ditemukan: " + ", ".join(hilang))
    st.stop()

fitur = data[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
valid = fitur.notna().all(axis=1)
if (~valid).any():
    st.warning(f"{int((~valid).sum())} baris dilewati karena ada nilai kosong atau bukan angka.")
if not valid.any():
    st.error("Tidak ada baris yang valid untuk diprediksi.")
    st.stop()

hasil = data.loc[valid].copy()
p = prediksi_proba(nama, fitur[valid])
hasil["Probabilitas baik"] = p
hasil["Prediksi"] = ["Baik" if v >= thr else "Biasa" for v in p]

c = st.columns(3)
c[0].metric("Wine diprediksi", fmt_int(len(hasil)), border=True)
c[1].metric("Diprediksi baik", fmt_int((hasil["Prediksi"] == "Baik").sum()), border=True)
c[2].metric("Persentase baik", fmt_pct((hasil["Prediksi"] == "Baik").mean()), border=True)

st.dataframe(hasil, width="stretch",
             column_config={"Probabilitas baik": st.column_config.ProgressColumn(
                 "Probabilitas baik", min_value=0.0, max_value=1.0, format="%.2f")})
st.download_button("Unduh hasil prediksi (CSV)", hasil.to_csv(index=False),
                   file_name="hasil_prediksi_wine.csv", mime="text/csv")

if "quality" in hasil.columns:
    q = pd.to_numeric(hasil["quality"], errors="coerce")
    ok = q.notna()
    y = (q[ok] >= THRESHOLD_SKOR).astype(int)
    if ok.any() and y.nunique() == 2:
        m = hitung_metrik(y, hasil.loc[ok, "Probabilitas baik"].to_numpy(), thr)
        st.subheader("Perbandingan dengan skor asli")
        st.dataframe(pd.DataFrame([m]), width="stretch", hide_index=True,
                     column_config={k: st.column_config.NumberColumn(format="%.3f") for k in m})
        st.caption("Jika file ini memuat wine yang sama dengan data latih, skor akan tampak terlalu bagus.")
