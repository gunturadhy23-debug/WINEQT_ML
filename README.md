# Prediksi Kualitas Wine Merah (WineQT) - Streamlit

Aplikasi web multi-halaman untuk tugas Machine Learning: eksplorasi data, preprocessing, PCA,
perbandingan lima algoritma klasifikasi, dan prediksi kualitas wine (baik / biasa).

## Isi repositori

```
app.py                 # pintu masuk + menu navigasi
wine_core.py           # identitas kelompok, data, dan pipeline machine learning
charts.py              # gaya grafik
views/                 # sembilan halaman aplikasi
data/WineQT.csv        # dataset
.streamlit/config.toml # tema warna
requirements.txt       # daftar library untuk deploy
```

## Mengubah identitas kelompok

Buka `wine_core.py`, ubah bagian paling atas: `KELOMPOK`, `ANGGOTA` (nama dan NIM), dan `DOSEN`.

## Menjalankan di komputer sendiri

```bash
py -3 -m venv env
env\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy ke Streamlit Community Cloud

1. Unggah seluruh isi folder ini ke repositori GitHub (public).
2. Buka https://share.streamlit.io lalu login dengan akun GitHub.
3. Klik **Create app**, pilih repositori dan branch (`main`), isi **Main file path** dengan `app.py`.
4. Klik **Deploy**. Instalasi pertama butuh beberapa menit, lalu aplikasi punya alamat `https://nama-app.streamlit.app`.

Kunjungan pertama ke halaman *Modelling & Evaluasi* butuh sekitar setengah menit karena model dilatih
dan di-cache; setelah itu halaman terbuka cepat.
