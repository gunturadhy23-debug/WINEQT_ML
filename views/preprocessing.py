import numpy as np
import pandas as pd
import streamlit as st

import charts as ch
from wine_core import (FEATURE_COLS, SEED, TEST_SIZE, THRESHOLD_SKOR, fmt_int, fmt_pct, load_clean,
                       load_raw, outlier_table, train_all)

raw = load_raw()
clean, info = load_clean()
res = train_all()

st.title("Preprocessing")
st.caption("Menyiapkan data agar adil dan bebas kebocoran sebelum dipakai melatih model.")
st.progress(1.0, text="Urutan: bersihkan data → periksa outlier → buat label → bagi data → skala fitur")

with st.expander("1. Data cleaning: buang Id dan hapus duplikat", expanded=True):
    st.markdown(
        "`Id` hanya penomor baris, sehingga dibuang. Setelah itu baris yang nilai fiturnya identik "
        "dihapus supaya baris kembar tidak jatuh di data latih **dan** data uji sekaligus (kebocoran data)."
    )
    tahap = pd.DataFrame({
        "Tahap": ["Data mentah", "Buang kolom Id", "Hapus baris duplikat"],
        "Baris": [info["n_raw"], info["n_raw"], info["n_clean"]],
        "Kolom": [raw.shape[1], raw.shape[1] - 1, raw.shape[1] - 1],
    })
    st.dataframe(tahap, width="stretch", hide_index=True)
    st.caption(f"{info['n_dup']} baris duplikat dihapus. Data bersih: {fmt_int(info['n_clean'])} baris.")
    st.dataframe(clean.head(8), width="stretch")

with st.expander("2. Penanganan outlier"):
    ot = outlier_table(clean)
    st.markdown(
        "Outlier dideteksi dengan aturan IQR (di luar Q1 − 1,5×IQR sampai Q3 + 1,5×IQR). "
        "Outlier **dipertahankan**: pada data kimia, nilai ekstrem bisa jadi wine yang memang berbeda, "
        "dan model berbasis pohon tahan terhadap nilai ekstrem."
    )
    st.dataframe(ot[["Fitur", "Skewness", "Outlier", "Batas bawah", "Batas atas"]],
                 width="stretch", hide_index=True,
                 column_config={"Skewness": st.column_config.NumberColumn(format="%.2f"),
                                "Batas bawah": st.column_config.NumberColumn(format="%.3f"),
                                "Batas atas": st.column_config.NumberColumn(format="%.3f")})
    z = (clean[FEATURE_COLS] - clean[FEATURE_COLS].mean()) / clean[FEATURE_COLS].std()
    fig, ax = ch.fig_baru(7.5, 3.6, grid="x")
    bp = ax.boxplot([z[c] for c in FEATURE_COLS], vert=False, tick_labels=FEATURE_COLS, patch_artist=True,
                    flierprops=dict(markersize=2.5, alpha=0.5), medianprops=dict(color=ch.INK))
    for b in bp["boxes"]:
        b.set_facecolor(ch.BIASA)
        b.set_alpha(0.55)
    ax.set_title("Sebaran fitur setelah distandarkan (z-score)")
    ax.set_xlabel("z-score")
    st.pyplot(fig)

with st.expander("3. Membuat label biner dan memeriksa ketidakseimbangan kelas"):
    n_baik = int(res["y"].sum())
    n_biasa = len(res["y"]) - n_baik
    st.markdown(
        f"Skor kualitas diubah menjadi label biner: **skor ≥ {THRESHOLD_SKOR} = Baik (1)**, "
        f"skor ≤ {THRESHOLD_SKOR - 1} = Biasa (0)."
    )
    st.code(f"y = (df['quality'] >= {THRESHOLD_SKOR}).astype(int)", language="python")
    c = st.columns(3)
    c[0].metric("Biasa (0)", fmt_int(n_biasa), border=True)
    c[1].metric("Baik (1)", fmt_int(n_baik), border=True)
    c[2].metric("Proporsi baik", fmt_pct(n_baik / len(res["y"])), border=True)
    st.markdown(
        "Kelas baik jauh lebih sedikit, sehingga model yang selalu menebak “Biasa” sudah akurat "
        f"{fmt_pct(n_biasa / len(res['y']))}. Karena itu model dinilai dengan **Recall, F1, dan ROC-AUC**."
    )

with st.expander("4. Membagi data dan menskalakan fitur"):
    c = st.columns(3)
    c[0].metric("Data latih", fmt_int(len(res["X_tr"])), border=True)
    c[1].metric("Data uji", fmt_int(len(res["X_te"])), border=True)
    c[2].metric("Proporsi baik latih / uji", f"{fmt_pct(res['y_tr'].mean())} / {fmt_pct(res['y_te'].mean())}", border=True)
    st.markdown(
        f"Pembagian {int((1 - TEST_SIZE) * 100)}% latih dan {int(TEST_SIZE * 100)}% uji memakai "
        f"`stratify=y` agar proporsi kelas sama di kedua bagian (`random_state={SEED}`). "
        "`StandardScaler` berada **di dalam Pipeline**, jadi hanya mempelajari rata-rata dan simpangan "
        "baku dari data latih dan tidak melihat data uji."
    )
    st.code(
        "Pipeline([\n    ('scaler', StandardScaler()),\n    ('model', RandomForestClassifier(class_weight='balanced'))\n])",
        language="python")
    ringkas = pd.DataFrame({"Rata-rata (latih)": res["X_tr"].mean(), "Std (latih)": res["X_tr"].std()})
    st.dataframe(ringkas, width="stretch")

with st.expander("5. Rangkuman keputusan preprocessing"):
    st.markdown(
        f"- Id dibuang, {info['n_dup']} duplikat dihapus → {fmt_int(info['n_clean'])} baris.\n"
        "- Outlier dipertahankan.\n"
        f"- Label biner dengan batas skor {THRESHOLD_SKOR}.\n"
        "- Split berstrata dan scaling di dalam Pipeline (tanpa kebocoran data).\n"
        "- Ketidakseimbangan ditangani dengan `class_weight='balanced'` (pada model yang mendukung) "
        "dan penyetelan threshold."
    )
