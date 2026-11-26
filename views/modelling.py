import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

import charts as ch
from wine_core import (MODEL_NOTES, cv_results, hitung_metrik, importance, kurva_threshold, train_all,
                       tuning_rf)

res = train_all()
met = res["metrics"]
best = res["best"]
modelnya = list(met.index)
FMT = {k: st.column_config.NumberColumn(format="%.3f") for k in met.columns}

st.title("Modelling & Evaluasi")
st.caption("Lima algoritma klasifikasi dibandingkan pada data uji yang sama.")

with st.expander("Lima algoritma yang diuji"):
    st.dataframe(pd.DataFrame({"Algoritma": modelnya, "Ide singkat": [MODEL_NOTES[m] for m in modelnya]}),
                 width="stretch", hide_index=True)

b = met.loc[best]
st.success(f"Model terbaik berdasarkan F1: **{best}** (F1 {b['F1']:.3f}, Recall {b['Recall']:.3f}, "
           f"Precision {b['Precision']:.3f}, ROC-AUC {b['ROC-AUC']:.3f}).")

tabs = st.tabs(["Perbandingan", "Confusion Matrix", "Kurva ROC & PR", "Cross-validation",
                "Tuning & Threshold", "Fitur Penting"])

with tabs[0]:
    st.dataframe(met.style.highlight_max(axis=0, color="#E8CDD3"), width="stretch", column_config=FMT)
    metrik = ["Akurasi", "Precision", "Recall", "F1", "ROC-AUC"]
    fig, ax = ch.fig_baru(8, 3.6)
    x = np.arange(len(modelnya))
    w = 0.16
    for j, (mt, col) in enumerate(zip(metrik, ch.PALET)):
        ax.bar(x + (j - 2) * w, met[mt], w, label=mt, color=col)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" ", "\n") for m in modelnya], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("Perbandingan metrik pada data uji")
    ax.legend(frameon=False, ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    st.pyplot(fig)
    st.caption("Akurasi saja menyesatkan karena kelas baik minoritas. Perhatikan Recall, F1, dan ROC-AUC.")

with tabs[1]:
    nama = st.selectbox("Pilih model", modelnya, index=modelnya.index(best), key="cm_model")
    thr = st.slider("Threshold klasifikasi", 0.05, 0.95, 0.5, 0.05, key="cm_thr")
    pred = (res["probas"][nama] >= thr).astype(int)
    cm = confusion_matrix(res["y_te"], pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    kiri, kanan = st.columns([3, 2])
    with kiri:
        fig, ax = ch.plt.subplots(figsize=(4.2, 3.6))
        im = ax.imshow(cm, cmap="Reds")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                        color="white" if cm[i, j] > cm.max() / 2 else ch.INK)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Biasa", "Baik"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Biasa", "Baik"])
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        ax.set_title(f"{nama} (threshold {thr:.2f})")
        for s in ax.spines.values():
            s.set_visible(False)
        st.pyplot(fig)
    with kanan:
        st.metric("TP: baik terdeteksi benar", int(tp), border=True,
                  help="Wine baik yang diprediksi baik.")
        st.metric("FN: baik terlewat", int(fn), border=True,
                  help="Wine baik yang diprediksi biasa (terlewat).")
        st.metric("FP: biasa diprediksi baik", int(fp), border=True,
                  help="Wine biasa yang diprediksi baik (salah tebak).")
        st.metric("TN: biasa terdeteksi benar", int(tn), border=True,
                  help="Wine biasa yang diprediksi biasa.")

with tabs[2]:
    pilih = st.multiselect("Model yang ditampilkan", modelnya, default=modelnya)
    if not pilih:
        st.info("Pilih minimal satu model.")
    else:
        kiri, kanan = st.columns(2)
        fig1, a1 = ch.fig_baru(4.6, 4, grid="both")
        fig2, a2 = ch.fig_baru(4.6, 4, grid="both")
        for nama, col in zip(pilih, ch.PALET * 2):
            p = res["probas"][nama]
            fpr, tpr, _ = roc_curve(res["y_te"], p)
            pr, rc, _ = precision_recall_curve(res["y_te"], p)
            a1.plot(fpr, tpr, color=col, label=f"{nama} ({met.loc[nama, 'ROC-AUC']:.3f})")
            a2.plot(rc, pr, color=col, label=f"{nama} ({met.loc[nama, 'PR-AUC']:.3f})")
        a1.plot([0, 1], [0, 1], color=ch.MUTE, linestyle="--", linewidth=1)
        a1.set(xlabel="False positive rate", ylabel="True positive rate", title="Kurva ROC")
        a2.axhline(res["y_te"].mean(), color=ch.MUTE, linestyle="--", linewidth=1)
        a2.set(xlabel="Recall", ylabel="Precision", title="Kurva Precision-Recall")
        for a in (a1, a2):
            a.legend(frameon=False, fontsize=7)
        kiri.pyplot(fig1)
        kanan.pyplot(fig2)
        st.caption("Garis putus-putus: tebakan acak. Makin ke kiri-atas (ROC) / kanan-atas (PR), makin baik.")

with tabs[3]:
    cv = cv_results()
    st.markdown("Cross-validation 5-fold berstrata pada seluruh data bersih. "
                "Scaling dilakukan di dalam tiap fold, sehingga hasilnya tidak bocor.")
    st.dataframe(cv, width="stretch", column_config={k: st.column_config.NumberColumn(format="%.3f") for k in cv.columns})
    terbaik = cv["F1 (rata-rata)"].idxmax()
    st.caption(f"Rata-rata F1 tertinggi: {terbaik} ({cv.loc[terbaik, 'F1 (rata-rata)']:.3f} ± {cv.loc[terbaik, 'F1 (std)']:.3f}). "
               "Simpangan baku yang besar menandakan hasil yang tidak terlalu stabil antar-fold.")

with tabs[4]:
    tr = tuning_rf()
    st.subheader("Hyperparameter tuning (GridSearchCV, skor F1)")
    st.dataframe(tr["table"], width="stretch", hide_index=True,
                 column_config={"F1 (CV)": st.column_config.NumberColumn(format="%.3f")})
    bp = ", ".join(f"`{k}={v}`" for k, v in tr["best_params"].items())
    st.markdown(f"Kombinasi terbaik: {bp} (F1 CV {tr['best_cv']:.3f}).")
    banding = pd.DataFrame({"Random Forest awal": tr["metrics_default"], "Random Forest tuning": tr["metrics_tuned"]}).T
    st.dataframe(banding, width="stretch", column_config=FMT)
    st.subheader("Threshold tuning")
    nama = st.selectbox("Model", modelnya, index=modelnya.index(best), key="thr_model")
    kt = kurva_threshold(res["y_te"], res["probas"][nama])
    fig, ax = ch.fig_baru(7, 3.5, grid="both")
    for kol, col in (("Precision", ch.BIASA), ("Recall", ch.BAIK), ("F1", ch.EMAS)):
        ax.plot(kt["Threshold"], kt[kol], marker="o", markersize=3, color=col, label=kol)
    ax.set_xlabel("Threshold")
    ax.set_title(f"Sweep threshold: {nama}")
    ax.legend(frameon=False)
    st.pyplot(fig)
    best_t = kt.loc[kt["F1"].idxmax()]
    st.caption(f"F1 tertinggi pada threshold {best_t['Threshold']:.2f} (F1 {best_t['F1']:.3f}). "
               "Menurunkan threshold menaikkan Recall tetapi menurunkan Precision.")

with tabs[5]:
    nama = st.selectbox("Model", modelnya, index=modelnya.index(best), key="imp_model")
    imp = importance(nama)
    fig, ax = ch.fig_baru(7, 4, grid="x")
    d = imp.iloc[::-1]
    ax.barh(d["Fitur"], d["Penurunan ROC-AUC"], xerr=d["Std"], color=ch.BAIK, alpha=0.85,
            error_kw=dict(ecolor=ch.MUTE, lw=1))
    ax.set_xlabel("Penurunan ROC-AUC bila fitur diacak")
    ax.set_title(f"Pentingnya fitur: {nama}")
    st.pyplot(fig)
    st.caption("Permutation importance pada data uji: makin besar penurunan ROC-AUC saat sebuah fitur diacak, "
               "makin penting fitur itu bagi model. Dihitung dengan cara yang sama untuk semua model.")
