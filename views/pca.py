import numpy as np
import pandas as pd
import streamlit as st

import charts as ch
from wine_core import FEATURE_COLS, fmt_pct, pca_info

pc = pca_info()
k90, k95 = pc["k90"], pc["k95"]
cum, ratio = pc["cum"], pc["ratio"]

st.title("Reduksi Dimensi dengan PCA")
st.caption("Meringkas 11 fitur yang saling berkorelasi menjadi komponen utama yang lebih sedikit.")

c = st.columns(4)
c[0].metric("Fitur asli", len(FEATURE_COLS), border=True)
c[1].metric("Komponen untuk 90% varians", k90, border=True)
c[2].metric("Komponen untuk 95% varians", k95, border=True)
c[3].metric("Varians PC1 + PC2", fmt_pct(cum[1]), border=True)

t1, t2, t3, t4 = st.tabs(["Scree Plot", "Interpretasi Komponen", "Benchmark Model", "Kesimpulan PCA"])

with t1:
    n = st.slider("Jumlah komponen ditampilkan", 2, len(ratio), len(ratio))
    fig, ax = ch.fig_baru(7, 3.6)
    idx = np.arange(1, n + 1)
    ax.bar(idx, ratio[:n] * 100, color=ch.BIASA, alpha=0.8, label="Varians per komponen")
    ax.plot(idx, cum[:n] * 100, color=ch.BAIK, marker="o", markersize=4, label="Kumulatif")
    for batas, ls in ((90, ":"), (95, "--")):
        ax.axhline(batas, color=ch.MUTE, linestyle=ls, linewidth=1)
        ax.text(n + 0.3, batas, f"{batas}%", va="center", fontsize=8, color=ch.MUTE)
    ax.set_xticks(idx)
    ax.set_xlabel("Komponen utama")
    ax.set_ylabel("Varians dijelaskan (%)")
    ax.set_title("Scree plot")
    ax.legend(frameon=False, loc="center right")
    st.pyplot(fig)
    st.markdown(f"Dengan **{k90} komponen**, {fmt_pct(cum[k90 - 1])} informasi data tetap terjaga; "
                f"dengan **{k95} komponen**, {fmt_pct(cum[k95 - 1])}.")

with t2:
    load = pc["loadings"]
    st.markdown("Fitur dengan *loading* terbesar (nilai mutlak) menentukan makna tiap komponen.")
    cols = st.columns(3)
    for i, col in enumerate(cols):
        nama = f"PC{i + 1}"
        top = load[nama].reindex(load[nama].abs().sort_values(ascending=False).index).head(4)
        with col.container(border=True):
            st.markdown(f"**{nama}** ({fmt_pct(ratio[i])})")
            st.dataframe(top.rename("Loading").to_frame(), width="stretch",
                         column_config={"Loading": st.column_config.NumberColumn(format="%+.3f")})
    fig, ax = ch.fig_baru(6, 4, grid="both")
    y = pc["y_tr"]
    for k, col, nm in ((0, ch.BIASA, "Biasa"), (1, ch.BAIK, "Baik")):
        ax.scatter(pc["proj"][y == k, 0], pc["proj"][y == k, 1], s=12, alpha=0.5, color=col, label=nm, edgecolor="none")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"Proyeksi 2D (varians {fmt_pct(cum[1])})")
    ax.legend(frameon=False)
    st.pyplot(fig)
    st.caption("Kelas baik cenderung bergeser ke satu sisi, tetapi masih banyak yang tumpang tindih; "
               "dua komponen saja tidak cukup untuk memisahkan kedua kelas.")

with t3:
    b = pc["bench"]
    st.markdown(f"Perbandingan kinerja pada data uji, **tanpa PCA** dan **dengan PCA ({k90} komponen)**.")
    pivot = b.pivot(index="Model", columns="Fitur", values=["F1", "ROC-AUC"])
    st.dataframe(b, width="stretch", hide_index=True,
                 column_config={k: st.column_config.NumberColumn(format="%.3f") for k in ("F1", "ROC-AUC", "Recall")})
    fig, axes = ch.plt.subplots(1, 2, figsize=(8, 3.3))
    for ax, metrik in zip(axes, ("F1", "ROC-AUC")):
        ch.gaya(ax)
        modelnya = list(dict.fromkeys(b["Model"]))
        x = np.arange(len(modelnya))
        for j, (fitur, col) in enumerate(zip(dict.fromkeys(b["Fitur"]), (ch.BIASA, ch.BAIK))):
            v = [b[(b["Model"] == m) & (b["Fitur"] == fitur)][metrik].iloc[0] for m in modelnya]
            ax.bar(x + (j - 0.5) * 0.36, v, 0.36, color=col, label=fitur)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(" ", "\n") for m in modelnya], fontsize=8)
        ax.set_title(metrik)
        ax.set_ylim(0, 1)
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    st.pyplot(fig)

    kv = pc["kurva"]
    fig, ax = ch.fig_baru(7, 3.3)
    ax.plot([k for k, _ in kv], [v for _, v in kv], marker="o", color=ch.BAIK, label="Logistic Regression + PCA")
    ax.axhline(pc["base_auc"], color=ch.MUTE, linestyle="--", label="Tanpa PCA")
    ax.axvline(k90, color=ch.BIASA, linestyle=":", label=f"k untuk 90% varians = {k90}")
    ax.set_xticks(range(1, len(FEATURE_COLS) + 1))
    ax.set_xlabel("Jumlah komponen k")
    ax.set_ylabel("ROC-AUC (CV 5-fold)")
    ax.set_title("Kinerja vs jumlah komponen")
    ax.legend(frameon=False)
    st.pyplot(fig)

with t4:
    b = pc["bench"]
    rf = b[b["Model"] == "Random Forest"].set_index("Fitur")["F1"]
    lr = b[b["Model"] == "Logistic Regression"].set_index("Fitur")["ROC-AUC"]
    st.markdown(
        f"- **Reduksi:** 11 fitur dapat diringkas menjadi {k90} komponen (90% varians) atau {k95} komponen (95%).\n"
        f"- **Logistic Regression:** ROC-AUC {lr.iloc[1]:.3f} dengan PCA vs {lr.iloc[0]:.3f} tanpa PCA.\n"
        f"- **Random Forest:** F1 {rf.iloc[1]:.3f} dengan PCA vs {rf.iloc[0]:.3f} tanpa PCA.\n"
        "- **Catatan:** dengan hanya 11 fitur, PCA tidak wajib. Manfaatnya lebih ke melihat struktur data "
        "dan mengurangi fitur yang berkorelasi, bukan menaikkan akurasi."
    )
    st.info("Pemodelan utama memakai **fitur asli** agar tiap fitur tetap bisa ditafsirkan; "
            "PCA dipakai sebagai analisis pembanding.")
