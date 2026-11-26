import numpy as np
import pandas as pd
import streamlit as st

import charts as ch
from wine_core import FEATURE_COLS, FEATURES, THRESHOLD_SKOR, fmt_int, fmt_pct, load_raw

raw = load_raw()
semua_skor = sorted(raw["quality"].unique())
a_min, a_max = float(raw["alcohol"].min()), float(raw["alcohol"].max())

with st.sidebar:
    st.subheader("Filter EDA")
    skor = st.multiselect("Skor kualitas", semua_skor, default=semua_skor)
    rentang = st.slider("Rentang alkohol (% vol)", a_min, a_max, (a_min, a_max))

df = raw[raw["quality"].isin(skor) & raw["alcohol"].between(*rentang)].copy()
if df.empty:
    st.title("Eksplorasi Data (EDA)")
    st.warning("Tidak ada data yang cocok dengan filter. Ubah filter di sidebar.")
    st.stop()
df["Kelas"] = np.where(df["quality"] >= THRESHOLD_SKOR, "Baik", "Biasa")

st.title("Eksplorasi Data (EDA)")
st.caption("Pola data yang membantu menjawab: apa yang membedakan wine baik dan wine biasa?")

c = st.columns(3)
c[0].metric("Sampel terfilter", fmt_int(len(df)), border=True)
c[1].metric("Wine baik", fmt_pct((df["Kelas"] == "Baik").mean()), border=True)
c[2].metric("Rata-rata alkohol", f"{df['alcohol'].mean():.2f} % vol", border=True)

tabs = st.tabs(["Distribusi Target", "Fitur Numerik", "Peluang Baik", "Korelasi", "Hubungan Fitur", "Peta (Ilustrasi)"])

with tabs[0]:
    skor_n = df["quality"].value_counts().sort_index()
    k, d = st.columns([3, 2])
    with k:
        fig, ax = ch.fig_baru(5.2, 3.4)
        warna = [ch.BAIK if s >= THRESHOLD_SKOR else ch.BIASA for s in skor_n.index]
        bars = ax.bar(skor_n.index.astype(str), skor_n.values, color=warna)
        ax.set_ylim(0, skor_n.max() * 1.15)
        ch.label_batang(ax, bars)
        ax.set_title("Jumlah wine per skor")
        ax.set_xlabel("Skor quality")
        st.pyplot(fig)
    with d:
        n_baik = int((df["Kelas"] == "Baik").sum())
        if 0 < n_baik < len(df):
            fig, ax = ch.fig_baru(4, 3.4, grid=None)
            ch.donat(ax, [len(df) - n_baik, n_baik], ["Biasa", "Baik"], [ch.BIASA, ch.BAIK], "Proporsi kelas")
            st.pyplot(fig)
        else:
            st.info("Hanya satu kelas pada data terfilter, proporsi tidak ditampilkan.")
    st.caption("Skor 5 dan 6 mendominasi; skor 3 dan 8 sangat jarang. Kelas baik (≥ 7) adalah kelas minoritas.")

with tabs[1]:
    fitur = st.selectbox("Pilih fitur", FEATURE_COLS, index=FEATURE_COLS.index("alcohol"), key="eda_fitur_num")
    kiri, kanan = st.columns(2)
    kelas = [k for k in ("Biasa", "Baik") if (df["Kelas"] == k).any()]
    warna = {"Biasa": ch.BIASA, "Baik": ch.BAIK}
    with kiri:
        fig, ax = ch.fig_baru(5, 3.3)
        bins = np.histogram_bin_edges(df[fitur], bins=25)
        for k in kelas:
            ax.hist(df.loc[df["Kelas"] == k, fitur], bins=bins, alpha=0.6, color=warna[k], label=k)
        ax.legend(frameon=False)
        ax.set_title(f"Distribusi {fitur}")
        ax.set_xlabel(f"{fitur} ({FEATURES[fitur][1]})")
        ax.set_ylabel("Jumlah wine")
        st.pyplot(fig)
    with kanan:
        fig, ax = ch.fig_baru(4.2, 3.3)
        data = [df.loc[df["Kelas"] == k, fitur] for k in kelas]
        bp = ax.boxplot(data, patch_artist=True, tick_labels=kelas, widths=0.5,
                        medianprops=dict(color=ch.INK), flierprops=dict(markersize=3, alpha=0.5))
        for patch, k in zip(bp["boxes"], kelas):
            patch.set_facecolor(warna[k])
            patch.set_alpha(0.6)
        ax.set_title(f"{fitur} menurut kelas")
        st.pyplot(fig)
    med = df.groupby("Kelas")[fitur].median()
    if len(med) == 2:
        st.caption(f"Median {fitur}: wine baik {med['Baik']:.3g} vs wine biasa {med['Biasa']:.3g} {FEATURES[fitur][1]}.")

with tabs[2]:
    fitur = st.selectbox("Pilih fitur", FEATURE_COLS, index=FEATURE_COLS.index("alcohol"), key="eda_fitur_peluang")
    n_bin = st.slider("Jumlah kelompok (kuantil)", 3, 8, 5)
    kel = pd.qcut(df[fitur], q=n_bin, duplicates="drop")
    tab = df.groupby(kel, observed=True).agg(n=("Kelas", "size"), baik=("Kelas", lambda s: (s == "Baik").mean()))
    fig, ax = ch.fig_baru(7, 3.5)
    label = [f"{i.left:.3g} – {i.right:.3g}" for i in tab.index]
    bars = ax.bar(label, tab["baik"] * 100, color=ch.BAIK, alpha=0.85)
    rata = (df["Kelas"] == "Baik").mean() * 100
    ax.axhline(rata, color=ch.INK, linestyle="--", linewidth=1, label=f"Rata-rata {rata:.1f}%")
    top = max(tab["baik"].max() * 100, rata) * 1.2 or 1
    ax.set_ylim(0, top)
    for b, v, n in zip(bars, tab["baik"] * 100, tab["n"]):
        ax.text(b.get_x() + b.get_width() / 2, v + top * 0.015, f"{v:.1f}%\n(n={n})", ha="center", va="bottom", fontsize=8)
    ax.legend(frameon=False)
    ax.set_title(f"Persentase wine baik per kelompok {fitur}")
    ax.set_ylabel("Wine baik (%)")
    ax.set_xlabel(f"{fitur} ({FEATURES[fitur][1]})")
    st.pyplot(fig)
    with st.expander("Lihat tabel angka"):
        st.dataframe(pd.DataFrame({"Rentang": label, "Jumlah wine": tab["n"].values,
                                   "Wine baik (%)": (tab["baik"] * 100).round(1).values}),
                     width="stretch", hide_index=True)

with tabs[3]:
    if len(df) < 10:
        st.info("Data terfilter terlalu sedikit untuk menghitung korelasi.")
    else:
        cols = FEATURE_COLS + ["quality"]
        corr = df[cols].corr().fillna(0)
        kiri, kanan = st.columns([3, 2])
        with kiri:
            fig, ax = ch.plt.subplots(figsize=(6.4, 5.4))
            im = ch.heatmap(ax, corr.values, cols, cols, fs=6.5)
            fig.colorbar(im, ax=ax, fraction=0.04)
            ax.set_title("Korelasi antar fitur numerik")
            st.pyplot(fig)
        with kanan:
            kq = corr["quality"].drop("quality").sort_values()
            st.markdown("**Cara membaca:** angka mendekati +1 berarti naik bersama, mendekati -1 berarti "
                        "berlawanan arah, dan 0 berarti tidak terkait secara linear.")
            st.markdown(f"- **Terkuat positif:** `{kq.index[-1]}` ({kq.iloc[-1]:+.2f}) dan `{kq.index[-2]}` ({kq.iloc[-2]:+.2f}).\n"
                        f"- **Terkuat negatif:** `{kq.index[0]}` ({kq.iloc[0]:+.2f}) dan `{kq.index[1]}` ({kq.iloc[1]:+.2f}).")
            st.caption("Korelasi bukan sebab-akibat.")

with tabs[4]:
    a, b = st.columns(2)
    x = a.selectbox("Sumbu X", FEATURE_COLS, index=FEATURE_COLS.index("alcohol"), key="eda_x")
    y = b.selectbox("Sumbu Y", FEATURE_COLS, index=FEATURE_COLS.index("volatile acidity"), key="eda_y")
    fig, ax = ch.fig_baru(7, 3.9, grid="both")
    for k, col in (("Biasa", ch.BIASA), ("Baik", ch.BAIK)):
        s = df[df["Kelas"] == k]
        ax.scatter(s[x], s[y], s=14, alpha=0.55, color=col, label=k, edgecolor="none")
    ax.legend(frameon=False)
    ax.set_xlabel(f"{x} ({FEATURES[x][1]})")
    ax.set_ylabel(f"{y} ({FEATURES[y][1]})")
    ax.set_title(f"{y} vs {x}")
    st.pyplot(fig)

with tabs[5]:
    st.caption("Ilustrasi saja: dataset WineQT tidak memiliki kolom koordinat, sehingga titik di bawah "
               "dibuat acak di sekitar Braga, wilayah Vinho Verde, Portugal. Bukan lokasi asli sampel.")
    rng = np.random.default_rng(42)
    titik = pd.DataFrame(rng.standard_normal((len(df), 2)) / 12 + [41.5454, -8.4265], columns=["lat", "lon"])
    st.map(titik)

with st.expander("Pengamatan kunci"):
    corr_all = raw[FEATURE_COLS + ["quality"]].corr()["quality"].drop("quality").sort_values()
    baik = raw["quality"] >= THRESHOLD_SKOR
    a_tinggi = baik[raw["alcohol"] >= 12].mean()
    a_rendah = baik[raw["alcohol"] < 10].mean()
    v_tinggi = baik[raw["volatile acidity"] > 0.7].mean()
    v_rendah = baik[raw["volatile acidity"] < 0.4].mean()
    st.markdown(
        f"- Fitur paling berkaitan dengan skor: **{corr_all.index[-1]}** ({corr_all.iloc[-1]:+.2f}) dan "
        f"**{corr_all.index[0]}** ({corr_all.iloc[0]:+.2f}).\n"
        f"- Wine beralkohol ≥ 12% vol: {fmt_pct(a_tinggi)} berkualitas baik, dibanding {fmt_pct(a_rendah)} pada alkohol < 10%.\n"
        f"- Wine dengan volatile acidity > 0.7: {fmt_pct(v_tinggi)} berkualitas baik, dibanding {fmt_pct(v_rendah)} pada < 0.4.\n"
        f"- Seluruh angka di atas dihitung dari data asli (tanpa filter sidebar)."
    )
