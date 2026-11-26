"""Konfigurasi, pemuatan data, dan pipeline machine learning untuk aplikasi WineQT.

Semua angka yang tampil di aplikasi dihitung dari data di sini, jadi tidak ada
angka yang ditulis manual. Fungsi berat memakai cache agar halaman terasa cepat.
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_validate, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Identitas kelompok  (EDIT bagian ini sesuai data kelompok)
# --------------------------------------------------------------------------
KELOMPOK = "9"
ANGGOTA = [
    ("Jhon Paulus Damanik", "NIM [isi NIM]"),
    ("Guntur Adhy Nugroho", "NIM [isi NIM]"),
    ("Nasywa Hafiz Syarief", "NIM [isi NIM]"),
]
DOSEN = "[isi nama dosen pengampu]"
PRODI = "Program Studi Teknik Informatika, Fakultas Ilmu Komputer, Universitas Pamulang"

# --------------------------------------------------------------------------
# Konfigurasi data
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "WineQT.csv"
THRESHOLD_SKOR = 7      # quality >= 7  ->  wine berkualitas "Baik"
SEED = 42
TEST_SIZE = 0.2

# nama kolom: (label, satuan, keterangan, step input, format input)
FEATURES = {
    "fixed acidity": ("Asam tetap", "g/dm³", "Asam non-volatil (mis. asam tartarat) yang memberi rasa asam stabil pada wine.", 0.1, "%.1f"),
    "volatile acidity": ("Asam volatil", "g/dm³", "Kadar asam asetat; jika terlalu tinggi memberi rasa seperti cuka.", 0.01, "%.2f"),
    "citric acid": ("Asam sitrat", "g/dm³", "Memberi kesegaran dan rasa buah pada wine.", 0.01, "%.2f"),
    "residual sugar": ("Gula sisa", "g/dm³", "Gula yang tersisa setelah fermentasi; menentukan tingkat kemanisan.", 0.1, "%.1f"),
    "chlorides": ("Klorida", "g/dm³", "Kadar garam (natrium klorida) dalam wine.", 0.001, "%.3f"),
    "free sulfur dioxide": ("SO₂ bebas", "mg/dm³", "Sulfur dioksida bebas yang berfungsi sebagai antioksidan dan antimikroba.", 1.0, "%.0f"),
    "total sulfur dioxide": ("SO₂ total", "mg/dm³", "Jumlah SO₂ bebas dan terikat; kadar tinggi dapat terasa pada aroma.", 1.0, "%.0f"),
    "density": ("Massa jenis", "g/cm³", "Kepadatan wine; dipengaruhi oleh kadar alkohol dan gula.", 0.0001, "%.4f"),
    "pH": ("pH", "-", "Tingkat keasaman wine (makin kecil makin asam).", 0.01, "%.2f"),
    "sulphates": ("Sulfat", "g/dm³", "Kalium sulfat sebagai pengawet yang menguatkan kerja SO₂.", 0.01, "%.2f"),
    "alcohol": ("Alkohol", "% vol", "Kadar alkohol dalam persen volume.", 0.1, "%.1f"),
}
FEATURE_COLS = list(FEATURES)

FOOTER = f"Tugas Machine Learning - Kelompok {KELOMPOK}\nTeknik Informatika, Universitas Pamulang"


def fmt_int(n):
    return f"{int(n):,}"


def fmt_pct(x, d=1):
    return f"{x * 100:.{d}f}%"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_raw() -> pd.DataFrame:
    """Dataset asli WineQT.csv (1.143 baris, 13 kolom)."""
    return pd.read_csv(DATA_PATH)


@st.cache_data(show_spinner=False)
def load_clean():
    """Data bersih: kolom Id dibuang dan baris duplikat dihapus."""
    raw = load_raw()
    tanpa_id = raw.drop(columns="Id")
    n_dup = int(tanpa_id.duplicated().sum())
    clean = tanpa_id.drop_duplicates().reset_index(drop=True)
    info = {"n_raw": len(raw), "n_dup": n_dup, "n_clean": len(clean)}
    return clean, info


def label_biner(quality) -> pd.Series:
    return (pd.Series(quality) >= THRESHOLD_SKOR).astype(int)


def outlier_table(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    """Ringkasan skewness dan jumlah outlier (aturan IQR) tiap fitur."""
    rows = []
    for c in (cols or FEATURE_COLS):
        s = df[c]
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        rows.append({
            "Fitur": c,
            "Min": s.min(), "Median": s.median(), "Maks": s.max(),
            "Skewness": s.skew(),
            "Outlier": int(((s < lo) | (s > hi)).sum()),
            "Batas bawah": lo, "Batas atas": hi,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def make_estimators() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                random_state=SEED, n_jobs=1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
    }


MODEL_NOTES = {
    "Logistic Regression": "Model linear; memberi bobot tiap fitur. Cepat dan mudah ditafsirkan.",
    "KNN": "Menilai wine dari 7 tetangga terdekatnya di ruang fitur.",
    "Decision Tree": "Pohon keputusan tunggal (kedalaman maks. 5); aturan mudah dibaca.",
    "Random Forest": "Gabungan 300 pohon keputusan yang memilih lewat voting.",
    "Gradient Boosting": "Pohon dibangun berurutan, tiap pohon memperbaiki kesalahan sebelumnya.",
}


def hitung_metrik(y_true, proba, thr=0.5) -> dict:
    pred = (np.asarray(proba) >= thr).astype(int)
    return {
        "Akurasi": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, proba),
        "PR-AUC": average_precision_score(y_true, proba),
    }


@st.cache_resource(show_spinner="Melatih lima model, mohon tunggu sebentar...")
def train_all() -> dict:
    clean, _ = load_clean()
    X = clean[FEATURE_COLS]
    y = label_biner(clean["quality"])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)

    pipes, probas, rows = {}, {}, {}
    for name, est in make_estimators().items():
        pipe = Pipeline([("scaler", StandardScaler()), ("model", est)]).fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        pipes[name], probas[name] = pipe, proba
        rows[name] = hitung_metrik(y_te, proba)
    metrics = pd.DataFrame(rows).T
    return {
        "X": X, "y": y, "X_tr": X_tr, "X_te": X_te, "y_tr": y_tr, "y_te": y_te,
        "pipes": pipes, "probas": probas, "metrics": metrics,
        "best": metrics["F1"].idxmax(),
    }


@st.cache_resource(show_spinner="Menjalankan cross-validation 5-fold...")
def cv_results() -> pd.DataFrame:
    res = train_all()
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows = []
    for name, est in make_estimators().items():
        pipe = Pipeline([("scaler", StandardScaler()), ("model", est)])
        sc = cross_validate(pipe, res["X"], res["y"], cv=cv, n_jobs=1,
                            scoring=["f1", "recall", "precision", "roc_auc"])
        rows.append({
            "Model": name,
            "F1 (rata-rata)": sc["test_f1"].mean(), "F1 (std)": sc["test_f1"].std(),
            "Recall (rata-rata)": sc["test_recall"].mean(),
            "Precision (rata-rata)": sc["test_precision"].mean(),
            "ROC-AUC (rata-rata)": sc["test_roc_auc"].mean(), "ROC-AUC (std)": sc["test_roc_auc"].std(),
        })
    return pd.DataFrame(rows).set_index("Model")


@st.cache_resource(show_spinner="Menjalankan GridSearchCV pada Random Forest...")
def tuning_rf() -> dict:
    res = train_all()
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("model", RandomForestClassifier(class_weight="balanced",
                                                      random_state=SEED, n_jobs=1))])
    grid = {"model__n_estimators": [200, 400],
            "model__max_depth": [None, 8],
            "model__min_samples_leaf": [1, 3]}
    gs = GridSearchCV(pipe, grid, scoring="f1", n_jobs=1,
                      cv=StratifiedKFold(3, shuffle=True, random_state=SEED))
    gs.fit(res["X_tr"], res["y_tr"])
    proba = gs.best_estimator_.predict_proba(res["X_te"])[:, 1]
    tabel = pd.DataFrame(gs.cv_results_)[
        ["param_model__n_estimators", "param_model__max_depth",
         "param_model__min_samples_leaf", "mean_test_score", "rank_test_score"]]
    tabel.columns = ["n_estimators", "max_depth", "min_samples_leaf", "F1 (CV)", "Peringkat"]
    tabel["max_depth"] = tabel["max_depth"].astype(str)
    return {
        "best_params": {k.replace("model__", ""): v for k, v in gs.best_params_.items()},
        "best_cv": gs.best_score_,
        "table": tabel.sort_values("Peringkat").reset_index(drop=True),
        "metrics_tuned": hitung_metrik(res["y_te"], proba),
        "metrics_default": res["metrics"].loc["Random Forest"].to_dict(),
    }


def kurva_threshold(y_true, proba) -> pd.DataFrame:
    rows = []
    for t in np.arange(0.05, 0.96, 0.05):
        m = hitung_metrik(y_true, proba, t)
        rows.append({"Threshold": round(float(t), 2), "Precision": m["Precision"],
                     "Recall": m["Recall"], "F1": m["F1"]})
    return pd.DataFrame(rows)


@st.cache_resource(show_spinner="Menghitung pentingnya fitur...")
def importance(name: str) -> pd.DataFrame:
    res = train_all()
    r = permutation_importance(res["pipes"][name], res["X_te"], res["y_te"],
                               scoring="roc_auc", n_repeats=10, random_state=SEED, n_jobs=1)
    return (pd.DataFrame({"Fitur": FEATURE_COLS, "Penurunan ROC-AUC": r.importances_mean,
                          "Std": r.importances_std})
            .sort_values("Penurunan ROC-AUC", ascending=False).reset_index(drop=True))


# --------------------------------------------------------------------------
# PCA
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Menghitung PCA...")
def pca_info() -> dict:
    res = train_all()
    scaler = StandardScaler().fit(res["X_tr"])
    Z = scaler.transform(res["X_tr"])
    pca = PCA().fit(Z)
    ratio = pca.explained_variance_ratio_
    cum = np.cumsum(ratio)
    k90 = int(np.argmax(cum >= 0.90) + 1)
    k95 = int(np.argmax(cum >= 0.95) + 1)
    loadings = pd.DataFrame(pca.components_.T, index=FEATURE_COLS,
                            columns=[f"PC{i + 1}" for i in range(len(FEATURE_COLS))])

    # benchmark: dengan vs tanpa PCA (k90)
    bench = []
    for name in ["Logistic Regression", "KNN", "Random Forest"]:
        est = make_estimators()[name]
        for dengan in (False, True):
            steps = [("scaler", StandardScaler())]
            if dengan:
                steps.append(("pca", PCA(n_components=k90, random_state=SEED)))
            pipe = Pipeline(steps + [("model", est)]).fit(res["X_tr"], res["y_tr"])
            proba = pipe.predict_proba(res["X_te"])[:, 1]
            m = hitung_metrik(res["y_te"], proba)
            bench.append({"Model": name, "Fitur": f"PCA ({k90} komponen)" if dengan else "Tanpa PCA (11 fitur)",
                          "F1": m["F1"], "ROC-AUC": m["ROC-AUC"], "Recall": m["Recall"]})
    bench = pd.DataFrame(bench)

    # kinerja Logistic Regression vs jumlah komponen (CV pada data latih)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    kurva = []
    for k in range(1, len(FEATURE_COLS) + 1):
        pipe = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=k, random_state=SEED)),
                         ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))])
        sc = cross_validate(pipe, res["X_tr"], res["y_tr"], cv=cv, scoring="roc_auc", n_jobs=1)
        kurva.append((k, sc["test_score"].mean()))
    base = cross_validate(
        Pipeline([("scaler", StandardScaler()),
                  ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        res["X_tr"], res["y_tr"], cv=cv, scoring="roc_auc", n_jobs=1)["test_score"].mean()

    return {"ratio": ratio, "cum": cum, "k90": k90, "k95": k95, "loadings": loadings,
            "proj": Z @ pca.components_[:2].T, "y_tr": res["y_tr"].to_numpy(),
            "bench": bench, "kurva": kurva, "base_auc": base}


# --------------------------------------------------------------------------
# Prediksi
# --------------------------------------------------------------------------
def prediksi_proba(nama_model: str, X: pd.DataFrame) -> np.ndarray:
    res = train_all()
    return res["pipes"][nama_model].predict_proba(X[FEATURE_COLS])[:, 1]
