"""
Data Preparation & Pipeline Project
===================================
Individual Assignment — AI Engineering Bootcamp

Pipeline ETL sederhana untuk dataset otomotif:
    Extract   : membaca data/raw/automobileEDA_dirty_training.csv
    Transform : inspeksi -> cleaning -> transformation
    Load      : menyimpan data/processed/automobileEDA_processed.csv

Cara menjalankan:
    python src/pipeline.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Konfigurasi path & parameter
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "raw" / "automobileEDA_dirty_training.csv"
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "automobileEDA_processed.csv"

# Skema kolom yang wajib ada pada dataset mentah (untuk validasi awal).
REQUIRED_COLUMNS = [
    "transaction_date", "symboling", "normalized-losses", "make", "aspiration",
    "num-of-doors", "body-style", "drive-wheels", "engine-location", "wheel-base",
    "length", "width", "height", "curb-weight", "engine-type", "num-of-cylinders",
    "engine-size", "fuel-system", "bore", "stroke", "compression-ratio",
    "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price", "city-L/100km",
    "horsepower-binned", "diesel", "gas",
]

# Urutan kolom pada dataset hasil (mengikuti dataset referensi).
COLUMN_ORDER = [
    "symboling", "normalized-losses", "num-of-doors", "wheel-base", "length",
    "width", "height", "curb-weight", "num-of-cylinders", "engine-size", "bore",
    "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg",
    "highway-mpg", "price", "city-L/100km", "diesel", "gas", "horsepower_ordinal",
]

# Kolom yang dinormalisasi dengan Min-Max Scaling.
MINMAX_COLS = [
    "symboling",
    "curb-weight",
    "engine-size",
    "horsepower",
    "peak-rpm",
    "city-mpg",
    "highway-mpg",
    "price",
    "num-of-cylinders",
]

# Kolom kategorikal yang di-encode dengan one-hot encoding.
ONEHOT_COLS = [
    "body-style",
    "drive-wheels",
    "aspiration",
    "engine-type",
    "engine-location",
    "fuel-system",
]

# Kolom yang tidak relevan untuk pemodelan dan tidak ada di dataset referensi.
DROP_COLS = ["transaction_date"]

DOOR_MAP = {"two": 2, "four": 4}
CYLINDER_MAP = {"two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "eight": 8, "twelve": 12}

# Batas bin horsepower-binned pada dataset (diverifikasi saat inspeksi).
HP_BINS = [-np.inf, 101, 155, np.inf]
HP_LABELS = ["Low", "Medium", "High"]

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 60)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def section(title: str) -> None:
    """Cetak judul tahap agar output terminal mudah dibaca."""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def log(message: str) -> None:
    print("  - " + message)


def relative_to_base(path: Path) -> Path | str:
    """Path relatif terhadap root project; kembalikan apa adanya bila di luar project."""
    try:
        return path.relative_to(BASE_DIR)
    except ValueError:
        return path


def count_missing_effective(df: pd.DataFrame) -> pd.Series:
    """Hitung missing value secara efektif.

    Dataset ini menyamarkan nilai kosong sebagai sel berisi spasi, sehingga
    ``df.isna()`` saja akan melaporkan 0. Nilai kosong dihitung sebagai:
    NaN + sel teks yang setelah di-strip menjadi string kosong.
    """
    counts = {}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            counts[col] = int(s.isna().sum())
        else:
            text = s.astype("string").fillna("").str.strip()
            counts[col] = int((text == "").sum())
    return pd.Series(counts)


# ---------------------------------------------------------------------------
# 1. EXTRACT
# ---------------------------------------------------------------------------
def load_data(path: Path = RAW_PATH) -> pd.DataFrame:
    """Membaca dataset mentah dan memvalidasi skema kolomnya."""
    section("[1/5] LOAD DATA")
    if not path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")

    df = pd.read_csv(path)

    # Validasi skema: bandingkan nama kolom setelah di-strip.
    available = {c.strip() for c in df.columns}
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in available]
    if missing_cols:
        raise ValueError(f"Kolom wajib tidak ditemukan pada dataset: {missing_cols}")

    log(f"File     : {relative_to_base(path)}")
    log(f"Ukuran   : {df.shape[0]} baris x {df.shape[1]} kolom")
    log(f"Skema    : {len(REQUIRED_COLUMNS)} kolom wajib terpenuhi")
    return df


# ---------------------------------------------------------------------------
# 2. DATA INSPECTION
# ---------------------------------------------------------------------------
def inspect_data(df: pd.DataFrame) -> None:
    """Memeriksa kondisi awal dataset dan menampilkan temuannya."""
    section("[2/5] DATA INSPECTION")

    print("\n-- Lima baris pertama (transpose agar terbaca) --")
    print(df.head(5).T.to_string())

    # Jumlah baris dan kolom
    print("\n-- Jumlah baris dan kolom --")
    log(f"Baris : {df.shape[0]}")
    log(f"Kolom : {df.shape[1]}")

    # Tipe data tiap kolom
    print("\n-- Tipe data setiap kolom --")
    print(df.dtypes.to_string())

    # Missing value (termasuk yang disamarkan sebagai sel kosong)
    print("\n-- Missing values per kolom (termasuk sel berisi spasi) --")
    missing = count_missing_effective(df)
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        log("Tidak ada missing value.")
    else:
        for col, n in missing.items():
            log(f"{col:<20} {n:>3}  ({n / len(df):.1%})")
        log(f"TOTAL missing value: {int(missing.sum())}")

    # Duplikat
    print("\n-- Duplicate records --")
    dup_all = int(df.duplicated().sum())
    log(f"Duplikat (semua kolom)          : {dup_all}")
    log(f"Duplikat (tanpa transaction_date): "
        f"{int(df.drop(columns=DROP_COLS, errors='ignore').duplicated().sum())}")

    print("\n-- Nilai unik kolom kategorikal --")
    text_cols = []       # kolom kategorikal sesungguhnya
    numeric_as_text = [] # kolom angka yang tersimpan sebagai teks
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        stripped = df[col].astype("string").str.strip()
        filled = stripped[stripped != ""]
        if not filled.empty and pd.to_numeric(filled, errors="coerce").notna().all():
            numeric_as_text.append(col)
        else:
            text_cols.append(col)

    for col in text_cols:
        values = sorted({str(v) for v in df[col].astype("string").str.strip()
                         if v != "" and pd.notna(v)})
        sample = ", ".join(values[:8])
        more = "" if len(values) <= 8 else f", ... (+{len(values) - 8})"
        log(f"{col:<20} {len(values):>2} unik | {sample}{more}")

    # Kolom yang punya varian penulisan (huruf besar/kecil)
    case_variant_cols = []
    for col in text_cols:
        values = {str(v) for v in df[col].astype("string").str.strip()
                  if v != "" and pd.notna(v)}
        if len({v.lower() for v in values}) < len(values):
            case_variant_cols.append(col)

    print("\n-- Statistik kolom numerik --")
    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        print(numeric.describe().T[["min", "max", "mean"]].round(2).to_string())

    print("\n-- AUDIT FORMAT (masalah yang tidak terlihat dari describe) --")
    padded_cols = [c for c in df.columns if c != c.strip()]
    log(f"Kolom dengan spasi berlebih pada nama  : {len(padded_cols)} / {len(df.columns)}")

    padded_cells = 0
    blank_cells = 0
    for col in [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]:
        s = df[col].astype("string").fillna("")
        padded_cells += int((s != s.str.strip()).sum())
        blank_cells += int((s.str.strip() == "").sum())
    log(f"Sel teks dengan spasi berlebih         : {padded_cells}")
    log(f"Sel teks kosong (spasi saja = missing) : {blank_cells}")

    log(f"Kolom angka tersimpan sebagai teks     : {numeric_as_text}")

    if "transaction_date" in df.columns:
        n_formats = df["transaction_date"].dropna().astype("string").str.strip().nunique()
        log(f"transaction_date: {n_formats} nilai berbeda, format tanggal campur")
        log(f"  contoh: {[str(v).strip() for v in df['transaction_date'].dropna().head(6)]}")

    print("\n-- Kesimpulan inspeksi (dihitung dari data) --")
    log(f"1. {len(padded_cols)} dari {len(df.columns)} nama kolom memiliki spasi berlebih.")
    log(f"2. {blank_cells} sel kosong (spasi) menyamarkan missing value.")
    text_names = [c.strip() for c in numeric_as_text]
    log(f"3. Kolom angka tersimpan sebagai teks: {text_names or 'tidak ada'}.")
    log(f"4. {len(case_variant_cols)} kolom kategorikal punya penulisan tidak konsisten "
        f"({[c.strip() for c in case_variant_cols]}).")
    log(f"5. Terdapat {dup_all} baris duplikat.")
    if not numeric.empty:
        spread = (numeric.max() - numeric.min()).sort_values(ascending=False)
        widest, narrowest = spread.index[0].strip(), spread.index[-1].strip()
        log(f"6. Skala numerik belum seragam: {widest} (rentang {spread.iloc[0]:,.0f}) "
            f"vs {narrowest} (rentang {spread.iloc[-1]:,.2f}).")


# ---------------------------------------------------------------------------
# 3. DATA CLEANING
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Membersihkan dataset: nama kolom, spasi, tipe data, kategori, duplikat, missing."""
    section("[3/5] DATA CLEANING")
    df = df.copy()  # jangan pernah mengubah DataFrame milik pemanggil
    rows_before = len(df)
    missing_before = int(count_missing_effective(df).sum())

    # 3.1 Nama kolom
    padded_cols = [c for c in df.columns if c != c.strip()]
    df = df.rename(columns={c: c.strip() for c in padded_cols})
    log(f"3.1 Nama kolom dirapikan            : {len(padded_cols)} kolom")
    missing_per_col = {c: n for c, n in count_missing_effective(df).items() if n > 0}

    # 3.2 Kolom yang tidak dipakai
    existing_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=existing_drop)
    log(f"3.2 Kolom dibuang                   : {existing_drop}")

    # 3.3 Spasi berlebih pada sel teks + sel kosong -> NaN
    text_cols = list(df.select_dtypes(exclude="number").columns)
    padded_cells = 0
    for col in text_cols:
        s = df[col].astype("string")
        padded_cells += int((s.fillna("") != s.fillna("").str.strip()).sum())
        df[col] = s.str.strip().replace("", pd.NA)
    log(f"3.3 Sel teks di-strip               : {padded_cells} sel")

    # 3.4 Tipe data: teks angka -> numerik
    converted = []
    for col in text_cols:
        probe = pd.to_numeric(df[col], errors="coerce")
        if probe.notna().sum() == df[col].notna().sum():
            df[col] = probe
            converted.append(col)
    log(f"3.4 Teks dikonversi ke numerik      : {converted}")

    # 3.5 Normalisasi penulisan kategori
    cat_cols = [c for c in df.select_dtypes(exclude="number").columns]
    unique_before = {c: df[c].nunique() for c in cat_cols}
    for col in cat_cols:
        df[col] = df[col].str.strip().str.lower()
    merged = {c: (unique_before[c], df[c].nunique()) for c in cat_cols
              if unique_before[c] != df[c].nunique()}
    log("3.5 Kategori diseragamkan (unik sebelum -> sesudah):")
    for col, (before, after) in merged.items():
        log(f"      {col:<20} {before:>3} -> {after:>3}")

    # 3.6 Duplikat (dievaluasi setelah normalisasi agar varian teks ikut tergabung)
    dup_count = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    log(f"3.6 Baris duplikat dihapus          : {dup_count}")

    # 3.7 Missing values: median (numerik), modus (kategorikal)
    for col in [c for c in df.select_dtypes(include="number").columns]:
        if df[col].isna().any():
            median = df[col].median()
            df[col] = df[col].fillna(median)
            log(f"3.7 {col:<20} diisi median = {median:g}")
    for col in [c for c in df.select_dtypes(exclude="number").columns]:
        if df[col].isna().any():
            mode = df[col].mode()[0]
            df[col] = df[col].fillna(mode)
            log(f"3.7 {col:<20} diisi modus  = {mode}")

    # 3.8 Kata -> angka untuk kolom ordinal (validasi nilai tak dikenal lebih dulu)
    for col, mapping in [("num-of-doors", DOOR_MAP), ("num-of-cylinders", CYLINDER_MAP)]:
        unknown = sorted(set(df[col].dropna()) - set(mapping))
        if unknown:
            raise ValueError(f"Nilai tidak dikenal pada kolom '{col}': {unknown}")
        df[col] = df[col].map(mapping)
    log("3.8 num-of-doors & num-of-cylinders: kata -> angka "
        f"(two={DOOR_MAP['two']}, four={DOOR_MAP['four']}; "
        f"three=3 ... twelve={CYLINDER_MAP['twelve']})")

    # 3.9 Konsistensi kolom turunan horsepower-binned
    old_labels = df["horsepower-binned"].astype("string").str.capitalize()
    recomputed = pd.cut(df["horsepower"], bins=HP_BINS, labels=HP_LABELS).astype("string")
    mask = (old_labels != recomputed).fillna(True).astype(bool)
    log(f"3.9 horsepower-binned direkonsiliasi: {int(mask.sum())} baris disesuaikan")
    for _, row in df.loc[mask].head(3).iterrows():
        log(f"      {row['horsepower']:.0f} HP  "
            f"label lama '{old_labels[row.name]}' -> '{recomputed[row.name]}'")
    df["horsepower-binned"] = recomputed

    # 3.10 Tipe data final untuk kolom hasil mapping
    for col in ["num-of-doors", "num-of-cylinders"]:
        df[col] = df[col].astype("int64")
    log("3.10 num-of-doors & num-of-cylinders: int64")

    # 3.11 Ringkasan
    print("\n-- Ringkasan cleaning --")
    log(f"Baris            : {rows_before} -> {len(df)}")
    log(f"Kolom            : {df.shape[1]}")
    log(f"Missing (total)  : {missing_before} -> {int(count_missing_effective(df).sum())}")
    log(f"Kolom terdampak  : {missing_per_col}")
    return df


# ---------------------------------------------------------------------------
# 4. DATA TRANSFORMATION
# ---------------------------------------------------------------------------
def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalisasi numerik (Min-Max) dan encoding kategorikal."""
    section("[4/5] DATA TRANSFORMATION")
    df = df.copy()
    sample = df.head(5)[["price", "horsepower", "num-of-cylinders",
                         "body-style", "make"]].copy()

    # 4.1 Encoding biner
    df["num-of-doors"] = (df["num-of-doors"] >= 4).astype(int)
    log("4.1 num-of-doors -> biner (two=0, four=1)")

    # 4.2 Encoding ordinal
    ordinal_map = {"low": 0, "medium": 1, "high": 2}
    df["horsepower_ordinal"] = (
        df["horsepower-binned"].astype("string").str.lower().map(ordinal_map).astype(int)
    )
    log("4.2 horsepower-binned -> horsepower_ordinal (Low=0, Medium=1, High=2)")

    # 4.3 Encoding frekuensi untuk kolom ber-kardinalitas tinggi
    freq = df["make"].value_counts(normalize=True)
    df["make_freq"] = df["make"].map(freq)
    log(f"4.3 make ({df['make'].nunique()} kategori) -> make_freq (proporsi kemunculan)")

    df = df.drop(columns=["make", "horsepower-binned"])
    log("4.4 Kolom 'make' & 'horsepower-binned' dibuang setelah di-encode")

    # 4.5 Min-Max Scaling
    print("\n-- Min-Max Scaling --")
    for col in MINMAX_COLS:
        lo, hi = df[col].min(), df[col].max()
        if hi > lo:
            df[col] = (df[col] - lo) / (hi - lo)
            log(f"{col:<18} {lo:>9.2f} .. {hi:>9.2f}  ->  0.00 .. 1.00")
        else:
            df[col] = 0.0
            log(f"{col:<18} nilai konstan ({lo:g}) -> 0.00 (scaling dilewati)")

    # 4.6 One-hot encoding
    df = pd.get_dummies(df, columns=ONEHOT_COLS, dtype=int)
    dummies = [c for c in df.columns if c.startswith(tuple(f"{c}_" for c in ONEHOT_COLS))]
    log(f"4.6 One-hot encoding {ONEHOT_COLS} -> {len(dummies)} kolom dummy")

    # 4.7 Urutan kolom mengikuti dataset referensi
    ordered = [c for c in COLUMN_ORDER if c in df.columns]
    ordered += [c for c in dummies if c in df.columns]
    if "make_freq" in df.columns:
        ordered.append("make_freq")
    ordered += [c for c in df.columns if c not in ordered]
    df = df[ordered]

    # 4.8 Contoh before -> after
    print("\n-- Contoh nilai sebelum -> sesudah transformasi --")
    after = df.head(5)
    print(pd.DataFrame({
        "price_asli": sample["price"].values,
        "price_scaled": after["price"].round(3).values,
        "hp_asli": sample["horsepower"].values,
        "hp_scaled": after["horsepower"].round(3).values,
        "cyl_asli": sample["num-of-cylinders"].values,
        "cyl_scaled": after["num-of-cylinders"].round(3).values,
        "hp_ordinal": after["horsepower_ordinal"].values,
        "make_asli": sample["make"].values,
        "make_freq": after["make_freq"].values,
        "body_style_asli": sample["body-style"].values,
    }).to_string(index=False))

    # 4.9 Validasi hasil (raise, bukan assert, agar tidak hilang saat python -O)
    n_nan = int(df.isna().sum().sum())
    if n_nan:
        raise ValueError(f"Masih ada {n_nan} NaN setelah transformasi")
    non_numeric = list(df.select_dtypes(exclude="number").columns)
    if non_numeric:
        raise ValueError(f"Masih ada kolom non-numerik: {non_numeric}")

    print("\n-- Hasil transformasi --")
    log(f"Shape akhir    : {df.shape[0]} baris x {df.shape[1]} kolom")
    log(f"Kolom non-numerik: {non_numeric or 'tidak ada'}")
    log(f"NaN tersisa    : {int(df.isna().sum().sum())}")
    return df


# ---------------------------------------------------------------------------
# 5. LOAD (SAVE)
# ---------------------------------------------------------------------------
def save_data(df: pd.DataFrame, path: Path = PROCESSED_PATH) -> None:
    """Menyimpan processed dataset ke folder data/processed."""
    section("[5/5] SAVE DATA")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    size_kb = path.stat().st_size / 1024
    log(f"Tersimpan di : {relative_to_base(path)}")
    log(f"Ukuran file  : {size_kb:.1f} KB")
    log(f"Shape        : {df.shape[0]} baris x {df.shape[1]} kolom")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> None:
    started = time.time()
    print("DATA PREPARATION & PIPELINE PROJECT")

    raw = load_data()
    inspect_data(raw)
    clean = clean_data(raw)
    processed = transform_data(clean)
    save_data(processed)

    section("SELESAI")
    log(f"Total waktu eksekusi: {time.time() - started:.2f} detik")
    log("Pipeline berhasil dijalankan end-to-end.")


if __name__ == "__main__":
    main()
