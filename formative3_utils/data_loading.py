"""
Shared Data Loading Utilities
==============================
load_education_data  — James Mukunzi  (Part 1)
load_imdb_data       — Favor          (Part 2)

Both loaders automatically try several common filenames so the team
never needs to rename CSV files manually.

Dataset sources
---------------
Education in Africa:
    https://www.kaggle.com/datasets/lydia70/education-in-africa?select=Education+in+General.csv

IMDb Movie Reviews:
    https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
"""

import os
import numpy as np
import pandas as pd


# ── Private helper ────────────────────────────────────────────────────────────

def _resolve_filepath(requested, fallbacks):
    """
    Return the first path that exists on disk.
    Tries `requested` first, then each entry in `fallbacks`.
    Raises a helpful FileNotFoundError if none are found.
    """
    for path in [requested] + fallbacks:
        if os.path.isfile(path):
            return path

    tried = "\n  ".join([requested] + fallbacks)
    raise FileNotFoundError(
        f"Dataset not found. Tried all of:\n  {tried}\n\n"
        "Download instructions are in README.md."
    )


# ── Part 1 — Education in Africa ─────────────────────────────────────────────

_EDUCATION_CANDIDATES = [
    "data/education_africa.csv",
    "education_africa.csv",
    "data/Education in General.csv",
    "Education in General.csv",
    "data/education_in_general.csv",
    "education_in_general.csv",
]


def load_education_data(filepath="data/education_africa.csv"):
    """
    Load and clean the Education in Africa dataset.

    Automatically tries several common filenames — no manual renaming needed.

    Returns
    -------
    pd.DataFrame  Cleaned dataframe ready for variable selection.
    """
    resolved = _resolve_filepath(filepath, _EDUCATION_CANDIDATES)
    print(f"[Education] Reading: '{resolved}'")

    df = pd.read_csv(resolved)

    id_cols = [c for c in df.columns if c in ("ISO_Code", "Country")]
    numeric_cols = df.columns.difference(id_cols)
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    before = df.shape[1]
    df = df.dropna(axis=1, how="all")
    dropped = before - df.shape[1]

    print(f"[Education] {df.shape[0]} rows × {df.shape[1]} columns "
          f"({dropped} empty columns removed)")
    print(f"[Education] Columns: {list(df.columns)}\n")
    return df


def select_variable_pair(df, min_rows=50, rho_min=0.50, rho_max=0.99):
    """
    Find the strongest-correlated numeric pair with |rho| in [rho_min, rho_max].

    Avoids near-perfect correlation (|rho| >= 1) which causes division by zero
    in the bivariate normal PDF formula.

    Returns
    -------
    (col_x, col_y, rho) : tuple[str, str, float]
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr().abs()
    pairs = corr.unstack()
    pairs = pairs[pairs < 1.0]

    candidates = []
    for (c1, c2), r in pairs.items():
        if c1 >= c2:
            continue
        n = df[[c1, c2]].dropna().shape[0]
        if n >= min_rows and rho_min <= r <= rho_max:
            candidates.append((r, c1, c2, n))

    if not candidates:
        raise RuntimeError(
            "No suitable pair found. Try relaxing rho_min / rho_max or min_rows."
        )

    candidates.sort(reverse=True)
    r, col_x, col_y, n = candidates[0]
    print(f"[Education] Best pair: '{col_x}'  vs  '{col_y}'")
    print(f"            |rho| = {r:.4f}  |  usable rows = {n}\n")
    return col_x, col_y, r


# ── Part 2 — IMDb Movie Reviews ───────────────────────────────────────────────

_IMDB_CANDIDATES = [
    "data/imdb_reviews.csv",
    "imdb_reviews.csv",
    "data/IMDB Dataset.csv",
    "IMDB Dataset.csv",
    "data/IMDB_Dataset.csv",
    "IMDB_Dataset.csv",
    "data/imdb_dataset.csv",
    "imdb_dataset.csv",
]


def load_imdb_data(filepath="data/imdb_reviews.csv"):
    """
    Load and validate the IMDb 50K Movie Reviews dataset.

    Automatically tries several common filenames — no manual renaming needed.

    Returns
    -------
    pd.DataFrame  Validated dataframe with normalised lowercase sentiment labels.
    """
    resolved = _resolve_filepath(filepath, _IMDB_CANDIDATES)
    print(f"[IMDb] Reading: '{resolved}'")

    df = pd.read_csv(resolved, engine="python", on_bad_lines="skip")

    df.columns = df.columns.str.lower().str.strip()

    missing = {"review", "sentiment"} - set(df.columns)
    if missing:
        raise ValueError(
            f"[IMDb] Missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    df["sentiment"] = df["sentiment"].str.lower().str.strip()

    nulls = df[["review", "sentiment"]].isnull().sum()
    if nulls.sum() > 0:
        print(f"[IMDb] Warning — nulls found:\n{nulls}")

    print(f"[IMDb] {df.shape[0]:,} reviews loaded")
    print(df["sentiment"].value_counts().to_string())
    print()
    return df