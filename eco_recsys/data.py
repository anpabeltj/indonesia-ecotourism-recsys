
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import load_npz
from typing import Tuple, Optional

@dataclass
class Artifacts:
    items: pd.DataFrame
    X: any           # scipy CSR
    vectorizer: any
    nbrs: Optional[any]

MIN_COLS = ["place_name","category","city","price","rating","place_img","place_map","combination","place_description"]

def ensure_min_columns(items: pd.DataFrame) -> None:
    for c in MIN_COLS:
        if c not in items.columns:
            items[c] = np.nan

def load_artifacts(art_dir: Path) -> Tuple[pd.DataFrame, any, any, Optional[any]]:
    items = pd.read_csv(art_dir / "items.csv")
    X     = load_npz(art_dir / "tfidf_matrix.npz").tocsr()
    try:
        vectorizer = joblib.load(art_dir / "vectorizer.joblib")
    except Exception:
        vectorizer = None
    try:
        nbrs = joblib.load(art_dir / "nbrs_cosine.joblib")
    except Exception:
        nbrs = None
    return items, X, vectorizer, nbrs
