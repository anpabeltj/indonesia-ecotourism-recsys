
import numpy as np
import pandas as pd
import math

def normalize_minmax(a):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return a
    mn, mx = np.nanmin(a), np.nanmax(a)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn:
        return np.zeros_like(a, dtype=float)
    return (a - mn) / (mx - mn)

def format_idr(x):
    if x is None or (isinstance(x, float) and (np.isnan(x) or math.isinf(x))):
        return "-"
    try:
        return "Rp{:,.0f}".format(float(x)).replace(",", ".")
    except Exception:
        return str(x)

def get_description(row: pd.Series) -> str:
    desc = row.get("place_description")
    if isinstance(desc, str) and desc.strip():
        return desc
    return str(row.get("combination") or "")
