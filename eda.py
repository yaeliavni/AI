# eda.py â€" Standalone EDA + cleaning + plots for LAPD dataset
# No external project dependencies. Uses numpy/pandas/matplotlib; optional: holidays, seaborn, h3.
import os, warnings, math
from typing import Optional, Dict, Any, Tuple, List
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional libs
try:
    import seaborn as sns
    _HAS_SEABORN = True
except Exception:
    _HAS_SEABORN = False

try:
    from holidays import UnitedStates as USHolidays
    _HAS_HOLIDAYS = True
except Exception:
    _HAS_HOLIDAYS = False

try:
    from sklearn.svm import OneClassSVM
    _HAS_OCSVM = True
except Exception:
    _HAS_OCSVM = False

try:
    import h3
    _HAS_H3 = True
except Exception:
    _HAS_H3 = False

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    _HAS_OUTLIER_DETECTORS = True
except Exception:
    _HAS_OUTLIER_DETECTORS = False

try:
    from imblearn.over_sampling import SMOTE
    _HAS_SMOTE = True
except Exception:
    _HAS_SMOTE = False

try:
    from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
    _HAS_MI = True
except Exception:
    _HAS_MI = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
    from sklearn.feature_selection import f_classif
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    _HAS_VIF = True
except Exception:
    _HAS_VIF = False

# ----------------------- Config -----------------------
DEFAULT_EDA_CONFIG = dict(
    use_us_holidays = True,
    add_h3 = False,
    h3_res = 8,
    iqr_clip = True,
    iqr_multiplier = 1.5,
    zscore_winsor = True,
    zscore_max = 4.0,
    drop_any_nan_after_features = True,
    svms_outlier_clip = False,
    svms_nu = 0.01,
    svms_gamma = "scale",
    preview_rows = 10_000,
    # New defaults
    outliers_iforest = True,
    outliers_lof = True,
    outliers_ocsvm = True,
    outliers_max_frac = 0.02,
    imbalance_target = None,
    plot_corr = True,
    plot_numeric_dists = True,
    vif_check = True,
)

# ----------------------- Utility Functions -----------------------
def _ensure_dirs(fig_dir: str, cache_dir: str):
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

def _save_df(df: pd.DataFrame, path: str) -> str:
    if path.lower().endswith(".parquet"):
        df.to_parquet(path, index=False)
    elif path.lower().endswith(".feather"):
        df.to_feather(path)
    else:
        df.to_csv(path, index=False)
    return path

def _plot_and_save(title: str, fig_dir: str):
    safe = (
        title.replace(' ', '-')
             .replace('×', 'x')   # real multiply sign
             .replace('Ã—', 'x')  # mojibake variant just in case
    )
    out = os.path.join(fig_dir, f"{safe}.png")
    try:
        plt.tight_layout()
        plt.savefig(out, dpi=160)
        print(f"🗂️ saved: {out}")
    except Exception as e:
        print("(skip save)", e)
    plt.show()


# ----------------------- Date/Time Functions -----------------------
def _parse_time_occ(s):
    """LAPD TIME OCC is often integer like 30, 930, 2359. Return (hour, minute) or (0,0) on fail."""
    try:
        if pd.isna(s): return 0, 0
        s = str(int(s)).zfill(4)  # "30" -> "0030", "930" -> "0930", "2359" -> "2359"
        hh = int(s[:2]); mm = int(s[2:])
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            return hh, mm
        return 0, 0
    except Exception:
        return 0, 0
from typing import Tuple  # (if not already imported at the top)

def infer_datetime_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Try to detect or construct a datetime column.
    Returns (df_with_dt, dt_col_name).
    Looks for 'date_dt' first, then common date columns.
    """
    # Already standardized by choose_best_date_column
    if "date_dt" in df.columns:
        return df, "date_dt"

    # Fallback: look for common date-like cols
    candidates = [
        "date", "date_occ", "date_occurred", "date_rptd",
        "DATE OCC", "Date OCC", "DATE", "Date", "DATE RPTD"
    ]
    for c in candidates:
        if c in df.columns:
            try:
                df[c] = pd.to_datetime(df[c], errors="coerce")
                if df[c].notna().mean() > 0.5:
                    return df, c
            except Exception:
                continue

    return df, None
def _parse_any_date(series):
    """Try multiple common formats and pandas fallback; return pd.Series[datetime64[ns]] with NaT on fail."""
    if np.issubdtype(series.dtype, np.datetime64):
        return pd.to_datetime(series, errors="coerce")
    
    if series.dtype == object and series.dropna().apply(lambda x: hasattr(x, "year")).all():
        return pd.to_datetime(series, errors="coerce")
    
    if pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series):
        arr = pd.to_numeric(series, errors='coerce')
        dt = pd.to_datetime(arr, unit='s', errors='coerce')
        # if most dates are around 1970, try ms
        if dt.notna().mean() < 0.5 or (dt.dropna().dt.year <= 1971).mean() > 0.8:
            dt_ms = pd.to_datetime(arr, unit='ms', errors='coerce')
            if dt_ms.notna().mean() > dt.notna().mean():
                dt = dt_ms
        return dt
    
    # Fast path: let pandas try
    s = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
    # If that did poorly (<80% valid), try manual formats
    valid_ratio = s.notna().mean()
    if valid_ratio >= 0.8:
        return s

    # manual attempts
    fmts = [
        "%m/%d/%Y %I:%M:%S %p",  # 12h with AM/PM
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%m/%d/%y",
        "%d/%m/%Y",
    ]
    best = s.copy()
    for fmt in fmts:
        try:
            t = pd.to_datetime(series, format=fmt, errors="coerce")
            if t.notna().mean() > best.notna().mean():
                best = t
        except Exception:
            pass
    return best

def _combine_date_time(df: pd.DataFrame, date_col: str, time_col: str) -> pd.Series:
    """Combine separate date and time columns into a single datetime."""
    d = _parse_any_date(df[date_col])
    t = df[time_col]
    # TIME OCC is often numeric HHMM
    if pd.api.types.is_numeric_dtype(t):
        t = t.fillna(0).astype(int).astype(str).str.zfill(4)
    else:
        t = t.astype(str).str.replace(r'[^0-9]', '', regex=True).str.zfill(4)
    hh = pd.to_numeric(t.str[:2], errors='coerce').fillna(0).clip(0, 23).astype(int)
    mm = pd.to_numeric(t.str[2:4], errors='coerce').fillna(0).clip(0, 59).astype(int)
    return pd.to_datetime(d + pd.to_timedelta(hh, unit='h') + pd.to_timedelta(mm, unit='m'), errors='coerce')

def choose_best_date_column(df: pd.DataFrame) -> pd.Series:
    """
    Search multiple candidate columns, pick the one with the highest fraction of
    valid dates in 2000–today
    """
    preferred = [
        "DATE OCC", "Date OCC", "date_occ", "date_occurred", "Date_occured", "Date_occurred", "Date Occured", "Date Occurred",
        "DATE RPTD", "Date Rptd", "date_rptd", "Date_Reported", "Date Reported",
        "DATE", "Date", "date", "date_only"
    ]
    parsed = {}
    for col in preferred:
        if col in df.columns:
            parsed[col] = _parse_any_date(df[col])

    # Add combined date/time column if present
    for dcol in ["DATE OCC", "Date OCC", "date_occ"]:
        for tcol in ["TIME OCC", "Time OCC", "time_occ"]:
            if dcol in df.columns and tcol in df.columns:
                parsed[f"{dcol}+{tcol}"] = _combine_date_time(df, dcol, tcol)

    # Fallback: parse any other string-like column
    if not parsed:
        for col in df.columns:
            s = df[col]
            if s.notna().any():
                p = _parse_any_date(s)
                if p.notna().mean() > 0.5:
                    parsed[col] = p

    def score(series: pd.Series) -> float:
        """Score by valid ratio Ã— (share in [2000, today]) minus penalty for single-date dominance."""
        if series is None:
            return 0.0
        valid_rate = series.notna().mean()
        if valid_rate == 0:
            return 0.0
        s = series.dropna()
        good = s.between(pd.Timestamp("2000-01-01"), pd.Timestamp.now()).mean()
        top_share = s.dt.date.value_counts(normalize=True).max()
        return valid_rate * (0.5 + 0.5 * good) - (0.5 if top_share > 0.8 else 0.0)

    best_key, best_series, best_score = None, None, 0
    for key, series in parsed.items():
        sc = score(series)
        if sc > best_score:
            best_key, best_series, best_score = key, series, sc

    if best_series is None:
        raise ValueError("No suitable date column found")
    print(f"Using date column: {best_key}")
    return best_series



def add_time_features(df: pd.DataFrame, dt_col: str = "dt_occ", use_us_holidays: bool = True) -> pd.DataFrame:
    d = df.copy()
    s = pd.to_datetime(d[dt_col], errors="coerce")
    d["year"] = s.dt.year
    d["month"] = s.dt.month
    d["weekday"] = s.dt.weekday
    d["hour"] = s.dt.hour
    d["is_weekend"] = d["weekday"].isin([5,6]).astype(int)
    if use_us_holidays and _HAS_HOLIDAYS:
        hs = USHolidays()
        d["is_holiday"] = s.dt.date.astype("datetime64[ns]").dt.date.map(lambda x: int(x in hs))
    else:
        d["is_holiday"] = 0
    return d

# ----------------------- Data Cleaning Functions -----------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
                 .str.replace(r'[^0-9a-zA-Z]+', '_', regex=True)
                 .str.lower()
    )
    return df

def validate_and_clean_coords(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    issues = []
    d = df.copy()
    lat_col = next((c for c in ["lat","latitude","latit","lat_num"] if c in d), None)
    lon_col = next((c for c in ["lon","longitude","long","lng","lon_num"] if c in d), None)
    if lat_col and lon_col:
        zz = (d[lat_col].fillna(0)==0) & (d[lon_col].fillna(0)==0)
        if zz.any(): issues.append(("zero_zero_coords", int(zz.sum())))
        # Los Angeles rough bbox
        bbox = (~((d[lat_col].between(33.3, 34.9)) & (d[lon_col].between(-119.1, -117.5))))
        if bbox.any(): issues.append(("outside_LA_bbox", int(bbox.sum())))
    issues_df = pd.DataFrame(issues, columns=["issue","count"]) if issues else pd.DataFrame(columns=["issue","count"])
    return d, issues_df

def add_h3_index(df: pd.DataFrame, h3_res: int = 8) -> pd.DataFrame:
    if not _HAS_H3:
        raise RuntimeError("h3 not installed")
    d = df.copy()
    lat_col = next((c for c in ["lat","latitude","latit","lat_num"] if c in d), None)
    lon_col = next((c for c in ["lon","longitude","long","lng","lon_num"] if c in d), None)
    if not lat_col or not lon_col: return d
    def _geo_to_h3(r):
        if pd.notna(r[lat_col]) and pd.notna(r[lon_col]):
            return h3.geo_to_h3(float(r[lat_col]), float(r[lon_col]), h3_res)
        return np.nan
    d["h3"] = d.apply(_geo_to_h3, axis=1)
    return d

def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)

def downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c in d.select_dtypes(include=[np.number]).columns:
        d[c] = pd.to_numeric(d[c], downcast="integer")
        d[c] = pd.to_numeric(d[c], downcast="float")
    return d

def iqr_outlier_clip(df: pd.DataFrame, num_cols: List[str], whisker: float = 1.5) -> pd.DataFrame:
    d = df.copy()
    for c in num_cols:
        x = d[c].astype(float)
        q1 = x.quantile(0.25); q3 = x.quantile(0.75); iqr = q3-q1
        lo = q1 - whisker*iqr; hi = q3 + whisker*iqr
        d[c] = x.clip(lo, hi)
    return d

def _zscore_winsorize(df: pd.DataFrame, cols, zmax: float) -> pd.DataFrame:
    d = df.copy()
    for c in cols:
        x = d[c].astype(float)
        mu = x.mean(); sd = x.std(ddof=0) or 1.0
        z = (x - mu)/sd
        z = z.clip(-zmax, zmax)
        d[c] = z*sd + mu
    return d

def clean_then_dropna_all(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(axis=0, how="any").reset_index(drop=True)

# ----------------------- Outlier Detection Functions -----------------------
def _cap_frac(mask: np.ndarray, max_frac: float) -> np.ndarray:
    # Limit removals to a cap (protects against overly aggressive detectors)
    if max_frac is None or max_frac <= 0: 
        return mask
    n = len(mask)
    bad_idx = np.where(~mask)[0]
    cap = int(max_frac * n)
    if len(bad_idx) > cap:
        keep_back = bad_idx[:len(bad_idx)-cap]
        mask[keep_back] = True
    return mask

def _iforest_outlier_filter(df: pd.DataFrame, cols, max_frac=0.02, random_state=42):
    if not cols or not _HAS_OUTLIER_DETECTORS: 
        return df, np.ones(len(df), bool)
    X = df[cols].astype(float).values
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    # contamination ~ expected outlier fraction; use cap as upper bound
    contamination = min(max_frac or 0.02, 0.1)
    mdl = IsolationForest(
        n_estimators=200, max_samples="auto", contamination=contamination, 
        random_state=random_state, n_jobs=-1
    )
    y = mdl.fit_predict(X)   # 1 inlier, -1 outlier
    mask = (y == 1)
    mask = _cap_frac(mask, max_frac)
    print(f"[IForest] flagged {(~mask).sum():,} / {len(mask):,} rows")
    return df.loc[mask].copy(), mask

def _lof_outlier_filter(df: pd.DataFrame, cols, max_frac=0.02):
    if not cols or not _HAS_OUTLIER_DETECTORS: 
        return df, np.ones(len(df), bool)
    X = df[cols].astype(float).values
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    # novelty=False allows fit_predict on the dataset (for EDA filtering)
    lof = LocalOutlierFactor(n_neighbors=20, contamination=max_frac or 0.02, novelty=False, n_jobs=-1)
    y = lof.fit_predict(X)   # 1 inlier, -1 outlier
    mask = (y == 1)
    mask = _cap_frac(mask, max_frac)
    print(f"[LOF] flagged {(~mask).sum():,} / {len(mask):,} rows")
    return df.loc[mask].copy(), mask

def _svm_outlier_filter(df: pd.DataFrame, cols, nu=0.01, gamma="scale"):
    if not _HAS_OCSVM or not cols:
        return df, np.ones(len(df), dtype=bool)
    X = df[cols].astype(float).values
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    oc = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
    y = oc.fit_predict(X)  # +1 inliers, -1 outliers
    mask = (y == 1)
    print(f"[OCSVM] flagged {(~mask).sum():,} / {len(mask):,} rows as outliers (nu≈{nu})")
    return df.loc[mask].copy(), mask

# ----------------------- Sampling Functions -----------------------
def _make_stratified_sample(df: pd.DataFrame, target_col: str, n: int, random_state: int) -> pd.DataFrame:
    """Use StratifiedShuffleSplit to select about n rows, preserving class proportions."""
    if n >= len(df) or not _HAS_SKLEARN:
        print(f"[EDA Runner] Requested sample size ({n}) >= dataset size ({len(df)}); using full data.")
        return df.copy()

    # Avoid NaNs in y
    y = df[target_col].astype(str).fillna("__NA__")
    frac = n / len(df)
    frac = max(min(frac, 0.9), 1e-6)  # keep in (0,1)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=frac, random_state=random_state)
    for _, test_idx in sss.split(np.zeros(len(df)), y):  # indices as X, y as labels
        sample = df.iloc[test_idx].copy()
        break

    # Trim if slightly over n
    if len(sample) > n:
        sample = sample.sample(n=n, random_state=random_state)
    print(f"[EDA Runner] Stratified sample using '{target_col}': {len(sample):,} rows.")
    return sample

def _make_random_sample(df: pd.DataFrame, n: int, random_state: int) -> pd.DataFrame:
    if n >= len(df):
        return df.copy()
    sample = df.sample(n=n, random_state=random_state)
    print(f"[EDA Runner] Random sample (fallback): {len(sample):,} rows.")
    return sample

def stratified_sample_for_task(df: pd.DataFrame, target_col: str, n: int, random_state: int = 42) -> pd.DataFrame:
    if target_col not in df: return df.sample(min(n, len(df)), random_state=random_state)
    groups = df[target_col].dropna().astype(str)
    frac = min(1.0, n/len(df))
    sampled = df.groupby(groups, group_keys=False).apply(lambda g: g.sample(max(1,int(len(g)*frac)), random_state=random_state))
    return sampled.reset_index(drop=True)

# ----------------------- Analysis Functions -----------------------
def vif_table(df: pd.DataFrame, max_vars: int = 30) -> pd.DataFrame:
    if not _HAS_VIF:
        return pd.DataFrame(columns=["feature","VIF"])
    num = df.select_dtypes(include=[np.number]).dropna().copy()
    if len(num.columns) == 0: 
        return pd.DataFrame(columns=["feature","VIF"])
    cols = num.columns[:max_vars]
    X = num[cols].astype(float).values
    v = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    out = pd.DataFrame({"feature": cols, "VIF": v}).sort_values("VIF", ascending=False)
    return out

def _balanced_preview(df: pd.DataFrame, target: str, method: str = "smote", max_n: int = 20000, random_state: int = 42) -> Optional[pd.DataFrame]:
    """
    Return a balanced *preview* DataFrame for EDA/modeling plots only.
    Does NOT alter the main cleaned dataset.
    """
    if target not in df.columns or not _HAS_SKLEARN:
        return None
    d = df.dropna(subset=[target]).copy()
    if len(d) == 0:
        return None

    # Choose numeric features for SMOTE; if not possible, fall back to undersample
    if method.lower() == "smote" and _HAS_SMOTE:
        num_cols = d.select_dtypes(include=[np.number]).columns.tolist()
        X = d[num_cols].fillna(d[num_cols].median()) if num_cols else None
        y = d[target].astype(str)
        if X is None or X.empty:
            method = "undersample"  # fallback
        else:
            # downsize for speed
            if len(d) > max_n:
                X, _, y, _ = train_test_split(X, y, train_size=max_n, stratify=y, random_state=random_state)
            try:
                Xr, yr = SMOTE(random_state=random_state).fit_resample(X, y)
                out = pd.concat([pd.DataFrame(Xr, columns=num_cols), yr.rename(target)], axis=1)
                print(f"[Balance] SMOTE preview: {len(out):,} rows.")
                return out
            except Exception as e:
                print("SMOTE failed; falling back to undersample:", e)
                method = "undersample"

    # Simple stratified undersample to the size of the minority class
    vc = d[target].astype(str).value_counts()
    if len(vc) < 2:
        print("[Balance] target has <2 classes; skip.")
        return None
    n_min = vc.min()
    parts = []
    for cls, n in vc.items():
        take = min(n_min, n)
        parts.append(d[d[target].astype(str) == cls].sample(n=take, random_state=random_state))
    out = pd.concat(parts, axis=0).sample(frac=1, random_state=random_state).reset_index(drop=True)
    if len(out) > max_n:
        out = out.sample(n=max_n, random_state=random_state)
    print(f"[Balance] Undersample preview: {len(out):,} rows.")
    return out
#add to eda.py
# === eda.py — ADD THESE HELPERS ==============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm

# ---------- Safety guards (avoid blank/weird plots) ----------
def _has_variation(series: pd.Series, min_unique: int = 2) -> bool:
    if series is None:
        return False
    s = pd.Series(series).dropna()
    return s.nunique() >= min_unique
def safe_plot_hist(series, ax, bins=30):
    s = series.dropna()
    if s.nunique() <= 1:
        ax.text(0.5, 0.5, "No variation", ha="center")
        return False
    ax.hist(s, bins=bins, alpha=0.8)
    return True

def safe_hist(ax, series: pd.Series, bins=30, title: str = ""):
    if _has_variation(series):
        ax.hist(series.dropna(), bins=bins, alpha=0.85)
        ax.set_title(title or series.name)
        return True
    return False  # caller may skip layout/save if False

def safe_corr_heatmap(df: pd.DataFrame, title: str = "Correlation (Pearson)"):
    num = df.select_dtypes(include=[np.number]).copy()
    if num.shape[1] < 2:
        print("↪  Skipping correlation heatmap (need ≥2 numeric columns).")
        return False
    corr = num.corr(method="pearson").replace([np.inf, -np.inf], np.nan).dropna(how="all").dropna(axis=1, how="all")
    if corr.shape[0] < 2 or corr.shape[1] < 2:
        print("↪  Skipping correlation heatmap (degenerate after cleaning).")
        return False
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, cmap="vlag", center=0, linewidths=0.5)
    plt.title(title)
    plt.tight_layout()
    return True
# Improved visualization functions to fix the odd-looking graphs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
import os

def plot_calendar_heatmap(df, date_col="date", value_col=None, title="Calendar Heatmap", out_path=None):
    """Create a proper calendar-style heatmap showing daily crime counts"""
    try:
        # Prepare daily counts if value_col not provided
        if value_col is None:
            daily_counts = df.groupby(df[date_col].dt.date).size().reset_index(name='count')
            daily_counts.columns = ['date', 'count']
            daily_counts['date'] = pd.to_datetime(daily_counts['date'])
            value_col = 'count'
        else:
            daily_counts = df[[date_col, value_col]].copy()
            daily_counts[date_col] = pd.to_datetime(daily_counts[date_col])
        
        # Create calendar features
        daily_counts['year'] = daily_counts[date_col].dt.year
        daily_counts['month'] = daily_counts[date_col].dt.month
        daily_counts['day'] = daily_counts[date_col].dt.day
        daily_counts['weekday'] = daily_counts[date_col].dt.weekday
        daily_counts['week_of_year'] = daily_counts[date_col].dt.isocalendar().week
        
        # Group by year and create subplots
        years = sorted(daily_counts['year'].unique())
        
        if len(years) == 1:
            # Single year - create weekly calendar
            year_data = daily_counts[daily_counts['year'] == years[0]]
            pivot = year_data.pivot_table(
                index='week_of_year', 
                columns='weekday', 
                values=value_col, 
                aggfunc='sum',
                fill_value=0
            )
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot, cmap='YlOrRd', cbar_kws={'label': 'Crime Count'}, 
                       annot=False, fmt='d', linewidths=0.1)
            plt.title(f'{title} - {years[0]}')
            plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
            plt.ylabel('Week of Year')
            
        else:
            # Multiple years - create year comparison
            fig, axes = plt.subplots(len(years), 1, figsize=(12, 4*len(years)))
            if len(years) == 1:
                axes = [axes]
            
            for i, year in enumerate(years):
                year_data = daily_counts[daily_counts['year'] == year]
                pivot = year_data.pivot_table(
                    index='week_of_year', 
                    columns='weekday', 
                    values=value_col, 
                    aggfunc='sum',
                    fill_value=0
                )
                
                sns.heatmap(pivot, ax=axes[i], cmap='YlOrRd', 
                           cbar_kws={'label': 'Crime Count'}, 
                           annot=False, linewidths=0.1)
                axes[i].set_title(f'{year}')
                axes[i].set_xlabel('Day of Week (0=Monday, 6=Sunday)')
                axes[i].set_ylabel('Week of Year')
            
            plt.suptitle(title, y=1.02)
        
        plt.tight_layout()
        
        if out_path:
            plt.savefig(out_path, dpi=160, bbox_inches='tight')
            print(f"Calendar heatmap saved to: {out_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"Calendar heatmap failed: {e}")

def plot_spatial_density_improved(df, lat_col=None, lon_col=None, out_prefix=None):
    """Create improved spatial density plots with better auto-detection and filtering"""
    try:
        # Auto-detect lat/lon columns with better candidates
        lat_candidates = ['LAT', 'Latitude', 'LATITUDE', 'lat', 'Y', 'y', 'lat_num']
        lon_candidates = ['LON', 'Longitude', 'LONGITUDE', 'lon', 'X', 'x', 'lon_num', 'lng']
        
        if lat_col is None:
            lat_col = next((col for col in lat_candidates if col in df.columns), None)
        if lon_col is None:
            lon_col = next((col for col in lon_candidates if col in df.columns), None)
        
        if lat_col is None or lon_col is None:
            print("Spatial density skipped: lat/lon columns not found")
            return
        
        # Filter valid coordinates for Los Angeles area specifically
        spatial_df = df[[lat_col, lon_col]].dropna()
        
        # More restrictive LA bounds
        la_bounds = {
            'lat_min': 33.7, 'lat_max': 34.3,
            'lon_min': -118.7, 'lon_max': -118.1
        }
        
        spatial_df = spatial_df[
            (spatial_df[lat_col].between(la_bounds['lat_min'], la_bounds['lat_max'])) & 
            (spatial_df[lon_col].between(la_bounds['lon_min'], la_bounds['lon_max']))
        ]
        
        if len(spatial_df) < 100:
            print("Spatial density skipped: insufficient valid coordinates")
            return
        
        # Remove obvious outliers (0,0 coordinates)
        spatial_df = spatial_df[~((spatial_df[lat_col] == 0) & (spatial_df[lon_col] == 0))]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # 1. Hexbin plot with better parameters
        hb = ax1.hexbin(spatial_df[lon_col], spatial_df[lat_col], 
                       gridsize=40, cmap='inferno', mincnt=1, alpha=0.8)
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.set_title('Crime Density (Hexbin)')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(hb, ax=ax1, label='Crime Count')
        
        # 2. Scatter plot with density coloring (sample if too many points)
        sample_size = min(10000, len(spatial_df))
        spatial_sample = spatial_df.sample(n=sample_size, random_state=42)
        
        scatter = ax2.scatter(spatial_sample[lon_col], spatial_sample[lat_col], 
                            s=1, alpha=0.5, c='red')
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        ax2.set_title(f'Crime Locations (Sample: {sample_size:,} points)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if out_prefix:
            plt.savefig(f"{out_prefix}_improved.png", dpi=160, bbox_inches='tight')
            print(f"Improved spatial plot saved to: {out_prefix}_improved.png")
        plt.show()
        
        # Print some statistics
        print(f"Spatial analysis summary:")
        print(f"- Valid coordinates: {len(spatial_df):,}")
        print(f"- Latitude range: {spatial_df[lat_col].min():.3f} to {spatial_df[lat_col].max():.3f}")
        print(f"- Longitude range: {spatial_df[lon_col].min():.3f} to {spatial_df[lon_col].max():.3f}")
        
    except Exception as e:
        print(f"Spatial density plotting failed: {e}")

def enhanced_ml_analysis_fixed(df_clean, target_col='is_night', output_dir=None):
    """Fixed version of ML analysis with better error handling and meaningful plots"""
    try:
        if target_col not in df_clean.columns:
            print(f"Target column '{target_col}' not found")
            return
        
        # Check if we have any meaningful features
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        # Filter columns with meaningful variation
        meaningful_features = []
        for col in numeric_cols:
            if df_clean[col].nunique() > 1:
                cv = df_clean[col].std() / (abs(df_clean[col].mean()) + 1e-8)
                if cv > 0.01:  # Coefficient of variation > 1%
                    meaningful_features.append(col)
        
        if len(meaningful_features) < 2:
            print(f"Insufficient meaningful features found: {len(meaningful_features)}")
            return
        
        print(f"Using {len(meaningful_features)} meaningful features")
        
        # Prepare data
        X = df_clean[meaningful_features].copy()
        y = df_clean[target_col].copy()
        
        # Basic preprocessing
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import mutual_info_classif
        from sklearn.inspection import permutation_importance
        from sklearn.model_selection import train_test_split
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        X_processed = pd.DataFrame(X_scaled, columns=meaningful_features)
        
        # 1. Mutual Information Analysis
        try:
            mi_scores = mutual_info_classif(X_processed, y, random_state=42)
            mi_results = pd.Series(mi_scores, index=meaningful_features).sort_values(ascending=False)
            
            # Only plot if we have meaningful scores
            mi_results = mi_results[mi_results > 1e-6]
            
            if len(mi_results) > 0:
                plt.figure(figsize=(10, max(6, len(mi_results)*0.4)))
                top_mi = mi_results.head(15)
                
                # Horizontal bar plot
                plt.barh(range(len(top_mi)), top_mi.values)
                plt.yticks(range(len(top_mi)), top_mi.index)
                plt.xlabel('Mutual Information Score')
                plt.title(f'Feature Mutual Information with {target_col}')
                plt.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                
                if output_dir:
                    plt.savefig(os.path.join(output_dir, 'mutual_info_fixed.png'), 
                               dpi=160, bbox_inches='tight')
                plt.show()
                
                print(f"Top MI features: {dict(top_mi.head(5))}")
            
        except Exception as e:
            print(f"Mutual information failed: {e}")
        
        # 2. Random Forest Feature Importance
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, test_size=0.3, stratify=y, random_state=42
            )
            
            # Train RF with reasonable parameters
            rf = RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            
            rf.fit(X_train, y_train)
            
            # Get feature importance
            importance = pd.Series(rf.feature_importances_, 
                                 index=meaningful_features).sort_values(ascending=False)
            
            if importance.max() > 1e-6:
                plt.figure(figsize=(10, max(6, len(importance)*0.4)))
                top_importance = importance.head(15)
                
                plt.barh(range(len(top_importance)), top_importance.values)
                plt.yticks(range(len(top_importance)), top_importance.index)
                plt.xlabel('Feature Importance')
                plt.title(f'Random Forest Feature Importance')
                plt.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                
                if output_dir:
                    plt.savefig(os.path.join(output_dir, 'rf_importance_fixed.png'), 
                               dpi=160, bbox_inches='tight')
                plt.show()
                
                # Model performance
                train_score = rf.score(X_train, y_train)
                test_score = rf.score(X_test, y_test)
                print(f"RF Training Accuracy: {train_score:.3f}")
                print(f"RF Testing Accuracy: {test_score:.3f}")
                print(f"Top RF features: {dict(top_importance.head(5))}")
            
        except Exception as e:
            print(f"Random Forest analysis failed: {e}")
        
        # 3. Permutation Importance (only if RF worked)
        try:
            if 'rf' in locals() and 'X_test' in locals():
                perm_importance = permutation_importance(
                    rf, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1
                )
                
                perm_results = pd.Series(
                    perm_importance.importances_mean, 
                    index=meaningful_features
                ).sort_values(ascending=False)
                
                perm_results = perm_results[perm_results > 1e-6]
                
                if len(perm_results) > 0:
                    plt.figure(figsize=(10, max(6, len(perm_results)*0.4)))
                    top_perm = perm_results.head(15)
                    
                    # Add error bars
                    top_indices = [meaningful_features.index(feat) for feat in top_perm.index]
                    errors = perm_importance.importances_std[top_indices]
                    
                    plt.barh(range(len(top_perm)), top_perm.values, xerr=errors, alpha=0.7)
                    plt.yticks(range(len(top_perm)), top_perm.index)
                    plt.xlabel('Permutation Importance')
                    plt.title(f'Permutation Importance (with error bars)')
                    plt.grid(axis='x', alpha=0.3)
                    plt.tight_layout()
                    
                    if output_dir:
                        plt.savefig(os.path.join(output_dir, 'permutation_importance_fixed.png'), 
                                   dpi=160, bbox_inches='tight')
                    plt.show()
                    
                    print(f"Top Permutation features: {dict(top_perm.head(5))}")
                
        except Exception as e:
            print(f"Permutation importance failed: {e}")
        
    except Exception as e:
        print(f"Enhanced ML analysis failed: {e}")

def fix_histogram_plots(df, output_dir, max_cols=16):
    """Create better histogram plots with proper binning and layout"""
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove columns with no variation or too few unique values
        good_cols = []
        for col in numeric_cols:
            if df[col].nunique() > 5:  # At least 5 unique values
                good_cols.append(col)
        
        if not good_cols:
            print("No suitable numeric columns for histograms")
            return
        
        # Limit number of plots
        good_cols = good_cols[:max_cols]
        
        # Calculate subplot layout
        n_cols = 4
        n_rows = (len(good_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
        
        for i, col in enumerate(good_cols):
            ax = axes[i]
            data = df[col].dropna()
            
            if len(data) > 0:
                # Determine appropriate number of bins
                n_bins = min(50, max(10, int(np.sqrt(len(data)))))
                
                # Plot histogram
                ax.hist(data, bins=n_bins, alpha=0.7, edgecolor='black', linewidth=0.5)
                ax.set_title(f'{col}', fontsize=12)
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
                ax.grid(axis='y', alpha=0.3)
                
                # Add statistics text
                mean_val = data.mean()
                std_val = data.std()
                ax.text(0.02, 0.98, f'μ={mean_val:.2f}\nσ={std_val:.2f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide empty subplots
        for i in range(len(good_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Improved Numeric Distributions', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'histograms_improved.png'), 
                       dpi=160, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"Histogram plotting failed: {e}")

# Function to replace the problematic functions in your main script
def replace_problematic_plots(df_clean, figs_dir):
    """Replace the problematic plots with improved versions"""
    
    print("Creating improved calendar heatmap...")
    if 'date' in df_clean.columns:
        plot_calendar_heatmap(df_clean, date_col='date', 
                            out_path=os.path.join(figs_dir, "calendar_heatmap_improved.png"))
    
    print("Creating improved spatial plots...")
    plot_spatial_density_improved(df_clean, out_prefix=os.path.join(figs_dir, "spatial_density"))
    
    print("Creating improved ML analysis...")
    enhanced_ml_analysis_fixed(df_clean, target_col='is_night', output_dir=figs_dir)
    
    print("Creating improved histograms...")
    fix_histogram_plots(df_clean, output_dir=figs_dir)
    
    plt.close('all')  # Clean up any remaining plots
# ---------- Calendar heatmap (month-grid style, no extra deps) ----------
def plot_calendar_heatmap_monthgrid(df: pd.DataFrame, date_col: str, value_col: str, out_path: str = None):
    """
    Render a calendar-style heatmap by month (rows = weeks, cols = weekdays).
    Expects one row per day in df with columns [date_col, value_col].
    """
    d = df[[date_col, value_col]].dropna().copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col])
    if d.empty:
        print("↪  Skipping calendar heatmap: no valid dates.")
        return

    d["date"] = d[date_col].dt.date
    d["year"] = d[date_col].dt.year
    d["month"] = d[date_col].dt.month
    d["weekday"] = d[date_col].dt.weekday  # 0=Mon ... 6=Sun
    # week index within month (Mon-based)
    d["week_of_month"] = ((d[date_col].dt.day - 1 + d[date_col].dt.to_period('M').dt.start_time.dt.weekday) // 7)

    # plot each month in a grid
    months = d.groupby(["year", "month"])
    n = len(months)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.2*rows), squeeze=False)

    vmax = d[value_col].max() if _has_variation(d[value_col]) else None
    for (idx, ((yr, mo), g)) in enumerate(months):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        pivot = g.pivot_table(index="week_of_month", columns="weekday", values=value_col, aggfunc="sum")
        sns.heatmap(pivot, ax=ax, cmap="YlOrRd", cbar=False, square=True, linewidths=0.5, vmax=vmax)
        ax.set_title(f"{pd.Timestamp(yr, mo, 1):%B %Y}")
        ax.set_xlabel("Weekday (Mon=0)")
        ax.set_ylabel("Week of month")
    # hide extra axes
    for k in range(n, rows*cols):
        r, c = divmod(k, cols)
        axes[r][c].axis("off")

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=160, bbox_inches="tight")
        print(f"🗓️  Saved calendar heatmap → {out_path}")
    plt.close(fig)

# ---------- Spatial density: hexbin + KDE ----------
def plot_spatial_density(
    df: pd.DataFrame,
    lat_col: str = None,
    lon_col: str = None,
    out_prefix: str = None,
    gridsize: int = 60
):
    """
    Makes (1) hexbin count map and (2) KDE heatmap (if seaborn can handle it).
    Tries to auto-detect lat/lon columns if not provided.
    """
    # auto-detect
    candidates_lat = ["LAT", "Latitude", "LATITUDE", "lat", "y", "gps_lat"]
    candidates_lon = ["LON", "Longitude", "LONGITUDE", "lon", "x", "gps_lon"]
    if lat_col is None:
        lat_col = next((c for c in candidates_lat if c in df.columns), None)
    if lon_col is None:
        lon_col = next((c for c in candidates_lon if c in df.columns), None)

    if lat_col is None or lon_col is None:
        print("↪  Skipping spatial density: lat/lon not found.")
        return

    dd = df[[lat_col, lon_col]].dropna().copy()
    # basic sanity range (LA-ish but lenient)
    dd = dd[(dd[lat_col].between(-90, 90)) & (dd[lon_col].between(-180, 180))]
    if len(dd) < 100:
        print("↪  Skipping spatial density: not enough valid points.")
        return

    # 1) Hexbin
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    hb = ax.hexbin(dd[lon_col], dd[lat_col], gridsize=gridsize, mincnt=1, cmap="inferno", norm=LogNorm())
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Spatial Hexbin Density")
    cb = fig.colorbar(hb, ax=ax); cb.set_label("Count (log scale)")
    plt.tight_layout()
    if out_prefix:
        p1 = f"{out_prefix}_hexbin.png"
        plt.savefig(p1, dpi=160, bbox_inches="tight")
        print(f"🗺️  Saved hexbin → {p1}")
    plt.close(fig)

    # 2) KDE (project to 2D plane)
    try:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        sns.kdeplot(
            x=dd[lon_col], y=dd[lat_col],
            fill=True, thresh=0.05, levels=50, bw_method="scott", ax=ax, cmap="mako"
        )
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.set_title("Spatial KDE Density")
        plt.tight_layout()
        if out_prefix:
            p2 = f"{out_prefix}_kde.png"
            plt.savefig(p2, dpi=160, bbox_inches="tight")
            print(f"🗺️  Saved KDE → {p2}")
        plt.close(fig)
    except Exception as e:
        print(f"↪  KDE plot skipped: {e}")
# --- eda.py ---------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True); return p

def plot_univariate_outliers(df: pd.DataFrame, output_dir: str, cols=None, z=3.0):
    """
    Saves per-feature boxplots + Z-score tails and a summary CSV of flagged rows.
    """
    out = _ensure_dir(os.path.join(output_dir, "eda_outliers"))
    num = df.select_dtypes(include=[np.number])
    if cols is None:
        cols = [c for c in num.columns if num[c].dropna().nunique() > 1]
    if not cols:
        print("EDA outliers: no numeric columns to plot."); return

    flagged = {}
    for c in cols:
        s = num[c].astype(float).dropna()
        if s.nunique() < 2: 
            continue

        # IQR whisker outliers
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
        iqr_idx = df.index[(df[c] < lo) | (df[c] > hi)]
        flagged[c] = list(map(int, iqr_idx))

        # Boxplot
        plt.figure(figsize=(6,3))
        plt.boxplot(s.values, vert=False, whis=1.5, showfliers=True)
        plt.title(f"Boxplot (IQR) — {c}")
        plt.tight_layout()
        plt.savefig(os.path.join(out, f"box_{c}.png"), dpi=160); plt.close()

        # Z-score tail histogram
        zscores = (s - s.mean()) / (s.std() if s.std() else 1.0)
        plt.figure(figsize=(6,3))
        plt.hist(zscores, bins=40, alpha=0.8)
        plt.axvline(z, ls="--"); plt.axvline(-z, ls="--")
        plt.title(f"Z-score histogram — {c} (|z|>{z} flagged)")
        plt.tight_layout()
        plt.savefig(os.path.join(out, f"z_{c}.png"), dpi=160); plt.close()

    # Summary file of row indices per column
    pd.Series({k: len(v) for k,v in flagged.items()}).rename("n_flagged") \
        .to_csv(os.path.join(out, "summary_counts.csv"))
    print(f"📦 EDA outliers saved → {out}")

def plot_bivariate_outliers(df: pd.DataFrame, output_dir: str, x: str, y: str, z=3.0):
    """
    Scatter with high-|z| points highlighted for both X/Y.
    """
    out = _ensure_dir(os.path.join(output_dir, "eda_outliers"))
    if any(c not in df.columns for c in [x,y]): 
        print("bivariate: columns not found"); return
    xy = df[[x,y]].dropna()
    if xy.shape[0] < 10 or xy.nunique().min() < 2:
        print("bivariate: not enough data/variance"); return
    zX = (xy[x]-xy[x].mean())/(xy[x].std() or 1.0)
    zY = (xy[y]-xy[y].mean())/(xy[y].std() or 1.0)
    mask = (zX.abs()>z) | (zY.abs()>z)

    plt.figure(figsize=(6,5))
    plt.scatter(xy[x], xy[y], s=10, alpha=0.5, label="inliers")
    if mask.any():
        plt.scatter(xy.loc[mask, x], xy.loc[mask, y], s=24, alpha=0.9, label="outliers")
    plt.legend(); plt.xlabel(x); plt.ylabel(y); plt.title(f"Outliers in {x} vs {y}")
    plt.tight_layout()
    plt.savefig(os.path.join(out, f"scatter_outliers_{x}_{y}.png"), dpi=170); plt.close()
def split_features(df: pd.DataFrame, cat_threshold: int = 20):
    """
    Split columns into categorical/numerical.
    Rule: if non-null unique values < cat_threshold → categorical,
          else numerical.
    """
    categorical, numerical = [], []
    for col in df.columns:
        nunique = df[col].dropna().nunique()
        if nunique == 0: 
            continue
        if nunique < cat_threshold:
            categorical.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            numerical.append(col)
        else:
            categorical.append(col)
    return categorical, numerical
# --- add near top of eda.py (if missing) ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def log_skipped(col: str, reason: str, prefix: str = "[EDA]"):
    try:
        print(f"{prefix} skipped {col}: {reason}")
    except Exception:
        pass


# ----------------------- Visualization Functions -----------------------
def plot_class_balance(df: pd.DataFrame, target: str, title: str = "Class balance"):
    if target in df:
        vc = df[target].astype(str).value_counts(dropna=False)
        plt.figure(figsize=(6,3)); vc.plot(kind="bar"); plt.title(title); plt.grid(alpha=0.2)

def _plot_class_balance(df: pd.DataFrame, target: str, title: str):
    if target in df.columns:
        vc = df[target].astype(str).value_counts(dropna=False)
        plt.figure(figsize=(6, 3))
        ax = vc.plot(kind="bar")
        ax.set_title(title); ax.set_xlabel(target); ax.set_ylabel("count"); ax.grid(alpha=0.2)

def eda_missingness(df: pd.DataFrame, title: str = "Missingness"):
    m = df.isna().mean().sort_values(ascending=False)
    plt.figure(figsize=(10,4)); m.plot(kind="bar"); plt.title(title); plt.ylabel("fraction missing"); plt.grid(alpha=0.2)

def eda_numeric_histograms(df: pd.DataFrame, title: str = "Numeric Distributions — cleaned"):
    nums = df.select_dtypes(include=[np.number]).columns.tolist()
    k = len(nums) or 1
    ncol = 4; nrow = math.ceil(k/ncol)
    plt.figure(figsize=(4*ncol, 3*nrow))
    for i,c in enumerate(nums,1):
        ax = plt.subplot(nrow,ncol,i)
        ax.hist(df[c].dropna().values, bins=30); ax.set_title(c); ax.grid(alpha=0.2)
    plt.suptitle(title, y=1.02)

def eda_top_categories(df: pd.DataFrame, col: str, top_n: int = 20):
    vc = df[col].astype(str).value_counts().head(top_n)
    plt.figure(figsize=(8,4)); vc.plot(kind="bar"); plt.title(f"Top categories — {col}"); plt.grid(alpha=0.2)

def eda_corr_mixed(df: pd.DataFrame, max_cat_card: int = 30):
    """Compute correlation between numeric (Pearson) and categorical (Cramer's V approx via Theil's U fallback)."""
    if not _HAS_SKLEARN:
        print("sklearn not available for mixed correlation analysis")
        return
        
    from sklearn.preprocessing import LabelEncoder
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    cats = [c for c in df.columns if df[c].dtype=="object" and df[c].nunique()<=max_cat_card]
    corr = pd.DataFrame(index=num+cats, columns=num+cats, dtype=float)
    # numeric-numeric
    if num:
        corr.loc[num,num] = df[num].corr()
    # cat-cat (use normalized mutual information as symmetric measure)
    try:
        if _HAS_MI:
            for c1 in cats:
                for c2 in cats:
                    if pd.isna(corr.loc[c1,c2]):
                        corr.loc[c1,c2] = normalized_mutual_info_score(df[c1].astype(str), df[c2].astype(str))
    except Exception:
        pass
    # num-cat: use ANOVA f-score proxy
    try:
        X = df[num].fillna(df[num].median()) if num else pd.DataFrame(index=df.index)
        for c in cats:
            y = pd.factorize(df[c].astype(str))[0]
            if num:
                f,_ = f_classif(X, y)
                corr.loc[num, c] = pd.Series(f, index=num)
                corr.loc[c, num] = pd.Series(f, index=num)
    except Exception:
        pass
    # plot heatmap
    plt.figure(figsize=(max(8,0.3*len(corr)), max(6,0.3*len(corr))))
    if _HAS_SEABORN:
        sns.heatmap(corr.astype(float), cmap="viridis", square=True); plt.title("Mixed-type correlation (proxies)")
    else:
        plt.imshow(corr.astype(float), cmap="viridis"); plt.colorbar(); plt.title("Mixed-type correlation (proxies)")
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90); plt.yticks(range(len(corr.index)), corr.index)

def _heatmap_numeric_corr(df: pd.DataFrame, method: str, title: str):
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        return
    c = num.corr(method=method)
    plt.figure(figsize=(max(6, 0.4*len(c)), max(5, 0.4*len(c))))
    if _HAS_SEABORN:
        sns.heatmap(c, cmap="coolwarm", center=0, square=True)
    else:
        plt.imshow(c.values, cmap="coolwarm"); plt.colorbar(); plt.xticks(range(len(c)), c.columns, rotation=90); plt.yticks(range(len(c)), c.index)
    plt.title(title)

def _mutual_info_to_target(df: pd.DataFrame, target: str, top_k: int = 30):
    """
    Plot mutual information (discrete) between each feature and the target.
    Works best when target is categorical.
    """
    if not _HAS_MI or target not in df.columns:
        return
    y = df[target].astype(str)
    scores = []
    for col in df.columns:
        if col == target:
            continue
        s = df[col]
        # Discretize numeric for MI (simple quantiles); categorical as-is
        if pd.api.types.is_numeric_dtype(s):
            q = pd.qcut(s, q=min(10, s.nunique()), duplicates="drop")
            val = mutual_info_score(y, q)
        else:
            val = mutual_info_score(y, s.astype(str))
        scores.append((col, val))
    if not scores:
        return
    scores = sorted(scores, key=lambda t: t[1], reverse=True)[:top_k]
    names = [n for n,_ in scores]; vals = [v for _,v in scores]
    plt.figure(figsize=(8, max(3, 0.25*len(scores))))
    plt.barh(names[::-1], vals[::-1]); plt.title(f"Mutual Information vs. {target}"); plt.xlabel("MI (higher = more informative)"); plt.grid(axis="x", alpha=0.2)

def _feature_importance_rf(df: pd.DataFrame, target: str, top_k: int = 25, random_state: int = 42):
    if target not in df.columns or not _HAS_SKLEARN:
        return
    # numeric-only quick model (robust & fast); fillna with medians
    X = df.select_dtypes(include=[np.number]).copy()
    if X.empty:
        return
    y = df[target].astype(str)
    X = X.fillna(X.median())
    n_estimators = 300 if X.shape[0] > 20000 else 200
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1, class_weight="balanced")
    try:
        rf.fit(X, y)
    except Exception as e:
        print("RF importance failed:", e); return
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)[:top_k]
    plt.figure(figsize=(8, max(3, 0.25*len(imp))))
    imp.iloc[::-1].plot(kind="barh")
    plt.title(f"RandomForest feature importance (top {len(imp)})"); plt.grid(axis="x", alpha=0.2)

    # Permutation importance on a small holdout for reliability
    try:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        rf.fit(Xtr, ytr)
        perm = permutation_importance(rf, Xte, yte, n_repeats=10, random_state=42, n_jobs=-1)
        pi = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)[:top_k]
        plt.figure(figsize=(8, max(3, 0.25*len(pi))))
        pi.iloc[::-1].plot(kind="barh")
        plt.title(f"Permutation importance (validation, top {len(pi)})"); plt.grid(axis="x", alpha=0.2)
    except Exception as e:
        print("Permutation importance skipped:", e)

def _kde_and_boxpanels(df: pd.DataFrame, title_prefix: str = "Distributions"):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return
    # KDE rows (cap to 12)
    show = num_cols[:12]
    plt.figure(figsize=(4*min(3, len(show)), 3*math.ceil(len(show)/3)))
    for i, c in enumerate(show, 1):
        ax = plt.subplot(math.ceil(len(show)/3), 3, i)
        s = df[c].dropna()
        if len(s) > 0:
            s.plot(kind="kde", ax=ax)
        ax.set_title(f"{c} - KDE"); ax.grid(alpha=0.2)
    plt.suptitle(f"{title_prefix}: KDE", y=1.02)

    # Boxplots (cap to 12)
    plt.figure(figsize=(max(8, 0.5*len(show)), 4))
    try:
        if _HAS_SEABORN:
            sns.boxplot(data=df[show], orient="h")
        else:
            # crude: individual boxplots
            plt.boxplot([df[c].dropna().values for c in show], vert=False, labels=show)
        plt.title(f"{title_prefix}: Boxplots")
        plt.grid(axis="x", alpha=0.2)
    except Exception as e:
        print("boxplots skipped:", e)

def _missing_matrix(df: pd.DataFrame, max_rows: int = 1000, title: str = "Missingness matrix"):
    m = df.isna()
    if m.empty:
        return
    if len(m) > max_rows:
        m = m.sample(n=max_rows, random_state=42)
    plt.figure(figsize=(min(16, 0.5*m.shape[1]+4), min(8, 0.008*m.shape[0]+3)))
    if _HAS_SEABORN:
        sns.heatmap(m, cbar=False); plt.title(title)
    else:
        plt.imshow(m.values, aspect="auto", interpolation="nearest"); plt.title(title)

def plot_hour_day_heatmap(df: pd.DataFrame, hour_col="hour", weekday_col="weekday", title="Hour x Day - raw"):
    if hour_col not in df or weekday_col not in df: return
    pivot = df.pivot_table(index=weekday_col, columns=hour_col, values=hour_col, aggfunc="count", fill_value=0)
    plt.figure(figsize=(10,4))
    if _HAS_SEABORN:
        sns.heatmap(pivot, cmap="magma"); plt.title(title)
    else:
        plt.imshow(pivot.values, aspect="auto", cmap="magma"); plt.title(title); plt.xlabel("hour"); plt.ylabel("weekday")


# ----------------------- Main EDA Pipeline -----------------------
def run_eda(
    raw_csv_path: str,
    fig_dir: str = "./figs_eda",
    cache_path: str = "./cache/lapd_clean.parquet",
    config_overrides: Optional[Dict[str, Any]] = None
) -> str:
    # --- config (existing + new defaults) ------------------------------------
    cfg = DEFAULT_EDA_CONFIG.copy()
    if config_overrides:
        cfg.update(config_overrides)

    # NEW toggles (safe defaults if not in DEFAULT_EDA_CONFIG)
    cfg.setdefault("imbalance_target", None)
    cfg.setdefault("balance_preview", True)           # build balanced preview for plots/importance
    cfg.setdefault("balance_method", "smote")         # "smote" or "undersample"
    cfg.setdefault("balance_max_n", 20000)

    cfg.setdefault("corr_pearson", True)
    cfg.setdefault("corr_spearman", True)
    cfg.setdefault("corr_mutual_info", True)          # MI to target (bar plot)

    cfg.setdefault("feature_importance", True)
    cfg.setdefault("perm_importance", True)           # (done inside helper; best-effort)

    cfg.setdefault("plot_numeric_dists", True)        # histograms (you have)
    cfg.setdefault("plot_kde_box", True)              # KDE + boxpanels (new)
    cfg.setdefault("missing_matrix", True)            # missingness heatmap

    cfg.setdefault("outliers_iforest", True)
    cfg.setdefault("outliers_lof", True)
    cfg.setdefault("outliers_ocsvm", True)
    cfg.setdefault("outliers_max_frac", 0.02)
    cfg.setdefault("plot_corr", True)                 # mixed-type proxies
    cfg.setdefault("vif_check", True)

    _ensure_dirs(fig_dir, os.path.dirname(cache_path) or ".")

    print("==== [EDA-1] Read CSV/ZIP ====")
    df = pd.read_csv(raw_csv_path, low_memory=False, compression="infer")

    # --- Canonical LAPD dates (one place) ------------------------------------
    date_dt = choose_best_date_column(df)

    # Filter to a sane date range
    mask = date_dt.between(pd.Timestamp("2000-01-01"), pd.Timestamp.now())
    df = df.loc[mask].copy()

    # Date/time feature scaffold (pre-normalize to keep downstream code happy)
    df["date_dt"]    = date_dt.loc[mask].values
    df["date"]       = df["date_dt"].dt.normalize()
    df["year"]       = df["date_dt"].dt.year
    df["month"]      = df["date_dt"].dt.month
    df["weekday"]    = df["date_dt"].dt.weekday
    df["hour"]       = df["date_dt"].dt.hour
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    print("shape:", df.shape)

    print("==== [EDA-2] Normalize column names ====")
    df = normalize_columns(df)

    print("==== [EDA-3] Build datetime + add time features ====")
    df, dt_col = infer_datetime_columns(df)
    if dt_col is None:
        raise RuntimeError("Could not infer a datetime column.")
    df = add_time_features(df, dt_col=dt_col, use_us_holidays=cfg.get("use_us_holidays", True) and _HAS_HOLIDAYS)

    # --- Class balance (raw) --------------------------------------------------
    if cfg.get("imbalance_target"):
        try:
            _plot_class_balance(df, cfg["imbalance_target"], title=f"Class balance - raw ({cfg['imbalance_target']})")
            _plot_and_save(f"class_balance_raw_{cfg['imbalance_target']}", fig_dir)
        except Exception as e:
            print("class balance plot skipped:", e)

    # Time rhythm previews (raw)
    try:
        plot_hour_day_heatmap(df, hour_col="hour", weekday_col="weekday", title="Hour x Day - raw")
        _plot_and_save("Hour x Day - raw", fig_dir)
        plot_calendar_heatmap(df, date_col="date", title="Calendar heatmap - raw")
        _plot_and_save("Calendar heatmap - raw", fig_dir)
    except Exception as e:
        print("preview time plots skipped:", e)

    # Correlations on raw
    if cfg.get("corr_pearson", True):
        try:
            _heatmap_numeric_corr(df, method="pearson", title="Numeric correlation (Pearson) - raw")
            _plot_and_save("Numeric correlation (Pearson) - raw", fig_dir)
        except Exception as e:
            print("pearson heatmap skipped:", e)
    if cfg.get("corr_spearman", True):
        try:
            _heatmap_numeric_corr(df, method="spearman", title="Numeric correlation (Spearman) - raw")
            _plot_and_save("Numeric correlation (Spearman) - raw", fig_dir)
        except Exception as e:
            print("spearman heatmap skipped:", e)
    if cfg.get("plot_corr", True):
        try:
            eda_corr_mixed(df)
            _plot_and_save("Mixed correlations (num + cat) - raw", fig_dir)
        except Exception as e:
            print("mixed-type corr skipped:", e)

    # Distributions (raw): hist + KDE + box
    if cfg.get("plot_numeric_dists", True):
        try:
            eda_numeric_histograms(df, title="Numeric distributions (hist) - raw")
            _plot_and_save("Numeric distributions (hist) - raw", fig_dir)
        except Exception as e:
            print("numeric hists (raw) skipped:", e)
    if cfg.get("plot_kde_box", True):
        try:
            _kde_and_boxpanels(df, title_prefix="Distributions - raw")
            _plot_and_save("Distributions KDE - raw", fig_dir)
            _plot_and_save("Distributions Box - raw", fig_dir)
        except Exception as e:
            print("kde/box (raw) skipped:", e)

    print("==== [EDA-4] Geo cleanup ====")
    df, issues = validate_and_clean_coords(df)
    print(issues)

    if cfg.get("add_h3", False):
        try:
            df = add_h3_index(df, h3_res=cfg.get("h3_res", 8))
        except Exception as e:
            print("H3 add failed:", e)

    print("==== [EDA-5] Dedupe & downcast ====")
    df = deduplicate(df)
    df = downcast_numeric(df)

    print("==== [EDA-6] Outliers (IQR clip + optional z-winsor) ====")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if cfg.get("iqr_clip", True) and num_cols:
        df = iqr_outlier_clip(df, num_cols=num_cols, whisker=cfg.get("iqr_multiplier", 1.5))
    if cfg.get("zscore_winsor", True) and num_cols:
        df = _zscore_winsorize(df, num_cols, cfg.get("zscore_max", 4.0))

    # ML-based outlier filters (IForest, LOF, OCSVM)
    if num_cols:
        used = [c for c in num_cols if df[c].notna().any()]
        if used:
            if cfg.get("outliers_iforest", False):
                df, _ = _iforest_outlier_filter(df, used, max_frac=cfg.get("outliers_max_frac", 0.02))
            if cfg.get("outliers_lof", False):
                df, _ = _lof_outlier_filter(df, used, max_frac=cfg.get("outliers_max_frac", 0.02))
            if cfg.get("outliers_ocsvm", False) and cfg.get("svms_outlier_clip", True):
                df, _ = _svm_outlier_filter(df, used, nu=cfg.get("svms_nu", 0.01), gamma=cfg.get("svms_gamma", "scale"))

    # VIF report
    if cfg.get("vif_check", True):
        try:
            vt = vif_table(df)
            out_vif = os.path.join(os.path.dirname(cache_path) or ".", "vif_table.csv")
            vt.to_csv(out_vif, index=False)
            print(f"VIF table saved: {out_vif}")
        except Exception as e:
            print("VIF skipped:", e)

    print("==== [EDA-7] STRICT drop-any-NaN (after feature creation) ====")
    if cfg.get("drop_any_nan_after_features", True):
        df = clean_then_dropna_all(df)

    # --- Plot previews on cleaned data (sample to keep things light) ----------
    n_preview = min(len(df), cfg.get("preview_rows", 5000))
    df_plot = df.sample(n_preview, random_state=42) if n_preview > 0 else df

    # Missingness summaries
    try:
        eda_missingness(df_plot, title="Missingness - cleaned")
        _plot_and_save("Missingness - cleaned", fig_dir)
    except Exception as e:
        print("missingness (bar) skipped:", e)
    if cfg.get("missing_matrix", True):
        try:
            _missing_matrix(df_plot, title="Missingness matrix - cleaned")
            _plot_and_save("Missingness matrix - cleaned", fig_dir)
        except Exception as e:
            print("missing matrix skipped:", e)

    # Distributions (cleaned)
    if cfg.get("plot_numeric_dists", True):
        try:
            eda_numeric_histograms(df_plot, title="Numeric distributions (hist) - cleaned")
            _plot_and_save("Numeric distributions (hist) - cleaned", fig_dir)
        except Exception as e:
            print("numeric hists (cleaned) skipped:", e)
    if cfg.get("plot_kde_box", True):
        try:
            _kde_and_boxpanels(df_plot, title_prefix="Distributions - cleaned")
            _plot_and_save("Distributions KDE - cleaned", fig_dir)
            _plot_and_save("Distributions Box - cleaned", fig_dir)
        except Exception as e:
            print("kde/box (cleaned) skipped:", e)

    # Correlations (cleaned)
    if cfg.get("corr_pearson", True):
        try:
            _heatmap_numeric_corr(df_plot, method="pearson", title="Numeric correlation (Pearson) - cleaned")
            _plot_and_save("Numeric correlation (Pearson) - cleaned", fig_dir)
        except Exception as e:
            print("pearson (cleaned) skipped:", e)
    if cfg.get("corr_spearman", True):
        try:
            _heatmap_numeric_corr(df_plot, method="spearman", title="Numeric correlation (Spearman) - cleaned")
            _plot_and_save("Numeric correlation (Spearman) - cleaned", fig_dir)
        except Exception as e:
            print("spearman (cleaned) skipped:", e)
    if cfg.get("plot_corr", True):
        try:
            eda_corr_mixed(df_plot)
            _plot_and_save("Mixed correlations (num + cat) - cleaned", fig_dir)
        except Exception as e:
            print("mixed-type corr (cleaned) skipped:", e)

    # Balanced preview (for plots/importance only - does NOT change df)
    df_bal = None
    if cfg.get("imbalance_target") and cfg.get("balance_preview", True):
        try:
            df_bal = _balanced_preview(
                df_plot,
                cfg["imbalance_target"],
                method=cfg.get("balance_method", "smote"),
                max_n=cfg.get("balance_max_n", 20000)
            )
            if df_bal is not None:
                _plot_class_balance(df_bal, cfg["imbalance_target"],
                                    title=f"Class balance - balanced preview ({cfg['imbalance_target']})")
                _plot_and_save(f"class_balance_balanced_{cfg['imbalance_target']}", fig_dir)
        except Exception as e:
            print("balanced preview skipped:", e)

    # Mutual Information to target (bar) and feature importance
    if cfg.get("imbalance_target"):
        source = df_bal if df_bal is not None else df_plot
        if cfg.get("corr_mutual_info", True):
            try:
                _mutual_info_to_target(source, cfg["imbalance_target"], top_k=30)
                _plot_and_save(f"Mutual Information vs {cfg['imbalance_target']}", fig_dir)
            except Exception as e:
                print("mutual information skipped:", e)
        if cfg.get("feature_importance", True):
            try:
                _feature_importance_rf(source, cfg["imbalance_target"], top_k=25)
                _plot_and_save("RF feature importance", fig_dir)
                # permutation done inside helper; figures saved via same hook
            except Exception as e:
                print("feature importance skipped:", e)

    print("==== [EDA-9] Save cleaned parquet ====")
    out = _save_df(df, cache_path)
    print(f"✅ Cleaned data saved to: {out}")
    return out
