
# eda.py — Standalone EDA + cleaning + plots for LAPD dataset
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

# ----------------------- Helpers & Config -----------------------
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
)

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
    out = os.path.join(fig_dir, f"{title.replace(' ','-')}.png")
    try:
        plt.tight_layout()
        plt.savefig(out, dpi=160)
        print(f"🖼️ saved: {out}")
    except Exception as e:
        print("(skip save)", e)
    plt.show()

# ----------------------- EDA primitives -----------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
                 .str.replace(r'[^0-9a-zA-Z]+', '_', regex=True)
                 .str.lower()
    )
    return df

def infer_datetime_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    """Try common LAPD column names; else infer from any column that parses to datetime."""
    df = df.copy()
    candidates = ["date_occ","date_rptd","date","time_occ","time_rptd","date_reported","date_occured","dt_occ","timestamp"]
    cand_present = [c for c in candidates if c in df.columns]
    dt_col = None
    # LAPD often has "date_occ" and "time_occ" as integers HHMM. Build dt_occ.
    if "date_occ" in df.columns and "time_occ" in df.columns:
        try:
            d = pd.to_datetime(df["date_occ"], errors="coerce")
            t = df["time_occ"].astype(str).str.zfill(4)
            hh = pd.to_numeric(t.str.slice(0,2), errors="coerce")
            mm = pd.to_numeric(t.str.slice(2,4), errors="coerce")
            dt = d + pd.to_timedelta(hh.fillna(0), unit="h") + pd.to_timedelta(mm.fillna(0), unit="m")
            df["dt_occ"] = dt
            dt_col = "dt_occ"
        except Exception:
            pass
    if dt_col is None:
        # pick the first column that parses to many valid datetimes
        best = None; best_ok = 0
        for c in df.columns:
            try:
                s = pd.to_datetime(df[c], errors="coerce")
                ok = s.notna().sum()
                if ok > best_ok and ok > 0.2*len(df):
                    best, best_ok = c, ok
            except Exception:
                continue
        if best is not None:
            df["dt_occ"] = pd.to_datetime(df[best], errors="coerce")
            dt_col = "dt_occ"
    # add date-only
    if dt_col:
        df["date"] = pd.to_datetime(df[dt_col]).dt.date
    return df, dt_col

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

def stratified_sample_for_task(df: pd.DataFrame, target_col: str, n: int, random_state: int = 42) -> pd.DataFrame:
    if target_col not in df: return df.sample(min(n, len(df)), random_state=random_state)
    groups = df[target_col].dropna().astype(str)
    frac = min(1.0, n/len(df))
    sampled = df.groupby(groups, group_keys=False).apply(lambda g: g.sample(max(1,int(len(g)*frac)), random_state=random_state))
    return sampled.reset_index(drop=True)

# ----------------------- EDA visualizations -----------------------
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
    from sklearn.preprocessing import LabelEncoder
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    cats = [c for c in df.columns if df[c].dtype=="object" and df[c].nunique()<=max_cat_card]
    corr = pd.DataFrame(index=num+cats, columns=num+cats, dtype=float)
    # numeric-numeric
    if num:
        corr.loc[num,num] = df[num].corr()
    # cat-cat (use normalized mutual information as symmetric measure)
    try:
        from sklearn.metrics import normalized_mutual_info_score
        for c1 in cats:
            for c2 in cats:
                if corr.loc[c1,c2] is np.nan: pass
                corr.loc[c1,c2] = normalized_mutual_info_score(df[c1].astype(str), df[c2].astype(str))
    except Exception:
        pass
    # num-cat: use ANOVA f-score proxy
    try:
        from sklearn.feature_selection import f_classif
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

def plot_hour_day_heatmap(df: pd.DataFrame, hour_col="hour", weekday_col="weekday", title="Hour × Day — raw"):
    if hour_col not in df or weekday_col not in df: return
    pivot = df.pivot_table(index=weekday_col, columns=hour_col, values=hour_col, aggfunc="count", fill_value=0)
    plt.figure(figsize=(10,4))
    if _HAS_SEABORN:
        sns.heatmap(pivot, cmap="magma"); plt.title(title)
    else:
        plt.imshow(pivot.values, aspect="auto", cmap="magma"); plt.title(title); plt.xlabel("hour"); plt.ylabel("weekday")

def plot_calendar_heatmap(df: pd.DataFrame, date_col="date", title="Calendar heatmap — raw"):
    if date_col not in df: return
    s = df.groupby(pd.to_datetime(df[date_col])).size()
    s = s.asfreq("D").fillna(0.0)
    plt.figure(figsize=(10,2)); plt.plot(s.index, s.values); plt.title(title); plt.grid(alpha=0.2)

# ----------------------- Optional SVM outliers -----------------------
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

# ----------------------- Public pipeline -----------------------
def run_eda(
    raw_csv_path: str,
    fig_dir: str = "./figs_eda",
    cache_path: str = "./cache/lapd_clean.parquet",
    config_overrides: Optional[Dict[str, Any]] = None
) -> str:
    cfg = DEFAULT_EDA_CONFIG.copy()
    if config_overrides:
        cfg.update(config_overrides)
    _ensure_dirs(fig_dir, os.path.dirname(cache_path) or ".")

    print("==== [EDA-1] Read CSV/ZIP ====")
    df = pd.read_csv(raw_csv_path, low_memory=False, compression='infer')
    print("shape:", df.shape)

    print("==== [EDA-2] Normalize column names ====")
    df = normalize_columns(df)

    print("==== [EDA-3] Build datetime + add time features ====")
    df, dt_col = infer_datetime_columns(df)
    if dt_col is None:
        raise RuntimeError("Could not infer a datetime column.")
    df = add_time_features(df, dt_col=dt_col, use_us_holidays=cfg["use_us_holidays"] and _HAS_HOLIDAYS)

    # Preview time rhythms
    try:
        plot_hour_day_heatmap(df, hour_col="hour", weekday_col="weekday", title="Hour × Day — raw"); _plot_and_save("Hour × Day — raw", fig_dir)
        plot_calendar_heatmap(df, date_col="date", title="Calendar heatmap — raw"); _plot_and_save("Calendar heatmap — raw", fig_dir)
    except Exception as e:
        print("preview time plots skipped:", e)

    print("==== [EDA-4] Geo cleanup ====")
    df, issues = validate_and_clean_coords(df); print(issues)

    if cfg["add_h3"]:
        try:
            df = add_h3_index(df, h3_res=cfg["h3_res"])
        except Exception as e:
            print("H3 add failed:", e)

    print("==== [EDA-5] Dedupe & downcast ====")
    df = deduplicate(df); df = downcast_numeric(df)

    print("==== [EDA-6] Outliers (IQR clip + optional z-winsor) ====")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if cfg["iqr_clip"] and num_cols:
        df = iqr_outlier_clip(df, num_cols=num_cols, whisker=cfg["iqr_multiplier"])
    if cfg["zscore_winsor"] and num_cols:
        df = _zscore_winsorize(df, num_cols, cfg["zscore_max"])

    print("==== [EDA-7] STRICT drop-any-NaN (after feature creation) ====")
    if cfg["drop_any_nan_after_features"]:
        df = clean_then_dropna_all(df)

    print("==== [EDA-8] Optional One-Class SVM outlier filter (post-clean) ====")
    if cfg["svms_outlier_clip"] and num_cols:
        df, _ = _svm_outlier_filter(df, num_cols, nu=cfg["svms_nu"], gamma=cfg["svms_gamma"])

    # Light cap for heavy plots
    df_plot = df.sample(min(len(df), cfg["preview_rows"]), random_state=42)

    print("==== [EDA-9] Rich EDA visuals ====")
    try: eda_missingness(df_plot, title="Missingness — cleaned"); _plot_and_save("Missingness — cleaned", fig_dir)
    except Exception as e: print("missingness plot skipped:", e)
    try: eda_numeric_histograms(df_plot, title="Numeric Distributions — cleaned"); _plot_and_save("Numeric Distributions — cleaned", fig_dir)
    except Exception as e: print("numeric hists skipped:", e)
    for c in ["crm_cd_desc","premis_desc","weapon_desc","status_desc","area_name"]:
        if c in df_plot.columns:
            try: eda_top_categories(df_plot, c, top_n=20); _plot_and_save(f"Top categories — {c}", fig_dir)
            except Exception: pass
    try: eda_corr_mixed(df_plot); _plot_and_save("Mixed correlations (num + cat)", fig_dir)
    except Exception as e: print("corr matrix skipped:", e)

    out = _save_df(df, cache_path)
    print(f"✓ Cleaned data saved to: {out}")
    return out
