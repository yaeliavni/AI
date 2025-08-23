# timeseries.py
# =============================================================================
# LAPD Daily Crime Forecasting (Colab-friendly, modular, optional deps safe)
# - Input : cleaned LAPD dataframe or file path (parquet/csv)
# - Output: forecasts, metrics, plots under OUT_DIR (default: /content/figs_ts)
# - Models: Naive (last value, 7d mean), SARIMAX (statsmodels), Prophet (optional),
#           LSTM (optional, Keras). Any missing dep is skipped gracefully.
# - Backtest: rolling-origin evaluation with multiple cut points
# - Horizon: configurable (default 30)
# =============================================================================

import os
import warnings
from typing import Dict, Optional, Union, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ------------------------------ Utils -----------------------------------------
def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p


def _smart_read(df_or_path: Union[pd.DataFrame, str]) -> pd.DataFrame:
    if isinstance(df_or_path, pd.DataFrame):
        return df_or_path.copy()
    if not os.path.exists(df_or_path):
        raise FileNotFoundError(f"File not found: {df_or_path}")
    if df_or_path.lower().endswith(".parquet"):
        return pd.read_parquet(df_or_path)
    return pd.read_csv(df_or_path, low_memory=False)


def _guess_datetime_column(df: pd.DataFrame) -> str:
    # Prefer the standardized fields from eda.py
    preferred = ["date_dt", "date"]
    for c in preferred:
        if c in df.columns:
            return c
    # Then fall back to common LAPD raw names
    candidates = [
        "DATE OCC", "Date OCC", "date_occ", "Date Occurred", "Date Occured",
        "DATE RPTD", "Date Rptd", "date_rptd",
        "date_only", "Date", "DATE"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # heuristic fallback
    for c in df.columns:
        try:
            s = pd.to_datetime(df[c], errors="coerce")
            if s.notna().mean() > 0.9:
                return c
        except Exception:
            pass
    raise ValueError("Could not locate a datetime column. Ensure eda.py created 'date_dt' or 'date'.")


def _fix_lapd_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robustly find & parse a valid LAPD date column and attach it as `date_fixed`.
    Prefers DATE OCC and DATE RPTD fields, handles epoch integers, combines date/time,
    and rejects columns dominated by a single placeholder date.
    """
    df = df.copy()

    def _parse_any(s):
        if np.issubdtype(s.dtype, np.datetime64):
            return pd.to_datetime(s, errors='coerce')
        if s.dtype == object and s.dropna().map(lambda x: hasattr(x, 'year')).all():
            return pd.to_datetime(s, errors='coerce')
        if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
            arr = pd.to_numeric(s, errors='coerce')
            dt = pd.to_datetime(arr, unit='s', errors='coerce')
            # fallback to ms if mostly 1970
            if dt.notna().mean() < 0.5 or (dt.dropna().dt.year <= 1971).mean() > 0.8:
                dt_ms = pd.to_datetime(arr, unit='ms', errors='coerce')
                if dt_ms.notna().mean() > dt.notna().mean():
                    dt = dt_ms
            return dt
        return pd.to_datetime(s, errors='coerce', infer_datetime_format=True)

    def _combine(date_col, time_col):
        d = _parse_any(df[date_col])
        t = df[time_col]
        if pd.api.types.is_numeric_dtype(t):
            t = t.fillna(0).astype(int).astype(str).str.zfill(4)
        else:
            t = t.astype(str).str.replace(r'[^0-9]', '', regex=True).str.zfill(4)
        hh = pd.to_numeric(t.str[:2], errors='coerce').fillna(0).clip(0, 23).astype(int)
        mm = pd.to_numeric(t.str[2:4], errors='coerce').fillna(0).clip(0, 59).astype(int)
        return pd.to_datetime(d + pd.to_timedelta(hh, unit='h') + pd.to_timedelta(mm, unit='m'), errors='coerce')

    candidates = {}
    primary = [
        'DATE OCC', 'Date OCC', 'date_occ', 'date_occurred',
        'DATE RPTD', 'Date Rptd', 'date_rptd',
        'DATE', 'Date', 'date', 'date_only'
    ]
    for col in primary:
        if col in df.columns:
            candidates[col] = _parse_any(df[col])
    for dcol in ['DATE OCC', 'Date OCC', 'date_occ']:
        for tcol in ['TIME OCC', 'Time OCC', 'time_occ']:
            if dcol in df.columns and tcol in df.columns:
                candidates[f'{dcol}+{tcol}'] = _combine(dcol, tcol)
    if not candidates:
        for col in df.columns:
            s = df[col]
            if s.notna().any():
                p = _parse_any(s)
                if p.notna().mean() > 0.5:
                    candidates[col] = p

    def score(s):
        if s is None:
            return 0
        valid_rate = s.notna().mean()
        if valid_rate == 0:
            return 0
        ss = s.dropna()
        good = ss.between(pd.Timestamp('2000-01-01'), pd.Timestamp.now()).mean()
        top_share = ss.dt.date.value_counts(normalize=True).max()
        return valid_rate * (0.5 + 0.5 * good) - (0.5 if top_share > 0.8 else 0)

    best_col, best_series, best_score = None, None, 0
    for k, s in candidates.items():
        sc = score(s)
        if sc > best_score:
            best_col, best_series, best_score = k, s, sc

    if best_series is None:
        raise ValueError('Could not parse any date column')

    print(f"Using date column: '{best_col}'")
    df['date_fixed'] = pd.to_datetime(best_series, errors='coerce')
    return df


def _daily_counts(df: pd.DataFrame) -> pd.Series:
    """
    Aggregate to daily counts from `date_fixed`, filling missing days with zeros.
    Attempts primary 2000–present range; widens to 1990–present as fallback.
    """
    df = _fix_lapd_dates(df)
    dates = pd.to_datetime(df['date_fixed'], errors='coerce').dropna()

    # Primary date window 2000–today
    good = dates[(dates >= pd.Timestamp('2000-01-01')) & (dates <= pd.Timestamp.now())]
    if good.empty:
        # widen to 1990-present or fallback to any valid date
        rescue = dates[(dates >= pd.Timestamp('1990-01-01')) & (dates <= pd.Timestamp.now())]
        good = rescue if not rescue.empty else dates
    if good.empty:
        raise ValueError('No usable dates after all rescue attempts')

    daily = good.dt.floor('D').value_counts().sort_index()
    daily.index = pd.to_datetime(daily.index)
    full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq='D')
    return daily.reindex(full_idx, fill_value=0)


def _add_time_features(df_y: pd.DataFrame) -> pd.DataFrame:
    """Add simple calendar/holiday features (for ML/LSTM; SARIMAX/Prophet handle seasonality internally)."""
    x = df_y.copy()
    x["dow"] = x.index.dayofweek
    x["month"] = x.index.month
    x["is_weekend"] = (x["dow"] >= 5).astype(int)
    try:
        # US Federal Holidays (optional)
        from pandas.tseries.holiday import USFederalHolidayCalendar
        cal = USFederalHolidayCalendar()
        hol = cal.holidays(start=x.index.min(), end=x.index.max())
        x["is_holiday"] = x.index.isin(hol).astype(int)
    except Exception:
        x["is_holiday"] = 0
    return x


def _metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    # avoid div by zero in MAPE
    denom = np.where(y_true == 0, 1.0, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / denom)) * 100.0
    return {"MAE": float(mae), "RMSE": float(rmse), "MAPE%": float(mape)}


def _plot_forecast(ds, y_true, y_pred, title, path):
    plt.figure(figsize=(10, 4))
    plt.plot(ds, y_true, label="Actual", lw=2)
    plt.plot(ds, y_pred, label="Forecast", lw=2)
    plt.title(title)
    plt.xlabel("Date"); plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ------------------------------ Models ----------------------------------------
def _naive_last(y: pd.Series, horizon: int) -> np.ndarray:
    return np.full(horizon, float(y.iloc[-1]))


def _naive_mean7(y: pd.Series, horizon: int) -> np.ndarray:
    tail = y.tail(7)
    val = float(tail.mean()) if len(tail) else float(y.iloc[-1])
    return np.full(horizon, val)


def _fit_predict_sarimax(y_train: pd.Series, horizon: int, seasonal: int = 7):
    try:
        import statsmodels.api as sm
    except Exception:
        return None, "statsmodels not available"
    # simple SARIMAX with weekly seasonality (tuned small search can be added if needed)
    try:
        mod = sm.tsa.statespace.SARIMAX(
            y_train, order=(1,1,1),
            seasonal_order=(1,1,1, seasonal),
            enforce_stationarity=False, enforce_invertibility=False
        ).fit(disp=False)
        y_fc = mod.forecast(steps=horizon).values
        return y_fc, None
    except Exception as e:
        return None, f"SARIMAX failed: {e}"


def _fit_predict_prophet(df_train: pd.DataFrame, horizon: int):
    try:
        from prophet import Prophet  # pip install prophet
    except Exception:
        return None, "prophet not available"
    try:
        m = Prophet(weekly_seasonality=True, daily_seasonality=False, yearly_seasonality=True)
        m.fit(df_train.reset_index().rename(columns={"ds": "ds", "y": "y"}))
        future = m.make_future_dataframe(periods=horizon, freq="D", include_history=False)
        fc = m.predict(future)
        y_fc = fc["yhat"].values
        return y_fc, None
    except Exception as e:
        return None, f"Prophet failed: {e}"


def _fit_predict_lstm(y_train: pd.Series, horizon: int, lookback: int = 60, epochs: int = 40):
    """Tiny univariate LSTM; only if TF/Keras is installed. Returns None if unavailable."""
    try:
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import LSTM, Dense
    except Exception:
        return None, "tensorflow/keras not available"

    try:
        # scale counts (robust)
        y = y_train.astype(float).values.reshape(-1, 1)
        mu = y.mean(); sd = y.std() if y.std() > 1e-8 else 1.0
        yz = (y - mu) / sd

        # build sequences
        Xs, ys = [], []
        for i in range(lookback, len(yz)):
            Xs.append(yz[i - lookback:i, 0])
            ys.append(yz[i, 0])
        Xs = np.array(Xs)[:, :, None]
        ys = np.array(ys)

        model = Sequential([
            LSTM(32, input_shape=(lookback, 1)),
            Dense(16, activation="relu"),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(Xs, ys, epochs=epochs, batch_size=32, verbose=0)

        # iterative forecast
        hist = yz[-lookback:, 0].tolist()
        preds = []
        for _ in range(horizon):
            x_in = np.array(hist[-lookback:])[None, :, None]
            p = float(model.predict(x_in, verbose=0)[0, 0])
            preds.append(p)
            hist.append(p)
        y_fc = np.array(preds) * sd + mu
        return y_fc, None
    except Exception as e:
        return None, f"LSTM failed: {e}"


# ------------------------------ Runner ----------------------------------------
def run_timeseries(
    df_or_path: Union[pd.DataFrame, str],
    out_dir: str = "/content/figs_ts",
    horizon: int = 30,
    eval_days: int = 90,
    rolling_splits: int = 3,
    run_models: Optional[List[str]] = None,
    date_col: Optional[str] = None,
    verbose: bool = True,
):
    """
    End-to-end: aggregate → split → fit → forecast → evaluate → plot.

    Parameters
    ----------
    df_or_path : DataFrame or str
        Cleaned LAPD dataframe or path to parquet/csv.
    out_dir : str
        Where to save figures/CSVs.
    horizon : int
        Forecast steps ahead (days).
    eval_days : int
        Total last days reserved for evaluation window.
    rolling_splits : int
        Number of rolling-origin evaluation splits.
    run_models : list[str]
        Subset of {"naive_last","naive_mean7","sarimax","prophet","lstm"}.
        Default runs all available.
    date_col : str or None
        If you want to force which column is the date column.
    verbose : bool
        Print progress.

    Returns
    -------
    dict with:
      - "metrics": DataFrame of backtest metrics (avg over splits)
      - "forecasts": dict of last-split forecasts (Series)
      - "y_test": last-split actuals (Series)
    """
    _ensure_dir(out_dir)
    charts_dir = _ensure_dir(os.path.join(out_dir, "charts"))
    tables_dir = _ensure_dir(os.path.join(out_dir, "tables"))

    if run_models is None:
        run_models = ["naive_last", "naive_mean7", "sarimax", "prophet", "lstm"]

    # 1) Load + aggregate to daily counts
    df = _smart_read(df_or_path)
    y = _daily_counts(df, date_col=date_col)  # Series indexed by daily ds, name 'y'
    y = y.astype(float).rename("y")

    if verbose:
        print("==== [TS-1] Daily series built ====")
        print(f"Span: {y.index.min().date()} → {y.index.max().date()}  (n={len(y)})")
        print(f"Mean={y.mean():.1f}, Std={y.std():.1f}, Zero-days={(y==0).sum()}")

    # 2) Rolling-origin backtests
    if eval_days < horizon + 7:
        eval_days = horizon + 7  # ensure eval window covers horizon
    split_points = np.linspace(len(y) - eval_days, len(y) - horizon, num=rolling_splits, dtype=int)
    split_points = np.unique(split_points)
    if verbose:
        print("\n==== [TS-2] Rolling backtests ====")
        print(f"Splits at indices: {split_points.tolist()}  (horizon={horizon}, eval_days={eval_days})")

    metrics_rows = []
    last_forecasts = {}
    last_test = None

    for si, split_idx in enumerate(split_points, start=1):
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:split_idx + horizon]
        if len(y_test) < horizon:
            # not enough future days; skip this split
            continue

        if verbose:
            print(f"\n[Split {si}/{len(split_points)}] train={y_train.index.min().date()}→{y_train.index.max().date()} "
                  f"| test={y_test.index.min().date()}→{y_test.index.max().date()}  (n_train={len(y_train)}, n_test={len(y_test)})")

        # Prepare Prophet frame for this split
        df_train = pd.DataFrame({"ds": y_train.index, "y": y_train.values})

        # 2.1 Baselines
        if "naive_last" in run_models:
            yhat = _naive_last(y_train, horizon)
            m = _metrics(y_test.values, yhat)
            m.update(split=si, model="naive_last")
            metrics_rows.append(m)
            if si == len(split_points):
                last_forecasts["naive_last"] = pd.Series(yhat, index=y_test.index)

        if "naive_mean7" in run_models:
            yhat = _naive_mean7(y_train, horizon)
            m = _metrics(y_test.values, yhat)
            m.update(split=si, model="naive_mean7")
            metrics_rows.append(m)
            if si == len(split_points):
                last_forecasts["naive_mean7"] = pd.Series(yhat, index=y_test.index)

        # 2.2 SARIMAX
        if "sarimax" in run_models:
            yhat, err = _fit_predict_sarimax(y_train, horizon)
            if yhat is None:
                if verbose: print(f"[SARIMAX] skipped: {err}")
            else:
                m = _metrics(y_test.values, yhat)
                m.update(split=si, model="sarimax")
                metrics_rows.append(m)
                if si == len(split_points):
                    last_forecasts["sarimax"] = pd.Series(yhat, index=y_test.index)

        # 2.3 Prophet
        if "prophet" in run_models:
            yhat, err = _fit_predict_prophet(df_train, horizon)
            if yhat is None:
                if verbose: print(f"[Prophet] skipped: {err}")
            else:
                m = _metrics(y_test.values, yhat)
                m.update(split=si, model="prophet")
                metrics_rows.append(m)
                if si == len(split_points):
                    last_forecasts["prophet"] = pd.Series(yhat, index=y_test.index)

        # 2.4 LSTM (optional)
        if "lstm" in run_models:
            yhat, err = _fit_predict_lstm(y_train, horizon)
            if yhat is None:
                if verbose: print(f"[LSTM] skipped: {err}")
            else:
                m = _metrics(y_test.values, yhat)
                m.update(split=si, model="lstm")
                metrics_rows.append(m)
                if si == len(split_points):
                    last_forecasts["lstm"] = pd.Series(yhat, index=y_test.index)

        # keep last test window for plotting
        if si == len(split_points):
            last_test = y_test.copy()

    # 3) Aggregate metrics
    met_df = pd.DataFrame(metrics_rows)
    if len(met_df):
        agg = met_df.groupby("model", as_index=False)[["MAE", "RMSE", "MAPE%"]].mean().sort_values("RMSE")
    else:
        agg = pd.DataFrame(columns=["model", "MAE", "RMSE", "MAPE%"])

    if verbose:
        print("\n==== [TS-3] Metrics (avg over splits) ====")
        if len(agg):
            print(agg.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        else:
            print("No metrics computed (check splits / horizon).")

    # 4) Save tables
    met_path = os.path.join(tables_dir, "rolling_metrics.csv")
    agg_path = os.path.join(tables_dir, "rolling_metrics_avg.csv")
    met_df.to_csv(met_path, index=False)
    agg.to_csv(agg_path, index=False)

    # 5) Plots for last split
    if last_test is not None and len(last_forecasts):
        for name, yhat in last_forecasts.items():
            _plot_forecast(
                last_test.index, last_test.values, yhat.values,
                f"{name} — Forecast vs Actual (last split)",
                os.path.join(charts_dir, f"forecast_{name}.png")
            )

        # overlay plot
        plt.figure(figsize=(10, 5))
        plt.plot(last_test.index, last_test.values, label="Actual", lw=2, color="black")
        for name, yhat in last_forecasts.items():
            plt.plot(yhat.index, yhat.values, label=name)
        plt.title("Forecast Comparison (last split)")
        plt.xlabel("Date"); plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        comp_path = os.path.join(charts_dir, "forecast_comparison.png")
        plt.savefig(comp_path, dpi=150); plt.close()

        if verbose:
            print("\n==== [TS-4] Plots saved ====")
            print(f"- {comp_path}")
            for name in last_forecasts.keys():
                print(f"- {os.path.join(charts_dir, f'forecast_{name}.png')}")

    # 6) Return
    return {
        "metrics": met_df,
        "metrics_avg": agg,
        "forecasts": last_forecasts,
        "y_test": last_test,
        "paths": {
            "charts": charts_dir,
            "tables": tables_dir
        }
    }


# ------------------------------ CLI/Colab helper ------------------------------
if __name__ == "__main__":
    # Example quick run inside Colab:
    # python timeseries.py
    CLEAN_PATH = "/content/lapd_clean.parquet"  # or _5k.parquet if you prefer
    OUT_DIR = "/content/figs_ts"
    res = run_timeseries(
        CLEAN_PATH,
        out_dir=OUT_DIR,
        horizon=30,
        eval_days=90,
        rolling_splits=3,
        run_models=["naive_last", "naive_mean7", "sarimax", "prophet"],  # add "lstm" if TF available
        verbose=True,
    )
    print("\n=== DONE (time series) ===")
    print("Tables:", res["paths"]["tables"])
    print("Charts:", res["paths"]["charts"])
