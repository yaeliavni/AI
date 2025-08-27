# timeseries.py
# =============================================================================
# LAPD Daily Crime Forecasting (Enhanced + Safe Optional Deps)
# - Functions exported:
#   run_enhanced_lapd_forecasting, run_timeseries,
#   diagnose_model_bias, create_ensemble_forecast, _metrics, analyze_residuals
# =============================================================================
import os
import warnings
from typing import Dict, Optional, Union, List, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ------------------------------ Utilities ------------------------------------
def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def _smart_read(df_or_path: Union[pd.DataFrame, str]) -> pd.DataFrame:
    if isinstance(df_or_path, pd.DataFrame):
        return df_or_path.copy()
    if not os.path.exists(df_or_path):
        raise FileNotFoundError(f"File not found: {df_or_path}")
    low = df_or_path.lower()
    if low.endswith((".zip", ".gz")):
        return pd.read_csv(df_or_path, low_memory=False, compression="infer")
    if low.endswith(".parquet"):
        return pd.read_parquet(df_or_path)
    return pd.read_csv(df_or_path, low_memory=False)

def _metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae   = float(np.mean(np.abs(y_pred - y_true)))
    rmse  = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mape  = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-10))) * 100)
    smape = float(np.mean(2 * np.abs(y_pred - y_true) /
                          np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-10)) * 100)
    return {"MAE": mae, "RMSE": rmse, "MAPE%": mape, "SMAPE%": smape}

# ------------------------------ Daily series ---------------------------------
def _daily_counts_enhanced(df: pd.DataFrame) -> Tuple[pd.Series, Dict]:
    """Build daily counts with robust date detection and fill missing days with 0."""
    # try common date columns
    date_cols = ['Date_occured', 'Date_Reported', 'date', 'Date', 'DATE', 'DATE OCC', 'Date OCC', 'DATE RPTD', 'Date Rptd']
    date_col = next((c for c in date_cols if c in df.columns), None)
    if date_col is None:
        # heuristic fallback: first column that parses mostly-valid datetimes
        candidates = []
        for c in df.columns:
            try:
                s = pd.to_datetime(df[c], errors='coerce')
                if s.notna().mean() > 0.6:
                    candidates.append((c, s))
            except Exception:
                pass
        if candidates:
            # choose the one with the highest validity rate
            date_col, parsed = max(candidates, key=lambda x: pd.to_datetime(df[x[0]], errors='coerce').notna().mean())
            df = df.copy()
            df[date_col] = parsed
        else:
            raise ValueError(f"No usable date column found. Available: {list(df.columns)}")
    else:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    original = len(df)
    df = df.dropna(subset=[date_col])
    valid = len(df)
    print(f"Using date column: '{date_col}'")
    print(f"Date parsing: {original} → {valid} records ({100*valid/original:.1f}% valid)")

    dates = df[date_col].dt.floor('D')
    daily = dates.value_counts().sort_index()
    if len(daily):
        full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq='D')
        daily = daily.reindex(full_idx, fill_value=0)
    daily.index = pd.DatetimeIndex(daily.index)

    meta = {
        "original_records": original,
        "valid_records": valid,
        "date_column": date_col,
        "date_range": (daily.index.min(), daily.index.max()),
        "total_days": len(daily),
        "zero_days": int((daily == 0).sum()),
        "max_daily": int(daily.max()),
        "min_daily": int(daily.min()),
        "mean_daily": float(daily.mean()),
        "std_daily": float(daily.std())
    }
    return daily.astype(float), meta

# ------------------------------ Time features --------------------------------
def _add_time_features(y: pd.Series) -> pd.DataFrame:
    """Add calendar features + lags/rollings for LSTM."""
    if not isinstance(y.index, pd.DatetimeIndex):
        y = y.copy()
        y.index = pd.to_datetime(y.index)  # <<< fix for Int64Index having no .month/.dayofweek
    df = pd.DataFrame({"crime_count": y.values}, index=y.index)
    df["dayofweek"]  = y.index.dayofweek
    df["dayofyear"]  = y.index.dayofyear
    df["month"]      = y.index.month
    df["quarter"]    = y.index.quarter
    df["year"]       = y.index.year
    df["sin_dayofweek"] = np.sin(2*np.pi*df["dayofweek"]/7)
    df["cos_dayofweek"] = np.cos(2*np.pi*df["dayofweek"]/7)
    df["sin_month"]     = np.sin(2*np.pi*df["month"]/12)
    df["cos_month"]     = np.cos(2*np.pi*df["month"]/12)
    df["sin_dayofyear"] = np.sin(2*np.pi*df["dayofyear"]/365.25)
    df["cos_dayofyear"] = np.cos(2*np.pi*df["dayofyear"]/365.25)
    df["is_weekend"]    = (df["dayofweek"] >= 5).astype(int)
    for lag in [1, 2, 3, 7, 14, 30]:
        df[f"lag_{lag}"] = y.shift(lag)
    for w in [7, 14, 30]:
        df[f"rolling_mean_{w}"] = y.rolling(w).mean()
        df[f"rolling_std_{w}"]  = y.rolling(w).std()
    return df

# ------------------------------ Models ---------------------------------------
def _naive_last(y: pd.Series, horizon: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    return np.full(horizon, float(y.iloc[-1])), None

def _naive_seasonal(y: pd.Series, horizon: int, season_length: int = 7) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if len(y) < season_length:
        return _naive_last(y, horizon)
    pred = []
    for h in range(horizon):
        vals = []
        for i in range(1, min(5, len(y)//season_length) + 1):
            idx = len(y) - i*season_length + (h % season_length)
            if idx >= 0:
                vals.append(y.iloc[idx])
        pred.append(np.mean(vals) if len(vals) else float(y.iloc[-1]))
    return np.asarray(pred), None

def _naive_adaptive(y: pd.Series, horizon: int, recent_weight: float = 0.7) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if len(y) < 14:
        return _naive_last(y, horizon)
    recent = y.iloc[-30:] if len(y) >= 30 else y.iloc[-len(y)//2:]
    older  = y.iloc[:-30] if len(y) >= 30 else y.iloc[:-len(y)//2]
    recent_mean = recent.mean(); older_mean = older.mean() if len(older) else recent_mean
    baseline = recent_weight*recent_mean + (1-recent_weight)*older_mean
    out = []
    for h in range(horizon):
        dow = (len(y)+h) % 7
        same = [recent.iloc[i] for i in range(len(recent)) if (len(y)-len(recent)+i) % 7 == dow]
        seasonal_adj = (np.mean(same) - recent_mean) if same else 0.0
        out.append(baseline + 0.3*seasonal_adj)
    return np.maximum(out, 0), None

def _fit_predict_enhanced_arima(y_train: pd.Series, horizon: int, order=(1,1,1)):
    try:
        from statsmodels.tsa.arima.model import ARIMA
        if y_train.std() < 1e-6:
            return np.full(horizon, y_train.mean()), None
        model = ARIMA(y_train, order=order).fit()
        fc = model.forecast(steps=horizon)
        ci = model.get_forecast(steps=horizon).conf_int()
        return fc.values, ci.values
    except Exception as e:
        return None, f"Enhanced ARIMA failed: {e}"

def _fit_predict_enhanced_sarimax(y_train: pd.Series, horizon: int, order=(1,1,1), seasonal_order=(1,1,1,7)):
    try:
        import statsmodels.api as sm
        mod = sm.tsa.statespace.SARIMAX(
            y_train, order=order, seasonal_order=seasonal_order,
            enforce_stationarity=False, enforce_invertibility=False, concentrate_scale=True
        )
        fit = mod.fit(disp=False, maxiter=100)
        fc = fit.forecast(steps=horizon)
        return fc.values, None
    except Exception as e:
        return None, f"Enhanced SARIMAX failed: {e}"

def _fit_predict_enhanced_prophet(y_train: pd.Series, horizon: int):
    try:
        from prophet import Prophet
    except Exception:
        return None, None
    try:
        dfp = pd.DataFrame({"ds": pd.DatetimeIndex(y_train.index), "y": y_train.values})
        m = Prophet(
            yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
            seasonality_mode="multiplicative", changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0, holidays_prior_scale=10.0, interval_width=0.95,
            mcmc_samples=0
        )
        m.fit(dfp)
        future = m.make_future_dataframe(periods=horizon, freq="D", include_history=False)
        fc = m.predict(future)
        return fc["yhat"].values, fc[["yhat_lower", "yhat_upper"]].values.T
    except Exception as e:
        print(f"Enhanced Prophet failed: {e}")
        return None, None

def _fit_predict_adaptive_seasonal(y_train: pd.Series, horizon: int, seasonality: int = 7, window: int = 4):
    if len(y_train) < seasonality*window:
        yhat, _ = _naive_seasonal(y_train, horizon, season_length=seasonality)
        return yhat, None
    last_idx = len(y_train)
    out = []
    for i in range(horizon):
        dow = (last_idx + i) % seasonality
        vals = y_train.iloc[(last_idx - seasonality*window + dow)::seasonality]
        out.append(vals.mean())
    return np.asarray(out), None

# ------------------------------ LSTM (optional) -------------------------------
def _fit_predict_enhanced_lstm(y_train: pd.Series, horizon: int, lookback: int = 90, epochs: int = 100):
    """Multifeature LSTM; returns (forecast, None) or (None, 'reason')."""
    try:
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return None, "tensorflow/keras not available"

    try:
        # ensure datetime index for feature engineering
        if not isinstance(y_train.index, pd.DatetimeIndex):
            y_train = y_train.copy()
            y_train.index = pd.to_datetime(y_train.index)  # <<< fix for '.month' error

        min_required = lookback + horizon + 30
        if len(y_train) < min_required:
            return None, f"Not enough data for LSTM: need {min_required}, have {len(y_train)}"

        feats = _add_time_features(y_train).fillna(method="bfill").fillna(method="ffill")
        cols = [c for c in feats.columns if c != "crime_count"]
        X_raw = feats[cols].values
        y_raw = feats["crime_count"].values

        fsc = StandardScaler(); tsc = StandardScaler()
        X = fsc.fit_transform(X_raw)
        y = tsc.fit_transform(y_raw.reshape(-1,1)).ravel()

        # build sequences (one-step ahead)
        Xs, ys = [], []
        for i in range(lookback, len(y)):
            Xs.append(X[i-lookback:i])
            ys.append(y[i])
        Xs = np.asarray(Xs); ys = np.asarray(ys)
        nfeat = Xs.shape[-1]

        # train/val split
        cut = int(0.85 * len(Xs))
        Xtr, Xva = Xs[:cut], Xs[cut:]
        ytr, yva = ys[:cut], ys[cut:]

        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(lookback, nfeat)),
            Dropout(0.2), BatchNormalization(),
            LSTM(32, return_sequences=True),
            Dropout(0.2), BatchNormalization(),
            LSTM(16, return_sequences=False),
            Dropout(0.1),
            Dense(32, activation="relu"), BatchNormalization(), Dropout(0.1),
            Dense(16, activation="relu"),
            Dense(1)
        ])
        model.compile(optimizer=Adam(1e-3), loss="mse", metrics=["mae"])
        cb = [EarlyStopping(patience=15, restore_best_weights=True, monitor="val_loss"),
              ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-7, monitor="val_loss")]
        model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=32, callbacks=cb, verbose=0)

        # iterative multi-step forecast
        seq = X[-lookback:]  # last lookback rows (already scaled)
        # build last window explicitly to avoid shape drift
        seq = X[-lookback:, :].copy()
        preds_scaled = []
        # We must roll features forward in a simple way; reuse last-feature vector with updated calendar terms
        cur_seq = seq.copy()
        last_date = y_train.index[-1]
        for step in range(horizon):
            # predict next scaled target
            x_in = cur_seq.reshape(1, lookback, nfeat)
            next_scaled = float(model.predict(x_in, verbose=0)[0,0])
            preds_scaled.append(next_scaled)

            # advance sequence by 1 row using last row as template (no ground-truth available here)
            cur_seq = np.roll(cur_seq, -1, axis=0)
            # we don't regenerate exact future exogenous features; keep last row (common for simple iterative LSTM)
            cur_seq[-1, :] = cur_seq[-2, :]

        preds = tsc.inverse_transform(np.asarray(preds_scaled).reshape(-1,1)).ravel()
        return np.maximum(preds, 0), None
    except Exception as e:
        return None, f"Enhanced LSTM failed: {e}"

# ------------------------------ Grid search -----------------------------------
def _enhanced_sarimax_grid(y: pd.Series, max_models: int = 150) -> Tuple[Tuple[int,int,int], Tuple[int,int,int,int]]:
    try:
        import statsmodels.api as sm
    except ImportError:
        return (1,1,1), (1,1,1,7)
    pR, dR, qR = [0,1,2,3], [0,1], [0,1,2]
    PR, DR, QR, s = [0,1,2], [0,1], [0,1,2], 7
    best_aic, best = np.inf, ((1,1,1),(1,1,1,7))
    tried = 0
    combos = [((p,d,q),(P,D,Q,s)) for p in pR for d in dR for q in qR for P in PR for D in DR for Q in QR]
    combos.sort(key=lambda x: sum(x[0])+sum(x[1][:3]))
    for order, seas in combos:
        if tried >= max_models: break
        try:
            fit = sm.tsa.statespace.SARIMAX(y, order=order, seasonal_order=seas,
                                            enforce_stationarity=False, enforce_invertibility=False,
                                            concentrate_scale=True).fit(disp=False, maxiter=50)
            if fit.aic < best_aic:
                best_aic, best = fit.aic, (order, seas)
            tried += 1
        except Exception:
            continue
    return best

# ------------------------------ Diagnostics/Plots -----------------------------
def diagnose_model_bias(y_train, y_test, forecasts_dict):
    print("=== BIAS DIAGNOSIS ===")
    print(f"Training period mean: {y_train.mean():.1f}")
    print(f"Test period mean: {y_test.mean():.1f}")
    shift = y_test.mean() - y_train.mean()
    pct = 100*shift/max(y_train.mean(), 1e-9)
    print(f"Shift: {shift:.1f} ({pct:+.1f}%)")
    recent_train = y_train.iloc[-90:]
    print(f"Recent training mean: {recent_train.mean():.1f}")
    print(f"Full training mean: {y_train.mean():.1f}\n")
    print("Model forecast biases:")
    for name, fc in forecasts_dict.items():
        bias = fc.mean() - y_test.mean()
        print(f"{name}: forecast={fc.mean():.1f}, bias={bias:+.1f}")

def analyze_residuals(actual, predicted, model_name):
    residuals = np.asarray(actual) - np.asarray(predicted)
    plt.figure(figsize=(15,10))
    plt.subplot(2,2,1); plt.plot(residuals, marker='o', alpha=0.7); plt.title(f'{model_name} - Residuals Over Time'); plt.grid(True, alpha=0.3)
    plt.subplot(2,2,2); plt.hist(residuals, bins=20, alpha=0.7, edgecolor='black'); plt.title(f'{model_name} - Residuals Distribution'); plt.grid(True, alpha=0.3)
    try:
        from scipy import stats
        plt.subplot(2,2,3); stats.probplot(residuals, dist="norm", plot=plt); plt.title(f'{model_name} - Q-Q Plot'); plt.grid(True, alpha=0.3)
    except Exception:
        pass
    plt.subplot(2,2,4); plt.scatter(actual, predicted, alpha=0.7); mn, mx = float(np.min(actual)), float(np.max(actual))
    plt.plot([mn, mx],[mn, mx], 'r--', lw=2); plt.title(f'{model_name} - Actual vs Predicted'); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt

# ------------------------------ Ensemble --------------------------------------
def create_ensemble_forecast(forecasts_dict: Dict[str, pd.Series], metrics_df: pd.DataFrame, top_n: int = 3) -> Optional[pd.Series]:
    corrected = {k: v for k, v in forecasts_dict.items() if 'corrected' in k}
    if not corrected:
        print("No corrected models found for ensembling.")
        return None
    top = metrics_df[metrics_df['model'].str.contains('corrected')].sort_values('RMSE').head(top_n)
    if top.empty:
        print("Not enough corrected models to create an ensemble.")
        return None
    top = top.copy()
    top['inv_rmse'] = 1.0 / top['RMSE']
    top['weight']   = top['inv_rmse'] / top['inv_rmse'].sum()
    base = list(corrected.values())[0]
    ens = pd.Series(np.zeros(len(base)), index=base.index)
    print(f"\nCreating ensemble from top {len(top)} models:")
    for _, r in top.iterrows():
        nm, w = r['model'], r['weight']
        print(f"  - {nm} (Weight: {w:.2f})")
        ens += corrected[nm].values * w
    ens = np.maximum(ens, 0)
    return pd.Series(ens, index=ens.index)

# ------------------------------ Enhanced Runner -------------------------------
def run_enhanced_lapd_forecasting(
    df_or_path: Union[pd.DataFrame, str],
    out_dir: str = "/content/figs_ts_enhanced",
    horizon: int = 30,
    eval_days: int = 365,
    rolling_splits: int = 5,
    run_models: Optional[List[str]] = None,
    cv_strategy: str = "expanding",
    rolling_train_days: Optional[int] = None,
    do_gridsearch: bool = True,
    verbose: bool = True,
):
    _ensure_dir(out_dir)
    charts_dir = _ensure_dir(os.path.join(out_dir, "charts"))
    tables_dir = _ensure_dir(os.path.join(out_dir, "tables"))

    if run_models is None:
        run_models = ["naive_last", "naive_seasonal", "arima", "sarimax", "prophet", "lstm", "adaptive"]

    # 1) Load data → daily series
    print("\n==== [ENHANCED-1] Loading and processing data ====")
    if isinstance(df_or_path, pd.DataFrame) and {'date','crime_count'}.issubset(df_or_path.columns):
        print("\n==== [ENHANCED-1] Using provided daily crime series DataFrame ====")
        if not pd.api.types.is_datetime64_any_dtype(df_or_path['date']):
            raise ValueError("'date' column must be datetime.")
        y = df_or_path.set_index("date")["crime_count"].astype(float)
        metadata = {'note': 'DataFrame provided directly'}
    else:
        print("\n==== [ENHANCED-1] Loading data and building daily crime series ====")
        df = _smart_read(df_or_path)
        y, metadata = _daily_counts_enhanced(df)

    print(f"Data span: {y.index.min().date()} → {y.index.max().date()}")
    print(f"Total days: {len(y)} | Mean: {y.mean():.1f} | Std: {y.std():.1f}")
    print(f"Zero days: {(y==0).sum()} | Max daily: {int(y.max())}")
    print(f"Data completeness: {100*(len(y)-(y==0).sum())/len(y):.1f}%")

    # 2) Stationarity checks (informational)
    print("\n==== [ENHANCED-2] Stationarity Analysis ====")
    try:
        from statsmodels.tsa.stattools import adfuller, kpss
        adf_stat, adf_p = adfuller(y.values, autolag='AIC')[:2]
        kpss_stat, kpss_p = kpss(y.values, regression='c')[:2]
        print(f"ADF test: stat={adf_stat:.3f}, p-value={adf_p:.3g}")
        print(f"KPSS test: stat={kpss_stat:.3f}, p-value={kpss_p:.3g}")
        print("✓ Series appears stationary (ADF)" if adf_p<0.05 else "⚠ Series may be non-stationary (ADF)")
        print("✓ Series appears stationary (KPSS)" if kpss_p>0.05 else "⚠ Series may be non-stationary (KPSS)")
    except Exception as e:
        print(f"Stationarity tests failed: {e}")

    # 3) CV setup
    total = len(y); min_train = max(200, horizon*3)
    if total < eval_days + horizon + min_train:
        eval_days = max(90, total - horizon - min_train)
        print(f"⚠ Adjusted eval_days to {eval_days} due to data constraints")
    start_eval = total - eval_days
    end_last  = total - horizon
    split_points = np.unique(np.linspace(start_eval, end_last, num=rolling_splits, dtype=int))
    split_points = [p for p in split_points if p >= min_train and p < total - horizon]
    if not split_points:
        raise ValueError("Cannot create valid CV splits with current parameters")

    print(f"\n==== [ENHANCED-3] Cross-Validation Setup ====")
    print(f"Strategy: {cv_strategy} | Splits: {len(split_points)} | Horizon: {horizon}")
    print(f"Training data range: {min_train} to {max(split_points)} days")

    # 4) Parameter search (first window)
    if do_gridsearch and len(split_points):
        print("\n==== [ENHANCED-4] Parameter Optimization ====")
        first_train = y.iloc[:split_points[0]]
        print("Optimizing SARIMAX parameters...")
        sarimax_params = _enhanced_sarimax_grid(first_train, max_models=150)
        arima_order = sarimax_params[0]
        print(f"Selected ARIMA order: {arima_order}")
        print(f"Selected SARIMAX: order={sarimax_params[0]}, seasonal={sarimax_params[1]}")
    else:
        arima_order = (1,1,1); sarimax_params = ((1,1,1),(1,1,1,7))

    # 5) Backtesting
    print("\n==== [ENHANCED-5] Model Backtesting ====")
    rows = []; last_forecasts = {}; last_test = None; last_cis = {}
    for s_idx, anchor in enumerate(split_points, start=1):
        train_start = max(0, anchor - rolling_train_days) if (cv_strategy.lower()=="rolling" and rolling_train_days) else 0
        y_train = y.iloc[train_start:anchor]; y_test = y.iloc[anchor:anchor+horizon]
        if len(y_test) < horizon: continue
        print(f"\n[Split {s_idx}/{len(split_points)}] Train: {len(y_train)} days | Test: {len(y_test)} days")
        print(f"  Period: {y_train.index.min().date()} → {y_train.index.max().date()} | {y_test.index.min().date()} → {y_test.index.max().date()}")

        for name in run_models:
            yhat, ci, err = None, None, None
            if name == "naive_last":
                yhat, ci = _naive_last(y_train, len(y_test))
            elif name == "naive_seasonal":
                yhat, ci = _naive_seasonal(y_train, len(y_test))
            elif name == "naive_adaptive":
                yhat, ci = _naive_adaptive(y_train, len(y_test))
            elif name == "arima":
                res = _fit_predict_enhanced_arima(y_train, len(y_test), order=arima_order)
                if isinstance(res, tuple) and res[0] is not None:
                    yhat, ci = res
                else:
                    yhat, err = (res[0] if isinstance(res, tuple) else None), (res[1] if isinstance(res, tuple) and len(res)>1 else None)
            elif name == "sarimax":
                yhat, err = _fit_predict_enhanced_sarimax(y_train, len(y_test), order=sarimax_params[0], seasonal_order=sarimax_params[1])
            elif name == "prophet":
                yhat, ci = _fit_predict_enhanced_prophet(y_train, len(y_test))
                if yhat is None: err = "Prophet failed"
            elif name == "lstm":
                yhat, err = _fit_predict_enhanced_lstm(y_train, len(y_test))
            elif name == "adaptive":
                yhat, ci = _fit_predict_adaptive_seasonal(y_train, len(y_test))
            else:
                continue

            if yhat is not None:
                yhat = np.maximum(yhat, 0)
                m = _metrics(y_test.values, yhat)
                m.update({"split": s_idx, "model": name, "train_size": len(y_train), "test_size": len(y_test)})
                rows.append(m)
                if s_idx == len(split_points):
                    last_forecasts[name] = pd.Series(yhat, index=y_test.index)
                    if ci is not None:
                        last_cis[name] = ci
                if verbose:
                    print(f"  ✓ {name.upper()}: MAE={m['MAE']:.1f}, RMSE={m['RMSE']:.1f}, MAPE={m['MAPE%']:.1f}%, SMAPE={m['SMAPE%']:.1f}%")
            else:
                if verbose:
                    print(f"  ✗ {name.upper()}: {err}")

        if s_idx == len(split_points):
            last_test = y_test

    if not rows:
        print("❌ No models produced forecasts"); return None

    metrics_df = pd.DataFrame(rows)
    display_metrics = (metrics_df.groupby("model")[["MAE","RMSE","MAPE%","SMAPE%"]]
                       .mean().reset_index().sort_values("RMSE"))
    _ensure_dir(tables_dir)
    metrics_df.to_csv(os.path.join(tables_dir, "detailed_metrics_enhanced.csv"), index=False)
    display_metrics.to_csv(os.path.join(tables_dir, "summary_metrics.csv"), index=False)

    print(f"\n==== [ENHANCED-6] Performance Summary ====")
    print("Average Performance (sorted by RMSE):")
    print(display_metrics.to_string(index=False))
    print(f"\n🏆 Best Model Ranking:")
    for i, r in display_metrics.iterrows():
        print(f"{i+1}. {r['model'].upper()}: RMSE={r['RMSE']:.1f}, MAE={r['MAE']:.1f}, MAPE={r['MAPE%']:.1f}%")

    return {
        "daily_series": y,
        "metadata": metadata,
        "metrics_detailed": metrics_df,
        "metrics_summary": display_metrics,
        "confidence_intervals": last_cis,
        "forecasts": last_forecasts,
        "actual_test": last_test,
        "model_params": {"arima_order": arima_order, "sarimax_params": sarimax_params},
        "paths": {"charts": charts_dir, "tables": tables_dir}
    }

# ------------------------------ Legacy Wrapper -------------------------------
def run_timeseries(
    df_or_path: Union[pd.DataFrame, str],
    out_dir: str = "/content/figs_ts",
    horizon: int = 30,
    eval_days: int = 90,
    rolling_splits: int = 3,
    run_models: Optional[List[str]] = None,
    date_col: Optional[str] = None,  # kept for compat, unused
    verbose: bool = True,
):
    """Thin wrapper around the enhanced runner to preserve older imports."""
    res = run_enhanced_lapd_forecasting(
        df_or_path=df_or_path,
        out_dir=out_dir,
        horizon=horizon,
        eval_days=eval_days,
        rolling_splits=rolling_splits,
        run_models=run_models or ["naive_last","naive_seasonal","sarimax","prophet","lstm"],
        cv_strategy="expanding",
        rolling_train_days=None,
        do_gridsearch=True,
        verbose=verbose
    )
    # adapt keys for old callers
    if not res:
        return None
    return {
        "metrics": res["metrics_detailed"],
        "metrics_avg": res["metrics_summary"],
        "forecasts": res["forecasts"],
        "y_test": res["actual_test"],
        "paths": res["paths"]
    }
