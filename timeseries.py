
# timeseries.py — Standalone daily aggregation + multi-model forecasting
import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _ensure_dir(d): os.makedirs(d, exist_ok=True)
def _plot_and_save(title, fig_dir):
    out = os.path.join(fig_dir, f"{title.replace(' ','-')}.png")
    try: plt.tight_layout(); plt.savefig(out, dpi=160); print(f"🖼️ saved: {out}")
    except Exception as e: print("(skip save)", e)
    plt.show()

# Optional libs
try:
    import pmdarima as pm
    _HAS_PM = True
except Exception:
    _HAS_PM = False

try:
    import statsmodels.api as sm
    _HAS_SM = True
except Exception:
    _HAS_SM = False

try:
    from prophet import Prophet
    _HAS_PROPHET = True
except Exception:
    _HAS_PROPHET = False

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    _HAS_TF = True
except Exception:
    _HAS_TF = False

def _mape(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    denom = np.maximum(1e-9, np.abs(y_true))
    return float(np.mean(np.abs(y_true - y_pred)/denom)) * 100.0

def _auto_arima_forecast(train: pd.Series, horizon: int):
    if not _HAS_PM: raise RuntimeError("pmdarima not installed")
    m = pm.auto_arima(train, seasonal=False, stepwise=True, suppress_warnings=True)
    pred = m.predict(horizon)
    return m, np.asarray(pred)

def _sarimax_forecast(train: pd.Series, horizon: int):
    if not _HAS_SM: raise RuntimeError("statsmodels not installed")
    m = sm.tsa.SARIMAX(train, order=(1,0,1), seasonal_order=(0,0,0,0)).fit(disp=False)
    pred = m.forecast(steps=horizon)
    return m, np.asarray(pred)

def _prophet_forecast(train: pd.Series, horizon: int):
    if not _HAS_PROPHET: raise RuntimeError("prophet not installed")
    df = pd.DataFrame({"ds": train.index, "y": train.values})
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=horizon, freq="D")
    fcst = m.predict(future).tail(horizon)["yhat"].values
    return m, np.asarray(fcst)

def _build_lstm_windowed(train: pd.Series, lookback: int = 60, horizon: int = 7, epochs: int = 60, verbose: int = 0):
    if not _HAS_TF: raise RuntimeError("tensorflow not installed")
    x, y = [], []
    a = train.values.astype("float32")
    for i in range(lookback, len(a)-horizon+1):
        x.append(a[i-lookback:i]); y.append(a[i:i+horizon])
    x = np.array(x)[..., None]; y = np.array(y)
    model = models.Sequential([
        layers.Input(shape=(lookback,1)),
        layers.LSTM(64),
        layers.Dense(horizon)
    ])
    model.compile(optimizer="adam", loss="mae")
    model.fit(x, y, epochs=epochs, batch_size=32, verbose=verbose)
    # last window
    last_win = a[-lookback:][None, :, None]
    pred = model.predict(last_win, verbose=0).ravel()
    return model, pred

def run_timeseries(
    data: str | pd.DataFrame,
    fig_dir: str = "./figs_ts",
    horizon: int = 7, lookback: int = 60, eval_days: int = 28, verbose: bool = False
) -> str:
    _ensure_dir(fig_dir)
    df = pd.read_parquet(data) if isinstance(data,str) and data.endswith(".parquet") else (pd.read_csv(data) if isinstance(data,str) else data.copy())
    if "date" not in df:
        raise ValueError("Data must contain a 'date' column from EDA output.")
    s = df.groupby(pd.to_datetime(df["date"]).dt.date).size()
    s.index = pd.to_datetime(s.index)
    s = s.asfreq("D").fillna(0.0)

    train = s.iloc[:-eval_days] if len(s) > eval_days else s
    val = s.iloc[-eval_days:] if len(s) > eval_days else None

    results = []

    # LSTM
    try:
        mdl, pred = _build_lstm_windowed(train, lookback=lookback, horizon=horizon, epochs=60, verbose=0)
        if val is not None and len(val) >= horizon:
            results.append(("LSTM", _mape(val.values[-horizon:], pred[:horizon])))
        plt.figure(); plt.plot(s.index[-len(pred):], pred); plt.title("LSTM forecast")
        _plot_and_save("LSTM forecast", fig_dir)
    except Exception as e:
        print("LSTM failed:", e)

    # auto_arima
    try:
        m, pred = _auto_arima_forecast(train, horizon=horizon)
        if val is not None and len(val) >= horizon:
            results.append(("auto_arima", _mape(val.values[-horizon:], pred[:horizon])))
        plt.figure(); plt.plot(s.index[-len(pred):], pred); plt.title("auto_arima forecast")
        _plot_and_save("auto_arima forecast", fig_dir)
    except Exception as e:
        print("auto_arima failed:", e)

    # SARIMAX
    try:
        m, pred = _sarimax_forecast(train, horizon=horizon)
        if val is not None and len(val) >= horizon:
            results.append(("sarimax", _mape(val.values[-horizon:], pred[:horizon])))
        plt.figure(); plt.plot(s.index[-len(pred):], pred); plt.title("SARIMAX forecast")
        _plot_and_save("SARIMAX forecast", fig_dir)
    except Exception as e:
        print("SARIMAX failed:", e)

    # Prophet
    try:
        m, pred = _prophet_forecast(train, horizon=horizon)
        if val is not None and len(val) >= horizon:
            results.append(("prophet", _mape(val.values[-horizon:], pred[:horizon])))
        plt.figure(); plt.plot(s.index[-len(pred):], pred); plt.title("Prophet forecast")
        _plot_and_save("Prophet forecast", fig_dir)
    except Exception as e:
        print("Prophet failed:", e)

    if results:
        dfres = pd.DataFrame(results, columns=["model","val_MAPE"]).sort_values("val_MAPE")
        out_csv = os.path.join(fig_dir, "timeseries_val_mape.csv")
        dfres.to_csv(out_csv, index=False)
        print(f"✓ Time-series validation results: {out_csv}")
        return out_csv
    else:
        print("No time-series result to save.")
        return ""
