
# supervised.py — Standalone binary supervised demo with SHAP preflight
import os, warnings, json
from typing import Tuple, Callable
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    RocCurveDisplay, PrecisionRecallDisplay, confusion_matrix
)

# Optional shap
try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

def _ensure_dir(d): os.makedirs(d, exist_ok=True)
def _plot_and_save(title, fig_dir):
    out = os.path.join(fig_dir, f"{title.replace(' ','-')}.png")
    try: plt.tight_layout(); plt.savefig(out, dpi=160); print(f"🖼️ saved: {out}")
    except Exception as e: print("(skip save)", e)
    plt.show()

def stratified_sample_for_task(df: pd.DataFrame, target_col: str, n: int, random_state: int = 42) -> pd.DataFrame:
    if target_col not in df: return df.sample(min(n, len(df)), random_state=random_state)
    groups = df[target_col].dropna().astype(str)
    frac = min(1.0, n/len(df))
    sampled = df.groupby(groups, group_keys=False).apply(lambda g: g.sample(max(1,int(len(g)*frac)), random_state=random_state))
    return sampled.reset_index(drop=True)

def _make_preprocessor(df: pd.DataFrame):
    num = sorted(list(df.select_dtypes(include=[np.number]).columns))
    cat = sorted([c for c in df.columns if df[c].dtype == "object"])
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])
    return pre

def _shap_beeswarm(model, X, fig_dir, title="SHAP beeswarm — preflight RF"):
    if not _HAS_SHAP: 
        print("SHAP not installed; skipping beeswarm."); return
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        plt.figure()
        shap.summary_plot(shap_values, X, show=False)
        _plot_and_save(title, fig_dir)
    except Exception as e:
        print("SHAP failed:", e)

def run_supervised(
    data: str | pd.DataFrame,
    fig_dir: str = "./figs_sup",
    target: Tuple[str, Callable[[pd.DataFrame], pd.Series]] = ("is_night", lambda df: (df["hour"].between(22,23) | df["hour"].between(0,5)).astype(int)),
    sample_n: int = 30_000
) -> str:
    _ensure_dir(fig_dir)
    df = pd.read_parquet(data) if isinstance(data,str) and data.endswith(".parquet") else (pd.read_csv(data) if isinstance(data,str) else data.copy())

    if target[0] not in df:
        df[target[0]] = target[1](df)
    df = df.dropna(subset=[target[0]])

    if len(df) > sample_n:
        df = stratified_sample_for_task(df, target_col=target[0], n=sample_n, random_state=42)

    y = df[target[0]].astype(int).values
    Xdf = df.drop(columns=[target[0]])

    pre = _make_preprocessor(Xdf)
    X = pre.fit_transform(Xdf)

    # SHAP preflight on RF
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    try:
        rf.fit(X, y)
        _shap_beeswarm(rf, X, fig_dir)
    except Exception as e:
        print("RF/SHAP preflight skipped:", e)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    models = [("LogReg", LogisticRegression(max_iter=1000)), ("RF", RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1))]

    report = {}; best_auc, best_name, best_proba = -np.inf, None, None
    for name, mdl in models:
        mdl.fit(X_train, y_train)
        proba = mdl.predict_proba(X_test)[:,1] if hasattr(mdl, "predict_proba") else mdl.decision_function(X_test)
        auc = roc_auc_score(y_test, proba); ap = average_precision_score(y_test, proba)
        report[name] = {"AUC": float(auc), "AP": float(ap)}
        RocCurveDisplay.from_predictions(y_test, proba); _plot_and_save(f"ROC — {name}", fig_dir)
        PrecisionRecallDisplay.from_predictions(y_test, proba); _plot_and_save(f"PR — {name}", fig_dir)
        if auc > best_auc:
            best_auc, best_name, best_proba = auc, name, proba

    if best_name is not None:
        pred = (best_proba >= 0.5).astype(int)
        cm = confusion_matrix(y_test, pred)
        print(f"Best model: {best_name} (AUC={best_auc:.3f})\nConfusion matrix:\n{cm}")

    out_json = os.path.join(fig_dir, "supervised_report.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"✓ Supervised metrics saved: {out_json}")
    return out_json
