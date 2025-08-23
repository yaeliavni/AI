
# unsupervised.py — Standalone reducer×clusterer search + plots + MI summary
import os, warnings, math
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, mutual_info_score

# Optional libs
try:
    import umap
    _HAS_UMAP = True
except Exception:
    _HAS_UMAP = False

try:
    import hdbscan
    _HAS_HDBSCAN = True
except Exception:
    _HAS_HDBSCAN = False

def _ensure_dir(d): os.makedirs(d, exist_ok=True)
def _plot_and_save(title, fig_dir):
    out = os.path.join(fig_dir, f"{title.replace(' ','-')}.png")
    try: plt.tight_layout(); plt.savefig(out, dpi=160); print(f"🖼️ saved: {out}")
    except Exception as e: print("(skip save)", e)
    plt.show()

def _prep_matrix(df: pd.DataFrame):
    num = sorted(list(df.select_dtypes(include=[np.number]).columns))
    cat = sorted([c for c in df.columns if df[c].dtype == "object"])
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])
    X = pre.fit_transform(df)
    return X

def _clusterers():
    out = [("kmeans_k8", KMeans(n_clusters=8, n_init=10, random_state=42)),
           ("agg_k8", AgglomerativeClustering(n_clusters=8)),
           ("dbscan", DBSCAN(eps=0.5, min_samples=10))]
    if _HAS_HDBSCAN:
        out.append(("hdbscan", hdbscan.HDBSCAN(min_cluster_size=30)))
    return out

def _reducers():
    out = [("pca2", PCA(n_components=2, random_state=42)),
           ("tsne2", TSNE(n_components=2, random_state=42, perplexity=30))]
    if _HAS_UMAP:
        out.append(("umap2", umap.UMAP(n_components=2, random_state=42)))
    return out

def run_unsupervised(data: str | pd.DataFrame, fig_dir: str = "./figs_unsup", max_samples: int = 30_000) -> str:
    _ensure_dir(fig_dir)
    df = pd.read_parquet(data) if isinstance(data,str) and data.endswith(".parquet") else (pd.read_csv(data) if isinstance(data,str) else data.copy())
    if len(df) > max_samples:
        df = df.sample(max_samples, random_state=42)

    X = _prep_matrix(df)

    rows = []
    best = (None, -np.inf, None, None)  # (name, silhouette, Xr, labels)

    for rname, red in _reducers():
        try:
            Xr = red.fit_transform(X)
        except Exception as e:
            print(f"{rname} failed:", e); continue
        for cname, clu in _clusterers():
            try:
                labels = clu.fit_predict(Xr)
            except Exception as e:
                print(f"{cname} failed:", e); continue

            # Valid labels check
            nclu = len(set(labels)) - (1 if -1 in labels else 0)
            if nclu < 2 or (labels == -1).all():
                sil = np.nan; ch = np.nan; db = np.nan
            else:
                mask = labels != -1
                Xeval = Xr[mask]; leval = labels[mask]
                if len(set(leval)) < 2 or len(leval) < 20:
                    sil = np.nan
                else:
                    sil = silhouette_score(Xeval, leval)
                ch = calinski_harabasz_score(Xeval, leval) if len(set(leval))>1 else np.nan
                db = davies_bouldin_score(Xeval, leval) if len(set(leval))>1 else np.nan

            rows.append({"reducer": rname, "clusterer": cname, "silhouette": sil, "calinski": ch, "davies": db, "n_clusters": nclu})
            if not np.isnan(sil) and sil > best[1]:
                best = (f"{rname}+{cname}", sil, Xr, labels)

    res = pd.DataFrame(rows)
    out_csv = os.path.join(fig_dir, "unsupervised_results.csv")
    res.to_csv(out_csv, index=False)
    print(f"✓ Results table saved: {out_csv}")

    if best[2] is not None:
        Xr, labels = best[2], best[3]
        plt.figure(figsize=(8,6))
        plt.scatter(Xr[:,0], Xr[:,1], c=labels, s=6, alpha=0.7, cmap="tab10")
        plt.title(f"Best embedding — {best[0]} (sil={best[1]:.3f})")
        _plot_and_save("Best embedding — color by label", fig_dir)

        # Simple MI of embedding-grid vs labels (proxy)
        try:
            # Bin coordinates and compute MI
            xbin = pd.qcut(Xr[:,0], q=20, duplicates="drop").astype(str)
            ybin = pd.qcut(Xr[:,1], q=20, duplicates="drop").astype(str)
            xy = (xbin + "|" + ybin)
            le = pd.Series(labels).astype(int)
            mi = mutual_info_score(xy, le)
            plt.figure(); plt.bar(["MI(emb bins, labels)"], [mi]); plt.title("Mutual information (proxy)")
            _plot_and_save("MI vs clusters — proxy", fig_dir)
        except Exception as e:
            print("MI plot skipped:", e)

    return out_csv
