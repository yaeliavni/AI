#!/usr/bin/env python3
"""
Unsupervised Analysis Runner

This script provides a simple interface to run comprehensive unsupervised learning analysis.
It uses the UnsupervisedAnalyzer backend to perform dimensionality reduction, clustering,
and result interpretation.

Usage:
    python runner.py --data data.csv --output ./results
    python runner.py --data data.parquet --output ./results --max-samples 10000
    python runner.py --help

Author: Your Name
"""
# unsupervised_backend.py
from __future__ import annotations
from __future__ import annotations

import os, sys, warnings
from pathlib import Path
from typing import Iterable, Optional, Dict, Any, List, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import os, sys, warnings, re
from pathlib import Path
from typing import Iterable, Optional, Dict, Any, List

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
import argparse
import sys
import os
from pathlib import Path

# Import the backend


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive unsupervised learning analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python runner.py --data data.csv --output ./results

  # Limit samples and specify random se..."""
)
    
#add to unsupervised.py
# === unsupervised.py — add after you compute y for the best 2D plot ===========
from sklearn.metrics import silhouette_samples


# --------------------------------------------------------------------------
# --- Core Class
# --------------------------------------------------------------------------

from typing import Optional, Iterable, Dict, Any, Tuple, List
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# ---------- helpers ----------
def _make_ohe():
    """Create a OneHotEncoder that always returns dense output, compatible across sklearn versions."""
    try:
        # sklearn >= 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def _try_umap():
    try:
        import umap
        return umap.UMAP
    except Exception:
        return None

#!/usr/bin/env python3
"""
Unsupervised Analysis Runner (single-file)

- NaN-safe preprocessing (imputers in the pipeline)
- Clustering grid (KMeans 2..10 + small DBSCAN sweep)
- Dimensionality reduction sweep: PCA/SVD with 2..4 components
- Picks the best (space, algorithm, config) by silhouette
"""



# ---------- helpers ----------
def _make_ohe():
    """OneHotEncoder that always returns dense, across sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

# --------------------------------------------------------------------------
# --- Core Class
# --------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import re
from typing import Optional, Dict, Any, Tuple, List, Iterable
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

warnings.filterwarnings("ignore")

def _make_ohe():
    """Helper to create OneHotEncoder safely."""
    return OneHotEncoder(handle_unknown='ignore', sparse_output=False)

class UnsupervisedAnalyzer:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.df: Optional[pd.DataFrame] = None
        self.df_processed: Optional[pd.DataFrame] = None
        self.ct: Optional[ColumnTransformer] = None
        self.preprocessed_path: Optional[str] = None
        self.metrics_df: Optional[pd.DataFrame] = None

    # ---------------- Load + Preprocess ----------------
    def load_data(self, data_path: str, max_samples: Optional[int] = None) -> pd.DataFrame:
        print(f"📦 Loading data from: {data_path}...")
        ext = str(data_path).lower()
        if ext.endswith(".parquet"):
            try:
                df = pd.read_parquet(data_path, engine="pyarrow")
            except Exception:
                df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(
                data_path, low_memory=False, compression="infer", encoding_errors="ignore"
            )
        if max_samples and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=self.random_state).reset_index(drop=True)
        self.df = df
        print(f"[LOAD] Dataset shape: {df.shape}")
        return df

    def preprocess(self, df: pd.DataFrame, exclude_columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
        print("🔧 Preprocessing data...")
        exclude_columns = list(exclude_columns) if exclude_columns else []

        df = df.replace([np.inf, -np.inf], np.nan)

        numeric_cols = [c for c in df.select_dtypes(include=np.number).columns if c not in exclude_columns]
        categorical_cols = [c for c in df.select_dtypes(include=['object', 'category', 'bool']).columns if c not in exclude_columns]

        num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')),
                             ('scaler', StandardScaler())])
        cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                             ('ohe', _make_ohe())])

        self.ct = ColumnTransformer(
            transformers=[
                ('num', num_pipe, numeric_cols),
                ('cat', cat_pipe, categorical_cols),
            ],
            remainder='drop'
        )

        X = self.ct.fit_transform(df)
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X)

        # Last-resort safety
        if np.isnan(X).any():
            X = np.nan_to_num(X, nan=0.0)
        if np.isinf(X).any():
            X[~np.isfinite(X)] = 0.0

        # feature names
        try:
            feature_names = [n.split('__', 1)[-1] for n in self.ct.get_feature_names_out().tolist()]
        except Exception:
            feature_names = [f"f{i}" for i in range(X.shape[1])]

        self.df_processed = pd.DataFrame(X, columns=feature_names, index=df.index)
        print(f"✅ Preprocessing complete. Matrix shape: {self.df_processed.shape}")
        return self.df_processed

    # ---------------- Clustering on a given space ----------------
    def _cluster_space(self, space_df: pd.DataFrame, space_tag: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Run clustering grid on a provided feature space and return (metrics, results)."""
        metrics: List[Dict[str, Any]] = []
        results: Dict[str, Any] = {}

        # KMeans sweep
        for k in range(2, 11):
            try:
                km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                labels = km.fit_predict(space_df)
                if len(np.unique(labels)) > 1:
                    sil = silhouette_score(space_df, labels)
                    db = davies_bouldin_score(space_df, labels)
                    ch = calinski_harabasz_score(space_df, labels)
                    metrics.append({
                        'space': space_tag, 'model': 'KMeans', 'config': f'k={k}',
                        'n_clusters': k, 'silhouette': sil, 'davies_bouldin': db, 'calinski_harabasz': ch
                    })
                    results[f'{space_tag}__kmeans_{k}'] = {'clusters': labels, 'model': km, 'space': space_tag}
            except Exception as e:
                print(f"[WARN] KMeans(k={k}) failed in {space_tag}: {e}")

        # DBSCAN tiny sweep
        for eps in (0.3, 0.5, 0.7, 1.0):
            for ms in (5, 10):
                try:
                    dbs = DBSCAN(eps=eps, min_samples=ms)
                    labels = dbs.fit_predict(space_df)
                    n_eff = len(np.unique(labels)) - (1 if -1 in labels else 0)
                    if n_eff > 1:
                        mask = labels != -1
                        if mask.sum() > 1 and len(np.unique(labels[mask])) > 1:
                            sil = silhouette_score(space_df.iloc[mask], labels[mask])
                            db = davies_bouldin_score(space_df.iloc[mask], labels[mask])
                            ch = calinski_harabasz_score(space_df.iloc[mask], labels[mask])
                            metrics.append({
                                'space': space_tag, 'model': 'DBSCAN', 'config': f'eps={eps},min={ms}',
                                'n_clusters': n_eff, 'silhouette': sil, 'davies_bouldin': db, 'calinski_harabasz': ch
                            })
                            results[f'{space_tag}__dbscan_eps{eps}_min{ms}'] = {'clusters': labels, 'model': dbs, 'space': space_tag}
                except Exception as e:
                    print(f"[WARN] DBSCAN(eps={eps},min={ms}) failed in {space_tag}: {e}")

        return pd.DataFrame(metrics), results

    # ---------------- DR helpers ----------------
    def _try_pca(self, X: pd.DataFrame, n: int) -> Optional[pd.DataFrame]:
        try:
            p = PCA(n_components=n, random_state=self.random_state).fit_transform(X)
            return pd.DataFrame(p, index=X.index, columns=[f"pca_{i+1}" for i in range(n)])
        except Exception as e:
            print(f"[WARN] PCA(n={n}) failed: {e}")
            return None

    def _try_svd(self, X: pd.DataFrame, n: int) -> Optional[pd.DataFrame]:
        try:
            s = TruncatedSVD(n_components=n, random_state=self.random_state).fit_transform(X)
            return pd.DataFrame(s, index=X.index, columns=[f"svd_{i+1}" for i in range(n)])
        except Exception as e:
            print(f"[WARN] SVD(n={n}) failed: {e}")
            return None

    # ---------------- Viz ----------------
    def visualize(self, df: pd.DataFrame, embedding2d: np.ndarray, clusters: np.ndarray, title: str, out_dir: str, fname: str):
        os.makedirs(out_dir, exist_ok=True)
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=embedding2d[:, 0], y=embedding2d[:, 1], hue=clusters, palette='viridis', legend='full')
        plt.title(title)
        plt.xlabel("Component 1"); plt.ylabel("Component 2")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname))
        plt.close()

    # ---------------- Anomaly ----------------
    def run_anomaly_detection(self, df_processed: pd.DataFrame, output_dir: str):
        print("🕵️‍♂️ Running anomaly detection...")
        os.makedirs(output_dir, exist_ok=True)

        num = df_processed.select_dtypes(include=[np.number]).copy()
        num = num.replace([np.inf, -np.inf], np.nan)
        if num.isna().any().any():
            imputer = SimpleImputer(strategy='median')
            num[:] = imputer.fit_transform(num)

        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import LocalOutlierFactor

        lof = LocalOutlierFactor(n_neighbors=20, contamination='auto')
        lof_score = lof.fit_predict(num.values)

        iso_forest = IsolationForest(contamination='auto', random_state=self.random_state)
        iso_score = iso_forest.fit_predict(num.values)

        df_out = num.copy()
        df_out['lof_score'] = lof_score
        df_out['iso_score'] = iso_score

        try:
            p2 = PCA(n_components=2, random_state=self.random_state).fit_transform(num.values)
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            sns.scatterplot(x=p2[:, 0], y=p2[:, 1], hue=df_out['lof_score'], palette='coolwarm')
            plt.title("LOF Anomaly Detection")
            plt.subplot(1, 2, 2)
            sns.scatterplot(x=p2[:, 0], y=p2[:, 1], hue=df_out['iso_score'], palette='coolwarm')
            plt.title("Isolation Forest Anomaly Detection")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "anomaly_detection_2d.png"))
            plt.close()
            print("✅ Anomaly detection visualizations saved.")
        except Exception as e:
            print(f"[WARN] Anomaly visualization failed: {e}")

        n = len(df_out)
        rank = pd.Series(np.arange(1, n + 1) if n > 0 else np.full(n, np.nan))
        df_out.assign(mean_rank=rank).sort_values("mean_rank")\
              .to_csv(os.path.join(output_dir, "anomaly_rank.csv"), index=False)
        print(f"🧭 Unsupervised outlier visuals saved → {output_dir}")

    # ---------------- Orchestrator ----------------
    def run_full_analysis(
        self,
        data_path: str,
        output_dir: str,
        exclude_columns: Optional[Iterable[str]] = None,
        max_samples: Optional[int] = None,
        silhouette_threshold: float = 0.5
    ) -> str:
        df = self.load_data(data_path, max_samples=max_samples)
        self.preprocess(df, exclude_columns)

        # NaN-free now
        self.run_anomaly_detection(self.df_processed.copy(), os.path.join(output_dir, 'anomalies'))

        all_metrics_frames: List[pd.DataFrame] = []
        all_results: Dict[str, Any] = {}

        # 0) RAW processed space
        raw_metrics, raw_results = self._cluster_space(self.df_processed, "raw")
        all_metrics_frames.append(raw_metrics); all_results.update(raw_results)

        # 1) PCA/SVD spaces with n=2..4
        spaces: List[Tuple[str, Optional[pd.DataFrame], int]] = []
        for n in (2, 3, 4):
            p = self._try_pca(self.df_processed, n)
            if p is not None:
                spaces.append((f"pca{n}", p, n))
            s = self._try_svd(self.df_processed, n)
            if s is not None:
                spaces.append((f"svd{n}", s, n))

        for tag, Xspace, _n in spaces:
            m, r = self._cluster_space(Xspace, tag)
            all_metrics_frames.append(m); all_results.update(r)

        # Combine metrics across all spaces
        self.metrics_df = pd.concat(all_metrics_frames, ignore_index=True) if all_metrics_frames else pd.DataFrame()

        if not len(self.metrics_df):
            print("🤷 No valid clustering results found.")
            self.preprocessed_path = None
            return "No good clusters found."

        # Pick best by silhouette
        best = self.metrics_df.sort_values("silhouette", ascending=False).iloc[0]
        print(f"\n🏆 Best combo: [{best['space']}] {best['model']} ({best['config']}) "
              f"→ silhouette={best['silhouette']:.3f}")

        # Retrieve labels
        key_prefix = f"{best['space']}__"
        # find the matching result key
        chosen_key = None
        for k in all_results.keys():
            if k.startswith(key_prefix) and best['model'].lower() in k.lower() and best['config'].split('=')[0] in k:
                chosen_key = k; break
        if chosen_key is None:
            # fallback: first space-matching key
            chosen_key = next(k for k in all_results if k.startswith(key_prefix))
        best_res = all_results[chosen_key]
        labels = best_res['clusters']

        # 2D embedding for viz
        if best['space'] == "raw":
            emb2 = self._try_pca(self.df_processed, 2)
            if emb2 is None:
                emb2 = self._try_svd(self.df_processed, 2)
            emb2_arr = emb2.values if emb2 is not None else self.df_processed.iloc[:, :2].values
            emb_tag = "raw2d"
        else:
            Xspace = next(sp for (t, sp, _n) in spaces if t == best['space'])
            # if that space has >2 dims, take first 2
            emb2_arr = Xspace.iloc[:, :2].values
            emb_tag = f"{best['space']}_2d"

        viz_dir = os.path.join(output_dir, "visualizations")
        title = f"{best['space'].upper()} • {best['model']} ({best['config']})"
        self.visualize(df, emb2_arr, labels, title, viz_dir, f"{best['space']}_{best['model']}.png")

        # Save augmented original df with clusters and 2D coords
        out_df = df.copy()
        out_df['cluster_label'] = labels
        out_df['x_2d_plot'] = emb2_arr[:, 0]
        out_df['y_2d_plot'] = emb2_arr[:, 1]
        self.preprocessed_path = os.path.join(output_dir, 'preprocessed_with_clusters.csv')
        os.makedirs(output_dir, exist_ok=True)
        out_df.to_csv(self.preprocessed_path, index=False)
        print(f"💾 Data with cluster labels saved to: {self.preprocessed_path}")

        # Optional simple cluster stats
        try:
            self.analyze_clusters(df, labels, output_dir, f"{best['space']}_{best['model']}")
        except Exception as e:
            print(f"[WARN] analyze_clusters failed: {e}")

        # Optionally filter on threshold
        if float(best['silhouette']) < silhouette_threshold:
            print(f"ℹ️ Best silhouette {best['silhouette']:.3f} < threshold {silhouette_threshold:.3f}")
        return self.preprocessed_path
    
    def analyze_clusters(self, df_original: pd.DataFrame, clusters: np.ndarray, output_dir: str, prefix: str):
        """
        Summarize clusters using only numeric columns for stats, and
        save per-feature boxplots safely.
        """
        print("🔎 Analyzing cluster characteristics...")

        os.makedirs(output_dir, exist_ok=True)

        # Attach labels
        df_clustered = df_original.copy()
        df_clustered["cluster"] = clusters

        # Use numeric columns only for aggregation/plots
        num_cols = [c for c in df_clustered.select_dtypes(include=[np.number]).columns if c != "cluster"]
        if not num_cols:
            print("[WARN] No numeric columns to analyze; skipping stats/plots.")
            return

        # Robust aggregation (numeric only)
        try:
            stats = (
                df_clustered.groupby("cluster")[num_cols]
                .agg(["count", "mean", "median", "std"])
            )
            stats.to_csv(os.path.join(output_dir, f"{prefix}_cluster_stats.csv"))
        except Exception as e:
            print(f"[WARN] Failed to compute cluster stats: {e}")

        # Boxplots per numeric column (guarded)
        for col in num_cols:
            try:
                plt.figure(figsize=(10, 6))
                sns.boxplot(x="cluster", y=col, data=df_clustered, showfliers=False)
                plt.title(f"Distribution of {col} by Cluster")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{prefix}_{col}_boxplot.png"))
                plt.close()
            except Exception as e:
                print(f"[WARN] Boxplot failed for {col}: {e}")

        # (Optional) quick categorical peek: top levels per cluster
        cat_cols = [c for c in df_clustered.select_dtypes(include=["object", "category", "bool"]).columns if c != "cluster"]
        if cat_cols:
            try:
                topk = {}
                for c in cat_cols[:20]: # cap to keep files small
                    vc = (
                        df_clustered.groupby("cluster")[c]
                        .apply(lambda s: s.astype(str).value_counts(normalize=True).head(3))
                        .unstack(fill_value=0)
                    )
                    topk[c] = vc
                # Save one wide CSV per categorical feature
                for c, table in topk.items():
                    table.to_csv(os.path.join(output_dir, f"{prefix}_topcats_{re.sub('[^A-Za-z0-9_]+','_', c)}.csv"))
            except Exception as e:
                print(f"[WARN] Categorical summary failed: {e}")

# --------------------------------------------------------------------------
# Convenience wrappers
# --------------------------------------------------------------------------
def run_simple_analysis(
    data_path: str,
    output_dir: str,
    max_samples: Optional[int] = None,
    exclude_columns: Optional[Iterable[str]] = None,
    seed: int = 42
) -> str:
    return UnsupervisedAnalyzer(random_state=seed).run_full_analysis(
        data_path=data_path,
        output_dir=output_dir,
        exclude_columns=exclude_columns,
        max_samples=max_samples,
        silhouette_threshold=0.6,
    )

def run_unsupervised(data, output, max_samples=None, seed=42, interpret=False, top_n=10):
    analyzer = UnsupervisedAnalyzer(random_state=seed)
    out = analyzer.run_full_analysis(
        data_path=data,
        output_dir=output,
        max_samples=max_samples
    )
    # Only attempt interpretation if a ClusterInterpreter is actually available
    if interpret:
        try:
            from unsupervised import ClusterInterpreter  # or your real path
            ClusterInterpreter(random_state=seed).interpret_top_combinations(
                analyzer=analyzer,
                df=analyzer.load_data(data)
            )
        except Exception as e:
            print(f"[WARN] Interpretation skipped: {e}")
    return out


def run_simple_analysis(
    data_path,
    output_dir: str,
    max_samples: Optional[int] = None,
    exclude_columns: Optional[Iterable[str]] = None,
    seed: int = 42
) -> str:
    return UnsupervisedAnalyzer(random_state=seed).run_full_analysis(
        data_path=data_path,
        output_dir=output_dir,
        exclude_columns=exclude_columns,
        max_samples=max_samples,
        silhouette_threshold=0.6,
    )

# replace the existing run_unsupervised with this version
def run_unsupervised(data, output, max_samples=None, seed=42, interpret=False, top_n=10):
    # use the locally defined classes – no external backend module
    analyzer = UnsupervisedAnalyzer(random_state=seed)
    out = analyzer.run_full_analysis(
        data_path=data,
        output_dir=output,
        max_samples=max_samples
    )
    if interpret:
        ClusterInterpreter(random_state=seed).interpret_top_combinations(
            analyzer=analyzer,
            df=analyzer.load_data(data)
        )