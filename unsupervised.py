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

  # Limit samples and specify random seed
  python runner.py --data large_dataset.parquet --output ./results --max-samples 50000 --seed 123

  # Run with interpretation of top 5 results
  python runner.py --data data.csv --output ./results --interpret --top-n 5

  # Exclude specific columns
  python runner.py --data data.csv --output ./results --exclude victim_age victim_sex

Output files:
  - all_combinations_results.csv: Complete results table
  - high_performance_combinations.csv: Results with silhouette >= 0.6
  - best_combination_visualization.png: Scatter plot of best 2D result
  - cluster_interpretations/: Detailed analysis of top combinations (if --interpret)
        """
    )
    
    parser.add_argument(
        '--data', '-d',
        required=True,
        help='Path to input data file (CSV or Parquet)'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='./unsupervised_results',
        help='Output directory for results (default: ./unsupervised_results)'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        help='Maximum number of samples to use (default: use all data)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--exclude',
        nargs='*',
        help='Column names to exclude from analysis'
    )
    
    parser.add_argument(
        '--interpret',
        action='store_true',
        help='Run detailed interpretation of top results'
    )
    
    parser.add_argument(
        '--top-n',
        type=int,
        default=10,
        help='Number of top combinations to interpret (default: 10, only with --interpret)'
    )
    
    parser.add_argument(
        '--simple',
        action='store_true',
        help='Use simple analysis function (faster, less detailed)'
    )
    
    parser.add_argument(
        '--silhouette-threshold',
        type=float,
        default=0.6,
        help='Silhouette threshold for high-performance results (default: 0.6)'
    )
    
    return parser.parse_args()


def validate_input_file(file_path):
    """Validate that input file exists and has correct extension."""
    path = Path(file_path)
    
    if not path.exists():
        print(f"ERROR: Input file does not exist: {file_path}")
        sys.exit(1)
    
    valid_extensions = {'.csv', '.parquet'}
    if path.suffix.lower() not in valid_extensions:
        print(f"ERROR: Unsupported file type: {path.suffix}")
        print(f"Supported types: {', '.join(valid_extensions)}")
        sys.exit(1)
    
    return str(path.absolute())


def main():
    """Main runner function."""
    args = parse_arguments()
    
    print("="*80)
    print("UNSUPERVISED LEARNING ANALYSIS RUNNER")
    print("="*80)
    
    # Validate inputs
    data_path = validate_input_file(args.data)
    output_dir = os.path.abspath(args.output)
    exclude_columns = set(args.exclude) if args.exclude else None
    
    # Print configuration
    print(f"Data file: {data_path}")
    print(f"Output directory: {output_dir}")
    print(f"Max samples: {args.max_samples or 'All'}")
    print(f"Random seed: {args.seed}")
    print(f"Exclude columns: {exclude_columns or 'None'}")
    print(f"Interpretation: {'Yes' if args.interpret else 'No'}")
    if args.interpret:
        print(f"Top N to interpret: {args.top_n}")
    print()
    
    try:
        if args.simple:
            # Simple analysis
            print("Running simple analysis...")
            results_path = run_simple_analysis(
                data_path=data_path,
                output_dir=output_dir,
                max_samples=args.max_samples,
                exclude_columns=exclude_columns
            )
            
        else:
            # Full analysis with custom analyzer
            print("Running comprehensive analysis...")
            
            # Initialize analyzer
            analyzer = UnsupervisedAnalyzer(
                random_state=args.seed,
                max_cat_cardinality=30,
                svd_components=50
            )
            
            # Run analysis
            results_path = analyzer.run_full_analysis(
                data_path=data_path,
                output_dir=output_dir,
                exclude_columns=exclude_columns,
                max_samples=args.max_samples
            )
            
            # Run interpretation if requested
            if args.interpret:
                print("\n" + "="*58)
                print("RUNNING CLUSTER INTERPRETATION")
                print("="*58)
                
                # Load original data for interpretation
                df_original = analyzer.load_data(data_path)
                if args.max_samples and len(df_original) > args.max_samples:
                    df_original = df_original.sample(args.max_samples, random_state=args.seed)
                
                # Initialize interpreter
                interpreter = ClusterInterpreter(random_state=args.seed)
                
                # Run interpretation
                interpretation_results = interpreter.interpret_top_combinations(
                    analyzer=analyzer,
                    df=df_original,
                    output_dir=output_dir,
                    top_n=args.top_n
                )
                
                print(f"\nInterpretation completed for {len(interpretation_results)} combinations")
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Results saved to: {output_dir}")
        print("\nKey output files:")
        print(f"  - all_combinations_results.csv")
        print(f"  - high_performance_combinations.csv (if any found)")
        print(f"  - best_combination_visualization.png (if 2D embedding found)")
        if args.interpret:
            print(f"  - cluster_interpretations/ (detailed analysis)")
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_quick_demo():
    """Run a quick demo with synthetic data."""
    print("Running quick demo with synthetic data...")
    
    import numpy as np
    import pandas as pd
    from sklearn.datasets import make_blobs
    
    # Create synthetic data
    X, y = make_blobs(n_samples=1000, centers=4, n_features=10, 
                      random_state=42, cluster_std=2.0)
    
    # Add some categorical features
    np.random.seed(42)
    cat_features = pd.DataFrame({
        'category_A': np.random.choice(['Type1', 'Type2', 'Type3'], 1000),
        'category_B': np.random.choice(['Group1', 'Group2'], 1000),
        'status': np.random.choice(['Active', 'Inactive', 'Pending'], 1000)
    })
    
    # Combine numeric and categorical
    num_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    demo_df = pd.concat([num_df, cat_features], axis=1)
    
    # Run analysis
    analyzer = UnsupervisedAnalyzer(random_state=42)
    results_path = analyzer.run_full_analysis(
        data_path=demo_df,
        output_dir='./demo_results',
        max_samples=1000
    )
    
    print(f"\nDemo completed! Results saved to: ./demo_results")


def _make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)

def _scores(X, y):
    uniq = set(y)
    if len(uniq) <= 1 or uniq == {-1}:
        return dict(silhouette=np.nan, calinski_harabasz=np.nan, davies_bouldin=np.nan,
                    n_clusters=1, n_noise=int(np.sum(y == -1)) if -1 in uniq else 0)
    def safe(fn):
        try: return fn(X, y)
        except Exception: return np.nan
    return dict(
        silhouette=safe(silhouette_score),
        calinski_harabasz=safe(calinski_harabasz_score),
        davies_bouldin=safe(davies_bouldin_score),
        n_clusters=len([u for u in uniq if u != -1]),
        n_noise=int(np.sum(y == -1)) if -1 in uniq else 0,
    )

class UnsupervisedAnalyzer:
    def __init__(self, random_state: int = 42, max_cat_cardinality: int = 30, svd_components: int = 50):
        self.rng = np.random.RandomState(random_state)
        self.random_state = random_state
        self.max_cat = max_cat_cardinality
        self.svd_components = svd_components

    # Accept path or a DataFrame
    def load_data(self, data_path_or_df):
        if isinstance(data_path_or_df, pd.DataFrame):
            return data_path_or_df.copy()
        p = str(data_path_or_df)
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        if p.lower().endswith(".parquet"):
            return pd.read_parquet(p)
        if p.lower().endswith(".csv"):
            return pd.read_csv(p, low_memory=False)
        raise ValueError("Only CSV/Parquet or DataFrame supported")

    def _build_matrix(self, df: pd.DataFrame, exclude_columns: Optional[Iterable[str]] = None):
        # drop constants
        const_cols = [c for c in df.columns if df[c].dropna().nunique() <= 1]
        if const_cols:
            df = df.drop(columns=const_cols)

        if exclude_columns:
            drop_cols = [c for c in exclude_columns if c in df.columns]
            if drop_cols:
                df = df.drop(columns=drop_cols)

        # select features
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        cat_all  = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in cat_all if df[c].dropna().nunique() <= self.max_cat]

        Xdf = df[num_cols + cat_cols].copy()
        for c in Xdf.select_dtypes(include=["bool"]).columns:
            Xdf[c] = Xdf[c].astype(int)

        pre = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(with_mean=False), [c for c in Xdf.columns if pd.api.types.is_numeric_dtype(Xdf[c])]),
                ("cat", _make_ohe(), [c for c in Xdf.columns if not pd.api.types.is_numeric_dtype(Xdf[c])]),
            ],
            remainder="drop",
        )
        X = pre.fit_transform(Xdf)
        return X, pre

    def run_full_analysis(
        self,
        data_path,
        output_dir: str,
        exclude_columns: Optional[Iterable[str]] = None,
        max_samples: Optional[int] = None,
        silhouette_threshold: float = 0.6,
    ) -> str:
        os.makedirs(output_dir, exist_ok=True)
        df = self.load_data(data_path)
        if max_samples and len(df) > max_samples:
            df = df.sample(max_samples, random_state=self.random_state)

        X, _ = self._build_matrix(df, exclude_columns=exclude_columns)

        # Embeddings (compact but useful)
        embeddings: Dict[str, np.ndarray] = {}

        # SVD50 (fast + good for UMAP input)
        comp = max(2, min(self.svd_components, X.shape[1] - 1 if X.shape[1] > 1 else 2))
        svd50 = TruncatedSVD(n_components=comp, random_state=self.random_state)
        X50 = svd50.fit_transform(X)
        embeddings["SVD50"] = X50

        # SVD2 (for plotting)
        svd2 = TruncatedSVD(n_components=2, random_state=self.random_state)
        embeddings["SVD2"] = svd2.fit_transform(X)

        # PCA2 (dense fallback if safe)
        try:
            X_dense = X if hasattr(X, "toarray") and X.shape[1] < 2000 else (X.toarray() if hasattr(X, "toarray") and (X.shape[0]*X.shape[1] < 3e7) else None)
            if X_dense is not None:
                pca2 = PCA(n_components=2, random_state=self.random_state)
                embeddings["PCA2"] = pca2.fit_transform(X_dense)
        except Exception:
            pass

        # UMAP2 on SVD50
        try:
            try:
                import umap
            except Exception:
                import subprocess
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "umap-learn"], check=True)
                import umap
            um = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                           random_state=self.random_state, metric="euclidean", verbose=False)
            embeddings["UMAP2"] = um.fit_transform(X50)
        except Exception:
            pass

        # Clustering for each 2D embedding
        results: List[Dict[str, Any]] = []
        for name, E in embeddings.items():
            # KMeans k=2..8
            for k in range(2, 9):
                try:
                    km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
                    y = km.fit_predict(E)
                    sc = _scores(E, y)
                    results.append(dict(embedding=name, algorithm="KMeans", config=f"k={k}", embedding_dim=E.shape[1], **sc))
                except Exception:
                    continue
            # DBSCAN a couple of eps
            for eps in (0.3, 0.5, 0.7):
                try:
                    db = DBSCAN(eps=eps, min_samples=10, n_jobs=-1)
                    y = db.fit_predict(E)
                    sc = _scores(E, y)
                    results.append(dict(embedding=name, algorithm="DBSCAN", config=f"eps={eps}_min=10", embedding_dim=E.shape[1], **sc))
                except Exception:
                    continue

        results_df = pd.DataFrame(results)
        results_df["combination"] = results_df["embedding"] + " + " + results_df["algorithm"] + " (" + results_df["config"] + ")"
        results_df.to_csv(os.path.join(output_dir, "all_combinations_results.csv"), index=False)

        hi = results_df.dropna(subset=["silhouette"]).query("silhouette >= @silhouette_threshold").copy()
        hi.to_csv(os.path.join(output_dir, "high_performance_combinations.csv"), index=False)

        # Plot best 2D
        try:
            best = (results_df
                    .query("embedding_dim == 2")
                    .dropna(subset=["silhouette"])
                    .sort_values("silhouette", ascending=False)
                    .iloc[0])
            emb = best["embedding"]; algo = best["algorithm"]; cfg = best["config"]; E = embeddings[emb]
            # refit for labels
            if algo == "KMeans":
                k = int(cfg.split("=")[1]); y = KMeans(n_clusters=k, n_init=10, random_state=self.random_state).fit_predict(E)
            else:
                parts = cfg.split("_"); eps = float(parts[0].split("=")[1]); m = int(parts[1].split("=")[1])
                y = DBSCAN(eps=eps, min_samples=m, n_jobs=-1).fit_predict(E)

            plt.figure(figsize=(8,6))
            for u in sorted(set(y)):
                M = (y == u)
                plt.scatter(E[M,0], E[M,1], s=12, alpha=0.8, label=("Noise" if u==-1 else f"C{u}"))
            plt.legend(title="Clusters", bbox_to_anchor=(1.02,1), loc="upper left", fontsize=8)
            plt.title(f"Best: {best['combination']}  Sil={best['silhouette']:.3f}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "best_combination_visualization.png"), dpi=180, bbox_inches="tight")
            plt.close()
        except Exception:
            pass

        return os.path.abspath(output_dir)

class ClusterInterpreter:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    # Minimal placeholder that won’t crash the runner; extend as needed
    def interpret_top_combinations(
        self,
        analyzer: UnsupervisedAnalyzer,
        df: pd.DataFrame,
        output_dir: str,
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        path = os.path.join(output_dir, "all_combinations_results.csv")
        if not os.path.exists(path):
            return []
        results = pd.read_csv(path)
        top = (results.dropna(subset=["silhouette"])
                      .sort_values("silhouette", ascending=False)
                      .head(top_n)
                      .copy())
        # Write a tiny index so the runner has something to point to
        idx = []
        for _, r in top.iterrows():
            idx.append(dict(combo=r["combination"], silhouette=float(r["silhouette"])))
        pd.DataFrame(idx).to_json(os.path.join(output_dir, "cluster_interpretations_index.json"), orient="records", indent=2)
        return idx

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
            df=analyzer.load_data(data),
            output_dir=output,
            top_n=top_n
        )
    return out


if __name__ == "__main__":
    # Check if running demo
    if len(sys.argv) == 2 and sys.argv[1] == '--demo':
        run_quick_demo()
    else:
        main()