# supervised_ml.py — Consolidated supervised learning framework with CV and tuning
"""
A comprehensive supervised machine learning framework with:
- Automated preprocessing and feature engineering
- Multiple model types with hyperparameter tuning
- Cross-validation and model comparison
- Visualization and results export
- Stratified sampling capabilities
"""

import os
import warnings
import json
import time
from typing import Tuple, Callable, Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Core sklearn imports
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV,
    learning_curve, validation_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, accuracy_score,
    RocCurveDisplay, PrecisionRecallDisplay, confusion_matrix, roc_curve,
    classification_report, precision_recall_curve
)

# Optional dependencies with graceful fallbacks
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    XGBClassifier = None
    _HAS_XGB = False

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

try:
    from sklearn.inspection import permutation_importance
    _HAS_PERM_IMPORTANCE = True
except ImportError:
    _HAS_PERM_IMPORTANCE = False


# ========================= CONFIGURATION =========================

@dataclass
class MLConfig:
    """Configuration for machine learning experiments."""
    # Core settings
    random_state: int = 42
    n_jobs: int = -1
    verbose: int = 1
    
    # Cross-validation settings
    n_splits: int = 5
    scoring: str = "roc_auc"
    
    # Hyperparameter search settings
    search_type: str = "random"  # "random" or "grid"
    n_iter: int = 30
    
    # Data sampling settings
    sample_n: Optional[int] = 5000
    sample_first: bool = True
    stratify_by: Optional[str] = None
    equalize_strata: bool = False
    
    # Evaluation settings
    test_size: Optional[float] = None
    
    # Output settings
    fig_dir: Optional[str] = None
    plot_curves: bool = True
    save_results: bool = True
    
    # Advanced settings
    class_weight: str = "balanced"
    handle_imbalance: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.test_size is not None and not (0 < self.test_size < 1):
            raise ValueError("test_size must be between 0 and 1")
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if self.search_type not in ["random", "grid"]:
            raise ValueError("search_type must be 'random' or 'grid'")


# ========================= UTILITY FUNCTIONS =========================

def ensure_dir(directory: str) -> Path:
    """Create directory if it doesn't exist and return Path object."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_and_save(title: str, fig_dir: str, dpi: int = 160) -> None:
    """Save the current Matplotlib figure to a PNG file.

    This helper ensures that the output directory exists and always uses
    a ``pathlib.Path`` for file operations.  Without converting
    ``ensure_dir``'s return value to a ``Path``, attempting to use the
    ``/`` operator on a plain string results in a ``TypeError`` like
    ``unsupported operand type(s) for /: 'str' and 'str'``.

    Args:
        title: Descriptive name for the plot; this becomes the base
            filename (sanitized) for the image.
        fig_dir: Destination directory where the plot should be saved.
            If falsy, the figure is not saved and the function
            returns immediately.
        dpi: Image resolution in dots per inch.

    Side effects:
        A ``.png`` file is written to ``fig_dir`` and the current
        Matplotlib figure is closed to free memory.
    """
    if not fig_dir:
        # No output directory provided -> skip saving
        return

    # Ensure the directory exists and normalise to a Path instance.  In
    # certain environments ``ensure_dir`` may return a string; convert to
    # Path so we can reliably use the ``/`` operator.
    out_dir = ensure_dir(fig_dir)
    if not isinstance(out_dir, Path):
        out_dir = Path(out_dir)

    # Sanitize the title to create a safe filename
    safe_title = "".join(c if c.isalnum() or c in "._-" else "_" for c in title)
    filename = out_dir / f"{safe_title}.png"

    try:
        plt.tight_layout()
        plt.savefig(filename, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"🖼️ Saved: {filename}")
    except Exception as e:
        print(f"❌ Failed to save {filename}: {e}")
    finally:
        # Close the figure whether saving succeeded or not
        plt.close()


def make_ohe() -> OneHotEncoder:
    """Create OneHotEncoder with version compatibility."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # Older sklearn versions
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def print_section(title: str, char: str = "=") -> None:
    """Print formatted section header."""
    print(f"\n{char * 60}")
    print(f"{title.upper().center(60)}")
    print(char * 60)


# ========================= DATA SAMPLING =========================

class DataSampler:
    """Handles various data sampling strategies."""
    
    @staticmethod
    def stratified_sample_by_target(
        df: pd.DataFrame,
        target_col: str,
        n: int = 5000,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Stratified sampling by target column with proportional allocation."""
        if target_col not in df.columns:
            raise KeyError(f"Target '{target_col}' not found in dataframe")

        # Remove rows with missing target values
        clean_df = df.dropna(subset=[target_col]).copy()
        if len(clean_df) <= n:
            return clean_df.sample(frac=1.0, random_state=random_state)

        # Calculate proportional allocation
        target_counts = clean_df[target_col].value_counts()
        total_samples = len(clean_df)
        
        sampled_parts = []
        rng = np.random.RandomState(random_state)
        
        for target_value, count in target_counts.items():
            # Proportional sample size with minimum of 1
            target_n = max(1, int((count / total_samples) * n))
            
            subset = clean_df[clean_df[target_col] == target_value]
            if len(subset) <= target_n:
                sampled_parts.append(subset)
            else:
                sampled_parts.append(subset.sample(n=target_n, random_state=rng))
        
        result = pd.concat(sampled_parts, ignore_index=True)
        return result.sample(frac=1.0, random_state=random_state)

    @staticmethod
    def stratified_sample_dual(
        df: pd.DataFrame,
        target_col: str,
        stratify_col: str,
        n: int = 5000,
        equalize_strata: bool = False,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Stratified sampling by both target and another column."""
        clean_df = df.dropna(subset=[target_col, stratify_col]).copy()
        if len(clean_df) <= n:
            return clean_df.sample(frac=1.0, random_state=random_state)

        if equalize_strata:
            # Equal samples per stratum
            strata_counts = clean_df[stratify_col].value_counts()
            n_strata = len(strata_counts)
            per_stratum = max(1, n // n_strata)
            
            sampled_parts = []
            for stratum_value in strata_counts.index:
                stratum_data = clean_df[clean_df[stratify_col] == stratum_value]
                if len(stratum_data) <= per_stratum:
                    sampled_parts.append(stratum_data)
                else:
                    # Within each stratum, maintain target proportions
                    sampled_parts.append(
                        DataSampler.stratified_sample_by_target(
                            stratum_data, target_col, per_stratum, random_state
                        )
                    )
        else:
            # Proportional to stratum size
            strata_counts = clean_df[stratify_col].value_counts()
            total_samples = len(clean_df)
            
            sampled_parts = []
            for stratum_value, count in strata_counts.items():
                stratum_n = max(1, int((count / total_samples) * n))
                stratum_data = clean_df[clean_df[stratify_col] == stratum_value]
                
                if len(stratum_data) <= stratum_n:
                    sampled_parts.append(stratum_data)
                else:
                    sampled_parts.append(
                        DataSampler.stratified_sample_by_target(
                            stratum_data, target_col, stratum_n, random_state
                        )
                    )
        
        result = pd.concat(sampled_parts, ignore_index=True)
        return result.sample(frac=1.0, random_state=random_state)


# ========================= PREPROCESSING =========================

class DataPreprocessor:
    """Handles data preprocessing and feature engineering."""
    
    def __init__(self):
        self.label_encoder = None
        self.preprocessor = None
        self.feature_names = None
    
    def build_preprocessor(self, df: pd.DataFrame) -> ColumnTransformer:
        """Build preprocessing pipeline for features."""
        # Identify numeric and categorical columns
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        categorical_cols = [c for c in df.columns if c not in numeric_cols]
        
        # Remove constant columns
        for col in numeric_cols[:]:
            if df[col].dropna().nunique() <= 1:
                numeric_cols.remove(col)
                print(f"⚠️ Removing constant numeric column: {col}")
        
        for col in categorical_cols[:]:
            if df[col].dropna().nunique() <= 1:
                categorical_cols.remove(col)
                print(f"⚠️ Removing constant categorical column: {col}")
        
        # Build preprocessor
        transformers = []
        if numeric_cols:
            numeric_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])
            transformers.append(("numeric", numeric_pipeline, numeric_cols))
            print(f"📊 Numeric columns ({len(numeric_cols)}): {numeric_cols[:5]}{'...' if len(numeric_cols) > 5 else ''}")
        
        if categorical_cols:
            categorical_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", make_ohe())
            ])
            transformers.append(("categorical", categorical_pipeline, categorical_cols))
            print(f"🏷️ Categorical columns ({len(categorical_cols)}): {categorical_cols[:5]}{'...' if len(categorical_cols) > 5 else ''}")
        
        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False
        )
        
        return self.preprocessor
    
    def encode_target(self, y: pd.Series) -> Tuple[pd.Series, bool]:
        """Encode target variable if needed."""
        if pd.api.types.is_numeric_dtype(y):
            return y, False
        
        self.label_encoder = LabelEncoder()
        y_encoded = pd.Series(self.label_encoder.fit_transform(y), index=y.index)
        print(f"🎯 Target encoded: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        return y_encoded, True


# ========================= MODEL DEFINITIONS =========================

class ModelFactory:
    """Factory class for creating models and their hyperparameter spaces."""
    
    @staticmethod
    def get_model_and_param_space(
        model_type: str,
        is_binary: bool = True,
        random_state: int = 42,
        class_weight: str = "balanced"
    ) -> Tuple[Any, Dict[str, Any]]:
        """Get model instance and hyperparameter search space."""
        model_type = model_type.lower()
        
        if model_type in ["logreg", "logistic", "lr"]:
            model = LogisticRegression(
                max_iter=1000,
                class_weight=class_weight,
                random_state=random_state
            )
            param_space = {
                "clf__C": np.logspace(-3, 2, 10),
                "clf__penalty": ["l2"],
                "clf__solver": ["lbfgs"]
            }
        
        elif model_type in ["rf", "random_forest"]:
            model = RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced_subsample" if class_weight == "balanced" else None,
                random_state=random_state,
                n_jobs=-1
            )
            param_space = {
                "clf__n_estimators": [100, 200, 400],
                "clf__max_depth": [None, 10, 20, 30],
                "clf__min_samples_split": [2, 5, 10],
                "clf__min_samples_leaf": [1, 2, 4],
                "clf__max_features": ["sqrt", "log2", None]
            }
        
        elif model_type in ["hgb", "hist_gradient_boosting"]:
            model = HistGradientBoostingClassifier(
                random_state=random_state,
                class_weight=class_weight if is_binary else None
            )
            param_space = {
                "clf__learning_rate": [0.05, 0.1, 0.2],
                "clf__max_depth": [None, 6, 10],
                "clf__max_leaf_nodes": [None, 31, 63],
                "clf__min_samples_leaf": [20, 50, 100],
                "clf__l2_regularization": [0.0, 0.1, 0.5]
            }
        
        elif model_type in ["svm", "svc"]:
            model = SVC(
                class_weight=class_weight,
                random_state=random_state,
                probability=True
            )
            param_space = {
                "clf__C": np.logspace(-2, 2, 5),
                "clf__kernel": ["rbf", "linear"],
                "clf__gamma": ["scale", "auto"]
            }
        
        elif model_type in ["mlp", "neural_network"]:
            model = MLPClassifier(
                max_iter=500,
                random_state=random_state,
                early_stopping=True
            )
            param_space = {
                "clf__hidden_layer_sizes": [(100,), (100, 50), (200,), (200, 100)],
                "clf__learning_rate_init": [0.001, 0.01, 0.1],
                "clf__alpha": [0.0001, 0.001, 0.01]
            }
        
        elif model_type in ["xgb", "xgboost"]:
            if not _HAS_XGB:
                raise ImportError("XGBoost not available. Install with: pip install xgboost")
            
            model = XGBClassifier(
                tree_method="hist",
                eval_metric="logloss",
                use_label_encoder=False,
                random_state=random_state,
                n_jobs=-1
            )
            param_space = {
                "clf__n_estimators": [200, 400, 600],
                "clf__max_depth": [3, 6, 9],
                "clf__learning_rate": [0.05, 0.1, 0.2],
                "clf__subsample": [0.8, 0.9, 1.0],
                "clf__colsample_bytree": [0.8, 0.9, 1.0],
                "clf__min_child_weight": [1, 3, 6]
            }
        
        else:
            raise ValueError(f"Unknown model type: {model_type}. Available: logreg, rf, hgb, svm, mlp, xgb")
        
        return model, param_space


# ========================= VISUALIZATION =========================

class MLVisualizer:
    """Handles all visualization tasks."""
    
    @staticmethod
    def plot_roc_curve(y_true, y_proba, model_name: str, config: MLConfig):
        """Plot ROC curve for binary classification."""
        if not config.fig_dir:
            return
            
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name.upper()} ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plot_and_save(f"{model_name}_roc_curve", config.fig_dir)
    
    @staticmethod
    def plot_precision_recall_curve(y_true, y_proba, model_name: str, config: MLConfig):
        """Plot Precision-Recall curve."""
        if not config.fig_dir:
            return
            
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        avg_precision = average_precision_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2, label=f'PR Curve (AP = {avg_precision:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{model_name.upper()} Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plot_and_save(f"{model_name}_pr_curve", config.fig_dir)
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, model_name: str, config: MLConfig, labels=None):
        """Plot a confusion matrix.

        If labels are provided, they are used for tick labels; otherwise the
        default tick labels generated by seaborn are used.  Passing
        ``None`` directly into ``xticklabels`` or ``yticklabels`` triggers
        an error because seaborn internally calls ``len`` on the labels
        parameter.  See https://github.com/mwaskom/seaborn/issues/2213 for
        details.
        """
        if not config.fig_dir:
            return
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        if labels is not None:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=labels, yticklabels=labels)
        else:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{model_name.upper()} Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plot_and_save(f"{model_name}_confusion_matrix", config.fig_dir)
    
    @staticmethod
    def plot_model_comparison(results_df: pd.DataFrame, config: MLConfig):
        """Plot model comparison results."""
        if not config.fig_dir or results_df.empty:
            return
        
        # Filter out failed models
        valid_results = results_df.dropna(subset=['cv_score']).copy()
        
        if len(valid_results) < 2:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # CV Score comparison
        valid_results.plot(x='model', y='cv_score', kind='bar', ax=ax1, color='skyblue')
        ax1.set_title('Cross-Validation Scores')
        ax1.set_ylabel('CV Score')
        ax1.tick_params(axis='x', rotation=45)
        
        # Training time comparison
        if 'duration_seconds' in valid_results.columns:
            valid_results.plot(x='model', y='duration_seconds', kind='bar', ax=ax2, color='lightcoral')
            ax2.set_title('Training Duration')
            ax2.set_ylabel('Time (seconds)')
            ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plot_and_save("model_comparison", config.fig_dir)


# ========================= MAIN TRAINING CLASS =========================

class SupervisedMLTrainer:
    """Main class for supervised ML training and evaluation."""
    
    def __init__(self, config: MLConfig = None):
        self.config = config or MLConfig()
        self.preprocessor = DataPreprocessor()
        self.sampler = DataSampler()
        self.visualizer = MLVisualizer()
        self.results_history = []
    
    def run_single_model(
        self,
        df: pd.DataFrame,
        target_col: str,
        model_type: str = "xgb"
    ) -> Tuple[Any, Dict[str, Any], float, Dict[str, Any]]:
        """
        Run supervised learning with cross-validation and hyperparameter tuning.
        
        Returns:
            best_estimator, best_params, best_cv_score, holdout_metrics
        """
        print_section(f"Training {model_type.upper()} on {target_col}")
        
        # Validate inputs
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found")
        
        # Sample data if requested
        df_work = self._prepare_data(df, target_col)
        
        # Prepare features and target
        y = df_work[target_col].copy()
        X = df_work.drop(columns=[target_col])
        
        # Remove rows with missing target values
        valid_mask = y.notna()
        X = X.loc[valid_mask]
        y = y.loc[valid_mask]
        
        # Encode target if needed
        y, target_encoded = self.preprocessor.encode_target(y)
        
        # Determine problem type
        n_classes = y.nunique()
        is_binary = n_classes == 2
        
        print(f"📊 Dataset shape: {X.shape}")
        print(f"🎯 Classes: {n_classes}, Binary: {is_binary}")
        print(f"📈 Target distribution:\n{y.value_counts().sort_index()}")
        
        # Build preprocessing pipeline
        preprocessor = self.preprocessor.build_preprocessor(X)
        
        # Get model and parameter space
        model, param_space = ModelFactory.get_model_and_param_space(
            model_type, is_binary, self.config.random_state, self.config.class_weight
        )
        
        # Handle class imbalance for XGBoost
        if model_type.lower() in ["xgb", "xgboost"] and is_binary and self.config.handle_imbalance:
            pos_count = (y == 1).sum()
            neg_count = (y == 0).sum()
            if pos_count > 0 and neg_count > 0:
                scale_pos_weight = neg_count / pos_count
                model.set_params(scale_pos_weight=scale_pos_weight)
                print(f"⚖️ XGB scale_pos_weight set to {scale_pos_weight:.3f}")
        
        # Create pipeline
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("clf", model)
        ])
        
        # Set up cross-validation
        cv = StratifiedKFold(
            n_splits=self.config.n_splits,
            shuffle=True,
            random_state=self.config.random_state
        )
        
        # Configure hyperparameter search
        search = self._setup_hyperparameter_search(pipeline, param_space, cv)
        
        # Fit the search
        print(f"🔍 Running {self.config.search_type} search with {self.config.n_splits}-fold CV...")
        start_time = time.time()
        search.fit(X, y)
        training_time = time.time() - start_time
        
        best_estimator = search.best_estimator_
        best_params = search.best_params_
        best_score = search.best_score_
        
        print(f"✅ Best CV score ({self.config.scoring}): {best_score:.4f}")
        print(f"⏱️ Training time: {training_time:.2f}s")
        print(f"🔧 Best parameters:")
        for param, value in best_params.items():
            print(f"   {param}: {value}")
        
        # Evaluate on holdout set if requested
        holdout_metrics = self._evaluate_holdout(
            X, y, best_estimator, is_binary, model_type
        )
        
        # Generate visualizations
        if self.config.plot_curves and is_binary:
            self._generate_plots(X, y, best_estimator, model_type, holdout_metrics)
        
        # Save results
        if self.config.save_results and self.config.fig_dir:
            self._save_results(
                model_type, target_col, best_score, best_params,
                holdout_metrics, n_classes, is_binary, training_time
            )
        
        return best_estimator, best_params, best_score, holdout_metrics
    
    def compare_models(
        self,
        df: pd.DataFrame,
        target_col: str,
        models: List[str] = None
    ) -> pd.DataFrame:
        """Compare multiple models on the same dataset."""
        if models is None:
            models = ["logreg", "rf", "hgb"]
            if _HAS_XGB:
                models.append("xgb")
        
        print_section("Model Comparison")
        print(f"🏁 Comparing models: {', '.join(models)}")
        
        results = []
        
        for model_type in models:
            try:
                start_time = time.time()
                _, best_params, best_score, holdout_metrics = self.run_single_model(
                    df, target_col, model_type
                )
                duration = time.time() - start_time
                
                result = {
                    "model": model_type,
                    "cv_score": best_score,
                    "duration_seconds": duration,
                    "best_params": str(best_params)
                }
                result.update(holdout_metrics)
                results.append(result)
                
            except Exception as e:
                print(f"❌ Failed to train {model_type}: {e}")
                results.append({
                    "model": model_type,
                    "cv_score": np.nan,
                    "duration_seconds": np.nan,
                    "error": str(e)
                })
        
        results_df = pd.DataFrame(results)
        if "cv_score" in results_df.columns:
            results_df = results_df.sort_values("cv_score", ascending=False, na_last=True)
        
        # Create comparison plots
        self.visualizer.plot_model_comparison(results_df, self.config)
        
        self.results_history.append(results_df)
        return results_df
    
    def _prepare_data(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Prepare data with sampling if requested."""
        df_work = df.copy()
        
        if self.config.sample_first and self.config.sample_n:
            original_size = len(df_work)
            
            if self.config.stratify_by and self.config.stratify_by in df.columns:
                df_work = self.sampler.stratified_sample_dual(
                    df_work, target_col, self.config.stratify_by,
                    self.config.sample_n, self.config.equalize_strata, 
                    self.config.random_state
                )
                print(f"📊 Using dual-stratified sample: {len(df_work):,} rows (from {original_size:,})")
            else:
                df_work = self.sampler.stratified_sample_by_target(
                    df_work, target_col, self.config.sample_n, self.config.random_state
                )
                print(f"📊 Using target-stratified sample: {len(df_work):,} rows (from {original_size:,})")
        
        return df_work
    
    def _setup_hyperparameter_search(self, pipeline, param_space, cv):
        """Setup hyperparameter search strategy."""
        if self.config.search_type.lower() == "grid":
            return GridSearchCV(
                pipeline, param_space, scoring=self.config.scoring, cv=cv,
                n_jobs=self.config.n_jobs, refit=True, verbose=self.config.verbose
            )
        else:
            return RandomizedSearchCV(
                pipeline, param_space, n_iter=self.config.n_iter, scoring=self.config.scoring,
                cv=cv, n_jobs=self.config.n_jobs, refit=True, verbose=self.config.verbose,
                random_state=self.config.random_state
            )
    
    def _evaluate_holdout(self, X, y, best_estimator, is_binary, model_type):
        """Evaluate model on holdout set if configured."""
        holdout_metrics = {}
        
        if not self.config.test_size or not (0 < self.config.test_size < 1):
            return holdout_metrics
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, stratify=y,
            random_state=self.config.random_state
        )
        
        # Refit on training portion
        best_estimator.fit(X_train, y_train)
        
        # Make predictions
        y_pred = best_estimator.predict(X_test)
        
        # Calculate metrics
        holdout_metrics["accuracy"] = accuracy_score(y_test, y_pred)
        holdout_metrics["macro_f1"] = f1_score(y_test, y_pred, average="macro")
        
        if is_binary and hasattr(best_estimator, "predict_proba"):
            y_proba = best_estimator.predict_proba(X_test)[:, 1]
            holdout_metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
            holdout_metrics["avg_precision"] = average_precision_score(y_test, y_proba)
            
            # Store for plotting
            holdout_metrics["_y_test"] = y_test
            holdout_metrics["_y_pred"] = y_pred
            holdout_metrics["_y_proba"] = y_proba
        
        print(f"🎯 Holdout metrics:")
        for metric, value in holdout_metrics.items():
            if not metric.startswith("_"):
                print(f"   {metric}: {value:.4f}")
        
        return holdout_metrics
    
    def _generate_plots(self, X, y, best_estimator, model_type, holdout_metrics):
        """Generate visualization plots."""
        if not self.config.fig_dir or not holdout_metrics:
            return
        
        try:
            # Extract test data from holdout metrics
            y_test = holdout_metrics.get("_y_test")
            y_pred = holdout_metrics.get("_y_pred")
            y_proba = holdout_metrics.get("_y_proba")
            
            if y_test is not None and y_proba is not None:
                # ROC Curve
                self.visualizer.plot_roc_curve(y_test, y_proba, model_type, self.config)
                
                # Precision-Recall Curve
                self.visualizer.plot_precision_recall_curve(y_test, y_proba, model_type, self.config)
                
                # Confusion Matrix
                if y_pred is not None:
                    self.visualizer.plot_confusion_matrix(y_test, y_pred, model_type, self.config)
                    
        except Exception as e:
            print(f"⚠️ Failed to generate plots: {e}")
    
    def _save_results(self, model_type, target_col, best_score, best_params, 
                     holdout_metrics, n_classes, is_binary, training_time):
        """Save results to JSON file."""
        try:
            # Clean holdout metrics (remove private keys)
            clean_holdout = {k: v for k, v in holdout_metrics.items() 
                           if not k.startswith("_") and not isinstance(v, (pd.Series, np.ndarray))}
            
            results = {
                "model_type": model_type,
                "target_column": target_col,
                "best_cv_score": float(best_score),
                "best_params": best_params,
                "scoring": self.config.scoring,
                "cv_splits": self.config.n_splits,
                "holdout_metrics": clean_holdout,
                "n_classes": int(n_classes),
                "is_binary": is_binary,
                "training_time_seconds": training_time,
                "config": {
                    "sample_n": self.config.sample_n,
                    "test_size": self.config.test_size,
                    "search_type": self.config.search_type,
                    "n_iter": self.config.n_iter,
                    "random_state": self.config.random_state
                }
            }
            
            results_file = Path(self.config.fig_dir) / f"{model_type}_results.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"💾 Results saved to {results_file}")
            
        except Exception as e:
            print(f"⚠️ Failed to save results: {e}")


# ========================= FEATURE IMPORTANCE =========================

class FeatureImportanceAnalyzer:
    """Analyze and visualize feature importance."""
    
    @staticmethod
    def get_feature_importance(estimator, X_test=None, y_test=None, method="default"):
        """Extract feature importance using various methods."""
        importance_dict = {}
        
        # Default model-based importance
        if hasattr(estimator.named_steps["clf"], "feature_importances_"):
            importance_dict["model_importance"] = estimator.named_steps["clf"].feature_importances_
        elif hasattr(estimator.named_steps["clf"], "coef_"):
            # For linear models
            coef = estimator.named_steps["clf"].coef_
            if coef.ndim > 1:
                coef = np.abs(coef).mean(axis=0)
            importance_dict["model_importance"] = np.abs(coef)
        
        # Permutation importance (if test data available)
        if X_test is not None and y_test is not None and _HAS_PERM_IMPORTANCE:
            try:
                perm_importance = permutation_importance(
                    estimator, X_test, y_test, n_repeats=5, random_state=42
                )
                importance_dict["permutation_importance"] = perm_importance.importances_mean
            except Exception as e:
                print(f"⚠️ Failed to calculate permutation importance: {e}")
        
        return importance_dict
    
    @staticmethod
    def plot_feature_importance(importance_dict, feature_names, model_name, config, top_k=20):
        """Plot feature importance."""
        if not config.fig_dir or not importance_dict:
            return
        
        n_methods = len(importance_dict)
        fig, axes = plt.subplots(1, n_methods, figsize=(6*n_methods, 8))
        if n_methods == 1:
            axes = [axes]
        
        for idx, (method, importance) in enumerate(importance_dict.items()):
            if len(importance) != len(feature_names):
                print(f"⚠️ Skipping {method}: length mismatch")
                continue
            
            # Get top features
            top_indices = np.argsort(importance)[-top_k:]
            top_importance = importance[top_indices]
            top_features = [feature_names[i] for i in top_indices]
            
            # Plot
            axes[idx].barh(range(len(top_features)), top_importance)
            axes[idx].set_yticks(range(len(top_features)))
            axes[idx].set_yticklabels(top_features)
            axes[idx].set_xlabel("Importance")
            axes[idx].set_title(f"{method.replace('_', ' ').title()}")
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_and_save(f"{model_name}_feature_importance", config.fig_dir)


# ========================= LEARNING CURVES =========================

class LearningCurveAnalyzer:
    """Analyze learning curves and model performance."""
    
    @staticmethod
    def plot_learning_curve(estimator, X, y, model_name, config, cv=5):
        """Plot learning curve to diagnose bias/variance."""
        if not config.fig_dir:
            return
        
        try:
            train_sizes = np.linspace(0.1, 1.0, 10)
            train_sizes, train_scores, val_scores = learning_curve(
                estimator, X, y, train_sizes=train_sizes, cv=cv,
                scoring=config.scoring, n_jobs=config.n_jobs
            )
            
            train_mean = train_scores.mean(axis=1)
            train_std = train_scores.std(axis=1)
            val_mean = val_scores.mean(axis=1)
            val_std = val_scores.std(axis=1)
            
            plt.figure(figsize=(10, 6))
            plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
            plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                           alpha=0.2, color='blue')
            
            plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
            plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                           alpha=0.2, color='red')
            
            plt.xlabel('Training Set Size')
            plt.ylabel(f'{config.scoring.upper()} Score')
            plt.title(f'{model_name.upper()} Learning Curve')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plot_and_save(f"{model_name}_learning_curve", config.fig_dir)
            
        except Exception as e:
            print(f"⚠️ Failed to plot learning curve: {e}")


# ========================= QUICK FUNCTIONS =========================

def quick_train(
    df: pd.DataFrame,
    target_col: str,
    model_type: str = "xgb",
    config: MLConfig = None
) -> Tuple[Any, Dict[str, Any], float, Dict[str, Any]]:
    """Quick training function for single model."""
    trainer = SupervisedMLTrainer(config)
    return trainer.run_single_model(df, target_col, model_type)


def quick_compare(
    df: pd.DataFrame,
    target_col: str,
    models: List[str] = None,
    config: MLConfig = None
) -> pd.DataFrame:
    """Quick comparison of multiple models."""
    trainer = SupervisedMLTrainer(config)
    return trainer.compare_models(df, target_col, models)


def quick_demo(
    data_path: str,
    target_col: str,
    fig_dir: str = "./ml_results",
    sample_n: int = 5000
) -> pd.DataFrame:
    """Quick demo function for testing the framework."""
    print_section("ML Framework Demo")
    
    # Load data
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        raise ValueError("Unsupported file format. Use .csv or .parquet")
    
    print(f"📊 Loaded dataset with shape: {df.shape}")
    print(f"🎯 Target column: {target_col}")
    print(f"📈 Target distribution:\n{df[target_col].value_counts()}")
    
    # Configure experiment
    config = MLConfig(
        sample_n=sample_n,
        test_size=0.2,
        fig_dir=fig_dir,
        n_iter=20,
        scoring="roc_auc" if df[target_col].nunique() == 2 else "accuracy",
        plot_curves=True,
        save_results=True
    )
    
    # Compare models
    results = quick_compare(df, target_col, config=config)
    
    print_section("Final Results")
    print(results.to_string(index=False, float_format='%.4f'))
    
    return results


# ========================= ADVANCED FEATURES =========================

class ModelExplainer:
    """Advanced model explanation using SHAP (if available)."""
    
    def __init__(self):
        self.has_shap = _HAS_SHAP
    
    def explain_predictions(self, estimator, X_test, model_type, config, max_display=10):
        """Generate SHAP explanations if available."""
        if not self.has_shap or not config.fig_dir:
            print("⚠️ SHAP not available for model explanations")
            return
        
        try:
            import shap
            
            # Choose appropriate explainer
            if model_type.lower() in ["rf", "random_forest", "xgb", "xgboost"]:
                explainer = shap.TreeExplainer(estimator.named_steps["clf"])
            else:
                explainer = shap.Explainer(estimator.named_steps["clf"])
            
            # Transform features
            X_transformed = estimator.named_steps["preprocessor"].transform(X_test)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_transformed[:max_display])
            
            # Generate plots
            shap.summary_plot(shap_values, X_transformed[:max_display], 
                            show=False, plot_size=(10, 6))
            plot_and_save(f"{model_type}_shap_summary", config.fig_dir)
            
            print(f"✅ SHAP explanations saved for {model_type}")
            
        except Exception as e:
            print(f"⚠️ Failed to generate SHAP explanations: {e}")

# === supervised.py — ADD near the bottom (before __main__) ====================
from typing import Optional, List
import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# If your file didn't already define this:
try:
    ensure_dir  # type: ignore[name-defined]
except NameError:
    def ensure_dir(p: str) -> str:
        os.makedirs(p, exist_ok=True)
        return p

def run_supervised(
    df: pd.DataFrame,
    target_col: str,
    out_dir: str = "./ml_results",
    models: Optional[List[str]] = None,
    test_size: float = 0.2,
    sample_n: Optional[int] = 5000,
    scoring: Optional[str] = None
) -> pd.DataFrame:
    """One-call supervised pipeline returning a comparison DataFrame."""
    if scoring is None:
        scoring = "roc_auc" if df[target_col].nunique() == 2 else "accuracy"

    cfg = MLConfig(
        sample_n=sample_n,
        test_size=test_size,
        n_splits=5,
        search_type="random",
        n_iter=30,
        scoring=scoring,
        fig_dir=out_dir,
        plot_curves=True,
        save_results=True,
        random_state=42
    )
    trainer = SupervisedMLTrainer(cfg)
    res = trainer.compare_models(df, target_col, models=models)
    # save tidy summary
    ensure_dir(out_dir)
    res.to_csv(os.path.join(out_dir, "supervised_model_comparison.csv"), index=False)
    print(f"💾 Saved supervised comparison → {out_dir}/supervised_model_comparison.csv")
    return res


# --- simple local dir helper for plots in the add-ons -------------------------
def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def plot_regression_influence(model, X: pd.DataFrame, y: pd.Series, output_dir: str,
                              title: str = "Linear model influence") -> None:
    """
    For linear models with statsmodels-fit results (preferred) or fallback using hat matrix.
    Saves leverage vs standardized residuals with Cook's D sizes.
    """
    out = _ensure_dir(os.path.join(output_dir, "sup_outliers"))
    try:
        # statsmodels path
        infl = model.get_influence()
        lev = np.asarray(infl.hat_matrix_diag)
        sr  = np.asarray(infl.resid_studentized_internal)
        cd  = np.asarray(infl.cooks_distance[0])
    except Exception:
        # fallback approximate with numpy (X must include intercept if needed)
        Xn = np.asarray(X, dtype=float)
        XtX_inv = np.linalg.pinv(Xn.T @ Xn)
        H = Xn @ XtX_inv @ Xn.T
        lev = np.clip(np.diag(H), 0, 1)
        # try sklearn-like model.coef_ else OLS
        coef = getattr(model, "coef_", None)
        if coef is None:
            coef = np.linalg.lstsq(Xn, y.values, rcond=None)[0]
        yhat  = Xn @ coef
        resid = y.values - yhat
        denom = resid.std(ddof=min(Xn.shape[1], len(y)-1)) or 1.0
        sr    = (resid - resid.mean()) / denom
        # crude Cook's D
        p   = Xn.shape[1]
        mse = (resid**2).sum() / max(len(y) - p, 1)
        cd  = (sr**2 * lev) / (p * np.maximum(1 - lev, 1e-9))

    plt.figure(figsize=(7, 5))
    sizes = 200 * (cd / (cd.max() or 1.0))
    plt.scatter(lev, sr, s=sizes, alpha=0.6)
    plt.xlabel("Leverage"); plt.ylabel("Std. residual")
    plt.title(title + " (size ~ Cook's D)")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "influence_scatter.png"), dpi=170)
    plt.close()
    print(f"📐 Regression influence saved → {out}")


def plot_misclassified_anomalies(X: pd.DataFrame, y_true, y_pred,
                                 output_dir: str, contamination: float = 0.05,
                                 random_state: int = 42) -> None:
    """
    Highlights misclassified points and whether they look anomalous in feature space.
    Saves histogram and a CSV of anomaly scores for misclassified vs correct points.
    """
    from sklearn.ensemble import IsolationForest

    out = _ensure_dir(os.path.join(output_dir, "sup_outliers"))
    err = (np.asarray(y_true) != np.asarray(y_pred))
    if err.sum() == 0:
        print("No misclassifications to analyze.")
        return

    # Fit on the raw feature space (after your usual preprocessing if you prefer)
    iso = IsolationForest(contamination=contamination, random_state=random_state)
    iso.fit(X)
    # higher = weirder
    anom = -iso.score_samples(X)

    df_out = pd.DataFrame({"misclassified": err, "anomaly_score": anom})
    df_out.to_csv(os.path.join(out, "misclassified_anomaly_scores.csv"), index=False)

    plt.figure(figsize=(6, 3))
    plt.hist(anom[~err], bins=40, alpha=0.7, label="correct")
    plt.hist(anom[err],  bins=40, alpha=0.7, label="misclassified")
    plt.legend(); plt.title("Anomaly scores by correctness")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "misclf_anomaly_hist.png"), dpi=160)
    plt.close()
    print(f"🧪 Misclassification anomaly plots saved → {out}")


# 1) Idempotent compatibility patch: translate 'na_last' -> 'na_position'
# Corrected patch_pandas_sort_values function
def patch_pandas_sort_values():
    """Patches pandas.DataFrame.sort_values to support `na_last` for older versions."""
    try:
        # Check for the modern na_position argument first
        if "na_position" in pd.DataFrame.sort_values.__code__.co_varnames:
            return  # The modern API is already supported

        # Capture the original function before patching
        _orig_sort_values = pd.DataFrame.sort_values

        def _sort_values_compat(self, *args, **kwargs):
            if "na_last" in kwargs:
                kwargs["na_position"] = "last" if kwargs.pop("na_last") else "first"
            return _orig_sort_values(self, *args, **kwargs)

        # Overwrite the function with our patched version
        pd.DataFrame.sort_values = _sort_values_compat

        print("Patch applied for pandas.DataFrame.sort_values.")
    except Exception as e:
        print(f"Warning: Could not apply pandas patch. {e}")

# 3) Leak-safety: one row per incident + drop target-like proxies/perfect keys
def make_leak_safe(df: pd.DataFrame, target: str, group_col: str) -> pd.DataFrame:
    """
    - Ensures one row per group_col (e.g., incident/case id) to avoid train/test bleed.
    - Drops columns that are obvious target proxies (by name) or perfectly determine the target.
    """
    df = df.copy()

    # One row per incident (prevents same case in train & test)
    if group_col in df.columns:
        time_cols = [c for c in df.columns if re.search(r"(date|time)", c, re.I)]
        if time_cols:
            df = df.sort_values(time_cols).drop_duplicates(subset=[group_col], keep="first")
        else:
            df = df.drop_duplicates(subset=[group_col], keep="first")

    # Drop columns that clearly encode the target (by name/proxy)
    proxy_name_cols = [c for c in df.columns if c != target and re.search(re.escape(target), c, re.I)]
    alias_cols = [c for c in df.columns
                  if c != target and (c.lower() in {"sex", "gender"} or re.search(r"\b(sex|gender)\b", c, re.I))]
    df = df.drop(columns=sorted(set(proxy_name_cols + alias_cols)), errors="ignore")

    # Drop features that perfectly determine target (bijective mapping) up to a reasonable cardinality
    max_card = 1000
    candidates = [c for c in df.columns if c != target and df[c].nunique(dropna=False) <= max_card]
    leaky_by_value = []
    for c in candidates:
        try:
            if df.groupby(c)[target].nunique(dropna=False).max() == 1:
                leaky_by_value.append(c)
        except Exception:
            pass
    df = df.drop(columns=leaky_by_value, errors="ignore")

    # Drop exact duplicates of the target vector
    dup_target_cols = [c for c in df.columns if c != target and df[c].equals(df[target])]
    df = df.drop(columns=dup_target_cols, errors="ignore")

    return df
# === Drop-in utilities: pandas hotfix + quick pipeline + diagnostics ===
from typing import Optional, List
import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# 0) Tiny util
def ensure_dir(path: str) -> str:
    """Create directory if missing and return it."""
    os.makedirs(path, exist_ok=True)
    return path


# ----------------------------------------------------------------------
# 1) Idempotent pandas hotfix: accept legacy kwarg `na_last` (map -> `na_position`)
def patch_pandas_sort_values() -> None:
    """
    Patch pandas.DataFrame.sort_values so code that passes `na_last=...`
    continues to work. Safe to call multiple times (no recursion).
    """
    # Only patch once; keep a stable reference to the original
    if not hasattr(pd.DataFrame, "_sort_values_orig"):
        pd.DataFrame._sort_values_orig = pd.DataFrame.sort_values  # type: ignore[attr-defined]

        def _sort_values_compat(self, *args, **kwargs):
            if "na_last" in kwargs:
                kwargs["na_position"] = "last" if kwargs.pop("na_last") else "first"
            # Call the true original every time
            return pd.DataFrame._sort_values_orig(self, *args, **kwargs)  # type: ignore[attr-defined]

        # Make it obvious we've patched this
        _sort_values_compat.__name__ = "_sort_values_compat"
        pd.DataFrame.sort_values = _sort_values_compat  # type: ignore[assignment]

# Apply immediately (no-op if already applied)
patch_pandas_sort_values()


# ----------------------------------------------------------------------
# 2) One-call supervised pipeline using this module's trainer/config
def run_supervised(
    df: pd.DataFrame,
    target_col: str,
    out_dir: str = "./ml_results",
    models: Optional[List[str]] = None,
    test_size: float = 0.2,
    sample_n: Optional[int] = 5000,
    scoring: Optional[str] = None
) -> pd.DataFrame:
    """
    One-call supervised pipeline returning a model-comparison DataFrame and
    saving a tidy CSV to `out_dir/supervised_model_comparison.csv`.

    Requires MLConfig and SupervisedMLTrainer to be defined in this module.
    """
    # Choose a reasonable default scorer
    if scoring is None:
        scoring = "roc_auc" if df[target_col].nunique(dropna=False) == 2 else "accuracy"

    cfg = MLConfig(
        sample_n=sample_n,
        test_size=test_size,
        n_splits=5,
        search_type="random",
        n_iter=30,
        scoring=scoring,
        fig_dir=out_dir,
        plot_curves=True,
        save_results=True,
        random_state=42,
    )
    trainer = SupervisedMLTrainer(cfg)
    res = trainer.compare_models(df, target_col, models=models)

    # Persist tidy summary
    ensure_dir(out_dir)
    out_csv = os.path.join(out_dir, "supervised_model_comparison.csv")
    res.to_csv(out_csv, index=False)
    print(f"💾 Saved supervised comparison → {out_csv}")
    return res


# ----------------------------------------------------------------------
# 3) Anti-leakage helper: one row per incident + drop target-like proxies/perfect keys
def make_leak_safe(
    df: pd.DataFrame,
    target: str,
    group_col: str,
    max_card: int = 1000
) -> pd.DataFrame:
    """
    Returns a copy of df that is safer against leakage:
      • Enforces one row per incident via `group_col` (keeps earliest by time if available)
      • Drops columns whose names clearly encode the target
      • Drops obvious aliases (e.g., sex/gender when predicting sex)
      • Drops features that perfectly determine the target via one-to-one mapping
      • Drops columns that are exact duplicates of the target
    """
    dfx = df.copy()

    # One row per incident (prevents same case appearing in both train & test)
    if group_col in dfx.columns:
        time_cols = [c for c in dfx.columns if re.search(r"(date|time)", c, flags=re.I)]
        if time_cols:
            dfx = dfx.sort_values(time_cols, na_position="last").drop_duplicates(subset=[group_col], keep="first")
        else:
            dfx = dfx.drop_duplicates(subset=[group_col], keep="first")

    # Drop columns whose names look like they encode the target
    proxy_name_cols = [c for c in dfx.columns if c != target and re.search(re.escape(target), c, flags=re.I)]
    alias_cols = [c for c in dfx.columns
                  if c != target and (c.lower() in {"sex", "gender"} or re.search(r"\b(sex|gender)\b", c, flags=re.I))]
    dfx.drop(columns=sorted(set(proxy_name_cols + alias_cols)), errors="ignore", inplace=True)

    # Drop features that perfectly determine the target (bijective by value) for moderate-cardinality cols
    candidates = [c for c in dfx.columns if c != target and dfx[c].nunique(dropna=False) <= max_card]
    leaky_by_value = []
    for c in candidates:
        try:
            # If for every value of c, target has a single unique value -> perfect key
            if dfx.groupby(c, dropna=False)[target].nunique(dropna=False).max() == 1:
                leaky_by_value.append(c)
        except Exception:
            # Non-groupable dtypes etc.
            pass
    dfx.drop(columns=leaky_by_value, errors="ignore", inplace=True)

    # Drop exact duplicates of the target series
    dup_target_cols = [c for c in dfx.columns if c != target and dfx[c].equals(dfx[target])]
    dfx.drop(columns=dup_target_cols, errors="ignore", inplace=True)

    return dfx


# ----------------------------------------------------------------------
# 4) Diagnostics: regression influence (leverage/residuals/Cook's D)
def plot_regression_influence(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: str,
    title: str = "Linear model influence",
) -> None:
    """
    For linear models with statsmodels fit results (preferred) or fallback
    using a hat-matrix approximation. Saves leverage vs std. residuals with
    point size ~ Cook's D.
    """
    out = ensure_dir(os.path.join(output_dir, "sup_outliers"))
    try:
        # statsmodels-style results
        infl = model.get_influence()
        lev = np.asarray(infl.hat_matrix_diag)
        sr = np.asarray(infl.resid_studentized_internal)
        cd = np.asarray(infl.cooks_distance[0])
    except Exception:
        # Fallback: approximate from X and coefficients
        Xn = np.asarray(X, dtype=float)
        XtX_inv = np.linalg.pinv(Xn.T @ Xn)
        H = Xn @ XtX_inv @ Xn.T
        lev = np.clip(np.diag(H), 0, 1)

        # Use coef_ if available, else least squares
        if hasattr(model, "coef_"):
            beta = getattr(model, "coef_")
            # If scikit pipeline: handle shape (n_features,) or (1, n_features)
            beta = np.ravel(beta)
        else:
            beta, *_ = np.linalg.lstsq(Xn, np.asarray(y), rcond=None)

        yhat = Xn @ beta
        resid = np.asarray(y) - yhat
        sr = (resid - resid.mean()) / (resid.std() if resid.std() else 1.0)

        p = Xn.shape[1]
        mse = (resid**2).sum() / max(len(y) - p, 1)
        cd = (sr**2 * lev) / (p * (1 - lev + 1e-9))

    plt.figure(figsize=(7, 5))
    sizes = 200 * (cd / (cd.max() if np.isfinite(cd).any() else 1.0))
    plt.scatter(lev, sr, s=sizes, alpha=0.6)
    plt.axhline(0, lw=1, ls="--")
    plt.xlabel("Leverage")
    plt.ylabel("Std. residual")
    plt.title(f"{title} (size ~ Cook's D)")
    plt.tight_layout()
    fp = os.path.join(out, "influence_scatter.png")
    plt.savefig(fp, dpi=170)
    plt.close()
    print(f"📐 Regression influence saved → {fp}")


# ----------------------------------------------------------------------
# 5) Diagnostics: misclassified vs anomaly scores
def plot_misclassified_anomalies(
    X: pd.DataFrame,
    y_true,
    y_pred,
    output_dir: str,
    contamination: float = 0.05,
    random_state: int = 42,
) -> None:
    """
    Highlights misclassified points and whether they look anomalous in feature space.
    Saves histogram comparison and a CSV with anomaly scores.
    """
    from sklearn.ensemble import IsolationForest

    out = ensure_dir(os.path.join(output_dir, "sup_outliers"))
    err = (np.asarray(y_true) != np.asarray(y_pred))
    if err.sum() == 0:
        print("No misclassifications to analyze.")
        return

    iso = IsolationForest(contamination=contamination, random_state=random_state)
    iso.fit(X)
    anom = -iso.score_samples(X)  # higher = weirder

    pd.DataFrame({"misclassified": err, "anomaly_score": anom}).to_csv(
        os.path.join(out, "misclassified_anomaly_scores.csv"), index=False
    )

    plt.figure(figsize=(6, 3))
    plt.hist(anom[~err], bins=40, alpha=0.7, label="correct")
    plt.hist(anom[err], bins=40, alpha=0.7, label="misclassified")
    plt.legend()
    plt.title("Anomaly scores by correctness")
    plt.tight_layout()
    fp = os.path.join(out, "misclf_anomaly_hist.png")
    plt.savefig(fp, dpi=160)
    plt.close()
    print(f"🧪 Misclassification anomaly plots saved → {fp}")

# ========================= MAIN EXECUTION =========================

if __name__ == "__main__":
    # Example usage and testing
    print("🚀 Supervised ML Framework - Organized Version")
    print("Available functions:")
    print("  - quick_train(df, target_col, model_type)")
    print("  - quick_compare(df, target_col, models)")
    print("  - quick_demo(data_path, target_col)")
    
    # Example configuration
    example_config = MLConfig(
        sample_n=10000,
        test_size=0.2,
        n_splits=5,
        search_type="random",
        n_iter=50,
        scoring="roc_auc",
        fig_dir="./ml_results",
        plot_curves=True,
        save_results=True,
        random_state=42
    )
    
    print("\n📋 Example configuration:")
    for field in example_config.__dataclass_fields__:
        value = getattr(example_config, field)
        print(f"   {field}: {value}")
    
    # Uncomment to run demo:
    # results = quick_demo("/path/to/your/data.csv", "target_column")
    pass