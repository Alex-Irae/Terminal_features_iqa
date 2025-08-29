"""
Centralized evaluation framework for consistent model evaluation, cross-validation,
and feature importance analysis across all calibration methods.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
from scipy import stats
from scipy.stats import spearmanr, pearsonr
import torch
import torch.nn as nn
from mmodels import StandardizedFeatureProcessor, LightweightCNN
from utils.mmisc import RESULTS, CACHE, STANDARD_PARAMS
class UnifiedEvaluator:
    """
    Centralized evaluator that ensures consistent evaluation across all models.
    This class handles:
    - Consistent cross-validation splits
    - Unified feature preparation
    - Standardized metrics computation
    - Feature importance analysis
    - Confidence interval calculation
    """
    def __init__(self, domain: str = "medical", random_state: int = 42, n_splits: int = 20, verbose: bool = False):
        """
        Initialize the unified evaluator.
        Args:
            domain: 'medical' or 'natural'
            random_state: Random seed for reproducibility
            n_splits: Number of CV folds
            verbose: Whether to print progress
        """
        n_splits = 20
        self.domain = domain
        self.random_state = random_state
        self.n_splits = n_splits
        self.verbose = verbose
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)
        self.results = {}
        self.feature_importances = {}
        self.cv_predictions = {}
        self.feature_processor = StandardizedFeatureProcessor(f"unified_{domain}_scaler")
    def evaluate_all_methods(
        self, df: pd.DataFrame, target_column: str, methods: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate all specified methods using consistent CV splits.
        Args:
            df: Input dataframe
            target_column: Target variable column name
            methods: List of methods to evaluate. If None, uses default set.
        Returns:
            Dictionary containing results for all methods
        """
        if methods is None:
            methods = ["linear", "random_forest", "xgboost", "lightweight_cnn"]
        X, feature_names, y, valid_indices = self._prepare_data(df, target_column)
        if len(valid_indices) < self.n_splits * 10:
            raise ValueError(f"Insufficient data for {self.n_splits}-fold CV: {len(valid_indices)} samples")
        cv_splits = self._create_cv_splits(X, y, df.iloc[valid_indices])
        self.cv_splits = cv_splits
        self.feature_names = feature_names
        for method in methods:
            if self.verbose:
                print(f"\nEvaluating {method}...")
            try:
                self.results[method] = self._evaluate_method(method, X, y, cv_splits, feature_names)
            except Exception as e:
                print(f"Error evaluating {method}: {str(e)}")
                self.results[method] = self._get_empty_results()
        self._compute_confidence_intervals()
        self._compute_feature_importances(X, y, feature_names)
        return self.results
    def _prepare_data(self, df: pd.DataFrame, target_column: str) -> Tuple:
        """
        Prepare data with consistent preprocessing for all methods.
        """
        valid_df = df.dropna(subset=[target_column]).copy()
        X, feature_names = self.feature_processor.prepare_features(valid_df, fit_scaler=True, return_numpy=True)
        y = pd.to_numeric(valid_df[target_column], errors="coerce").values
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]
        valid_indices = valid_df.index[valid_mask].tolist()
        if self.domain == "medical" and target_column in ["dice_score", "dice"]:
            y = np.clip(y, 0, 1)
        if self.verbose:
            print(f"Prepared {len(y)} samples with {len(feature_names)} features")
            print(f"Target range: [{y.min():.3f}, {y.max():.3f}]")
        return X, feature_names, y, valid_indices
    def _create_cv_splits(self, X: np.ndarray, y: np.ndarray, df_subset: pd.DataFrame) -> List[Tuple]:
        """
        Create consistent CV splits for all methods.
        Uses stratification if modality column is available.
        """
        cv_splits = []
        if "modality" in df_subset.columns and not df_subset["modality"].isna().all():
            try:
                cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
                splits = cv.split(X, df_subset["modality"].values)
            except:
                cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
                splits = cv.split(X)
        else:
            cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            splits = cv.split(X)
        for train_idx, val_idx in splits:
            cv_splits.append((train_idx.copy(), val_idx.copy()))
        return cv_splits
    def _evaluate_method(
        self, method: str, X: np.ndarray, y: np.ndarray, cv_splits: List[Tuple], feature_names: List[str]
    ) -> Dict:
        """
        Evaluate a single method using the provided CV splits.
        """
        cv_scores = {"r2": [], "mse": [], "rmse": [], "mae": [], "spearman": [], "pearson": []}
        all_predictions = []
        all_true_values = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            model = self._get_model(method, input_dim=X.shape[1])
            if method == "lightweight_cnn":
                y_pred = self._train_lightweight_cnn(model, X_train, y_train, X_val, y_val)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
            metrics = self._compute_metrics(y_val, y_pred)
            for key, value in metrics.items():
                cv_scores[key].append(value)
            all_predictions.extend(y_pred)
            all_true_values.extend(y_val)
            if self.verbose:
                print(f"  Fold {fold_idx + 1}: R = {metrics['r2']:.4f}")
        results = {
            "method": method,
            "domain": self.domain,
            "n_folds": len(cv_splits),
            "n_samples": len(y),
            "n_features": len(feature_names),
            "feature_names": feature_names,
            "cv_scores": cv_scores,
            "all_predictions": np.array(all_predictions),
            "all_true_values": np.array(all_true_values),
        }
        for metric_name, scores in cv_scores.items():
            results[f"{metric_name}_mean"] = np.mean(scores)
            results[f"{metric_name}_std"] = np.std(scores)
        return results
    def _get_model(self, method: str, input_dim: Optional[int] = None):
        """
        Get a fresh model instance for the specified method.
        """
        if method == "linear":
            return Ridge(alpha=1.0, random_state=self.random_state)
        elif method == "random_forest":
            return RandomForestRegressor(
                n_estimators=STANDARD_PARAMS["n_estimators"],
                max_depth=STANDARD_PARAMS["max_depth"],
                random_state=self.random_state,
                n_jobs=STANDARD_PARAMS["n_jobs"],
                min_samples_split=5,
                min_samples_leaf=2,
            )
        elif method == "xgboost":
            return xgb.XGBRegressor(
                n_estimators=STANDARD_PARAMS["n_estimators"],
                max_depth=STANDARD_PARAMS["max_depth"],
                random_state=self.random_state,
                n_jobs=STANDARD_PARAMS["n_jobs"],
                learning_rate=0.1,
                objective="reg:squarederror",
                subsample=0.8,
                colsample_bytree=0.8,
            )
        elif method == "lightweight_cnn":
            if input_dim is None:
                raise ValueError("input_dim required for LightweightCNN")
            return LightweightCNN(input_dim=input_dim, hidden_dims=[128, 64, 32], use_attention=True)
        else:
            raise ValueError(f"Unknown method: {method}")
    def _train_lightweight_cnn(
        self,
        model: LightweightCNN,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        max_epochs: int = 150,
        patience: int = 25,
    ) -> np.ndarray:
        """
        Train LightweightCNN with consistent settings.
        """
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
        X_val_t = torch.FloatTensor(X_val).to(device)
        y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.BCELoss()
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=0.01, epochs=max_epochs, steps_per_epoch=1, pct_start=0.1, anneal_strategy="cos"
        )
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None
        for epoch in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_t)
            loss = criterion(outputs, y_train_t)
            l2_lambda = 1e-5
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            loss = loss + l2_lambda * l2_norm
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
            if patience_counter >= patience:
                break
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        model.eval()
        with torch.no_grad():
            predictions = model(X_val_t).cpu().numpy().flatten()
        predictions = np.clip(predictions, 0, 1)
        return predictions
    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute all evaluation metrics consistently.
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]
        if len(y_true) < 2:
            return {"r2": np.nan, "mse": np.nan, "rmse": np.nan, "mae": np.nan, "spearman": np.nan, "pearson": np.nan}
        mse = mean_squared_error(y_true, y_pred)
        return {
            "r2": r2_score(y_true, y_pred),
            "mse": mse,
            "rmse": np.sqrt(mse),
            "mae": mean_absolute_error(y_true, y_pred),
            "spearman": spearmanr(y_true, y_pred)[0],
            "pearson": pearsonr(y_true, y_pred)[0],
        }
    def _compute_confidence_intervals(self, confidence_level: float = 0.95):
        """
        Compute confidence intervals for all methods.
        """
        for method, result in self.results.items():
            if "cv_scores" not in result:
                continue
            ci_results = {}
            for metric, scores in result["cv_scores"].items():
                scores = np.array(scores)
                n = len(scores)
                if n < 2:
                    ci_results[f"{metric}_ci"] = (np.nan, np.nan)
                    continue
                mean = np.mean(scores)
                sem = stats.sem(scores)
                ci_param = stats.t.interval(confidence_level, n - 1, loc=mean, scale=sem)
                bootstrap_means = []
                for _ in range(10000):
                    bootstrap_sample = np.random.choice(scores, size=n, replace=True)
                    bootstrap_means.append(np.mean(bootstrap_sample))
                ci_bootstrap = np.percentile(
                    bootstrap_means, [(1 - confidence_level) / 2 * 100, (1 + confidence_level) / 2 * 100]
                )
                ci_results[f"{metric}_ci_parametric"] = ci_param
                ci_results[f"{metric}_ci_bootstrap"] = ci_bootstrap
            result.update(ci_results)
    def _compute_feature_importances(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]):
        """
        Compute feature importances for tree-based methods using full dataset.
        """
        for method in ["random_forest", "xgboost"]:
            if method not in self.results:
                continue
            model = self._get_model(method)
            model.fit(X, y)
            importances = model.feature_importances_
            importance_dict = {name: float(imp) for name, imp in zip(feature_names, importances)}
            sorted_importances = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            self.feature_importances[method] = {
                "importances": importance_dict,
                "sorted": sorted_importances,
                "top_5": sorted_importances[:5],
            }
            self.results[method]["feature_importances"] = importance_dict
    def _get_empty_results(self) -> Dict:
        """
        Return empty results structure for failed methods.
        """
        return {
            "method": "failed",
            "domain": self.domain,
            "n_folds": 0,
            "n_samples": 0,
            "n_features": 0,
            "cv_scores": {},
            "r2_mean": np.nan,
            "r2_std": np.nan,
            "mse_mean": np.nan,
            "mse_std": np.nan,
        }
    def create_comparison_dataframe(self) -> pd.DataFrame:
        """
        Create a comparison dataframe of all methods.
        """
        rows = []
        for method, result in self.results.items():
            row = {
                "Method": method,
                "Domain": self.domain,
                "R_mean": result.get("r2_mean", np.nan),
                "R_std": result.get("r2_std", np.nan),
                "MSE_mean": result.get("mse_mean", np.nan),
                "MSE_std": result.get("mse_std", np.nan),
                "MAE_mean": result.get("mae_mean", np.nan),
                "MAE_std": result.get("mae_std", np.nan),
                "Spearman_mean": result.get("spearman_mean", np.nan),
                "Spearman_std": result.get("spearman_std", np.nan),
                "N_samples": result.get("n_samples", 0),
                "N_features": result.get("n_features", 0),
            }
            if "r2_ci_parametric" in result:
                row["R_CI_lower"] = result["r2_ci_parametric"][0]
                row["R_CI_upper"] = result["r2_ci_parametric"][1]
            rows.append(row)
        df = pd.DataFrame(rows)
        df = df.sort_values("R_mean", ascending=False)
        return df
    def print_summary(self):
        """
        Print a formatted summary of results.
        """
        print("\n" + "=" * 70)
        print(f"EVALUATION SUMMARY - {self.domain.upper()} DOMAIN")
        print("=" * 70)
        df = self.create_comparison_dataframe()
        for _, row in df.iterrows():
            method = row["Method"]
            print(f"\n{method.upper()}:")
            print(f"  R Score: {row['R_mean']:.4f}  {row['R_std']:.4f}")
            if "R_CI_lower" in row and not np.isnan(row["R_CI_lower"]):
                print(f"  95% CI: [{row['R_CI_lower']:.4f}, {row['R_CI_upper']:.4f}]")
            print(f"  MSE: {row['MSE_mean']:.6f}  {row['MSE_std']:.6f}")
            print(f"  MAE: {row['MAE_mean']:.6f}  {row['MAE_std']:.6f}")
            print(f"  Spearman: {row['Spearman_mean']:.4f}  {row['Spearman_std']:.4f}")
            if method in self.feature_importances:
                print(f"  Top 3 Features:")
                for feat, imp in self.feature_importances[method]["top_5"][:3]:
                    print(f"    - {feat}: {imp:.4f}")
        print("\n" + "=" * 70)
        print(f"Best Method: {df.iloc[0]['Method']} (R = {df.iloc[0]['R_mean']:.4f})")
        print("=" * 70)
def run_unified_evaluation(
    medical_df: pd.DataFrame = None,
    natural_df: pd.DataFrame = None,
    medical_target: str = "dice_score",
    natural_target: str = "quality_score",
    methods: List[str] = None,
    verbose: bool = True,
) -> Dict:
    """
    Run unified evaluation on both domains with consistent methodology.
    Args:
        medical_df: Medical domain dataframe
        natural_df: Natural domain dataframe
        medical_target: Target column for medical domain
        natural_target: Target column for natural domain
        methods: List of methods to evaluate
        verbose: Whether to print progress
    Returns:
        Dictionary containing results for both domains
    """
    results = {}
    if medical_df is not None and len(medical_df) > 50:
        print("\n" + "=" * 70)
        print("EVALUATING MEDICAL DOMAIN")
        print("=" * 70)
        evaluator_med = UnifiedEvaluator(domain="medical", random_state=42, n_splits=5, verbose=verbose)
        if medical_target not in medical_df.columns or medical_df[medical_target].isna().all():
            for alt_target in ["dice", "iou", "segmentation_score"]:
                if alt_target in medical_df.columns and not medical_df[alt_target].isna().all():
                    medical_target = alt_target
                    print(f"Using alternative target: {medical_target}")
                    break
        try:
            results["medical"] = evaluator_med.evaluate_all_methods(medical_df, medical_target, methods)
            if verbose:
                evaluator_med.print_summary()
            if CACHE:
                df_results = evaluator_med.create_comparison_dataframe()
                save_path = Path(RESULTS) / "medical" / "analysis_logs" / "unified_evaluation.csv"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                df_results.to_csv(save_path, index=False)
                print(f"\nResults saved to: {save_path}")
        except Exception as e:
            print(f"Error evaluating medical domain: {str(e)}")
            results["medical"] = {}
    if natural_df is not None and len(natural_df) > 50:
        print("\n" + "=" * 70)
        print("EVALUATING NATURAL DOMAIN")
        print("=" * 70)
        evaluator_nat = UnifiedEvaluator(domain="natural", random_state=42, n_splits=5, verbose=verbose)
        if natural_target not in natural_df.columns or natural_df[natural_target].isna().all():
            for alt_target in ["mos", "dmos", "quality", "score"]:
                if alt_target in natural_df.columns and not natural_df[alt_target].isna().all():
                    natural_target = alt_target
                    print(f"Using alternative target: {natural_target}")
                    break
        try:
            results["natural"] = evaluator_nat.evaluate_all_methods(natural_df, natural_target, methods)
            if verbose:
                evaluator_nat.print_summary()
            if CACHE:
                df_results = evaluator_nat.create_comparison_dataframe()
                save_path = Path(RESULTS) / "natural" / "analysis_logs" / "unified_evaluation.csv"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                df_results.to_csv(save_path, index=False)
                print(f"\nResults saved to: {save_path}")
        except Exception as e:
            print(f"Error evaluating natural domain: {str(e)}")
            results["natural"] = {}
    if "medical" in results and "natural" in results and verbose:
        print("\n" + "=" * 70)
        print("CROSS-DOMAIN COMPARISON")
        print("=" * 70)
        for method in ["linear", "random_forest", "xgboost", "lightweight_cnn"]:
            med_r2 = results.get("medical", {}).get(method, {}).get("r2_mean", np.nan)
            nat_r2 = results.get("natural", {}).get(method, {}).get("r2_mean", np.nan)
            if not np.isnan(med_r2) and not np.isnan(nat_r2):
                diff = nat_r2 - med_r2
                print(f"\n{method.upper()}:")
                print(f"  Medical R: {med_r2:.4f}")
                print(f"  Natural R: {nat_r2:.4f}")
                print(f"  Difference: {diff:+.4f} {'(Natural better)' if diff > 0 else '(Medical better)'}")
    return results
def example_usage():
    """
    Example of how to use the unified evaluation framework.
    """
    medical_df = pd.read_csv("medical_results.csv")
    natural_df = pd.read_csv("natural_results.csv")
    results = run_unified_evaluation(
        medical_df=medical_df,
        natural_df=natural_df,
        medical_target="dice_score",
        natural_target="quality_score",
        methods=["linear", "random_forest", "xgboost", "lightweight_cnn"],
        verbose=True,
    )
    if "medical" in results:
        med_rf = results["medical"].get("random_forest", {})
        print(f"\nMedical RF R: {med_rf.get('r2_mean', 0):.4f}")
        print(f"Medical RF Top Features: {med_rf.get('feature_importances', {})}")
    return results
if __name__ == "__main__":
    print("Unified Evaluation Framework")
    print("Use run_unified_evaluation() to evaluate your models")
