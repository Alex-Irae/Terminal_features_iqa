import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
from scipy.stats import pearsonr, spearmanr
from pathlib import Path
from aablation import AblationCalibrator
from models.mmodels import SklearnCalibrator
from utils.mmisc import RESULTS, CACHE, STANDARD_PARAMS
import os
import torch
import torch.nn as nn
from ccalibration import LightweightCNNCalibrator
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from models.mmodels import StandardizedFeatureProcessor
from sklearn.model_selection import KFold
from models.unified_eval import UnifiedEvaluator
class DomainAgnosticCalibrator:
    """
    Calibrator specifically designed for domain-agnostic evaluation.
    Uses only IQA metrics without modality/pathology/distortion information.
    """
    def __init__(self, model_type="neural"):
        self.model_type = model_type
        self.model = None
        self.is_fitted = False
        self.feature_processor = StandardizedFeatureProcessor(f"agnostic_{model_type}_scaler")
    def fit(self, iqa_features_or_df, targets=None, validation_split=0.2):
        """Support both numpy array and DataFrame input with standardized processing"""
        if isinstance(iqa_features_or_df, np.ndarray):
            feature_names = StandardizedFeatureProcessor.STANDARD_IQA_FEATURES[: iqa_features_or_df.shape[1]]
            temp_df = pd.DataFrame(iqa_features_or_df, columns=feature_names)
            X_scaled, _ = self.feature_processor.prepare_features(temp_df, fit_scaler=True)
        else:
            X_scaled, feature_names = self.feature_processor.prepare_features(iqa_features_or_df, fit_scaler=True)
        if targets is not None:
            valid_mask = ~np.isnan(targets)
            if not np.all(valid_mask):
                print(f"Removing {np.sum(~valid_mask)} samples with NaN targets")
                X_scaled = X_scaled[valid_mask]
                targets = targets[valid_mask]
        if self.model_type == "neural":
            self.model = SimpleMLP(input_dim=X_scaled.shape[1])
            self._train_neural_model(X_scaled, targets, validation_split)
        elif self.model_type == "rf":
            from sklearn.ensemble import RandomForestRegressor
            from utils.mmisc import STANDARD_PARAMS
            self.model = RandomForestRegressor(
                n_estimators=STANDARD_PARAMS["n_estimators"],
                max_depth=STANDARD_PARAMS["max_depth"],
                random_state=STANDARD_PARAMS["random_state"],
                n_jobs=STANDARD_PARAMS["n_jobs"],
                min_samples_split=5,
                min_samples_leaf=2,
            )
            self.model.fit(X_scaled, targets)
        elif self.model_type == "linear":
            from sklearn.linear_model import Ridge
            self.model = Ridge(alpha=1.0, random_state=42)
            self.model.fit(X_scaled, targets)
        elif self.model_type == "xgb":
            import xgboost as xgb
            from utils.mmisc import STANDARD_PARAMS
            self.model = xgb.XGBRegressor(
                n_estimators=STANDARD_PARAMS["n_estimators"],
                max_depth=STANDARD_PARAMS["max_depth"],
                random_state=STANDARD_PARAMS["random_state"],
                n_jobs=STANDARD_PARAMS["n_jobs"],
                learning_rate=0.1,
                objective="reg:squarederror",
                subsample=0.8,
                colsample_bytree=0.8,
            )
            self.model.fit(X_scaled, targets)
        self.is_fitted = True
    def predict(self, iqa_features_or_df):
        """Consistent prediction method with standardized processing"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        if isinstance(iqa_features_or_df, np.ndarray):
            feature_names = StandardizedFeatureProcessor.STANDARD_IQA_FEATURES[: iqa_features_or_df.shape[1]]
            temp_df = pd.DataFrame(iqa_features_or_df, columns=feature_names)
            X_scaled, _ = self.feature_processor.prepare_features(temp_df, fit_scaler=False)
        else:
            X_scaled, _ = self.feature_processor.prepare_features(iqa_features_or_df, fit_scaler=False)
        if self.model_type == "neural":
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(torch.FloatTensor(X_scaled))
                return predictions.numpy().flatten()
        else:
            return self.model.predict(X_scaled)
    def _train_neural_model(self, X, y, validation_split):
        """Train neural model with early stopping"""
        from sklearn.model_selection import train_test_split
        import torch.nn as nn
        if validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=42)
        else:
            X_train, X_val, y_train, y_val = X, None, y, None
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        best_val_loss = float("inf")
        patience_counter = 0
        for epoch in range(200):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    X_val_tensor = torch.FloatTensor(X_val)
                    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
                    val_outputs = self.model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    if patience_counter >= 20:
                        break
class SimpleMLP(nn.Module):
    """Simple MLP for domain-agnostic calibration"""
    def __init__(self, input_dim, hidden_dims=[64, 32]):
        super().__init__()
        layers = []
        dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2)])
            dim = hidden_dim
        layers.append(nn.Linear(dim, 1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)
class CrossModalityFramework:
    """
    Modified to use UnifiedEvaluator for all cross-modality experiments.
    """
    def __init__(self, results_df, cache_dir=None, domain="medical"):
        ssim_col = "ssim"
        psnr_col = "psnr"
        self.results_df = results_df.dropna(subset=["dice_score", psnr_col, ssim_col]).copy()
        if cache_dir is None:
            self.cache_dir = Path(RESULTS / domain / "cache")
        self.modalities = ["T1", "T2", "FLAIR"]
        self.iqa_features = [psnr_col, ssim_col, "mse", "mae", "snr", "cnr", "gradient_mag", "laplacian_var"]
        self.available_iqa_features = [feat for feat in self.iqa_features if feat in self.results_df.columns]
        self.evaluator = UnifiedEvaluator(domain=domain, random_state=42, n_splits=5, verbose=False)
        self._validate_dataset()
    def _domain_aware_generalization(self, domain):
        """Domain-aware generalization using UnifiedEvaluator properly"""
        results = {}
        for test_mod in self.modalities:
            train_mods = [m for m in self.modalities if m != test_mod]
            train_df = self.results_df[self.results_df["modality"].isin(train_mods)].copy()
            test_df = self.results_df[self.results_df["modality"] == test_mod].copy()
            if len(train_df) < 50 or len(test_df) < 20:
                continue
            method_results = {}
            method_configs = {
                "lightweight_cnn_full": "lightweight_cnn",
                "rf_domain_aware": "random_forest",
                "linear_domain_aware": "linear",
                "xgb_domain_aware": "xgboost",
            }
            for method_name, method_type in method_configs.items():
                try:
                    evaluator = UnifiedEvaluator(domain=domain, random_state=42, n_splits=1, verbose=False)
                    X_train, feature_names, y_train, train_idx = evaluator._prepare_data(train_df, "dice_score")
                    X_test, _, y_test, test_idx = evaluator._prepare_data(test_df, "dice_score")
                    model = evaluator._get_model(method_type, input_dim=X_train.shape[1])
                    if method_type == "lightweight_cnn":
                        split_idx = int(0.8 * len(X_train))
                        predictions = evaluator._train_lightweight_cnn(
                            model, X_train[:split_idx], y_train[:split_idx], X_test, y_test
                        )
                    else:
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                    metrics = evaluator._compute_metrics(y_test, predictions)
                    method_results[method_name] = metrics
                except Exception as e:
                    continue
            if method_results:
                results[test_mod] = {
                    "train_modalities": train_mods,
                    "test_samples": len(test_df),
                    "train_samples": len(train_df),
                    "methods": method_results,
                    "experiment_type": "domain_aware",
                }
        return results
    def _domain_agnostic_robustness(self, domain):
        """Domain-agnostic robustness testing using only IQA features"""
        results = {}
        for test_mod in self.modalities:
            train_mods = [m for m in self.modalities if m != test_mod]
            train_df = self.results_df[self.results_df["modality"].isin(train_mods)].copy()
            test_df = self.results_df[self.results_df["modality"] == test_mod].copy()
            if len(train_df) < 50 or len(test_df) < 20:
                continue
            X_train_iqa = train_df[self.available_iqa_features].values
            X_test_iqa = test_df[self.available_iqa_features].values
            y_train = train_df["dice_score"].values
            y_test = test_df["dice_score"].values
            method_results = {}
            calibrators = {
                "neural_iqa_only": DomainAgnosticCalibrator("neural"),
                "rf_iqa_only": DomainAgnosticCalibrator("rf"),
                "linear_iqa_only": DomainAgnosticCalibrator("linear"),
                "xgboost_iqa_only": DomainAgnosticCalibrator("xgb"),
            }
            for method_name, calibrator in calibrators.items():
                try:
                    calibrator.fit(X_train_iqa, y_train, validation_split=0.2)
                    predictions = calibrator.predict(X_test_iqa)
                    evaluator = UnifiedEvaluator(domain=domain)
                    metrics = evaluator._compute_metrics(y_test, predictions)
                    method_results[method_name] = metrics
                except Exception as e:
                    continue
            if method_results:
                results[test_mod] = {
                    "train_modalities": train_mods,
                    "test_samples": len(test_df),
                    "train_samples": len(train_df),
                    "methods": method_results,
                    "experiment_type": "domain_agnostic",
                }
        return results
    def _few_shot_adaptation(self, domain):
        """Few-shot adaptation experiments"""
        results = {}
        shot_sizes = [5, 10, 20, 50]
        for test_mod in self.modalities:
            train_mods = [m for m in self.modalities if m != test_mod]
            base_train_df = self.results_df[self.results_df["modality"].isin(train_mods)].copy()
            test_df = self.results_df[self.results_df["modality"] == test_mod].copy()
            if len(base_train_df) < 100 or len(test_df) < 60:
                continue
            shot_results = {}
            for n_shots in shot_sizes:
                if n_shots >= len(test_df) * 0.8:
                    continue
                kf = KFold(n_splits=min(5, len(test_df) // (n_shots + 10)), shuffle=True, random_state=42)
                r2_scores = []
                for fold_idx, (adapt_idx, eval_idx) in enumerate(kf.split(test_df)):
                    try:
                        adapt_df = test_df.iloc[adapt_idx]
                        if len(adapt_df) < n_shots:
                            continue
                        few_shot_df = adapt_df.sample(n=n_shots, random_state=42 + fold_idx)
                        eval_df = test_df.iloc[eval_idx]
                        combined_train = pd.concat(
                            [
                                base_train_df.sample(n=min(200, len(base_train_df)), random_state=42 + fold_idx),
                                few_shot_df,
                            ]
                        )
                        evaluator = UnifiedEvaluator(
                            domain=domain, random_state=42 + fold_idx, n_splits=1, verbose=False
                        )
                        X_train, _, y_train, _ = evaluator._prepare_data(combined_train, "dice_score")
                        X_eval, _, y_eval, _ = evaluator._prepare_data(eval_df, "dice_score")
                        model = evaluator._get_model("lightweight_cnn", input_dim=X_train.shape[1])
                        split_idx = int(0.9 * len(X_train))
                        predictions = evaluator._train_lightweight_cnn(
                            model, X_train[:split_idx], y_train[:split_idx], X_eval, y_eval, max_epochs=100, patience=15
                        )
                        r2 = r2_score(y_eval, predictions)
                        r2_scores.append(r2)
                    except Exception as e:
                        continue
                if r2_scores:
                    shot_results[f"{n_shots}_shot"] = {
                        "r2": np.mean(r2_scores),
                        "r2_std": np.std(r2_scores),
                        "n_folds": len(r2_scores),
                    }
            if shot_results:
                results[test_mod] = {
                    "train_modalities": train_mods,
                    "methods": shot_results,
                    "experiment_type": "few_shot",
                }
        return results
    def _analyze_cross_modal_stability(self, all_results, domain):
        """
        Analyze stability and consistency of cross-modal performance.
        """
        stability_analysis = {
            "overall_stability": {},
            "method_consistency": {},
            "modality_gaps": {},
            "best_configurations": {},
        }
        for exp_type in ["domain_aware", "domain_agnostic", "few_shot"]:
            if exp_type not in all_results or not all_results[exp_type]:
                continue
            exp_data = all_results[exp_type]
            method_performances = {}
            for modality, modality_data in exp_data.items():
                methods = modality_data.get("methods", {})
                for method_name, metrics in methods.items():
                    if method_name not in method_performances:
                        method_performances[method_name] = []
                    r2_score = metrics.get("r2", np.nan)
                    if not np.isnan(r2_score):
                        method_performances[method_name].append(r2_score)
            for method_name, scores in method_performances.items():
                if len(scores) > 1:
                    stability_analysis["method_consistency"][f"{exp_type}_{method_name}"] = {
                        "mean": np.mean(scores),
                        "std": np.std(scores),
                        "cv": np.std(scores) / (np.mean(scores) + 1e-10),
                        "range": max(scores) - min(scores),
                        "n_modalities": len(scores),
                    }
        best_r2 = -np.inf
        best_config = None
        for exp_type, exp_data in all_results.items():
            for modality, modality_data in exp_data.items():
                for method_name, metrics in modality_data.get("methods", {}).items():
                    r2_score = metrics.get("r2", -np.inf)
                    if r2_score > best_r2:
                        best_r2 = r2_score
                        best_config = {
                            "experiment": exp_type,
                            "test_modality": modality,
                            "method": method_name,
                            "r2": r2_score,
                            "train_modalities": modality_data.get("train_modalities", []),
                        }
        stability_analysis["best_configurations"]["overall"] = best_config
        for modality in self.modalities:
            modality_scores = []
            for exp_type, exp_data in all_results.items():
                if modality in exp_data:
                    methods = exp_data[modality].get("methods", {})
                    for method_name, metrics in methods.items():
                        r2_score = metrics.get("r2", np.nan)
                        if not np.isnan(r2_score):
                            modality_scores.append(r2_score)
            if modality_scores:
                stability_analysis["modality_gaps"][modality] = {
                    "best": max(modality_scores),
                    "worst": min(modality_scores),
                    "gap": max(modality_scores) - min(modality_scores),
                    "mean": np.mean(modality_scores),
                }
        return stability_analysis
    def _create_comprehensive_figures(self, all_results, stability_analysis, domain):
        """
        Create comprehensive visualization figures for cross-modality analysis.
        """
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Cross-Modality Analysis - {domain.capitalize()} Domain", fontsize=16)
        ax = axes[0, 0]
        for exp_type in ["domain_aware", "domain_agnostic"]:
            if exp_type not in all_results:
                continue
            modalities = []
            scores = []
            for modality in self.modalities:
                if modality in all_results[exp_type]:
                    best_score = max(
                        [m.get("r2", 0) for m in all_results[exp_type][modality].get("methods", {}).values()]
                    )
                    modalities.append(modality)
                    scores.append(best_score)
            if modalities:
                ax.plot(modalities, scores, marker="o", label=exp_type.replace("_", " ").title())
        ax.set_xlabel("Test Modality")
        ax.set_ylabel("Best R Score")
        ax.set_title("Cross-Modal Performance")
        ax.legend()
        ax.grid(alpha=0.3)
        ax = axes[0, 1]
        if stability_analysis["method_consistency"]:
            methods = list(stability_analysis["method_consistency"].keys())
            cvs = [data["cv"] for data in stability_analysis["method_consistency"].values()]
            ax.bar(range(len(methods)), cvs, alpha=0.7)
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels([m.split("_", 1)[1] for m in methods], rotation=45, ha="right")
            ax.set_ylabel("Coefficient of Variation")
            ax.set_title("Method Stability (Lower is Better)")
            ax.grid(axis="y", alpha=0.3)
        ax = axes[1, 0]
        if stability_analysis["modality_gaps"]:
            modalities = list(stability_analysis["modality_gaps"].keys())
            gaps = [data["gap"] for data in stability_analysis["modality_gaps"].values()]
            ax.bar(modalities, gaps, color="coral", alpha=0.7)
            ax.set_ylabel("Performance Gap (R)")
            ax.set_title("Performance Variability by Modality")
            ax.grid(axis="y", alpha=0.3)
        ax = axes[1, 1]
        ax.axis("off")
        summary = "KEY FINDINGS:\n\n"
        if stability_analysis["best_configurations"]["overall"]:
            best = stability_analysis["best_configurations"]["overall"]
            summary += f"Best Configuration:\n"
            summary += f"  Method: {best['method']}\n"
            summary += f"  Test: {best['test_modality']}\n"
            summary += f"  R: {best['r2']:.4f}\n\n"
        if stability_analysis["method_consistency"]:
            most_stable = min(stability_analysis["method_consistency"].items(), key=lambda x: x[1]["cv"])
            summary += f"Most Stable: {most_stable[0].split('_', 1)[1]}\n"
            summary += f"  CV: {most_stable[1]['cv']:.3f}\n"
        ax.text(
            0.1,
            0.9,
            summary,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        plt.tight_layout()
        save_path = Path(RESULTS / domain / "figures" / "cross_modality_comprehensive.png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    def _print_comprehensive_results(self, all_results, stability_analysis, domain):
        """
        Print comprehensive analysis results.
        """
        print("\n" + "=" * 70)
        print(f"CROSS-MODALITY COMPREHENSIVE RESULTS - {domain.upper()}")
        print("=" * 70)
        if stability_analysis["best_configurations"]["overall"]:
            best = stability_analysis["best_configurations"]["overall"]
            print(f"\nBest Overall Configuration:")
            print(f"  Experiment Type: {best['experiment']}")
            print(f"  Method: {best['method']}")
            print(f"  Test Modality: {best['test_modality']}")
            print(f"  Training Modalities: {', '.join(best['train_modalities'])}")
            print(f"  R Score: {best['r2']:.4f}")
        if stability_analysis["method_consistency"]:
            print("\nMethod Stability Analysis:")
            for method, stats in stability_analysis["method_consistency"].items():
                print(f"  {method}:")
                print(f"    Mean R: {stats['mean']:.4f}")
                print(f"    Std Dev: {stats['std']:.4f}")
                print(f"    CV: {stats['cv']:.3f}")
        if stability_analysis["modality_gaps"]:
            print("\nModality-Specific Performance Gaps:")
            for modality, gaps in stability_analysis["modality_gaps"].items():
                print(f"  {modality}:")
                print(f"    Best R: {gaps['best']:.4f}")
                print(f"    Worst R: {gaps['worst']:.4f}")
                print(f"    Gap: {gaps['gap']:.4f}")
    def run_comprehensive_cross_modality_analysis(self, domain):
        """
        Modified main analysis function using unified evaluation throughout.
        """
        all_results = {}
        try:
            all_results["domain_aware"] = self._domain_aware_generalization(domain=domain)
        except Exception as e:
            print(f"Domain-aware failed: {str(e)}")
            all_results["domain_aware"] = {}
        try:
            all_results["domain_agnostic"] = self._domain_agnostic_robustness(domain=domain)
        except Exception as e:
            print(f"Domain-agnostic failed: {str(e)}")
            all_results["domain_agnostic"] = {}
        try:
            all_results["few_shot"] = self._few_shot_adaptation(domain=domain)
        except Exception as e:
            print(f"Few-shot failed: {str(e)}")
            all_results["few_shot"] = {}
        try:
            stability_analysis = self._analyze_cross_modal_stability(all_results, domain)
            self._create_comprehensive_figures(all_results, stability_analysis, domain)
            self._print_comprehensive_results(all_results, stability_analysis, domain)
        except Exception as e:
            print(f"Analysis failed: {str(e)}")
            stability_analysis = {}
        return all_results, stability_analysis
    def _validate_dataset(self):
        """
        Validate that the dataset has required columns and sufficient data.
        """
        required_cols = ["dice_score", "psnr", "ssim", "modality"]
        missing_cols = [col for col in required_cols if col not in self.results_df.columns]
        if missing_cols:
            print(f"Warning: Missing columns: {missing_cols}")
        if "modality" in self.results_df.columns:
            modality_counts = self.results_df["modality"].value_counts()
            for modality in self.modalities:
                if modality not in modality_counts or modality_counts[modality] < 10:
                    print(f"Warning: Insufficient data for {modality}: {modality_counts.get(modality, 0)} samples")
        if len(self.results_df) < 50:
            raise ValueError(f"Insufficient data: {len(self.results_df)} samples (need at least 50)")
        print(f" Dataset validated: {len(self.results_df)} samples across {len(self.modalities)} modalities")
def run_cross_modality_generalization_test(results_df, cache_file=None, domain="medical"):
    """
    Modified to use unified evaluation framework.
    """
    cache_file_path = cache_file
    domain_name = domain
    if cache_file_path is None:
        cache_file_path = os.path.join(Path(RESULTS / domain_name / "analysis_logs"), "cross_modality_enhanced.csv")
    if cache_file_path and os.path.exists(cache_file_path) and CACHE:
        try:
            cached_df = pd.read_csv(cache_file_path)
            results = {}
            framework = CrossModalityFramework(results_df)
            stability = framework._analyze_cross_modal_stability(results, domain_name)
            return results, stability
        except Exception as e:
            print(f"Cache loading failed: {str(e)}")
    framework = CrossModalityFramework(results_df)
    results, stability = framework.run_comprehensive_cross_modality_analysis(domain_name)
    if cache_file_path and CACHE:
        try:
            os.makedirs(os.path.dirname(cache_file_path), exist_ok=True)
            rows = []
            for exp_type, exp_results in results.items():
                for modality, modality_data in exp_results.items():
                    for method, method_metrics in modality_data.get("methods", {}).items():
                        row = {
                            "experiment_type": exp_type,
                            "test_modality": modality,
                            "method": method,
                            "train_modalities": "+".join(modality_data.get("train_modalities", [])),
                            "train_samples": modality_data.get("train_samples", 0),
                            "test_samples": modality_data.get("test_samples", 0),
                            **method_metrics,
                        }
                        rows.append(row)
            if rows:
                cache_df = pd.DataFrame(rows)
                cache_df.to_csv(cache_file_path, index=False)
        except Exception as e:
            print(f"Cache saving failed: {str(e)}")
    figure, analysis = create_cross_modality_figure(results, domain="medical")
    return results, stability
def plot_cross_modality_matrix(results, domain="medical", save_path=None):
    """
    Create a standalone figure showing cross-modality generalization table
    for Experiment 1 (Domain-Aware) results.
    Format: 3 rows (training combinations) x 4 columns (model types)
    Parameters:
    -----------
    results : dict
        Results from run_comprehensive_cross_modality_analysis
    domain : str
        Domain name for saving
    save_path : str, optional
        Custom save path for the figure
    """
    domain_aware_results = results.get("domain_aware", {})
    if not domain_aware_results:
        print("No domain-aware results found!")
        return
    modalities = ["T1", "T2", "FLAIR"]
    all_methods = set()
    for modality_data in domain_aware_results.values():
        all_methods.update(modality_data["methods"].keys())
    methods = sorted(list(all_methods))
    print(f"Found methods: {methods}")
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    training_combinations = [
        {"modalities": ["T1", "T2"], "label": "T1+T2", "test_modality": "FLAIR"},
        {"modalities": ["T1", "FLAIR"], "label": "T1+FLAIR", "test_modality": "T2"},
        {"modalities": ["T2", "FLAIR"], "label": "T2+FLAIR", "test_modality": "T1"},
    ]
    n_rows = len(training_combinations)
    n_cols = len(methods)
    matrix_data = np.full((n_rows, n_cols), np.nan)
    matrix_labels = np.full((n_rows, n_cols), "", dtype=object)
    for row_idx, train_combo in enumerate(training_combinations):
        test_modality = train_combo["test_modality"]
        train_modalities = train_combo["modalities"]
        if test_modality in domain_aware_results:
            modality_data = domain_aware_results[test_modality]
            actual_train_modalities = modality_data.get("train_modalities", [])
            if set(actual_train_modalities) == set(train_modalities):
                for col_idx, method in enumerate(methods):
                    if method in modality_data["methods"]:
                        r2_score = modality_data["methods"][method].get("r2", np.nan)
                        if not np.isnan(r2_score):
                            matrix_data[row_idx, col_idx] = r2_score
                            matrix_labels[row_idx, col_idx] = f"{r2_score:.3f}"
                        else:
                            matrix_labels[row_idx, col_idx] = "N/A"
    cmap = plt.cm.RdYlGn
    im = ax.imshow(matrix_data, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(n_rows):
        for j in range(n_cols):
            if matrix_labels[i, j]:
                if matrix_labels[i, j] != "N/A":
                    color = "white" if matrix_data[i, j] < 0.5 else "black"
                    ax.text(
                        j, i, matrix_labels[i, j], ha="center", va="center", color=color, fontweight="bold", fontsize=14
                    )
                else:
                    ax.text(j, i, "N/A", ha="center", va="center", color="gray", fontsize=12, alpha=0.7)
    clean_method_names = [method.replace("_", " ").title() for method in methods]
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(clean_method_names, fontweight="bold", fontsize=12)
    ax.set_yticks(range(n_rows))
    row_labels = [f"{combo['label']}\n {combo['test_modality']}" for combo in training_combinations]
    ax.set_yticklabels(row_labels, fontweight="bold", fontsize=12)
    ax.set_xlabel("Model Type", fontweight="bold", fontsize=14)
    ax.set_ylabel("Training  Test", fontweight="bold", fontsize=14)
    ax.grid(False)
    for i in range(n_rows + 1):
        ax.axhline(i - 0.5, color="white", linewidth=2)
    for j in range(n_cols + 1):
        ax.axvline(j - 0.5, color="white", linewidth=2)
    ax.set_title(
        "Cross-Modality Generalization Performance\n(Domain-Aware Methods)", fontweight="bold", fontsize=16, pad=20
    )
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=30)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("R Score", fontweight="bold", fontsize=12)
    fig.text(
        0.02,
        0.02,
        "Interpretation: Each cell shows R when models trained on the row's modality combination\n"
        "are tested on the target modality. Higher values (green) indicate better generalization.",
        fontsize=11,
        style="italic",
        wrap=True,
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    if save_path is None:
        save_path = Path(f"cross_modality_table_{domain}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Cross-modality table saved to: {save_path}")
    return fig
def analyze_cross_modality_performance(results):
    """
    Analyze and print summary statistics for cross-modality performance
    """
    domain_aware_results = results.get("domain_aware", {})
    if not domain_aware_results:
        print("No domain-aware results to analyze!")
        return
    print("\n" + "=" * 80)
    print("CROSS-MODALITY PERFORMANCE ANALYSIS")
    print("=" * 80)
    method_scores = {}
    for test_modality, modality_data in domain_aware_results.items():
        train_modalities = modality_data.get("train_modalities", [])
        train_combo = "+".join(sorted(train_modalities))
        print(f"\nTest Modality: {test_modality} (trained on {train_combo})")
        print(
            f"Samples: {modality_data.get('train_samples', 'N/A')} train / {modality_data.get('test_samples', 'N/A')} test"
        )
        for method, metrics in modality_data["methods"].items():
            r2 = metrics.get("r2", np.nan)
            mae = metrics.get("mae", np.nan)
            if method not in method_scores:
                method_scores[method] = []
            if not np.isnan(r2):
                method_scores[method].append(r2)
            if r2 >= 0.8:
                performance = "EXCELLENT"
            elif r2 >= 0.7:
                performance = "GOOD"
            elif r2 >= 0.6:
                performance = "FAIR"
            else:
                performance = "POOR"
            print(f"  {method:25s}: R^2={r2:.3f}, MAE={mae:.3f} [{performance}]")
    print(f"\n{'METHOD RANKING (Average R^2 across modalities)':^50}")
    print("-" * 50)
    method_averages = []
    for method, scores in method_scores.items():
        if scores:
            avg_r2 = np.mean(scores)
            std_r2 = np.std(scores)
            method_averages.append((method, avg_r2, std_r2, len(scores)))
    method_averages.sort(key=lambda x: x[1], reverse=True)
    for rank, (method, avg_r2, std_r2, n_tests) in enumerate(method_averages, 1):
        print(f"{rank}. {method:25s}: {avg_r2:.3f} +/- {std_r2:.3f} (n={n_tests})")
    return method_scores
def create_cross_modality_figure(results, domain="medical"):
    """
    Main function to create the cross-modality matrix figure
    Parameters:
    -----------
    results : dict
        Results from CrossModalityFramework.run_comprehensive_cross_modality_analysis()
    domain : str
        Domain name for file naming
    """
    fig = plot_cross_modality_matrix(results, domain=domain)
    method_scores = analyze_cross_modality_performance(results)
    return fig, method_scores
