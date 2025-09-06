import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from utils.mmisc import RESULTS
import json
import time
from models.mmodels import StandardizedFeatureProcessor
from models.unified_eval import UnifiedEvaluator
def run_shared_parameter_grid_search(df, target_column="dice_score", domain="medical", n_splits=5, save_results=True):
    """
    Grid search to find the best shared parameters for both Random Forest and XGBoost.
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with IQA features and target
    target_column : str
        Target column name (e.g., 'dice_score', 'quality_score', 'mos')
    domain : str
        Domain name for saving results
    n_splits : int
        Number of CV folds
    save_results : bool
        Whether to save results to disk
    Returns:
    --------
    dict: Grid search results with best parameters
    """
    print(f"Target: {target_column}")
    print(f"Cross-validation folds: {n_splits}")
    if target_column not in df.columns:
        print(f"ERROR: Target column '{target_column}' not found in dataframe")
        return {}
    evaluator = UnifiedEvaluator(domain=domain, random_state=42, n_splits=n_splits, verbose=False)
    try:
        X, feature_names, y, valid_indices = evaluator._prepare_data(df, target_column)
    except Exception as e:
        print(f"ERROR in data preparation: {str(e)}")
        return {}
    print(f"Dataset prepared: X.shape={X.shape}, y.shape={y.shape}")
    print(f"Features: {feature_names[:5]}..." if len(feature_names) > 5 else f"Features: {feature_names}")
    print(f"Target range: [{y.min():.3f}, {y.max():.3f}]")
    param_grid = {
        "n_estimators": [100, 150, 200, 250, 300, 350, 400, 450, 500],
        "max_depth": [3, 5, 8, 10, 12, 15, 20, None],
    }
    print(f"\nParameter grid:")
    print(f"  n_estimators: {param_grid['n_estimators']}")
    print(f"  max_depth: {param_grid['max_depth']}")
    total_combinations = len(param_grid["n_estimators"]) * len(param_grid["max_depth"])
    print(f"Total combinations to test: {total_combinations}")
    results = []
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    combination_count = 0
    start_time = time.time()
    for n_est in param_grid["n_estimators"]:
        for max_d in param_grid["max_depth"]:
            combination_count += 1
            print(
                f"\nTesting combination {combination_count}/{total_combinations}: "
                f"n_estimators={n_est}, max_depth={max_d}"
            )
            try:
                rf_fold_scores = []
                xgb_fold_scores = []
                for fold, (train_idx, test_idx) in enumerate(kfold.split(X)):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    temp_evaluator = UnifiedEvaluator(domain=domain, random_state=42, n_splits=1, verbose=False)
                    def get_rf_with_params(method, input_dim=None):
                        from sklearn.ensemble import RandomForestRegressor
                        return RandomForestRegressor(
                            n_estimators=n_est,
                            max_depth=max_d,
                            random_state=42,
                            n_jobs=-1,
                            min_samples_split=5,
                            min_samples_leaf=2,
                        )
                    def get_xgb_with_params(method, input_dim=None):
                        from xgboost import XGBRegressor
                        xgb_max_depth = max_d
                        return XGBRegressor(
                            n_estimators=n_est,
                            max_depth=xgb_max_depth,
                            random_state=42,
                            n_jobs=-1,
                            learning_rate=0.1,
                            objective="reg:squarederror",
                            subsample=0.8,
                            colsample_bytree=0.8,
                        )
                    rf_model = get_rf_with_params("random_forest")
                    rf_model.fit(X_train, y_train)
                    rf_pred = rf_model.predict(X_test)
                    rf_metrics = temp_evaluator._compute_metrics(y_test, rf_pred)
                    rf_fold_scores.append(rf_metrics["r2"])
                    xgb_model = get_xgb_with_params("xgboost")
                    xgb_model.fit(X_train, y_train)
                    xgb_pred = xgb_model.predict(X_test)
                    xgb_metrics = temp_evaluator._compute_metrics(y_test, xgb_pred)
                    xgb_fold_scores.append(xgb_metrics["r2"])
                rf_scores = np.array(rf_fold_scores)
                xgb_scores = np.array(xgb_fold_scores)
                rf_r2_mean = rf_scores.mean()
                rf_r2_std = rf_scores.std()
                xgb_r2_mean = xgb_scores.mean()
                xgb_r2_std = xgb_scores.std()
                combined_r2_mean = (rf_r2_mean + xgb_r2_mean) / 2
                combined_r2_std = (rf_r2_std + xgb_r2_std) / 2
                result = {
                    "n_estimators": n_est,
                    "max_depth": max_d if max_d is not None else 0,
                    "max_depth_value": max_d if max_d is not None else 0,
                    "max_depth_display": str(max_d),
                    "rf_max_depth_actual": max_d,
                    "xgb_max_depth_actual": max_d if max_d is not None else 0,
                    "rf_r2_mean": rf_r2_mean,
                    "rf_r2_std": rf_r2_std,
                    "xgb_r2_mean": xgb_r2_mean,
                    "xgb_r2_std": xgb_r2_std,
                    "combined_r2_mean": combined_r2_mean,
                    "combined_r2_std": combined_r2_std,
                    "rf_scores": rf_scores.tolist(),
                    "xgb_scores": xgb_scores.tolist(),
                }
                results.append(result)
                print(f"  RF  R: {rf_r2_mean:.4f}  {rf_r2_std:.4f}")
                print(f"  XGB R: {xgb_r2_mean:.4f}  {xgb_r2_std:.4f}")
                print(f"  Combined: {combined_r2_mean:.4f}  {combined_r2_std:.4f}")
            except Exception as e:
                print(f"  ERROR: {str(e)}")
                continue
    elapsed_time = time.time() - start_time
    print(f"\nGrid search completed in {elapsed_time:.1f} seconds")
    if not results:
        print("ERROR: No successful combinations found!")
        return {}
    best_result = max(results, key=lambda x: x["combined_r2_mean"])
    print(f"\n{'='*60}")
    print("BEST SHARED PARAMETERS FOUND:")
    print(f"{'='*60}")
    print(f"n_estimators: {best_result['n_estimators']}")
    print(f"max_depth: {best_result['max_depth_display']}")
    print(f"")
    print(f"Performance with best parameters:")
    print(f"  Random Forest R: {best_result['rf_r2_mean']:.4f}  {best_result['rf_r2_std']:.4f}")
    print(f"  XGBoost R:       {best_result['xgb_r2_mean']:.4f}  {best_result['xgb_r2_std']:.4f}")
    print(f"  Combined R:      {best_result['combined_r2_mean']:.4f}  {best_result['combined_r2_std']:.4f}")
    rf_best = max(results, key=lambda x: x["rf_r2_mean"])
    xgb_best = max(results, key=lambda x: x["xgb_r2_mean"])
    print(f"\nComparison with individual best:")
    print(
        f"  RF individual best:  n_est={rf_best['n_estimators']}, "
        f"max_depth={rf_best['max_depth_display']}, R={rf_best['rf_r2_mean']:.4f}"
    )
    print(
        f"  XGB individual best: n_est={xgb_best['n_estimators']}, "
        f"max_depth={xgb_best['max_depth_display']}, R={xgb_best['xgb_r2_mean']:.4f}"
    )
    rf_loss = rf_best["rf_r2_mean"] - best_result["rf_r2_mean"]
    xgb_loss = xgb_best["xgb_r2_mean"] - best_result["xgb_r2_mean"]
    print(f"\nPerformance cost of shared parameters:")
    print(f"  RF performance loss:  {rf_loss:.4f} ({rf_loss/rf_best['rf_r2_mean']*100:.1f}%)")
    print(f"  XGB performance loss: {xgb_loss:.4f} ({xgb_loss/xgb_best['xgb_r2_mean']*100:.1f}%)")
    verification_evaluator = UnifiedEvaluator(domain=domain, random_state=42, n_splits=n_splits, verbose=False)
    original_get_model = verification_evaluator._get_model
    def get_model_with_best_params(method, input_dim=None):
        if method == "random_forest":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(
                n_estimators=best_result["n_estimators"],
                max_depth=best_result["rf_max_depth_actual"],
                random_state=42,
                n_jobs=-1,
                min_samples_split=5,
                min_samples_leaf=2,
            )
        elif method == "xgboost":
            from xgboost import XGBRegressor
            return XGBRegressor(
                n_estimators=best_result["n_estimators"],
                max_depth=best_result["xgb_max_depth_actual"],
                random_state=42,
                n_jobs=-1,
                learning_rate=0.1,
                objective="reg:squarederror",
                subsample=0.8,
                colsample_bytree=0.8,
            )
        else:
            return original_get_model(method, input_dim)
    verification_evaluator._get_model = get_model_with_best_params
    verification_results = verification_evaluator.evaluate_all_methods(
        df, target_column, methods=["random_forest", "xgboost"]
    )
    if "random_forest" in verification_results:
        rf_verified = verification_results["random_forest"]["r2_mean"]
        print(f"  RF verified R:  {rf_verified:.4f} (grid search: {best_result['rf_r2_mean']:.4f})")
    if "xgboost" in verification_results:
        xgb_verified = verification_results["xgboost"]["r2_mean"]
        print(f"  XGB verified R: {xgb_verified:.4f} (grid search: {best_result['xgb_r2_mean']:.4f})")
    if save_results:
        create_grid_search_visualizations(results, domain, target_column)
        results_dict = {
            "domain": domain,
            "target_column": target_column,
            "best_parameters": {
                "n_estimators": best_result["n_estimators"],
                "max_depth": best_result["max_depth_display"],
                "rf_max_depth_actual": best_result["rf_max_depth_actual"],
                "xgb_max_depth_actual": best_result["xgb_max_depth_actual"],
            },
            "best_performance": {
                "rf_r2_mean": best_result["rf_r2_mean"],
                "rf_r2_std": best_result["rf_r2_std"],
                "xgb_r2_mean": best_result["xgb_r2_mean"],
                "xgb_r2_std": best_result["xgb_r2_std"],
                "combined_r2_mean": best_result["combined_r2_mean"],
                "combined_r2_std": best_result["combined_r2_std"],
            },
            "verification_with_unified": {
                "rf_verified": rf_verified if "rf_verified" in locals() else None,
                "xgb_verified": xgb_verified if "xgb_verified" in locals() else None,
            },
            "individual_best": {
                "rf_best": {
                    "n_estimators": rf_best["n_estimators"],
                    "max_depth": rf_best["max_depth_display"],
                    "r2_mean": rf_best["rf_r2_mean"],
                },
                "xgb_best": {
                    "n_estimators": xgb_best["n_estimators"],
                    "max_depth": xgb_best["max_depth_display"],
                    "r2_mean": xgb_best["xgb_r2_mean"],
                },
            },
            "performance_cost": {"rf_loss": rf_loss, "xgb_loss": xgb_loss},
            "all_results": results,
            "search_info": {
                "n_combinations_tested": len(results),
                "total_combinations": total_combinations,
                "elapsed_time_seconds": elapsed_time,
                "n_cv_folds": n_splits,
                "dataset_size": len(y),
                "n_features": len(feature_names),
                "features": feature_names,
            },
        }
        save_path = Path(RESULTS / domain / "analysis_logs" / f"shared_param_grid_search_{target_column}.json")
        save_path.parent.mkdir(exist_ok=True, parents=True)
        with open(save_path, "w") as f:
            json.dump(results_dict, f, indent=2, default=str)
        print(f"\nResults saved to: {save_path}")
    return results_dict
def create_grid_search_visualizations(results, domain, target_column):
    """Create heatmaps showing grid search results - FIXED for proper max_depth handling"""
    df_results = pd.DataFrame(results)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        f"Grid Search Results - {domain.capitalize()} Domain ({target_column})", fontsize=16, fontweight="bold"
    )
    n_estimators_vals = sorted(df_results["n_estimators"].unique())
    max_depth_unique = df_results["max_depth_display"].unique()
    max_depth_vals = []
    for val in max_depth_unique:
        if val == "None" or val is None:
            max_depth_vals.append(999)
        else:
            try:
                max_depth_vals.append(int(val))
            except:
                max_depth_vals.append(int(float(val)))
    max_depth_vals = sorted(set(max_depth_vals))
    max_depth_labels = [str(v) if v != 999 else "None" for v in max_depth_vals]
    def create_pivot(metric_col):
        pivot = df_results.pivot_table(
            values=metric_col,
            index="max_depth_display",
            columns="n_estimators",
            aggfunc="mean",
        )
        pivot = pivot.reindex(max_depth_labels)
        pivot = pivot[n_estimators_vals]
        return pivot
    ax = axes[0, 0]
    pivot_rf = create_pivot("rf_r2_mean")
    sns.heatmap(pivot_rf, annot=True, fmt=".3f", cmap="RdYlGn", ax=ax, cbar_kws={"label": "R^2 Score"}, vmin=0, vmax=1)
    ax.set_title("Random Forest R^2 Scores", fontweight="bold")
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("max_depth")
    ax = axes[0, 1]
    pivot_xgb = create_pivot("xgb_r2_mean")
    sns.heatmap(pivot_xgb, annot=True, fmt=".3f", cmap="RdYlGn", ax=ax, cbar_kws={"label": "R^2 Score"}, vmin=0, vmax=1)
    ax.set_title("XGBoost R^2 Scores", fontweight="bold")
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("max_depth")
    ax = axes[1, 0]
    pivot_combined = create_pivot("combined_r2_mean")
    sns.heatmap(
        pivot_combined,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        ax=ax,
        cbar_kws={"label": "Combined R^2 Score"},
        vmin=0,
        vmax=1,
    )
    ax.set_title("Combined R^2 Scores (Average)", fontweight="bold")
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("max_depth")
    best_result = max(results, key=lambda x: x["combined_r2_mean"])
    best_n_est = best_result["n_estimators"]
    best_max_depth = best_result["max_depth_display"]
    try:
        col_pos = n_estimators_vals.index(best_n_est)
        row_pos = max_depth_labels.index(best_max_depth)
        from matplotlib.patches import Rectangle
        rect = Rectangle((col_pos, row_pos), 1, 1, linewidth=3, edgecolor="blue", facecolor="none")
        ax.add_patch(rect)
        ax.text(
            col_pos + 0.5, row_pos + 0.5, "BEST", ha="center", va="center", fontweight="bold", color="blue", fontsize=8
        )
    except ValueError:
        print(f"Warning: Could not mark best combination in heatmap")
    ax = axes[1, 1]
    sorted_results = sorted(results, key=lambda x: x["combined_r2_mean"], reverse=True)[:5]
    x_labels = []
    rf_scores = []
    xgb_scores = []
    combined_scores = []
    for i, result in enumerate(sorted_results):
        label = f"({result['n_estimators']}, {result['max_depth_display']})"
        x_labels.append(label)
        rf_scores.append(result["rf_r2_mean"])
        xgb_scores.append(result["xgb_r2_mean"])
        combined_scores.append(result["combined_r2_mean"])
    x = np.arange(len(x_labels))
    width = 0.25
    ax.bar(x - width, rf_scores, width, label="Random Forest", color="lightblue", alpha=0.8)
    ax.bar(x, xgb_scores, width, label="XGBoost", color="lightcoral", alpha=0.8)
    ax.bar(x + width, combined_scores, width, label="Combined", color="lightgreen", alpha=0.8)
    ax.set_xlabel("(n_estimators, max_depth)")
    ax.set_ylabel("R^2 Score")
    ax.set_title("Top 5 Parameter Combinations", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim([0, 1])
    plt.tight_layout()
    save_path = Path(RESULTS / domain / "figures" / f"grid_search_results_{target_column}.png")
    save_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Visualization saved to: {save_path}")
def find_best_medical_params(medical_df):
    """Find best shared parameters for medical domain"""
    target_options = ["dice_score", "dice", "iou"]
    for target in target_options:
        if target in medical_df.columns and not medical_df[target].isna().all():
            print(f"Running grid search for medical domain with target: {target}")
            return run_shared_parameter_grid_search(medical_df, target_column=target, domain="medical")
    print("ERROR: No valid target column found for medical domain")
    return {}
def find_best_natural_params(natural_df):
    """Find best shared parameters for natural domain"""
    target_options = ["quality_score", "mos", "dmos"]
    for target in target_options:
        if target in natural_df.columns and not natural_df[target].isna().all():
            print(f"Running grid search for natural domain with target: {target}")
            return run_shared_parameter_grid_search(natural_df, target_column=target, domain="natural")
    print("ERROR: No valid target column found for natural domain")
    return {}
def update_standard_params_with_best(grid_search_results):
    """
    Update STANDARD_PARAMS with the best found parameters.
    Call this after running grid search.
    """
    if not grid_search_results or "best_parameters" not in grid_search_results:
        print("ERROR: Invalid grid search results")
        return None
    best_params = grid_search_results["best_parameters"]
    updated_params = {
        "n_estimators": best_params["n_estimators"],
        "max_depth": best_params["max_depth"],
        "random_state": 42,
        "n_jobs": -1,
    }
    print(f"Updated STANDARD_PARAMS:")
    print(f"  n_estimators: {updated_params['n_estimators']}")
    print(f"  max_depth: {updated_params['max_depth']}")
    return updated_params
def run_comparison_with_default_params(df, target_column, domain, grid_search_results):
    """
    Compare grid search optimized parameters with default UnifiedEvaluator parameters.
    This verifies that the optimization actually improves performance.
    """
    print(f"\n{'='*60}")
    print("COMPARING OPTIMIZED VS DEFAULT PARAMETERS")
    print(f"{'='*60}")
    print("\n1. Evaluating with DEFAULT parameters...")
    default_evaluator = UnifiedEvaluator(domain=domain, random_state=42, n_splits=5, verbose=False)
    default_results = default_evaluator.evaluate_all_methods(df, target_column, methods=["random_forest", "xgboost"])
    print("\n2. Evaluating with OPTIMIZED parameters...")
    best_params = grid_search_results["best_parameters"]
    optimized_evaluator = UnifiedEvaluator(domain=domain, random_state=42, n_splits=5, verbose=False)
    original_get_model = optimized_evaluator._get_model
    def get_optimized_model(method, input_dim=None):
        if method == "random_forest":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(
                n_estimators=best_params["n_estimators"],
                max_depth=best_params["rf_max_depth_actual"],
                random_state=42,
                n_jobs=-1,
                min_samples_split=5,
                min_samples_leaf=2,
            )
        elif method == "xgboost":
            from xgboost import XGBRegressor
            return XGBRegressor(
                n_estimators=best_params["n_estimators"],
                max_depth=best_params["xgb_max_depth_actual"],
                random_state=42,
                n_jobs=-1,
                learning_rate=0.1,
                objective="reg:squarederror",
                subsample=0.8,
                colsample_bytree=0.8,
            )
        else:
            return original_get_model(method, input_dim)
    optimized_evaluator._get_model = get_optimized_model
    optimized_results = optimized_evaluator.evaluate_all_methods(
        df, target_column, methods=["random_forest", "xgboost"]
    )
    print(f"\n{'='*60}")
    print("RESULTS COMPARISON:")
    print(f"{'='*60}")
    for method in ["random_forest", "xgboost"]:
        if method in default_results and method in optimized_results:
            default_r2 = default_results[method]["r2_mean"]
            optimized_r2 = optimized_results[method]["r2_mean"]
            improvement = optimized_r2 - default_r2
            print(f"\n{method.upper()}:")
            print(f"  Default R:   {default_r2:.4f}")
            print(f"  Optimized R: {optimized_r2:.4f}")
            print(f"  Improvement:  {improvement:+.4f} ({improvement/default_r2*100:+.1f}%)")
    return {"default_results": default_results, "optimized_results": optimized_results}
