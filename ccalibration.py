from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import spearmanr
import torch, os, hashlib, joblib
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import torch.nn.functional as F
from models.unified_eval import UnifiedEvaluator
from models.mmodels import StandardizedFeatureProcessor
from models.mmodels import (
    load_models_for_eval,
    register_hooks_for_resnet_encoder,
    LightweightCNN,
)
from utils.mmisc import (
    linear_cka,
    feature_grid,
    hd95,
    _adapt_for_model_input,
    _to_tensor_nchw,
    _prep_image_uint8_to_float01,
    _as_logits,
    dice_coefficient,
    jaccard_index,
    RESULTS,
    CACHE,
    STANDARD_PARAMS,
)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
def cache_model(model, model_name="calibration_model.pth", domain=None):
    """Cache model with proper validation - handles both PyTorch and sklearn models"""
    model_cache_path = os.path.join(Path(RESULTS / domain / "cache"), model_name)
    if hasattr(model, "state_dict"):
        torch.save(model.state_dict(), model_cache_path)
    else:
        import joblib
        sklearn_path = model_cache_path.replace(".pth", ".joblib")
        joblib.dump(model, sklearn_path)
        model_cache_path = sklearn_path
    print(f" Model cached at {model_cache_path}")
def load_cached_model(model_class, model_params, model_name="calibration_model.pth", domain=None):
    """Load cached model with architecture validation - handles both PyTorch and sklearn models"""
    model_cache_path = os.path.join(Path(RESULTS / domain / "cache"), model_name)
    is_pytorch_model = model_class.__name__ == "HierarchicalCalibrationNetwork"
    if not is_pytorch_model:
        import joblib
        sklearn_path = model_cache_path.replace(".pth", ".joblib")
        if os.path.exists(sklearn_path):
            try:
                model = joblib.load(sklearn_path)
                print(f" Loaded cached sklearn model from {sklearn_path}")
                return model
            except Exception as e:
                print(f"! Failed to load cached sklearn model: {e}")
                return None
        else:
            print(f"! Sklearn model not found in cache: {sklearn_path}")
            return None
    if os.path.exists(model_cache_path):
        try:
            device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
            model = model_class(**model_params).to(device)
            state_dict = torch.load(model_cache_path, map_location=device)
            model_keys = set(model.state_dict().keys())
            cache_keys = set(state_dict.keys())
            if model_keys != cache_keys:
                print(f"! Architecture mismatch detected. Model keys: {len(model_keys)}, Cache keys: {len(cache_keys)}")
                missing_keys = model_keys - cache_keys
                unexpected_keys = cache_keys - model_keys
                if missing_keys:
                    print(f"  Missing keys: {list(missing_keys)[:5]}...")
                if unexpected_keys:
                    print(f"  Unexpected keys: {list(unexpected_keys)[:5]}...")
                print("! Cache invalid due to architecture changes. Will retrain.")
                return None
            for key in model_keys:
                if state_dict[key].shape != model.state_dict()[key].shape:
                    print(
                        f"! Shape mismatch for {key}: cache {state_dict[key].shape} vs model {model.state_dict()[key].shape}"
                    )
                    print("! Cache invalid due to shape changes. Will retrain.")
                    return None
            model.load_state_dict(state_dict)
            print(f" Loaded cached PyTorch model from {model_cache_path} to {device}")
            return model
        except Exception as e:
            print(f"! Failed to load cached PyTorch model: {e}")
            print("! Will retrain from scratch.")
            return None
    else:
        print(f"! PyTorch model not found in cache: {model_cache_path}")
        return None
def get_cache_key(method, fold_idx, train_data_hash):
    """Generate unique cache key for each fold and data combination"""
    cache_key = f"calibration_{method}_fold{fold_idx}_{train_data_hash[:8]}"
    return cache_key
def get_data_hash(df):
    """Create hash of dataframe to detect data changes"""
    content_str = f"{len(df)}_{df.columns.tolist()}_{df.dtypes.to_dict()}"
    if "dice_score" in df.columns:
        content_str += f"_{df['dice_score'].sum():.6f}"
    if "modality" in df.columns:
        content_str += f"_{df['modality'].value_counts().to_dict()}"
    return hashlib.md5(content_str.encode()).hexdigest()
def main_calibration_pipeline(results_df, methods=None, domain="medical"):
    """
    Modified main calibration pipeline to use unified evaluation.
    """
    if methods is None:
        if domain == "medical":
            methods = ["random_forest", "xgboost", "linear", "lightweight_cnn"]
        elif domain == "natural":
            methods = ["random_forest", "xgboost", "linear", "lightweight_cnn"]
        else:
            methods = ["random_forest", "xgboost", "linear", "lightweight_cnn"]
    cache_file = os.path.join(Path(RESULTS / domain / "analysis_logs"), f"segmentation_results_{domain}.csv")
    if os.path.exists(cache_file) and CACHE:
        results_df = pd.read_csv(cache_file)
    else:
        if domain == "medical":
            results_df = evaluate_real_segmentation(results_df, domain=domain)
        if CACHE:
            results_df.to_csv(cache_file, index=False)
    if domain == "medical":
        target_column = "dice_score"
        if target_column not in results_df.columns or results_df[target_column].isna().all():
            return results_df
    else:
        target_column = None
        for col in ["mos", "dmos", "quality_score"]:
            if col in results_df.columns and not results_df[col].isna().all():
                target_column = col
                break
        if target_column is None:
            return results_df
    valid_df = results_df.dropna(subset=[target_column, "psnr", "ssim"]).copy()
    if len(valid_df) < 50:
        if len(valid_df) < 10:
            return results_df
    calibration_results, comparison_df = evaluate_calibration_methods(
        results_df=valid_df, methods=methods, domain=domain, target_column=target_column
    )
    if calibration_results:
        create_calibration_figure(calibration_results, domain=domain)
        best_method = (
            comparison_df.loc[comparison_df["R2_mean"].idxmax(), "Method"] if len(comparison_df) > 0 else methods[0]
        )
        analyze_all_methods_feature_importance(valid_df, methods, target_column=target_column, domain=domain)
        if domain == "medical":
            analyze_modality_specific_calibration(valid_df, best_method, target_column=target_column, domain=domain)
            analyze_distortion_sensitivity(valid_df, best_method, target_column=target_column, domain=domain)
        elif domain == "natural":
            if "lightweight_cnn" in methods:
                lightweight_r2 = comparison_df.loc[comparison_df["Method"] == "lightweight_cnn", "R2_mean"].values
                if len(lightweight_r2) > 0:
                    print(f"Lightweight CNN R2: {lightweight_r2[0]:.4f}")
            analyze_natural_domain_calibration(valid_df, best_method, target_column=target_column, domain=domain)
        comparison_df["domain"] = domain
        comparison_df["target_metric"] = target_column
        if CACHE:
            comparison_df.to_csv(Path(RESULTS / domain / "analysis_logs/calibration_comparison_.csv"), index=False)
            pd.DataFrame(calibration_results).T.to_csv(
                Path(RESULTS / domain / "analysis_logs/calibration_results_full_.csv")
            )
    return calibration_results
def analyze_distortion_severity_calibration(results_df, best_method, target_column, domain):
    """Analyze calibration performance across distortion severity levels."""
    print(f"\n{'-'*40}")
    print("DISTORTION SEVERITY ANALYSIS")
    print(f"{'-'*40}")
    valid_df = results_df.copy()
    if "severity" in valid_df.columns:
        severity_groups = valid_df.groupby("severity")
    else:
        y_values = pd.to_numeric(valid_df[target_column], errors="coerce")
        severity_labels = pd.cut(y_values, bins=3, labels=["High_Severity", "Medium_Severity", "Low_Severity"])
        valid_df["inferred_severity"] = severity_labels
        severity_groups = valid_df.groupby("inferred_severity")
    severity_results = {}
    for severity, severity_df in severity_groups:
        if len(severity_df) < 30:
            print(f"  ! Insufficient data for {severity} ({len(severity_df)} samples)")
            continue
        print(f"\nAnalyzing {severity} severity...")
        train_df = valid_df[~valid_df.index.isin(severity_df.index)].copy()
        try:
            calibrator = DomainAdaptiveCalibrator(method=best_method, domain=domain)
            calibrator.fit(train_df, target_column=target_column, validation_split=0.2)
            predictions = calibrator.predict(severity_df)
            true_values = severity_df[target_column].values
            mse = mean_squared_error(true_values, predictions)
            r2 = r2_score(true_values, predictions)
            rmse = np.sqrt(mse)
            severity_results[severity] = {"mse": mse, "r2": r2, "rmse": rmse, "sample_size": len(severity_df)}
            print(f"  Results - MSE: {mse:.6f}, R^2: {r2:.4f}, Samples: {len(severity_df)}")
        except Exception as e:
            print(f"  ! Error: {str(e)}")
    if severity_results:
        print(f"\n{'='*50}")
        print("DISTORTION SEVERITY CALIBRATION SUMMARY")
        print(f"{'='*50}")
        print(f"{'Severity':<15} {'R^2':<8} {'MSE':<12} {'RMSE':<8} {'Samples':<8}")
        print("-" * 55)
        for severity, res in severity_results.items():
            print(
                f"{str(severity):<15} {res['r2']:<8.4f} {res['mse']:<12.6f} {res['rmse']:<8.4f} {res['sample_size']:<8}"
            )
    return severity_results
def create_comprehensive_analysis_figure(all_results, domain):
    """Create comprehensive visualization of all analysis results."""
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE ANALYSIS VISUALIZATION - {domain.upper()}")
    print(f"{'='*60}")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f"Comprehensive Calibration Analysis - {domain.title()} Domain", fontsize=16, fontweight="bold")
    plt.tight_layout()
    Path(RESULTS / domain / "figures").mkdir(exist_ok=True)
    plt.savefig(RESULTS / domain / "figures" / f"comprehensive_analysis_{domain}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f" Comprehensive analysis figure saved for {domain} domain")
def analyze_natural_domain_calibration(
    results_df, best_method="random_forest", target_column="quality_score", domain="natural"
):
    """
    Natural domain specific calibration analysis.
    Adapts medical domain analyses for natural image characteristics.
    """
    print(f"\n{'='*60}")
    print("NATURAL DOMAIN CALIBRATION ANALYSIS")
    print(f"{'='*60}")
    valid_df = results_df.dropna(subset=[target_column, "psnr", "ssim"]).copy()
    analyze_quality_range_calibration(valid_df, best_method, target_column, domain)
    if "content_type" in valid_df.columns:
        analyze_content_type_calibration(valid_df, best_method, target_column, domain)
    analyze_distortion_severity_calibration(valid_df, best_method, target_column, domain)
def analyze_quality_range_calibration(results_df, best_method, target_column, domain):
    """Analyze calibration performance across different quality ranges."""
    print(f"\n{'-'*40}")
    print("QUALITY RANGE ANALYSIS")
    print(f"{'-'*40}")
    valid_df = results_df.copy()
    y_values = pd.to_numeric(valid_df[target_column], errors="coerce")
    q25 = y_values.quantile(0.25)
    q75 = y_values.quantile(0.75)
    ranges = {
        "Low Quality": valid_df[y_values <= q25].copy(),
        "Medium Quality": valid_df[(y_values > q25) & (y_values < q75)].copy(),
        "High Quality": valid_df[y_values >= q75].copy(),
    }
    range_results = {}
    for range_name, range_df in ranges.items():
        if len(range_df) < 30:
            print(f"  ! Insufficient data for {range_name} ({len(range_df)} samples)")
            continue
        print(f"\nAnalyzing {range_name} range...")
        train_df = pd.concat([df for name, df in ranges.items() if name != range_name])
        if len(train_df) < 50:
            print(f"  ! Insufficient training data")
            continue
        try:
            calibrator = DomainAdaptiveCalibrator(method=best_method, domain=domain)
            calibrator.fit(train_df, target_column=target_column, validation_split=0.2)
            predictions = calibrator.predict(range_df)
            true_values = range_df[target_column].values
            mse = mean_squared_error(true_values, predictions)
            mae = mean_absolute_error(true_values, predictions)
            r2 = r2_score(true_values, predictions)
            rmse = np.sqrt(mse)
            rho, _ = spearmanr(true_values, predictions)
            range_results[range_name] = {
                "mse": mse,
                "mae": mae,
                "r2": r2,
                "rmse": rmse,
                "spearman": rho,
                "sample_size": len(range_df),
                "quality_range": f"[{range_df[target_column].min():.3f}, {range_df[target_column].max():.3f}]",
            }
            print(f"  Results - MSE: {mse:.6f}, R^2: {r2:.4f}, Samples: {len(range_df)}")
        except Exception as e:
            print(f"  ! Error: {str(e)}")
    if range_results:
        print(f"\n{'='*60}")
        print("QUALITY RANGE CALIBRATION SUMMARY")
        print(f"{'='*60}")
        print(f"{'Range':<15} {'R^2':<8} {'MSE':<12} {'RMSE':<8} {'Samples':<8} {'Quality Range'}")
        print("-" * 70)
        for range_name, res in range_results.items():
            print(
                f"{range_name:<15} {res['r2']:<8.4f} {res['mse']:<12.6f} {res['rmse']:<8.4f} {res['sample_size']:<8} {res['quality_range']}"
            )
    return range_results
def analyze_content_type_calibration(results_df, best_method, target_column, domain):
    """Analyze calibration across different content types."""
    print(f"\n{'-'*40}")
    print("CONTENT TYPE ANALYSIS")
    print(f"{'-'*40}")
    valid_df = results_df.copy()
    content_types = valid_df["content_type"].unique()
    content_results = {}
    for content_type in content_types:
        print(f"\nAnalyzing {content_type} content...")
        content_df = valid_df[valid_df["content_type"] == content_type].copy()
        if len(content_df) < 30:
            print(f"  ! Insufficient data for {content_type} ({len(content_df)} samples)")
            continue
        train_df = valid_df[valid_df["content_type"] != content_type].copy()
        try:
            calibrator = DomainAdaptiveCalibrator(method=best_method, domain=domain)
            calibrator.fit(train_df, target_column=target_column, validation_split=0.2)
            predictions = calibrator.predict(content_df)
            true_values = content_df[target_column].values
            mse = mean_squared_error(true_values, predictions)
            r2 = r2_score(true_values, predictions)
            content_results[content_type] = {"mse": mse, "r2": r2, "sample_size": len(content_df)}
            print(f"  Results - MSE: {mse:.6f}, R^2: {r2:.4f}")
        except Exception as e:
            print(f"  ! Error: {str(e)}")
    return content_results
def analyze_all_methods_feature_importance(results_df, methods, target_column="dice_score", domain=None):
    """
    Analyze feature importance for all methods that support it.
    Prints top 10 features for each method.
    FIXED: Removed encoder dependencies, using StandardizedFeatureProcessor
    """
    print(f"\n{'='*80}")
    print("FEATURE IMPORTANCE ANALYSIS - ALL METHODS")
    print(f"{'='*80}")
    df = results_df.dropna(subset=[target_column, "psnr", "ssim"]).copy()
    if len(df) < 50:
        print("! Insufficient data for feature importance analysis.")
        return
    tree_methods = ["random_forest", "xgboost"]
    for method in methods:
        if method not in tree_methods:
            continue
        print(f"\n{'-'*60}")
        print(f"FEATURE IMPORTANCE: {method.upper()}")
        print(f"{'-'*60}")
        try:
            processor = StandardizedFeatureProcessor(f"importance_{method}_scaler")
            X_scaled, feature_names = processor.prepare_features(df, fit_scaler=True)
            y = pd.to_numeric(df[target_column], errors="coerce").astype(np.float32).values
            valid_mask = ~np.isnan(y)
            if not np.all(valid_mask):
                print(f"Removing {np.sum(~valid_mask)} samples with NaN targets")
                X_scaled = X_scaled[valid_mask]
                y = y[valid_mask]
            if method == "random_forest":
                model = RandomForestRegressor(
                    n_estimators=STANDARD_PARAMS["n_estimators"],
                    max_depth=STANDARD_PARAMS["max_depth"],
                    random_state=STANDARD_PARAMS["random_state"],
                    n_jobs=STANDARD_PARAMS["n_jobs"],
                    min_samples_split=5,
                    min_samples_leaf=2,
                )
            elif method == "xgboost":
                model = xgb.XGBRegressor(
                    n_estimators=STANDARD_PARAMS["n_estimators"],
                    max_depth=STANDARD_PARAMS["max_depth"],
                    random_state=STANDARD_PARAMS["random_state"],
                    n_jobs=STANDARD_PARAMS["n_jobs"],
                    learning_rate=0.1,
                    objective="reg:squarederror",
                    subsample=0.8,
                    colsample_bytree=0.8,
                )
            model.fit(X_scaled, y)
            importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
                by="importance", ascending=False
            )
            print("Top 10 Most Important Features:")
            print(f"{'Rank':<5} {'Feature':<20} {'Importance':<12}")
            print("-" * 40)
            for idx, (_, row) in enumerate(feature_importance_df.head(10).iterrows(), 1):
                print(f"{idx:<5} {row['feature']:<20} {row['importance']:<12.6f}")
            iqa_features = ["psnr", "ssim", "mse", "mae", "snr", "cnr", "gradient_mag", "laplacian_var"]
            iqa_importance = feature_importance_df[feature_importance_df["feature"].isin(iqa_features)][
                "importance"
            ].sum()
            engineered_importance = feature_importance_df[
                feature_importance_df["feature"].str.contains("ratio|engineered", case=False, na=False)
            ]["importance"].sum()
            print(f"\nFeature Category Summary:")
            print(f"  IQA Metrics Total Importance: {iqa_importance:.6f}")
            print(f"  Engineered Features Total:    {engineered_importance:.6f}")
            if engineered_importance > 0:
                print(f"  IQA vs Engineered Ratio:      {iqa_importance/engineered_importance:.2f}:1")
            if CACHE:
                Path(RESULTS / domain / "analysis_logs").mkdir(exist_ok=True)
                feature_importance_df.to_csv(
                    RESULTS / domain / "analysis_logs" / f"feature_importance_{method}_{domain}.csv", index=False
                )
        except Exception as e:
            print(f"! Error analyzing {method}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    print(f"\n{'='*80}")
def analyze_feature_importance(results_df, best_method="random_forest", target_column="dice_score", domain=None):
    """
    Trains the best performing model on the full dataset and analyzes
    the importance of each feature.
    FIXED: Using StandardizedFeatureProcessor
    """
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60 + "\n")
    df = results_df.dropna(subset=[target_column, "psnr", "ssim"]).copy()
    if len(df) < 50:
        print("Insufficient data for feature importance analysis.")
        return
    processor = StandardizedFeatureProcessor(f"feature_importance_{best_method}_scaler")
    X_scaled, feature_names = processor.prepare_features(df, fit_scaler=True)
    y = pd.to_numeric(df[target_column], errors="coerce").astype(np.float32).values
    valid_mask = ~np.isnan(y)
    if not np.all(valid_mask):
        print(f"Removing {np.sum(~valid_mask)} samples with NaN targets")
        X_scaled = X_scaled[valid_mask]
        y = y[valid_mask]
    if best_method == "random_forest":
        model = RandomForestRegressor(
            n_estimators=STANDARD_PARAMS["n_estimators"],
            max_depth=STANDARD_PARAMS["max_depth"],
            random_state=STANDARD_PARAMS["random_state"],
            n_jobs=STANDARD_PARAMS["n_jobs"],
            min_samples_split=5,
            min_samples_leaf=2,
        )
    elif best_method == "xgboost":
        model = xgb.XGBRegressor(
            n_estimators=STANDARD_PARAMS["n_estimators"],
            max_depth=STANDARD_PARAMS["max_depth"],
            random_state=STANDARD_PARAMS["random_state"],
            n_jobs=STANDARD_PARAMS["n_jobs"],
            learning_rate=0.1,
            objective="reg:squarederror",
            subsample=0.8,
            colsample_bytree=0.8,
        )
    else:
        print(f"Feature importance for '{best_method}' is not implemented.")
        return
    model.fit(X_scaled, y)
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
        by="importance", ascending=False
    )
    print("Top 10 Most Important Features:")
    print(feature_importance_df.head(10))
    plt.figure(figsize=(10, 8))
    sns.barplot(x="importance", y="feature", data=feature_importance_df.head(15), palette="viridis")
    plt.title("Top 15 Feature Importances for Predicting Dice Score", fontsize=16, fontweight="bold")
    plt.xlabel("Importance Score (Gini Importance)", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    fig_path = RESULTS / domain / "figures/figure6_feature_importance.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"\n Figure 6: Feature importance plot saved to {fig_path}")
def cache_segmentation_results(
    results_df,
    domain,
    file_name="segmentation_results.csv",
):
    """Cache the complete results dataframe"""
    cache_path = os.path.join(Path(RESULTS / domain / "analysis_logs"), file_name)
    results_df.to_csv(cache_path, index=False)
    print(f" Segmentation results cached at {cache_path}")
def load_cached_segmentation_results(domain, file_name="segmentation_results.csv"):
    """Load cached segmentation results if available"""
    cache_path = os.path.join(Path(RESULTS / domain / "analysis_logs"), file_name)
    if os.path.exists(cache_path):
        results_df = pd.read_csv(cache_path)
        print(" Loaded cached segmentation results.")
        return results_df
    return None
@torch.no_grad()
def evaluate_real_segmentation(results_df, domain, threshold=0.5, batch_size=1, amp=True):
    """
    Runs real inference on each image in results_df, compares to the original tumor mask,
    and updates results_df with Dice/IoU/HD95 directly.
    """
    if CACHE:
        cached_results = load_cached_segmentation_results(domain=domain)
        if cached_results is not None:
            required_cols = ["dice", "iou", "hd95", "dice_score"]
            if all(col in cached_results.columns for col in required_cols):
                print(" Using cached segmentation results with all required columns")
                return cached_results
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    models_list = load_models_for_eval(device, domain=domain)
    if len(models_list) == 0:
        raise RuntimeError("No models available for evaluation. Train Phase 1 (U-Net) or provide checkpoints.")
    sample_img = None
    for idx, row in results_df.iterrows():
        if pd.notna(row["path_img"]) and Path(row["path_img"]).exists():
            sample_img = row["path_img"]
            break
    if sample_img:
        run_feature_probe(sample_img, models=models_list, domain=domain)
    rows_done = 0
    use_amp = amp and device.type == "cuda:1"
    for idx, row in results_df.iterrows():
        img_path = row["path_img"]
        msk_path = row["path_mask_original"]
        if pd.isna(img_path) or str(img_path).strip() == "":
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        mask = None
        mp = Path(str(msk_path)) if msk_path is not None else None
        if mp is not None and mp.exists():
            mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask = (mask > 0).astype(np.uint8)
        if mask is None:
            continue
        x = _prep_image_uint8_to_float01(img)
        t = _to_tensor_nchw(x).to(device=device, dtype=torch.float32)
        logits = None
        if use_amp:
            with torch.amp.autocast(device_type="cuda:1"):
                for name, model in models_list:
                    t_in = _adapt_for_model_input(t, model)
                    y = model(t_in)
                    y = _as_logits(y)
                    logits = y if logits is None else logits + y
        else:
            for name, model in models_list:
                t_in = _adapt_for_model_input(t, model)
                y = model(t_in)
                y = _as_logits(y)
                logits = y if logits is None else logits + y
        logits /= float(len(models_list))
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        if prob.ndim == 4:
            prob = prob[0]
        if prob.ndim == 3 and prob.shape[0] == 1:
            prob = prob[0]
        if prob.ndim == 3 and prob.shape[0] > 1:
            prob = prob[0]
        pred = (prob >= threshold).astype(np.uint8)
        d = float(dice_coefficient(pred, mask))
        iou = float(jaccard_index(pred, mask))
        h95 = hd95(pred, mask, spacing=(1.0, 1.0))
        results_df.loc[idx, "dice"] = d
        results_df.loc[idx, "iou"] = iou
        results_df.loc[idx, "hd95"] = h95
        results_df.loc[idx, "dice_score"] = d
        rows_done += 1
        if rows_done % 100 == 0:
            print(f"Processed {rows_done} images...")
    if CACHE:
        cache_segmentation_results(results_df, domain=domain)
    return results_df
def validate_no_encoding_leakage(calibration_model):
    """Check that pathology encoding isn't being used"""
    if hasattr(calibration_model, "feature_importances_"):
        feature_names = calibration_model.feature_names_
        if any("pathology" in name for name in feature_names[:5]):
            print("WARNING: Pathology encoding detected in top features!")
            return False
    print(" No pathology encoding detected")
    return True
def evaluate_calibration_methods(results_df, methods=None, domain="medical", target_column=None):
    """
    Modified to use unified evaluation framework for consistent results.
    Replaces the original scattered evaluation logic.
    """
    domain_suitability = "domain_suitability"
    n_folds = "n_folds"
    mae_std = "mae_std"
    mae_mean = "mae_mean"
    if methods is None:
        methods = ["lightweight_cnn", "hierarchical", "random_forest", "linear", "xgboost"]
    if target_column is None:
        if domain == "medical":
            target_column = "dice_score"
        else:
            for col in ["mos", "dmos", "quality_score"]:
                if col in results_df.columns and not results_df[col].isna().all():
                    target_column = col
                    break
    valid_df = results_df.dropna(subset=[target_column, "psnr", "ssim"]).copy()
    if len(valid_df) < 50:
        return {}, pd.DataFrame()
    evaluator = UnifiedEvaluator(domain=domain, random_state=42, n_splits=5, verbose=False)
    method_mapping = {
        "lightweight_cnn": "lightweight_cnn",
        "hierarchical": "lightweight_cnn",
        "random_forest": "random_forest",
        "linear": "linear",
        "xgboost": "xgboost",
    }
    unified_methods = [method_mapping.get(m, m) for m in methods if method_mapping.get(m)]
    unified_methods = list(set(unified_methods))
    results = evaluator.evaluate_all_methods(valid_df, target_column, unified_methods)
    calibration_results = {}
    for method in methods:
        mapped_method = method_mapping.get(method, method)
        if mapped_method in results:
            unified_result = results[mapped_method]
            calibration_results[method] = {
                "mse_mean": float(unified_result.get("mse_mean", np.nan)),
                "mse_std": float(unified_result.get("mse_std", np.nan)),
                "rmse_mean": float(unified_result.get("rmse_mean", np.nan)),
                "rmse_std": float(unified_result.get("rmse_std", np.nan)),
                mae_mean: float(unified_result.get("mae_mean", np.nan)),
                mae_std: float(unified_result.get("mae_std", np.nan)),
                "r2_mean": float(unified_result.get("r2_mean", np.nan)),
                "r2_std": float(unified_result.get("r2_std", np.nan)),
                "spearman_mean": float(unified_result.get("spearman_mean", np.nan)),
                "spearman_std": float(unified_result.get("spearman_std", np.nan)),
                "all_predictions": unified_result.get("all_predictions", []),
                "all_true_values": unified_result.get("all_true_values", []),
                "cv_scores": unified_result.get("cv_scores", {}).get("r2", []),
                "domain": domain,
                "target_column": target_column,
                n_folds: unified_result.get("n_folds", 5),
            }
            calibration_results[method][domain_suitability] = calculate_domain_suitability(
                method, domain, calibration_results[method]
            )
    comparison_df = pd.DataFrame(
        [
            {
                "Method": method,
                "Domain": domain,
                "Target": target_column,
                "MSE_mean": calibration_results[method]["mse_mean"],
                "MSE_std": calibration_results[method]["mse_std"],
                "RMSE_mean": calibration_results[method]["rmse_mean"],
                "RMSE_std": calibration_results[method]["rmse_std"],
                "MAE_mean": calibration_results[method][mae_mean],
                "MAE_std": calibration_results[method][mae_std],
                "R2_mean": calibration_results[method]["r2_mean"],
                "R2_std": calibration_results[method]["r2_std"],
                "Spearman_mean": calibration_results[method]["spearman_mean"],
                "Spearman_std": calibration_results[method]["spearman_std"],
                "Domain_Suitability": calibration_results[method][domain_suitability],
                "N_Folds": calibration_results[method][n_folds],
            }
            for method in calibration_results.keys()
        ]
    )
    if CACHE:
        Path(RESULTS / domain / "analysis_logs").mkdir(exist_ok=True)
        comparison_df.to_csv(RESULTS / domain / "analysis_logs" / f"calibration_comparison_{domain}.csv", index=False)
    return calibration_results, comparison_df
def compare_calibration_improvements(old_results, new_results, domain="natural"):
    """
    Compare the performance improvements between old and new calibration methods
    """
    print("\n" + "=" * 60)
    print("CALIBRATION IMPROVEMENT ANALYSIS")
    print("=" * 60)
    improvements = {}
    common_methods = set(old_results.keys()) & set(new_results.keys())
    print(f"Comparing {len(common_methods)} common methods:")
    for method in common_methods:
        old_r2 = old_results[method].get("r2_mean", 0)
        new_r2 = new_results[method].get("r2_mean", 0)
        improvement = new_r2 - old_r2
        improvement_pct = (improvement / abs(old_r2) * 100) if old_r2 != 0 else 0
        improvements[method] = {
            "old_r2": old_r2,
            "new_r2": new_r2,
            "absolute_improvement": improvement,
            "relative_improvement_pct": improvement_pct,
        }
        status = "" if improvement > 0 else "" if improvement < 0 else ""
        print(f"{status} {method}: {old_r2:.4f}  {new_r2:.4f} ({improvement:+.4f}, {improvement_pct:+.1f}%)")
    new_methods = set(new_results.keys()) - set(old_results.keys())
    if new_methods:
        print(f"\nNew methods added:")
        for method in new_methods:
            r2 = new_results[method].get("r2_mean", 0)
            print(f" {method}: R^2 = {r2:.4f}")
    if improvements:
        best_method = max(improvements.keys(), key=lambda m: improvements[m]["absolute_improvement"])
        best_improvement = improvements[best_method]
        print(f"\n Best improvement: {best_method}")
        print(f"   Absolute: +{best_improvement['absolute_improvement']:.4f} R^2")
        print(f"   Relative: +{best_improvement['relative_improvement_pct']:.1f}%")
    return improvements
def create_calibration_figure(results, domain=None):
    print("\n" + "=" * 60)
    print("FIGURE 4: CALIBRATION EFFECTIVENESS RESULTS")
    print("=" * 60)
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    fig.suptitle("Calibration Method Comparison", fontsize=16, fontweight="bold")
    methods = list(results.keys())
    colors = ["C0", "C1", "C2"]
    print("Calibration method performance comparison:")
    print("-" * 50)
    for i, method in enumerate(methods):
        res = results[method]
        print(f"\n{method.upper()} METHOD:")
        print(f"      MSE: {res['mse_mean']:.8f} +/- {res['mse_std']:.8f}")
        print(f"     RMSE: {res['rmse_mean']:.8f} +/- {res['rmse_std']:.8f}")
        print(f"      MAE: {res['mae_mean']:.8f} +/- {res['mae_std']:.8f}")
        print(f"      R^2: {res['r2_mean']:.8f} +/- {res['r2_std']:.8f}")
        print(f" Spearman: {res['spearman_mean']:.8f} +/- {res['spearman_std']:.8f}")
        print(f"  Samples: {len(res['all_predictions'])}")
    best_method = max(methods, key=lambda m: results[m]["r2_mean"])
    worst_method = min(methods, key=lambda m: results[m]["r2_mean"])
    print(f"\nBEST METHOD: {best_method.upper()} (R^2 = {results[best_method]['r2_mean']:.8f})")
    print(f"WORST METHOD: {worst_method.upper()} (R^2 = {results[worst_method]['r2_mean']:.8f})")
    improvement = results[best_method]["r2_mean"] - results[worst_method]["r2_mean"]
    improvement_pct = (
        (improvement / abs(results[worst_method]["r2_mean"])) * 100 if results[worst_method]["r2_mean"] != 0 else 0
    )
    print(f"IMPROVEMENT: {improvement:.8f} R^2 points ({improvement_pct:.2f}%)")
    ax1 = axes[0, 0]
    mse_means = [results[m]["mse_mean"] for m in methods]
    mse_stds = [results[m]["mse_std"] for m in methods]
    bars1 = ax1.bar(methods, mse_means, yerr=mse_stds, capsize=5, alpha=0.7)
    ax1.set_title("Mean Squared Error")
    ax1.set_ylabel("MSE")
    ax1.tick_params(axis="x", rotation=45)
    for i, (bar, mean, std) in enumerate(zip(bars1, mse_means, mse_stds)):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std / 2,
            f"{mean:.6f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax2 = axes[0, 1]
    rmse_means = [results[m]["rmse_mean"] for m in methods]
    rmse_stds = [results[m]["rmse_std"] for m in methods]
    bars2 = ax2.bar(methods, rmse_means, yerr=rmse_stds, capsize=5, alpha=0.7)
    ax2.set_title("Root Mean Squared Error (RMSE)")
    ax2.set_ylabel("RMSE")
    ax2.tick_params(axis="x", rotation=45)
    for i, (bar, mean, std) in enumerate(zip(bars2, rmse_means, rmse_stds)):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std / 2,
            f"{mean:.6f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax3 = axes[1, 0]
    r2_means = [results[m]["r2_mean"] for m in methods]
    r2_stds = [results[m]["r2_std"] for m in methods]
    bars3 = ax3.bar(methods, r2_means, yerr=r2_stds, capsize=5, alpha=0.7)
    ax3.set_title("R^2 Score")
    ax3.set_ylabel("R^2")
    ax3.tick_params(axis="x", rotation=45)
    for i, (bar, mean, std) in enumerate(zip(bars3, r2_means, r2_stds)):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std / 2,
            f"{mean:.6f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax4 = axes[1, 1]
    true_vals = results[best_method]["all_true_values"]
    pred_vals = results[best_method]["all_predictions"]
    print(f"\n{best_method.upper()} SCATTER PLOT STATISTICS:")
    print(f"  Predictions range: {min(pred_vals):.8f} to {max(pred_vals):.8f}")
    print(f"  True values range: {min(true_vals):.8f} to {max(true_vals):.8f}")
    print(f"  Mean absolute error: {np.mean(np.abs(np.array(pred_vals) - np.array(true_vals))):.8f}")
    print(f"  Root mean squared error: {np.sqrt(np.mean((np.array(pred_vals) - np.array(true_vals))**2)):.8f}")
    correlation = np.corrcoef(true_vals, pred_vals)[0, 1]
    print(f"  Pearson correlation: {correlation:.8f}")
    ax4.scatter(true_vals, pred_vals, alpha=0.5)
    vmin, vmax = min(min(true_vals), min(pred_vals)), max(max(true_vals), max(pred_vals))
    ax4.plot([vmin, vmax], [vmin, vmax], "r--", linewidth=2)
    ax4.set_xlabel("True Dice Score")
    ax4.set_ylabel("Predicted Dice Score")
    ax4.set_title(f"Best Method: {best_method.upper()}")
    ax4.grid(True, alpha=0.3)
    ax4.text(
        0.05,
        0.95,
        f"R^2 = {results[best_method]['r2_mean']:.6f}",
        transform=ax4.transAxes,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax5 = axes[2, 0]
    residuals = np.array(pred_vals) - np.array(true_vals)
    print(f"\nRESIDUAL ANALYSIS FOR {best_method.upper()}:")
    print(f"  Mean residual: {np.mean(residuals):.8f}")
    print(f"  Std residual: {np.std(residuals):.8f}")
    print(f"  Min residual: {np.min(residuals):.8f}")
    print(f"  Max residual: {np.max(residuals):.8f}")
    print(f"  Median residual: {np.median(residuals):.8f}")
    print(f"  MAE of residuals: {np.mean(np.abs(residuals)):.8f}")
    from scipy.stats import shapiro
    if len(residuals) >= 3:
        try:
            shapiro_stat, shapiro_p = shapiro(residuals)
            print(f"  Shapiro-Wilk normality test: stat={shapiro_stat:.6f}, p-value={shapiro_p:.6f}")
        except:
            print("  Could not compute normality test")
    ax5.scatter(pred_vals, residuals, alpha=0.5)
    ax5.axhline(y=0, color="r", linestyle="--", linewidth=2)
    ax5.set_xlabel("Predicted Dice Score")
    ax5.set_ylabel("Residuals")
    ax5.set_title("Residual Analysis")
    ax5.grid(True, alpha=0.3)
    ax6 = axes[2, 1]
    spearman_means = [results[m]["spearman_mean"] for m in methods]
    spearman_stds = [results[m]["spearman_std"] for m in methods]
    bars6 = ax6.bar(methods, spearman_means, yerr=spearman_stds, capsize=5, alpha=0.7)
    ax6.set_title("Spearman Correlation")
    ax6.set_ylabel("Spearman ")
    ax6.tick_params(axis="x", rotation=45)
    for i, (bar, mean, std) in enumerate(zip(bars6, spearman_means, spearman_stds)):
        ax6.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std / 2,
            f"{mean:.6f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.tight_layout()
    Path(RESULTS / domain / "figures").mkdir(exist_ok=True)
    plt.savefig(RESULTS / domain / "figures/figure4_calibration_effectiveness.png", dpi=300, bbox_inches="tight")
    plt.savefig(RESULTS / domain / "figures/figure4_calibration_effectiveness.pdf", bbox_inches="tight")
    plt.close()
    print(" Figure 4: Calibration effectiveness saved")
def analyze_modality_specific_calibration(
    results_df, best_method="hierarchical", target_column="dice_score", domain=None
):
    """Analyze calibration effectiveness across different modalities with proper CV"""
    print("\n" + "=" * 60)
    print("MODALITY-SPECIFIC CALIBRATION ANALYSIS")
    print("=" * 60)
    modalities = ["T1", "T2", "FLAIR"]
    modality_results = {}
    valid_df = results_df.dropna(subset=[target_column, "psnr", "ssim"]).copy()
    for modality in modalities:
        print(f"\nAnalyzing {modality} modality...")
        modality_df = valid_df[valid_df["modality"] == modality].copy()
        if len(modality_df) < 50:
            print(f"  Insufficient data for {modality} ({len(modality_df)} samples)")
            continue
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_predictions = []
        fold_true_values = []
        for fold, (train_idx, test_idx) in enumerate(kfold.split(modality_df)):
            train_df = modality_df.iloc[train_idx].copy()
            test_df = modality_df.iloc[test_idx].copy()
            if best_method == "lightweight_cnn":
                calibrator = LightweightCNNCalibrator(domain=domain)
            else:
                calibrator = DomainAdaptiveCalibrator(method=best_method, domain=domain)
            calibrator.fit(train_df, target_column=target_column, validation_split=0.2)
            predictions = calibrator.predict(test_df)
            true_values = test_df[target_column].values
            fold_predictions.extend(predictions)
            fold_true_values.extend(true_values)
        mse = mean_squared_error(fold_true_values, fold_predictions)
        mae = mean_absolute_error(fold_true_values, fold_predictions)
        r2 = r2_score(fold_true_values, fold_predictions)
        rmse = np.sqrt(mse)
        rho, _ = spearmanr(fold_true_values, fold_predictions)
        modality_results[modality] = {
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "rmse": rmse,
            "spearman": rho,
            "predictions": fold_predictions,
            "true_values": fold_true_values,
            "sample_size": len(modality_df),
        }
        print(f"  Results - MSE: {mse:.8f}")
        print(f"           RMSE: {rmse:.8f}")
        print(f"            MAE: {mae:.8f}")
        print(f"            R^2: {r2:.8f}")
        print(f"       Spearman: {rho:.8f}")
        print(f"    Sample size: {len(modality_df)}")
    if modality_results:
        create_modality_calibration_figure(modality_results, domain=domain)
    return modality_results
def analyze_distortion_sensitivity(results_df, best_method="hierarchical", target_column="dice_score", domain=None):
    """Analyze how calibration performs across different distortion types with proper CV"""
    print("\n" + "=" * 60)
    print("DISTORTION SENSITIVITY ANALYSIS")
    print("=" * 60)
    valid_df = results_df.dropna(subset=[target_column, "psnr", "ssim"]).copy()
    distortion_types = valid_df["distortion"].unique()
    distortion_results = {}
    for distortion in distortion_types:
        if distortion == "original":
            continue
        print(f"\nAnalyzing {distortion} distortion...")
        distortion_df = valid_df[valid_df["distortion"] == distortion].copy()
        if len(distortion_df) < 30:
            print(f"  Insufficient data for {distortion} ({len(distortion_df)} samples)")
            continue
        train_df = valid_df[valid_df["distortion"] != distortion].copy()
        kfold = KFold(n_splits=min(5, len(train_df) // 10), shuffle=True, random_state=42)
        best_model = None
        best_score = -np.inf
        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_df)):
            fold_train_df = train_df.iloc[train_idx].copy()
            fold_val_df = train_df.iloc[val_idx].copy()
            calibrator = DomainAdaptiveCalibrator(method=best_method, domain=domain)
            calibrator.fit(fold_train_df, target_column=target_column, validation_split=0.2)
            val_pred = calibrator.predict(fold_val_df)
            val_true = fold_val_df[target_column].values
            val_score = r2_score(val_true, val_pred)
            if val_score > best_score:
                best_score = val_score
                best_model = calibrator
        predictions = best_model.predict(distortion_df)
        true_values = distortion_df[target_column].values
        mse = mean_squared_error(true_values, predictions)
        mae = mean_absolute_error(true_values, predictions)
        r2 = r2_score(true_values, predictions)
        rmse = np.sqrt(mse)
        rho, _ = spearmanr(true_values, predictions)
        distortion_results[distortion] = {
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "rmse": rmse,
            "spearman": rho,
            "sample_size": len(distortion_df),
        }
        print(f"  Results - MSE: {mse:.8f}")
        print(f"           RMSE: {rmse:.8f}")
        print(f"            MAE: {mae:.8f}")
        print(f"            R^2: {r2:.8f}")
        print(f"       Spearman: {rho:.8f}")
    print("\n" + "=" * 80)
    print("DISTORTION SENSITIVITY SUMMARY")
    print("=" * 80)
    print(f"{'Distortion':<15} {'MSE':<12} {'RMSE':<12} {'MAE':<12} {'R^2':<10} {'Spearman':<10} {'Samples':<8}")
    print("-" * 80)
    for distortion, res in distortion_results.items():
        print(
            f"{distortion:<15} {res['mse']:<12.8f} {res['rmse']:<12.8f} {res['mae']:<12.8f} {res['r2']:<10.6f} {res['spearman']:<10.6f} {res['sample_size']:<8}"
        )
    return distortion_results
def run_feature_probe(sample_img_path, models, save_dir=None, domain=None):
    """
    Runs one image through models, captures ResNet encoder features, saves grids,
    and computes CKA between models at each stage.
    Only models with a .encoder (ResNet) will yield features (DeepLabV3+, UNet-R101).
    """
    if save_dir == None:
        save_dir = Path(RESULTS / domain / "figures/feature_probe")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    img = cv2.imread(sample_img_path, cv2.IMREAD_GRAYSCALE)
    x = _prep_image_uint8_to_float01(img)
    t = _to_tensor_nchw(x)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    t = t.to(device=device, dtype=torch.float32)
    feats_all = {}
    hooks = []
    try:
        for name, model in models:
            fdict, h = register_hooks_for_resnet_encoder(model, name)
            feats_all[name] = fdict
            hooks += h
        with torch.no_grad():
            for name, model in models:
                _ = model(_adapt_for_model_input(t, model))
        model_names = list(feats_all.keys())
        layers = ["conv1", "layer1", "layer2", "layer3", "layer4"]
        cka_rows = []
        for ln in layers:
            for mname in model_names:
                fmap = feats_all[mname].get(f"{mname}:{ln}", None)
                if fmap is None:
                    continue
                grid = feature_grid(t, fmap, max_channels=8)
                outp = Path(save_dir) / f"{Path(sample_img_path).stem}_{mname}_{ln}.png"
                cv2.imwrite(str(outp), grid)
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    f1 = feats_all[model_names[i]].get(f"{model_names[i]}:{ln}")
                    f2 = feats_all[model_names[j]].get(f"{model_names[j]}:{ln}")
                    if f1 is None or f2 is None:
                        continue
                    cka = linear_cka(f1, f2)
                    cka_rows.append(
                        {
                            "image": sample_img_path,
                            "layer": ln,
                            "model_a": model_names[i],
                            "model_b": model_names[j],
                            "cka_linear": cka,
                        }
                    )
        if cka_rows:
            df = pd.DataFrame(cka_rows)
            Path(RESULTS / domain / "analysis_logs").mkdir(exist_ok=True)
            out_csv = Path(RESULTS / domain / "analysis_logs") / "feature_probe_cka.csv"
            if out_csv.exists():
                old = pd.read_csv(out_csv)
                df = pd.concat([old, df], ignore_index=True)
            df.to_csv(out_csv, index=False)
            print(f" Saved CKA rows to {out_csv}")
        print(f" Feature probe saved grids to {save_dir}")
    finally:
        for h in hooks:
            h.remove()
class DomainAdaptiveCalibrator:
    """
    Modified to use UnifiedEvaluator internally for consistency.
    """
    def __init__(self, method="hierarchical", domain=None):
        self.method = method
        self.model = None
        self.is_fitted = False
        self.model_params = {}
        self.domain = domain
        self.feature_processor = StandardizedFeatureProcessor(f"{method}_scaler")
        self.evaluator = UnifiedEvaluator(domain=domain or "medical", random_state=42, n_splits=5, verbose=False)
    def prepare_features(self, df, fit_encoders=False):
        """
        Use standardized feature preparation.
        fit_encoders parameter kept for backward compatibility but ignored.
        """
        X_scaled, feature_names = self.feature_processor.prepare_features(df, fit_scaler=fit_encoders)
        return X_scaled, None, None
    def fit(self, df, target_column="dice_score", validation_split=0.2, fold_idx=None):
        """Modified to ensure consistency with unified evaluation"""
        X_features, feature_names, _ = self.prepare_features(df, fit_encoders=True)
        if self.method == "random_forest":
            self.model = self.evaluator._get_model("random_forest")
        elif self.method == "linear":
            self.model = self.evaluator._get_model("linear")
        elif self.method == "xgboost":
            self.model = self.evaluator._get_model("xgboost")
        else:
            raise ValueError(f"Unknown method: {self.method}")
        y_target = pd.to_numeric(df[target_column], errors="coerce").astype(np.float32).values
        valid_mask = ~np.isnan(y_target)
        if not np.all(valid_mask):
            X_features = X_features[valid_mask]
            y_target = y_target[valid_mask]
        if len(y_target) < 10:
            raise ValueError(f"Insufficient samples for training: {len(y_target)}")
        self.model.fit(X_features, y_target)
        self.is_fitted = True
    def predict(self, df):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predictions")
        X_features, _, _ = self.prepare_features(df, fit_encoders=False)
        predictions = self.model.predict(X_features)
        return predictions
def create_modality_calibration_figure(modality_results, domain=None):
    """Create modality-specific calibration visualization (Figure 5)."""
    print("\n" + "=" * 60)
    print("FIGURE 5: MODALITY-SPECIFIC CALIBRATION RESULTS")
    print("=" * 60)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Modality-Specific Calibration Performance", fontsize=16, fontweight="bold")
    modalities = list(modality_results.keys())
    print("Modality-specific calibration performance:")
    print("-" * 60)
    for idx, modality in enumerate(modalities):
        ax = axes[idx]
        result = modality_results[modality]
        true_vals = result["true_values"]
        pred_vals = result["predictions"]
        r2 = result["r2"]
        mse = result["mse"]
        mae = result["mae"]
        rmse = result.get("rmse", np.sqrt(mse))
        spearman = result.get("spearman", 0)
        sample_size = result["sample_size"]
        print(f"\n{modality} MODALITY RESULTS:")
        print(f"  Sample size: {sample_size}")
        print(f"          MSE: {mse:.8f}")
        print(f"         RMSE: {rmse:.8f}")
        print(f"          MAE: {mae:.8f}")
        print(f"          R^2: {r2:.8f}")
        print(f"     Spearman: {spearman:.8f}")
        print(f"  Prediction range: [{min(pred_vals):.6f}, {max(pred_vals):.6f}]")
        print(f"  True value range: [{min(true_vals):.6f}, {max(true_vals):.6f}]")
        residuals = np.array(pred_vals) - np.array(true_vals)
        correlation = np.corrcoef(true_vals, pred_vals)[0, 1]
        print(f"  Mean residual: {np.mean(residuals):.8f}")
        print(f"  Std residual: {np.std(residuals):.8f}")
        print(f"  Pearson correlation: {correlation:.8f}")
        ax.scatter(true_vals, pred_vals, alpha=0.6, s=30)
        vmin, vmax = min(min(true_vals), min(pred_vals)), max(max(true_vals), max(pred_vals))
        ax.plot([vmin, vmax], [vmin, vmax], "r--", linewidth=2)
        ax.set_xlabel("True Dice Score")
        ax.set_ylabel("Predicted Dice Score")
        ax.set_title(f"{modality} (R^2 = {r2:.6f})")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(RESULTS / domain / "figures/figure5_modality_calibration.png", dpi=300, bbox_inches="tight")
    plt.savefig(RESULTS / domain / "figures/figure5_modality_calibration.pdf", bbox_inches="tight")
    plt.close()
    print(" Figure 5: Modality-specific calibration saved")
def calculate_domain_suitability(method, domain, performance):
    """Calculate how suitable a method is for a specific domain"""
    suitability_matrix = {
        "medical": {
            "random_forest": 0.95,
            "xgboost": 0.90,
            "linear": 0.75,
            "hierarchical": 0.20,
            "lightweight_cnn": 0.30,
        },
        "natural": {
            "random_forest": 0.60,
            "xgboost": 0.65,
            "linear": 0.50,
            "hierarchical": 0.75,
            "lightweight_cnn": 0.85,
        },
    }
    base_suitability = suitability_matrix.get(domain, {}).get(method, 0.5)
    r2 = performance.get("r2_mean", 0)
    performance_adjustment = r2 * 0.3
    final_suitability = base_suitability * 0.7 + performance_adjustment
    return min(1.0, max(0.0, final_suitability))
class LightweightCNNCalibrator:
    """
    Modified to use UnifiedEvaluator's training logic for consistency.
    """
    def __init__(self, use_attention=True, use_mixup=True, domain=None):
        self.model = None
        self.is_fitted = False
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.best_model_state = None
        self.training_history = {"train_loss": [], "val_loss": [], "val_r2": []}
        self.use_attention = use_attention
        self.use_mixup = use_mixup
        self.feature_importance = None
        self.domain = domain
        self.model_params = {}
        self.feature_processor = StandardizedFeatureProcessor("lightweightcnn_scaler")
        self.evaluator = UnifiedEvaluator(domain=domain or "medical", random_state=42, n_splits=5, verbose=False)
    def prepare_features(self, df, fit_encoders=False):
        """
        Wrapper method to use feature processor's prepare_features.
        Returns only X for backward compatibility with existing code.
        """
        X_scaled, feature_names = self.feature_processor.prepare_features(df, fit_scaler=fit_encoders)
        return X_scaled
    def fit(self, df, target_column="dice_score", validation_split=0.2, fold_idx=None):
        """Modified to match unified evaluator's training approach"""
        X_features = self.prepare_features(df, fit_encoders=True)
        y_target = pd.to_numeric(df[target_column], errors="coerce").fillna(0).values
        valid_mask = ~np.isnan(y_target)
        if not np.all(valid_mask):
            X_features = X_features[valid_mask]
            y_target = y_target[valid_mask]
        if validation_split > 0:
            train_idx, val_idx = train_test_split(
                np.arange(len(X_features)), test_size=validation_split, random_state=42
            )
        else:
            train_idx = np.arange(len(X_features))
            val_idx = []
        X_train, y_train = X_features[train_idx], y_target[train_idx]
        X_val, y_val = X_features[val_idx], y_target[val_idx] if len(val_idx) > 0 else (None, None)
        input_dim = X_features.shape[1]
        self.model = LightweightCNN(input_dim=input_dim, use_attention=self.use_attention).to(self.device)
        if X_val is not None:
            predictions = self.evaluator._train_lightweight_cnn(
                self.model, X_train, y_train, X_val, y_val, max_epochs=150, patience=25
            )
        self.is_fitted = True
        self._calculate_feature_importance(
            torch.FloatTensor(X_train).to(self.device), torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        )
    def predict(self, df):
        """Enhanced prediction with uncertainty estimation"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predictions")
        X = self.prepare_features(df)
        X_tensor = torch.FloatTensor(X).to(self.device)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(X_tensor)
            mean_pred = pred.cpu().numpy().flatten()
            mean_pred = np.clip(mean_pred, 0, 1)
        return mean_pred
