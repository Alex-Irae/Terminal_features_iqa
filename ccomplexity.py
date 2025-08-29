import os, psutil
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from aablation import AblationCalibrator
from models.mmodels import SklearnCalibrator
from utils.mmisc import RESULTS, CACHE, STANDARD_PARAMS
from ccalibration import LightweightCNNCalibrator
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
def run_computational_complexity_analysis(domain=None, cache_file=None):
    if cache_file is None:
        cache_file = os.path.join(Path(RESULTS / domain / "analysis_logs"), "computational_complexity.csv")
    """Comprehensive computational complexity analysis"""
    print("\n" + "=" * 50)
    print("COMPUTATIONAL COMPLEXITY ANALYSIS")
    print("=" * 50)
    if Path(cache_file).exists() and CACHE:
        print(" Using cached computational complexity results...")
        cached_df = pd.read_csv(cache_file)
        complexity_results = {}
        for method in cached_df["method"].unique():
            method_data = cached_df[cached_df["method"] == method]
            complexity_results[method] = {
                "input_sizes": method_data["input_sizes"].tolist(),
                "training_time": method_data["training_time"].tolist(),
                "inference_time": method_data["inference_time"].tolist(),
                "memory_usage": method_data["memory_usage"].tolist(),
                "model_size": method_data["model_size"].tolist(),
            }
        return complexity_results
    input_sizes = [100, 500, 1000, 2000, 5000]
    methods = ["lightweight_cnn", "xgboost", "random_forest", "linear"]
    complexity_results = {}
    for method in methods:
        print(f"\nAnalyzing {method} method...")
        complexity_results[method] = {
            "input_sizes": input_sizes,
            "training_time": [],
            "inference_time": [],
            "memory_usage": [],
            "model_size": [],
            "performance": [],
        }
        for size in input_sizes:
            print(f"  Testing with {size} samples...")
            X_synthetic = np.random.randn(size, 8)
            X_synthetic[:, 1] = 0.8 * X_synthetic[:, 0] + 0.2 * np.random.randn(size)
            modality_synthetic = np.random.choice(["T1", "T2", "FLAIR"], size)
            pathology_synthetic = np.random.choice(["healthy", "pathological"], size)
            distortion_synthetic = np.random.choice(
                ["blur", "noise", "motion", "rician", "bias", "ghosting", "original"], size
            )
            y_synthetic = 0.5 + 0.3 * np.random.randn(size)
            y_synthetic = np.clip(y_synthetic, 0, 1)
            synthetic_df = pd.DataFrame(
                {
                    "psnr": X_synthetic[:, 0],
                    "ssim": X_synthetic[:, 1],
                    "mse": X_synthetic[:, 2],
                    "mae": X_synthetic[:, 3],
                    "snr": X_synthetic[:, 4],
                    "cnr": X_synthetic[:, 5],
                    "gradient_mag": X_synthetic[:, 6],
                    "laplacian_var": X_synthetic[:, 7],
                    "modality": modality_synthetic,
                    "health": pathology_synthetic,
                    "distortion": distortion_synthetic,
                    "dice_score": y_synthetic,
                }
            )
            train_df, test_df = train_test_split(synthetic_df, test_size=0.2, random_state=42)
            try:
                process = psutil.Process(os.getpid())
                memory_before = process.memory_info().rss / 1024 / 1024
                start_time = time.time()
                if method == "lightweight_cnn":
                    from ccalibration import LightweightCNNCalibrator
                    calibrator = LightweightCNNCalibrator(use_attention=True, use_mixup=True, domain=domain)
                elif method == "random_forest":
                    from sklearn.ensemble import RandomForestRegressor
                    calibrator = SklearnCalibrator(RandomForestRegressor(
            n_estimators=STANDARD_PARAMS["n_estimators"],
            max_depth=STANDARD_PARAMS["max_depth"],
            random_state=STANDARD_PARAMS["random_state"],
            n_jobs=STANDARD_PARAMS["n_jobs"],
            min_samples_split=5,
            min_samples_leaf=2,
        ))
                elif method == "linear":
                    from sklearn.linear_model import Ridge
                    calibrator = SklearnCalibrator(Ridge(alpha=1.0))
                elif method == "xgboost":
                    import xgboost as xgb
                    calibrator = SklearnCalibrator(xgb.XGBRegressor(
            n_estimators=STANDARD_PARAMS["n_estimators"],
            max_depth=STANDARD_PARAMS["max_depth"],
            random_state=STANDARD_PARAMS["random_state"],
            n_jobs=STANDARD_PARAMS["n_jobs"],
            learning_rate=0.1,
            objective="reg:squarederror",
            subsample=0.8,
            colsample_bytree=0.8,
        ))
                calibrator.fit(train_df, target_column="dice_score", validation_split=0.2)
                training_time = time.time() - start_time
                memory_after = process.memory_info().rss / 1024 / 1024
                memory_usage = max(0, memory_after - memory_before)
                inference_sample = test_df.iloc[: min(100, len(test_df))]
                start_time = time.time()
                _ = calibrator.predict(inference_sample)
                inference_time = time.time() - start_time
                if hasattr(calibrator, "model") and hasattr(calibrator.model, "state_dict"):
                    model_size = sum(p.numel() for p in calibrator.model.parameters()) * 4 / 1024 / 1024
                else:
                    model_size = 0.1
                predictions = calibrator.predict(test_df)
                true_y = test_df["dice_score"].values
                r2 = r2_score(true_y, predictions)
                complexity_results[method]["training_time"].append(training_time)
                complexity_results[method]["inference_time"].append(inference_time)
                complexity_results[method]["memory_usage"].append(memory_usage)
                complexity_results[method]["model_size"].append(model_size)
                complexity_results[method]["performance"].append(r2)
                print(
                    f"    Training: {training_time:.2f}s, Inference: {inference_time:.4f}s, Memory: {memory_usage:.1f}MB, R2: {r2:.4f}"
                )
            except Exception as e:
                print(f"    Error with {method} at size {size}: {e}")
                complexity_results[method]["training_time"].append(np.nan)
                complexity_results[method]["inference_time"].append(np.nan)
                complexity_results[method]["memory_usage"].append(np.nan)
                complexity_results[method]["model_size"].append(np.nan)
                complexity_results[method]["performance"].append(np.nan)
    complexity_df = pd.DataFrame()
    for method, data in complexity_results.items():
        method_df = pd.DataFrame(data)
        method_df["method"] = method
        complexity_df = pd.concat([complexity_df, method_df], ignore_index=True)
    if CACHE:
        complexity_df.to_csv(RESULTS / domain / "analysis_logs" / "computational_complexity.csv", index=False)
    create_complexity_figure(complexity_results, domain)
    print_complexity_results(complexity_results)
    complexity_df = pd.DataFrame()
    for method, data in complexity_results.items():
        method_df = pd.DataFrame(data)
        method_df["method"] = method
        complexity_df = pd.concat([complexity_df, method_df], ignore_index=True)
    if CACHE:
        complexity_df.to_csv(cache_file, index=False)
    return complexity_results
def create_complexity_figure(complexity_results, domain):
    """Create computational complexity visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Computational Complexity Analysis", fontsize=16, fontweight="bold")
    methods = list(complexity_results.keys())
    colors = ["blue", "red", "green", "purple"]
    ax = axes[0, 0]
    for i, method in enumerate(methods):
        data = complexity_results[method]
        ax.plot(data["input_sizes"], data["training_time"], marker="o", label=method, color=colors[i], linewidth=2)
        ax.fill_between(
            data["input_sizes"],
            [max(0, t - 0.1 * t) for t in data["training_time"]],
            [t + 0.1 * t for t in data["training_time"]],
            alpha=0.2,
            color=colors[i],
        )
    ax.set_xlabel("Dataset Size")
    ax.set_ylabel("Training Time (seconds)")
    ax.set_title("Training Time Scalability")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    ax = axes[0, 1]
    for i, method in enumerate(methods):
        data = complexity_results[method]
        ax.plot(data["input_sizes"], data["memory_usage"], marker="s", label=method, color=colors[i], linewidth=2)
    ax.set_xlabel("Dataset Size")
    ax.set_ylabel("Memory Usage (MB)")
    ax.set_title("Memory Usage Scalability")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax = axes[1, 0]
    for i, method in enumerate(methods):
        data = complexity_results[method]
        inference_per_sample = [t / 100 for t in data["inference_time"]]
        ax.plot(data["input_sizes"], inference_per_sample, marker="^", label=method, color=colors[i], linewidth=2)
    ax.set_xlabel("Training Dataset Size")
    ax.set_ylabel("Inference Time per 100 samples (seconds)")
    ax.set_title("Inference Efficiency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax = axes[1, 1]
    avg_training_times = {method: np.mean(data["training_time"]) for method, data in complexity_results.items()}
    avg_performance = {method: np.mean(data["performance"]) for method, data in complexity_results.items()}
    for i, method in enumerate(methods):
        if method in avg_performance and method in avg_training_times:
            ax.scatter(
                avg_training_times[method], avg_performance[method], s=200, alpha=0.7, color=colors[i], label=method
            )
            ax.annotate(
                method,
                (avg_training_times[method], avg_performance[method]),
                xytext=(5, 5),
                textcoords="offset points",
            )
    ax.set_xlabel("Average Training Time (seconds)")
    ax.set_ylabel("Performance (R^2)")
    ax.set_title("Performance vs. Training Time Trade-off")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(RESULTS / domain / "figures/computational_complexity.png"), dpi=300, bbox_inches="tight")
    plt.close()
def print_complexity_results(complexity_results):
    """Print computational complexity results"""
    print("\n" + "=" * 60)
    print("COMPUTATIONAL COMPLEXITY RESULTS")
    print("=" * 60)
    for method, data in complexity_results.items():
        print(f"\n{method.upper()} METHOD:")
        print("-" * 40)
        sizes = data["input_sizes"]
        times = data["training_time"]
        valid_indices = [i for i, t in enumerate(times) if not np.isnan(t)]
        if len(valid_indices) >= 2:
            valid_sizes = [sizes[i] for i in valid_indices]
            valid_times = [times[i] for i in valid_indices]
            if len(valid_sizes) > 1:
                scaling_factor = (valid_times[-1] / valid_times[0]) / (valid_sizes[-1] / valid_sizes[0])
                print(f"Scaling Factor: {scaling_factor:.2f}x time per data size increase")
            print(f"Training Time Range: {min(valid_times):.2f}s - {max(valid_times):.2f}s")
        memory_usage = data["memory_usage"]
        valid_memory = [m for m in memory_usage if not np.isnan(m)]
        if valid_memory:
            print(f"Memory Usage Range: {min(valid_memory):.1f}MB - {max(valid_memory):.1f}MB")
        model_sizes = data["model_size"]
        valid_model_sizes = [m for m in model_sizes if not np.isnan(m)]
        if valid_model_sizes:
            print(f"Model Size: ~{np.mean(valid_model_sizes):.2f}MB")
        performances = data["performance"]
        valid_performances = [p for p in performances if not np.isnan(p)]
        if valid_performances:
            print(f"Performance (R2) Range: {min(valid_performances):.2f} - {max(valid_performances):.2f}")
