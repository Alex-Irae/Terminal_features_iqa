from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import cv2
from skimage.metrics import structural_similarity as ssim
from scipy.stats import f_oneway, spearmanr
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import lpips
import matplotlib.pyplot as plt
import pingouin as pg
from scipy import stats
from scipy.stats import spearmanr, pearsonr
from sklearn.feature_selection import mutual_info_regression
import warnings
lpips_model = lpips.LPIPS(net="alex").eval()
from utils.mmisc import (
    safe_symlink_or_copy,
    dataset_signature,
    resolve_path,
    dice_coefficient,
    RESULTS,
    CACHE,
)
from models.mmodels import UNet
class IQAMetrics:
    """Domain-agnostic IQA metrics evaluation"""
    @staticmethod
    def psnr(img1, img2):
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float("inf")
        max_pixel = 255.0
        return 20 * np.log10(max_pixel / np.sqrt(mse))
    @staticmethod
    def ssim_metric(img1, img2):
        return ssim(img1, img2, data_range=img1.max() - img1.min())
    @staticmethod
    def mse_metric(img1, img2):
        return np.mean((img1 - img2) ** 2)
    @staticmethod
    def mae_metric(img1, img2):
        return np.mean(np.abs(img1 - img2))
    @staticmethod
    def snr(image):
        """Signal-to-Noise Ratio"""
        mean_signal = np.mean(image)
        std_noise = np.std(image)
        if std_noise == 0:
            return float("inf")
        return 20 * np.log10(mean_signal / std_noise)
    @staticmethod
    def cnr(image, mask=None):
        """Contrast-to-Noise Ratio - adapted for natural images"""
        if mask is None:
            h, w = image.shape
            center_mask = np.zeros_like(image, dtype=bool)
            center_h, center_w = h // 4, w // 4
            center_mask[center_h : 3 * center_h, center_w : 3 * center_w] = True
            foreground = image[center_mask]
            background = image[~center_mask]
        else:
            if np.sum(mask) == 0:
                return 0
            foreground = image[mask > 0]
            background = image[mask == 0]
        if len(foreground) == 0 or len(background) == 0:
            return 0
        mean_fg = np.mean(foreground)
        mean_bg = np.mean(background)
        std_bg = np.std(background)
        if std_bg == 0:
            return float("inf")
        return abs(mean_fg - mean_bg) / std_bg
    @staticmethod
    def gradient_magnitude(image):
        """Image sharpness via gradient magnitude"""
        img32 = image.astype(np.float32, copy=False)
        gx = cv2.Sobel(img32, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
        gy = cv2.Sobel(img32, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)
        grad_mag = np.sqrt(gx * gx + gy * gy)
        return grad_mag.mean()
    @staticmethod
    def laplacian_variance(image):
        """Laplacian variance for blur detection"""
        x = image.astype(np.float32, copy=False)
        lap = cv2.Laplacian(x, cv2.CV_32F, ksize=3)
        return float(lap.var())
    @staticmethod
    def brisque_approximation(image):
        """Simplified BRISQUE-like metric for natural image quality"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mu = cv2.GaussianBlur(image.astype(np.float32), (3, 3), 1.166)
        mu_sq = mu * mu
        sigma = cv2.GaussianBlur(image.astype(np.float32) * image.astype(np.float32), (3, 3), 1.166)
        sigma = np.sqrt(np.abs(sigma - mu_sq))
        sigma[sigma == 0] = 1
        structdis = (image.astype(np.float32) - mu) / (sigma + 1)
        return float(np.var(structdis))
    @staticmethod
    def entropy(image):
        """Image entropy as quality measure"""
        hist, _ = np.histogram(image, bins=256, range=(0, 256), density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))
    @staticmethod
    def edge_density(image):
        """Edge density as image complexity measure"""
        edges = cv2.Canny(image.astype(np.uint8), 50, 150)
        return np.sum(edges > 0) / (image.shape[0] * image.shape[1])
    @staticmethod
    def intersection_over_union(pred, target):
        """IoU for segmentation masks - medical only"""
        if pred is None or target is None:
            return np.nan
        pred = pred.flatten()
        target = target.flatten()
        intersection = np.sum(pred * target)
        union = np.sum(pred) + np.sum(target) - intersection
        return intersection / (union + 1e-6)
    @staticmethod
    def root_mean_squared_error(pred, target):
        """RMSE between predicted and target values - medical only"""
        if pred is None or target is None:
            return np.nan
        return np.sqrt(np.mean((pred - target) ** 2))
    @staticmethod
    def spearman_rank_correlation(pred, target):
        """Spearman's rank correlation - medical only"""
        if pred is None or target is None:
            return np.nan
        try:
            rho, _ = spearmanr(pred, target)
            return rho if not np.isnan(rho) else 0.0
        except:
            return np.nan
    @staticmethod
    def dice_(pred, target):
        """Dice coefficient - medical only"""
        if pred is None or target is None:
            return np.nan
        return dice_coefficient(pred, target)
def to_tensor(img):
    """
    Convert a HxWxC NumPy image to a PyTorch tensor in [-1, 1], shape (1,3,H,W).
    If grayscale, repeats channel to get 3-channel.
    """
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
    img = img * 2.0 - 1.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return tensor
def natural_image_target(ref, dist):
    max_val = 255.0 if ref.dtype == np.uint8 else 1.0
    s_msssim = ms_ssim(ref, dist, max_val=max_val)
    psnr_val = IQAMetrics.psnr(ref * 255.0, dist * 255.0)
    s_psnr = 1 / (1 + np.exp(-0.25 * (psnr_val - 30)))
    lpips_val = lpips_model(to_tensor(ref), to_tensor(dist)).item()
    s_lpips = 1 / (1 + np.exp(-12 * (0.15 - lpips_val)))
    target = 0.35 * s_msssim + 0.10 * s_psnr + 0.55 * s_lpips
    return target
def ms_ssim(img1, img2, max_val=255, weights=None, levels=5):
    """
    Multi-scale SSIM (MS-SSIM) approximation using skimage.metrics.ssim.
    img1, img2: uint8 or float images (H, W) or (H, W, C) in same dtype/range
    max_val: max pixel value (255 for uint8, 1 for floats)
    weights: per-scale weights; if None, uses Wang et al. default for 5 scales
    """
    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    if img1.ndim == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    mssim_vals = []
    mcs_vals = []
    for level in range(levels):
        ssim_val, ssim_map = ssim(img1, img2, data_range=max_val, full=True)
        mssim_vals.append(ssim_val)
        mcs_vals.append(ssim_val)
        if level < levels - 1:
            img1 = cv2.pyrDown(img1)
            img2 = cv2.pyrDown(img2)
    mssim_vals = np.array(mssim_vals)
    mcs_vals = np.array(mcs_vals)
    return np.prod(mcs_vals[:-1] ** weights[:-1]) * (mssim_vals[-1] ** weights[-1])
class DomainAwareImageDataset(Dataset):
    """Dataset that handles both medical and natural images"""
    def __init__(self, df, domain="medical", transform=None, load_masks=True):
        self.df = df.reset_index(drop=True)
        self.domain = domain
        self.transform = transform
        self.load_masks = load_masks and (domain == "medical")
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = resolve_path(row["path_img"])
        if self.domain == "natural":
            image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
        mask = None
        if self.load_masks and self.domain == "medical":
            mask_path = resolve_path(row["path_mask_original"])
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask = (mask > 0).astype(np.uint8)
        if self.transform:
            image = self.transform(image)
        if self.domain == "natural":
            if len(image.shape) == 3:
                image_t = torch.from_numpy(image.copy()).permute(2, 0, 1)
            else:
                image_t = torch.from_numpy(image.copy()).unsqueeze(0)
        else:
            image_t = torch.from_numpy(image.copy()).unsqueeze(0)
        mask_t = None
        if mask is not None:
            mask_t = torch.from_numpy(mask.copy()).unsqueeze(0)
        return {"image": image_t, "mask": mask_t, "metadata": row.to_dict()}
def get_original_image_path(row, domain="medical"):
    """
    Get the path to the original (reference) image for a given distorted image.
    Args:
        row: DataFrame row containing image metadata
        domain: "medical" or "natural"
    Returns:
        Path to original image
    """
    if row["distortion"] == "original":
        return row["path_img"]
    base_path = Path(row["path_img"]).parent
    original_filename = f"{row['base_id']}.png"
    return base_path / original_filename
def run_iqa_analysis(
    results_df, *, force_cache=False, epochs=65, skip_train=False, sample_size=None, domain="medical", csv_path
):
    """
    Domain-aware IQA analysis driver - FIXED VERSION
    Args:
        results_df: DataFrame with image metadata
        domain: "medical" or "natural" - determines which metrics to compute
        csv_path: Path to CSV file (REQUIRED - no default value)
        force_cache: Whether to force cache refresh
        epochs: Number of training epochs (ignored for natural domain)
        skip_train: Skip model training
        sample_size: Limit analysis to N samples
    """
    print(f"Starting comprehensive {domain} IQA analysis...")
    Path(RESULTS / domain / "analysis_logs").mkdir(exist_ok=True)
    Path(RESULTS / domain / "figures").mkdir(exist_ok=True)
    if domain not in ["medical", "natural"]:
        raise ValueError(f"Domain must be 'medical' or 'natural', got '{domain}'")
    if csv_path is None:
        raise ValueError("csv_path parameter is required")
    csv_path = Path(csv_path)
    eff_sample = len(results_df) if sample_size is None else int(sample_size)
    model = None
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    if domain == "medical" and not skip_train:
        model = train_segmentation_model(results_df, num_epochs=epochs, force_cache=force_cache, csv_path=csv_path)
    results_df = analyze_iqa_metrics(
        results_df,
        sample_size=eff_sample,
        force_cache=force_cache,
        domain=domain,
        csv_path=csv_path,
        model=model,
        device=device,
    )
    if domain == "medical":
        modality_bias_analysis(results_df)
        correlation_results = analyze_feature_target_correlations(
            df=results_df,
            target_column="dice_score",
            correlation_threshold=0.95,
            p_value_threshold=0.001,
            domain=domain,
            save_results=True,
        )
    print("\n" + "=" * 80)
    print(f"{domain.upper()} IQA ANALYSIS COMPLETED")
    print("=" * 80)
    return results_df
def analyze_iqa_metrics(
    results_df,
    sample_size=1000,
    force_cache=False,
    domain="medical",
    csv_path=Path("PHD/mri_csv.csv"),
    model=None,
    device=None,
):
    """
    Domain-aware comprehensive IQA metrics analysis
    """
    print("\n" + "=" * 50)
    print(f"{domain.upper()} IQA METRICS ANALYSIS")
    print("=" * 50)
    sig = dataset_signature(Path(csv_path), extra={"phase": "iqa", "sample_size": int(sample_size), "domain": domain})
    hashed_csv = Path(RESULTS / domain / "analysis_logs" / f"iqa_metrics_results_{domain}_{sig}.csv")
    stable_csv = Path(RESULTS / domain / "analysis_logs" / f"iqa_metrics_results_{domain}.csv")
    if hashed_csv.exists() and not force_cache and CACHE:
        print(f" Using cached IQA metrics: {hashed_csv.name}")
        cached_results = pd.read_csv(hashed_csv)
        if domain == "medical":
            iqa_cols = [
                "psnr",
                "ssim",
                "mse",
                "mae",
                "snr",
                "cnr",
                "gradient_mag",
                "laplacian_var",
                "iou",
                "rmse",
                "spearman",
                "dice_score",
                "is_original",
                "brisque_approx",
                "entropy",
                "edge_density",
                "quality_score",
            ]
        else:
            iqa_cols = [
                "psnr",
                "ssim",
                "mse",
                "mae",
                "snr",
                "cnr",
                "gradient_mag",
                "laplacian_var",
                "brisque_approx",
                "entropy",
                "edge_density",
                "is_original",
                "quality_score",
            ]
        for col in iqa_cols:
            if col in cached_results.columns:
                for idx, row in results_df.iterrows():
                    matching_rows = cached_results[cached_results["path_img"] == row["path_img"]]
                    if len(matching_rows) > 0:
                        results_df.loc[idx, col] = matching_rows.iloc[0][col]
        safe_symlink_or_copy(hashed_csv, stable_csv)
        return results_df
    if len(results_df) > sample_size:
        sample_indices = results_df.sample(n=sample_size, random_state=42).index
        print(f"Analyzing random sample of {sample_size} images")
    else:
        sample_indices = results_df.index
        print(f"Analyzing all {len(results_df)} images")
    print(f"\nProcessing {domain} images...")
    processed_count = 0
    for idx in sample_indices:
        row = results_df.loc[idx]
        processed_count += 1
        if processed_count % 100 == 0:
            print(f"Processed {processed_count}/{len(sample_indices)} images...")
        try:
            img_path = resolve_path(row["path_img"])
            if domain == "natural":
                distorted_img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if distorted_img is not None:
                    distorted_img = cv2.cvtColor(distorted_img, cv2.COLOR_BGR2RGB)
                    distorted_gray = cv2.cvtColor(distorted_img, cv2.COLOR_RGB2GRAY)
                else:
                    continue
            else:
                distorted_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if distorted_gray is None:
                    continue
            mask = None
            if domain == "medical":
                mask_path = resolve_path(row["path_mask_original"])
                if Path(mask_path).exists():
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        mask = (mask > 0).astype(np.uint8)
            is_orig = row["distortion"] == "original"
            results_df.loc[idx, "is_original"] = is_orig
            if not is_orig:
                if domain == "natural":
                    original_path = get_original_image_path(row, domain)
                    if Path(original_path).exists():
                        original_img = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
                        if original_img is not None:
                            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                            original_gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
                        else:
                            original_gray = distorted_gray.copy()
                    else:
                        print(f"Warning: Original not found for {row['base_id']}, using distorted as reference")
                        original_gray = distorted_gray.copy()
                else:
                    original_rows = results_df[
                        (results_df["base_id"] == row["base_id"]) & (results_df["distortion"] == "original")
                    ]
                    if len(original_rows) == 0:
                        original_gray = distorted_gray.copy()
                    else:
                        orig_path = original_rows.iloc[0]["path_img"]
                        original_gray = cv2.imread(str(orig_path), cv2.IMREAD_GRAYSCALE)
                        if original_gray is None:
                            original_gray = distorted_gray.copy()
            else:
                original_gray = distorted_gray.copy()
            distorted_gray = distorted_gray.astype(np.float32) / 255.0
            original_gray = original_gray.astype(np.float32) / 255.0
            if not is_orig:
                results_df.loc[idx, "psnr"] = IQAMetrics.psnr(original_gray * 255, distorted_gray * 255)
                results_df.loc[idx, "ssim"] = IQAMetrics.ssim_metric(original_gray, distorted_gray)
                results_df.loc[idx, "mse"] = IQAMetrics.mse_metric(original_gray, distorted_gray)
                results_df.loc[idx, "mae"] = IQAMetrics.mae_metric(original_gray, distorted_gray)
                results_df.loc[idx, "quality_score"] = natural_image_target(original_gray, distorted_gray)
                if domain == "natural":
                    results_df.loc[idx, "quality_score"] = natural_image_target(original_gray, distorted_gray)
                else:
                    results_df.loc[idx, "quality_score"] = np.nan
            else:
                results_df.loc[idx, "psnr"] = np.nan
                results_df.loc[idx, "quality_score"] = np.nan
                results_df.loc[idx, "ssim"] = np.nan
                results_df.loc[idx, "mse"] = np.nan
                results_df.loc[idx, "mae"] = np.nan
            results_df.loc[idx, "snr"] = IQAMetrics.snr(distorted_gray * 255)
            if domain == "natural":
                results_df.loc[idx, "cnr"] = IQAMetrics.cnr(distorted_gray * 255, mask=None)
            else:
                results_df.loc[idx, "cnr"] = IQAMetrics.cnr(distorted_gray * 255, mask)
            results_df.loc[idx, "gradient_mag"] = IQAMetrics.gradient_magnitude(distorted_gray * 255)
            results_df.loc[idx, "laplacian_var"] = IQAMetrics.laplacian_variance(distorted_gray * 255)
            results_df.loc[idx, "brisque_approx"] = IQAMetrics.brisque_approximation(distorted_gray * 255)
            results_df.loc[idx, "entropy"] = IQAMetrics.entropy(distorted_gray * 255)
            results_df.loc[idx, "edge_density"] = IQAMetrics.edge_density(distorted_gray * 255)
            if domain == "medical" and model is not None and mask is not None:
                image_t = torch.from_numpy(distorted_gray).unsqueeze(0).unsqueeze(0).float()
                model.eval()
                with torch.no_grad():
                    output = model(image_t.to(device))
                    pred = (output.sigmoid() > 0.5).float().cpu().numpy().squeeze()
                target = mask.astype(np.float32)
                results_df.loc[idx, "iou"] = IQAMetrics.intersection_over_union(pred=pred, target=target)
                results_df.loc[idx, "rmse"] = IQAMetrics.root_mean_squared_error(pred=pred, target=target)
                results_df.loc[idx, "spearman"] = IQAMetrics.spearman_rank_correlation(
                    pred=pred.flatten(), target=target.flatten()
                )
                results_df.loc[idx, "dice_score"] = IQAMetrics.dice_(pred=pred, target=target)
            elif domain == "natural":
                results_df.loc[idx, "iou"] = np.nan
                results_df.loc[idx, "rmse"] = np.nan
                results_df.loc[idx, "dice_score"] = np.nan
                results_df.loc[idx, "spearman"] = np.nan
        except Exception as e:
            print(f"Error processing {row.get('path_img', '?')}: {e}")
            continue
    non_original_results = results_df[results_df["distortion"] != "original"].copy()
    print(f"\nSuccessfully analyzed {len(non_original_results)} images")
    print(f"\n{domain.upper()} IQA METRICS SUMMARY:")
    print("-" * 30)
    if domain == "medical":
        summary_cols = [
            "psnr",
            "ssim",
            "mse",
            "mae",
            "snr",
            "cnr",
            "gradient_mag",
            "laplacian_var",
            "rmse",
            "spearman",
            "dice_score",
            "iou",
            "brisque_approx",
            "entropy",
            "edge_density",
            "quality_score",
        ]
    else:
        summary_cols = [
            "psnr",
            "ssim",
            "mse",
            "mae",
            "snr",
            "cnr",
            "gradient_mag",
            "laplacian_var",
            "brisque_approx",
            "entropy",
            "edge_density",
            "quality_score",
        ]
    available_cols = [col for col in summary_cols if col in non_original_results.columns]
    if available_cols:
        print(non_original_results[available_cols].describe())
    if CACHE:
        non_original_results.to_csv(hashed_csv, index=False)
        safe_symlink_or_copy(hashed_csv, stable_csv)
        print(f" Saved {domain} IQA cache as {hashed_csv.name} and updated {stable_csv.name}")
    return results_df
def modality_bias_analysis(results_df):
    """Analyze IQA metric biases across modalities - medical domain only"""
    print("\n" + "=" * 50)
    print("MODALITY BIAS ANALYSIS")
    print("=" * 50)
    metrics = [
        "psnr",
        "ssim",
        "mse",
        "mae",
        "snr",
        "cnr",
        "gradient_mag",
        "laplacian_var",
        "iou",
        "rmse",
        "spearman",
        "dice_score",
        "brisque_approx",
        "entropy",
        "edge_density",
    ]
    if "modality" not in results_df.columns:
        print("Warning: No modality column found, skipping modality bias analysis")
        return results_df
    modalities = results_df["modality"].unique()
    if len(modalities) < 2:
        print("Warning: Less than 2 modalities found, skipping bias analysis")
        return results_df
    distorted = results_df[results_df["distortion"] != "original"].copy()
    distorted = distorted.replace([np.inf, -np.inf], np.nan)
    for m in metrics:
        if m in distorted.columns:
            distorted[m] = pd.to_numeric(distorted[m], errors="coerce")
    print("\nMODALITY-WISE METRIC STATISTICS:")
    print("-" * 40)
    for metric in metrics:
        if metric not in distorted.columns:
            continue
        print(f"\n{metric.upper()}:")
        sub = distorted[["modality", metric]].dropna()
        if len(sub) == 0:
            print("No valid data for this metric")
            continue
        modality_stats = sub.groupby("modality")[metric].agg(["mean", "std", "count"])
        print(modality_stats)
        groups = [
            sub.loc[sub["modality"] == m, metric].values
            for m in modalities
            if (sub["modality"] == m).any() and len(sub.loc[sub["modality"] == m]) > 1
        ]
        if len(groups) >= 2:
            try:
                f_stat, p_val = f_oneway(*groups)
                print(f"ANOVA F-statistic: {f_stat:.4f}, p-value: {p_val:.6f}")
            except Exception as e:
                print(f"ANOVA failed: {e}")
    return results_df
def train_segmentation_model(
    results_df,
    num_epochs=50,
    batch_size=64,
    num_workers=4,
    force_cache=False,
    csv_path=Path("PHD/mri_csv.csv"),
    domain="medical",
):
    """Train U-Net for segmentation task (medical domain only)"""
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    print("\n" + "=" * 50)
    print("SEGMENTATION MODEL TRAINING (MEDICAL DOMAIN)")
    print("=" * 50)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    pin = device.type == "cuda:1"
    original_df = results_df[results_df["distortion"] == "original"].copy()
    print(f"Training on {len(original_df)} original images")
    dataset = DomainAwareImageDataset(original_df, domain="medical", transform=None, load_masks=True)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4,
    )
    sig = dataset_signature(Path(csv_path), extra={"phase": "unet", "epochs": int(num_epochs), "B": int(batch_size)})
    models_dir = Path(RESULTS / domain / "models")
    models_dir.mkdir(parents=True, exist_ok=True)
    ckpt_hashed = models_dir / f"unet_e{num_epochs}_{sig}.pth"
    hist_hashed = models_dir / f"unet_e{num_epochs}_{sig}_history.json"
    ckpt_stable = Path(RESULTS / domain / "analysis_logs") / "unet_model.pth"
    if ckpt_hashed.exists() and not force_cache and CACHE:
        print(f" Using cached model: {ckpt_hashed.name}")
        safe_symlink_or_copy(ckpt_hashed, ckpt_stable)
        model = UNet(in_channels=1, out_channels=1).to(device)
        model.load_state_dict(torch.load(ckpt_hashed, map_location=device))
        return model.eval()
    model = UNet(in_channels=1, out_channels=1).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_dice": []}
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            try:
                imgs = batch["image"]
                msks = batch["mask"]
                if imgs is None or msks is None:
                    continue
                if imgs.dim() == 3:
                    imgs = imgs.unsqueeze(1)
                if msks.dim() == 3:
                    msks = msks.unsqueeze(1)
                imgs = imgs.float().to(device) / 255.0
                msks = (msks > 0).float().to(device)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(imgs)
                loss = criterion(outputs, msks)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            except Exception as e:
                print(f"Training batch error: {e}")
                continue
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                try:
                    imgs = batch["image"]
                    msks = batch["mask"]
                    if imgs.dim() == 3:
                        imgs = imgs.unsqueeze(1)
                    if msks.dim() == 3:
                        msks = msks.unsqueeze(1)
                    imgs = imgs.float().to(device) / 255.0
                    msks = (msks > 0).float().to(device)
                    outputs = model(imgs)
                    loss = criterion(outputs, msks)
                    val_loss += loss.item()
                    preds = (outputs > 0.5).float()
                    val_dice += sum(
                        dice_coefficient(preds[i].cpu(), msks[i].cpu()).item() for i in range(preds.size(0))
                    )
                    val_count += preds.size(0)
                except Exception:
                    continue
        avg_train_loss = train_loss / max(1, len(train_loader))
        avg_val_loss = val_loss / max(1, len(val_loader))
        avg_val_dice = val_dice / max(1, val_count)
        print(
            f"Epoch {epoch+1}/{num_epochs}: Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}"
        )
        history["epoch"].append(int(epoch + 1))
        history["train_loss"].append(float(avg_train_loss))
        history["val_loss"].append(float(avg_val_loss))
        history["val_dice"].append(float(avg_val_dice))
    if CACHE:
        torch.save(model.state_dict(), ckpt_hashed)
        with open(hist_hashed, "w") as f:
            json.dump({"created": datetime.now().isoformat(), **history}, f)
        Path(RESULTS / domain / "analysis_logs").mkdir(exist_ok=True)
        safe_symlink_or_copy(ckpt_hashed, ckpt_stable)
        print(f" Saved model to {ckpt_hashed.name} and updated {ckpt_stable.name}")
    return model
def validate_natural_image_setup(results_df, base_dir=None, domain="natural"):
    """
    Validate that natural image dataset is properly structured
    Args:
        results_df: DataFrame with image metadata
        base_dir: Base directory for images (optional, auto-detected from paths)
    Returns:
        dict: Validation results and recommendations
    """
    print("\n" + "=" * 50)
    print("NATURAL IMAGE DATASET VALIDATION")
    print("=" * 50)
    validation_results = {"valid": True, "warnings": [], "errors": [], "stats": {}}
    required_cols = ["path_img", "base_id", "distortion"]
    missing_cols = [col for col in required_cols if col not in results_df.columns]
    if missing_cols:
        validation_results["errors"].append(f"Missing required columns: {missing_cols}")
        validation_results["valid"] = False
    original_count = len(results_df[results_df["distortion"] == "original"])
    distorted_count = len(results_df[results_df["distortion"] != "original"])
    validation_results["stats"]["original_images"] = original_count
    validation_results["stats"]["distorted_images"] = distorted_count
    if original_count == 0:
        validation_results["errors"].append("No original images found (distortion == 'original')")
        validation_results["valid"] = False
    unique_base_ids = results_df["base_id"].nunique()
    validation_results["stats"]["unique_base_ids"] = unique_base_ids
    missing_originals = []
    if base_dir is None:
        first_path = Path(results_df.iloc[0]["path_img"])
        base_dir = first_path.parent
    distorted_rows = results_df[results_df["distortion"] != "original"]
    for _, row in distorted_rows.head(20).iterrows():
        original_path = get_original_image_path(row, domain="natural")
        if not Path(original_path).exists():
            missing_originals.append((row["base_id"], original_path))
    if missing_originals:
        validation_results["warnings"].append(
            f"Missing original images for {len(missing_originals)} distorted images. "
            f"Examples: {missing_originals[:3]}"
        )
    extensions = results_df["path_img"].apply(lambda x: Path(x).suffix.lower()).value_counts()
    validation_results["stats"]["file_extensions"] = dict(extensions)
    if len(extensions) > 1:
        validation_results["warnings"].append(
            f"Multiple image formats detected: {dict(extensions)}. " "Consider standardizing to a single format."
        )
    distortion_counts = results_df["distortion"].value_counts()
    validation_results["stats"]["distortion_distribution"] = dict(distortion_counts)
    print(f" Dataset contains {len(results_df)} total images")
    print(f" {original_count} original images, {distorted_count} distorted images")
    print(f" {unique_base_ids} unique base images")
    print(f" File extensions: {dict(extensions)}")
    print(f" Distortion types: {dict(distortion_counts)}")
    if validation_results["warnings"]:
        print("\nWARNINGS:")
        for warning in validation_results["warnings"]:
            print(f"  {warning}")
    if validation_results["errors"]:
        print("\nERRORS:")
        for error in validation_results["errors"]:
            print(f" {error}")
    else:
        print("\n Dataset validation passed!")
    return validation_results
def natural_image_quality_analysis(results_df, domain="natural"):
    """
    Natural image specific quality analysis
    Args:
        results_df: DataFrame with computed IQA metrics for natural images
    """
    print("\n" + "=" * 50)
    print("NATURAL IMAGE QUALITY ANALYSIS")
    print("=" * 50)
    distorted = results_df[results_df["distortion"] != "original"].copy()
    if len(distorted) == 0:
        print("No distorted images found for analysis")
        return
    natural_metrics = ["brisque_approx", "entropy", "edge_density", "gradient_mag", "laplacian_var"]
    available_metrics = [m for m in natural_metrics if m in distorted.columns]
    print("\nNATURAL IMAGE QUALITY METRICS SUMMARY:")
    print("-" * 40)
    if available_metrics:
        print(distorted[available_metrics].describe())
    if "distortion" in distorted.columns:
        print("\nQUALITY BY DISTORTION TYPE:")
        print("-" * 30)
        for metric in ["psnr", "ssim", "brisque_approx", "entropy"]:
            if metric in distorted.columns:
                print(f"\n{metric.upper()} by distortion:")
                distortion_stats = distorted.groupby("distortion")[metric].agg(["mean", "std", "count"])
                print(distortion_stats)
    reference_metrics = [
        "psnr",
        "ssim",
    ]
    noref_metrics = ["brisque_approx", "entropy", "edge_density", "gradient_mag", "laplacian_var"]
    available_ref = [m for m in reference_metrics if m in distorted.columns]
    available_noref = [m for m in noref_metrics if m in distorted.columns]
    if len(available_ref) > 0 and len(available_noref) > 0:
        print("\nCORRELATION BETWEEN REFERENCE AND NO-REFERENCE METRICS:")
        print("-" * 55)
        correlation_matrix = distorted[available_ref + available_noref].corr()
        for ref_metric in available_ref:
            print(f"\n{ref_metric.upper()} correlations:")
            for noref_metric in available_noref:
                if noref_metric in correlation_matrix.columns:
                    corr = correlation_matrix.loc[ref_metric, noref_metric]
                    print(f"  {noref_metric}: {corr:.3f}")
    print("\nOUTLIER DETECTION:")
    print("-" * 20)
    for metric in ["psnr", "ssim", "brisque_approx", "quality_score"]:
        if metric in distorted.columns:
            metric_data = distorted[metric].dropna()
            if len(metric_data) > 10:
                q1, q3 = metric_data.quantile([0.25, 0.75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = metric_data[(metric_data < lower_bound) | (metric_data > upper_bound)]
                print(f"{metric}: {len(outliers)} outliers ({len(outliers)/len(metric_data)*100:.1f}%)")
    from utils.vvizualization import create_natural_distortion_impact_figure
    create_natural_distortion_impact_figure(results_df, domain)
    _analyze_natural_image_characteristics(results_df, domain)
def export_natural_analysis_report(results_df, output_path=None, domain="natural"):
    """
    Export comprehensive analysis report for natural images
    Args:
        results_df: DataFrame with IQA analysis results
        output_path: Path to save report (optional)
    """
    if output_path is None:
        output_path = Path(RESULTS / domain / "analysis_logs") / "natural_iqa_report.json"
    distorted = results_df[results_df["distortion"] != "original"].copy()
    original = results_df[results_df["distortion"] == "original"].copy()
    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_images": len(results_df),
            "original_images": len(original),
            "distorted_images": len(distorted),
            "unique_base_ids": results_df["base_id"].nunique(),
            "distortion_types": results_df["distortion"].value_counts().to_dict(),
        },
        "quality_metrics": {},
        "distortion_analysis": {},
        "correlations": {},
    }
    quality_metrics = ["psnr", "ssim", "mse", "mae", "brisque_approx", "entropy", "edge_density"]
    for metric in quality_metrics:
        if metric in distorted.columns:
            metric_data = pd.to_numeric(distorted[metric], errors="coerce").dropna()
            if len(metric_data) > 0:
                report["quality_metrics"][metric] = {
                    "mean": float(metric_data.mean()),
                    "std": float(metric_data.std()),
                    "min": float(metric_data.min()),
                    "max": float(metric_data.max()),
                    "median": float(metric_data.median()),
                    "q25": float(metric_data.quantile(0.25)),
                    "q75": float(metric_data.quantile(0.75)),
                }
    for distortion_type in distorted["distortion"].unique():
        if pd.isna(distortion_type):
            continue
        subset = distorted[distorted["distortion"] == distortion_type]
        report["distortion_analysis"][distortion_type] = {
            "count": len(subset),
            "avg_psnr": (
                float(pd.to_numeric(subset["psnr"], errors="coerce").mean()) if "psnr" in subset.columns else None
            ),
            "avg_ssim": (
                float(pd.to_numeric(subset["ssim"], errors="coerce").mean()) if "ssim" in subset.columns else None
            ),
            "avg_brisque": (
                float(pd.to_numeric(subset["brisque_approx"], errors="coerce").mean())
                if "brisque_approx" in subset.columns
                else None
            ),
        }
    correlation_metrics = ["psnr", "ssim", "brisque_approx", "entropy", "edge_density", "gradient_mag"]
    available_corr_metrics = [m for m in correlation_metrics if m in distorted.columns]
    if len(available_corr_metrics) >= 2:
        corr_data = distorted[available_corr_metrics].select_dtypes(include=[np.number])
        if not corr_data.empty:
            correlation_matrix = corr_data.corr()
            report["correlations"] = correlation_matrix.round(3).to_dict()
    if CACHE:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
    print(f"\n Natural image analysis report saved to: {output_path}")
    return report
def _analyze_natural_image_characteristics(results_df, domain):
    """Analyze characteristics specific to natural images"""
    print("\n" + "=" * 50)
    print("NATURAL IMAGE CHARACTERISTICS ANALYSIS")
    print("=" * 50)
    valid_df = results_df[results_df["distortion"] != "original"].copy()
    if len(valid_df) == 0:
        print("No distorted natural images found for analysis")
        return
    if "brisque_approx" in valid_df.columns and "entropy" in valid_df.columns:
        print("\nTEXTURE COMPLEXITY ANALYSIS:")
        print("-" * 30)
        valid_df["texture_complexity"] = valid_df["brisque_approx"] / (valid_df["entropy"] + 1e-6)
        complexity_stats = valid_df.groupby("distortion")["texture_complexity"].agg(["mean", "std", "count"])
        print(complexity_stats)
        if "quality_score" in valid_df.columns:
            corr = valid_df["texture_complexity"].corr(valid_df["quality_score"])
            print(f"\nTexture complexity vs quality correlation: {corr:.4f}")
    perceptual_metrics = ["brisque_approx", "entropy", "edge_density"]
    available_perceptual = [m for m in perceptual_metrics if m in valid_df.columns]
    if available_perceptual:
        print("\nPERCEPTUAL QUALITY METRICS CORRELATION:")
        print("-" * 45)
        perceptual_corr = valid_df[available_perceptual].corr()
        print(perceptual_corr.round(3))
    if "quality_score" in valid_df.columns:
        print("\nQUALITY SCORE DISTRIBUTION BY DISTORTION:")
        print("-" * 45)
        quality_stats = valid_df.groupby("distortion")["quality_score"].agg(["mean", "std", "min", "max", "count"])
        print(quality_stats.round(4))
def analyze_feature_target_correlations(
    df,
    target_column,
    feature_columns=None,
    correlation_threshold=0.95,
    p_value_threshold=0.001,
    domain="medical",
    save_results=True,
):
    """
    Analyze correlations between IQA features and target to detect potential data leakage.
    Uses pingouin for robust correlation analysis.
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with features and target
    target_column : str
        Target column name (e.g., 'dice_score', 'quality_score')
    feature_columns : list, optional
        List of feature columns to analyze. If None, uses IQA features.
    correlation_threshold : float
        Threshold for flagging high correlations (default: 0.95)
    p_value_threshold : float
        Threshold for statistical significance (default: 0.001)
    domain : str
        Domain name for saving results
    save_results : bool
        Whether to save results to disk
    Returns:
    --------
    dict: Analysis results with correlation metrics and leakage warnings
    """
    print(f"\n{'='*80}")
    print("FEATURE-TARGET CORRELATION ANALYSIS (DATA LEAKAGE DETECTION)")
    print(f"{'='*80}")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")
    if feature_columns is None:
        iqa_features = [
            "psnr",
            "ssim",
            "mse",
            "mae",
            "snr",
            "cnr",
            "gradient_mag",
            "laplacian_var",
            "brisque_approx",
            "entropy",
            "edge_density",
        ]
        feature_columns = [f for f in iqa_features if f in df.columns]
    print(f"Analyzing correlations for {len(feature_columns)} features with target '{target_column}'")
    print(f"Features: {feature_columns}")
    print(f"Correlation threshold for leakage warning: {correlation_threshold}")
    print(f"P-value threshold for significance: {p_value_threshold}")
    analysis_columns = feature_columns + [target_column]
    valid_df = df[df.get("distortion", "original") != "original"][analysis_columns].dropna()
    if len(valid_df) < 30:
        warnings.warn(f"Only {len(valid_df)} valid samples for correlation analysis")
        return {}
    print(f"Valid samples for analysis: {len(valid_df)}")
    correlation_results = {
        "target_column": target_column,
        "domain": domain,
        "n_samples": len(valid_df),
        "feature_correlations": [],
        "high_correlations": [],
        "leakage_warnings": [],
        "feature_intercorrelations": [],
        "summary_stats": {},
        "analysis_timestamp": pd.Timestamp.now().isoformat(),
    }
    target_values = pd.to_numeric(valid_df[target_column], errors="coerce").dropna()
    print(f"\nTarget variable '{target_column}' statistics:")
    print(f"  Range: [{target_values.min():.4f}, {target_values.max():.4f}]")
    print(f"  Mean: {target_values.mean():.4f}  {target_values.std():.4f}")
    print(f"  Skewness: {stats.skew(target_values):.4f}")
    print(f"  Kurtosis: {stats.kurtosis(target_values):.4f}")
    if len(target_values) >= 5000:
        jb_stat, jb_p = stats.jarque_bera(target_values)
        print(f"  Normality (Jarque-Bera): p-value = {jb_p:.6f}")
        normality_test = "jarque_bera"
        normality_p = jb_p
    else:
        sw_stat, sw_p = stats.shapiro(target_values.values)
        print(f"  Normality (Shapiro-Wilk): p-value = {sw_p:.6f}")
        normality_test = "shapiro_wilk"
        normality_p = sw_p
    correlation_results["target_stats"] = {
        "min": float(target_values.min()),
        "max": float(target_values.max()),
        "mean": float(target_values.mean()),
        "std": float(target_values.std()),
        "skewness": float(stats.skew(target_values)),
        "kurtosis": float(stats.kurtosis(target_values)),
        "normality_test": normality_test,
        "normality_p_value": float(normality_p),
    }
    print(f"\n{'-'*80}")
    print("FEATURE-TARGET CORRELATIONS:")
    print(f"{'-'*80}")
    print(
        f"{'Feature':<15} {'Pearson r':<10} {'p-value':<10} {'Spearman ':<12} {'p-value':<10} {'MI Score':<10} {'Warning':<15}"
    )
    print("-" * 90)
    for feature in feature_columns:
        if feature not in valid_df.columns:
            continue
        feature_values = pd.to_numeric(valid_df[feature], errors="coerce")
        if feature_values.var() == 0 or feature_values.isna().sum() > len(feature_values) * 0.5:
            print(f"{feature:<15} {'N/A':<10} {'N/A':<10} {'N/A':<12} {'N/A':<10} {'N/A':<10} {'No variance':<15}")
            continue
        aligned_data = pd.DataFrame(
            {"feature": feature_values, "target": pd.to_numeric(valid_df[target_column], errors="coerce")}
        ).dropna()
        if len(aligned_data) < 10:
            print(
                f"{feature:<15} {'N/A':<10} {'N/A':<10} {'N/A':<12} {'N/A':<10} {'N/A':<10} {'Insufficient data':<15}"
            )
            continue
        feature_clean = aligned_data["feature"].values
        target_clean = aligned_data["target"].values
        try:
            pearson_result = pg.corr(feature_clean, target_clean, method="pearson")
            pearson_r = pearson_result["r"].iloc[0]
            pearson_p = pearson_result["p-val"].iloc[0]
        except Exception as e:
            print(f"Pingouin Pearson failed for {feature}: {e}")
            pearson_r, pearson_p = pearsonr(feature_clean, target_clean)
        try:
            spearman_result = pg.corr(feature_clean, target_clean, method="spearman")
            spearman_r = spearman_result["r"].iloc[0]
            spearman_p = spearman_result["p-val"].iloc[0]
        except Exception as e:
            print(f"Pingouin Spearman failed for {feature}: {e}")
            spearman_r, spearman_p = spearmanr(feature_clean, target_clean)
        try:
            mi_score = mutual_info_regression(feature_clean.reshape(-1, 1), target_clean, random_state=42)[0]
        except Exception:
            mi_score = np.nan
        warning_flags = []
        if abs(pearson_r) >= correlation_threshold:
            warning_flags.append("HIGH_PEARSON")
        if abs(spearman_r) >= correlation_threshold:
            warning_flags.append("HIGH_SPEARMAN")
        if pearson_p < p_value_threshold and abs(pearson_r) > 0.8:
            warning_flags.append("SIG_HIGH_CORR")
        warning_str = "|".join(warning_flags) if warning_flags else "OK"
        print(
            f"{feature:<15} {pearson_r:<10.4f} {pearson_p:<10.2e} {spearman_r:<12.4f} "
            f"{spearman_p:<10.2e} {mi_score:<10.4f} {warning_str:<15}"
        )
        feature_corr = {
            "feature": feature,
            "n_valid_pairs": len(aligned_data),
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
            "mutual_info": float(mi_score) if not np.isnan(mi_score) else None,
            "warnings": warning_flags,
            "feature_stats": {
                "mean": float(feature_clean.mean()),
                "std": float(feature_clean.std()),
                "min": float(feature_clean.min()),
                "max": float(feature_clean.max()),
            },
        }
        correlation_results["feature_correlations"].append(feature_corr)
        if warning_flags:
            correlation_results["high_correlations"].append(feature_corr)
        if abs(pearson_r) >= correlation_threshold or abs(spearman_r) >= correlation_threshold:
            leak_warning = {
                "feature": feature,
                "correlation_type": "pearson" if abs(pearson_r) >= correlation_threshold else "spearman",
                "correlation_value": float(pearson_r if abs(pearson_r) >= correlation_threshold else spearman_r),
                "p_value": float(pearson_p if abs(pearson_r) >= correlation_threshold else spearman_p),
                "severity": "HIGH" if abs(max(pearson_r, spearman_r, key=abs)) >= 0.98 else "MODERATE",
                "recommendation": "Consider removing feature or investigating relationship",
            }
            correlation_results["leakage_warnings"].append(leak_warning)
    print(f"\n{'-'*60}")
    print("INTER-FEATURE CORRELATION ANALYSIS:")
    print(f"{'-'*60}")
    available_features = [f for f in feature_columns if f in valid_df.columns]
    if len(available_features) >= 2:
        feature_data = valid_df[available_features].select_dtypes(include=[np.number])
        corr_matrix = feature_data.corr()
        high_intercorr_pairs = []
        for i, feat1 in enumerate(available_features):
            for j, feat2 in enumerate(available_features[i + 1 :], i + 1):
                if feat1 in corr_matrix.columns and feat2 in corr_matrix.columns:
                    corr_val = corr_matrix.loc[feat1, feat2]
                    if abs(corr_val) >= 0.8:
                        high_intercorr_pairs.append(
                            {
                                "feature1": feat1,
                                "feature2": feat2,
                                "correlation": float(corr_val),
                                "abs_correlation": float(abs(corr_val)),
                            }
                        )
        correlation_results["feature_intercorrelations"] = high_intercorr_pairs
        if high_intercorr_pairs:
            print(f"Found {len(high_intercorr_pairs)} high inter-feature correlations:")
            for pair in high_intercorr_pairs[:10]:
                print(f"  {pair['feature1']}  {pair['feature2']}: r = {pair['correlation']:.4f}")
        else:
            print("No concerning inter-feature correlations found.")
    print(f"\n{'='*60}")
    print("DATA LEAKAGE ASSESSMENT SUMMARY:")
    print(f"{'='*60}")
    n_high_corr = len(correlation_results["high_correlations"])
    n_warnings = len(correlation_results["leakage_warnings"])
    correlation_results["summary_stats"] = {
        "total_features_analyzed": len([f for f in feature_columns if f in valid_df.columns]),
        "features_with_high_correlation": n_high_corr,
        "leakage_warnings": n_warnings,
        "max_abs_correlation": float(
            max([abs(fc["pearson_r"]) for fc in correlation_results["feature_correlations"]] + [0])
        ),
        "features_above_threshold": [
            fc["feature"]
            for fc in correlation_results["feature_correlations"]
            if abs(fc["pearson_r"]) >= correlation_threshold or abs(fc["spearman_r"]) >= correlation_threshold
        ],
    }
    if n_warnings == 0:
        print(" No data leakage concerns detected!")
        print("   All feature-target correlations are below the threshold.")
    else:
        print(f"  {n_warnings} potential data leakage warnings detected!")
        print("   Features with concerning correlations:")
        for warning in correlation_results["leakage_warnings"]:
            print(f"    {warning['feature']}: {warning['correlation_type']} r = {warning['correlation_value']:.4f}")
            print(f"     Severity: {warning['severity']}, p-value: {warning['p_value']:.2e}")
            print(f"     Recommendation: {warning['recommendation']}")
    if save_results:
        create_correlation_visualizations(correlation_results, domain, target_column)
    if save_results:
        from utils.mmisc import RESULTS
        save_path = Path(RESULTS / domain / "analysis_logs" / f"feature_target_correlations_{target_column}.json")
        save_path.parent.mkdir(exist_ok=True, parents=True)
        with open(save_path, "w") as f:
            json.dump(correlation_results, f, indent=2, default=str)
        print(f"\n Correlation analysis saved to: {save_path}")
    return correlation_results
def create_correlation_visualizations(correlation_results, domain, target_column):
    """Create visualizations for correlation analysis results"""
    from utils.mmisc import RESULTS
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"Feature-Target Correlation Analysis - {domain.capitalize()} Domain ({target_column})",
        fontsize=16,
        fontweight="bold",
    )
    features = [fc["feature"] for fc in correlation_results["feature_correlations"]]
    pearson_rs = [fc["pearson_r"] for fc in correlation_results["feature_correlations"]]
    spearman_rs = [fc["spearman_r"] for fc in correlation_results["feature_correlations"]]
    pearson_ps = [fc["pearson_p"] for fc in correlation_results["feature_correlations"]]
    mutual_infos = [
        fc["mutual_info"] for fc in correlation_results["feature_correlations"] if fc["mutual_info"] is not None
    ]
    mi_features = [fc["feature"] for fc in correlation_results["feature_correlations"] if fc["mutual_info"] is not None]
    ax = axes[0, 0]
    x = np.arange(len(features))
    width = 0.35
    bars1 = ax.bar(x - width / 2, pearson_rs, width, label="Pearson r", alpha=0.8, color="skyblue")
    bars2 = ax.bar(x + width / 2, spearman_rs, width, label="Spearman ", alpha=0.8, color="lightcoral")
    threshold = 0.95
    for i, (p_r, s_r) in enumerate(zip(pearson_rs, spearman_rs)):
        if abs(p_r) >= threshold:
            bars1[i].set_color("red")
            bars1[i].set_alpha(1.0)
        if abs(s_r) >= threshold:
            bars2[i].set_color("darkred")
            bars2[i].set_alpha(1.0)
    ax.set_xlabel("Features")
    ax.set_ylabel("Correlation Coefficient")
    ax.set_title("Feature-Target Correlations")
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=threshold, color="red", linestyle="--", alpha=0.5, label=f"Threshold ({threshold})")
    ax.axhline(y=-threshold, color="red", linestyle="--", alpha=0.5)
    ax.set_ylim([-1.1, 1.1])
    ax = axes[0, 1]
    log_p_values = [-np.log10(p) if p > 0 else 10 for p in pearson_ps]
    colors = ["red" if abs(r) >= 0.8 else "blue" for r in pearson_rs]
    scatter = ax.scatter(pearson_rs, log_p_values, c=colors, alpha=0.7, s=60)
    ax.set_xlabel("Pearson Correlation (r)")
    ax.set_ylabel("-log10(p-value)")
    ax.set_title("Correlation Significance")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=-np.log10(0.001), color="red", linestyle="--", alpha=0.5, label="p=0.001")
    ax.axvline(x=0.8, color="orange", linestyle="--", alpha=0.5, label="|r|=0.8")
    ax.axvline(x=-0.8, color="orange", linestyle="--", alpha=0.5)
    ax.legend()
    for i, (r, log_p, feat) in enumerate(zip(pearson_rs, log_p_values, features)):
        if abs(r) >= 0.8 or log_p >= -np.log10(0.001):
            ax.annotate(feat, (r, log_p), xytext=(5, 5), textcoords="offset points", fontsize=8, alpha=0.8)
    ax = axes[1, 0]
    if mutual_infos and len(mutual_infos) > 0:
        bars = ax.bar(range(len(mi_features)), mutual_infos, alpha=0.7, color="green")
        ax.set_xlabel("Features")
        ax.set_ylabel("Mutual Information Score")
        ax.set_title("Mutual Information with Target")
        ax.set_xticks(range(len(mi_features)))
        ax.set_xticklabels(mi_features, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.3)
        for i, mi in enumerate(mutual_infos):
            if mi >= 0.5:
                bars[i].set_color("darkgreen")
    else:
        ax.text(
            0.5,
            0.5,
            "No Mutual Information\nscores available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_title("Mutual Information with Target")
    ax = axes[1, 1]
    if correlation_results["leakage_warnings"]:
        warning_features = [w["feature"] for w in correlation_results["leakage_warnings"]]
        warning_values = [abs(w["correlation_value"]) for w in correlation_results["leakage_warnings"]]
        warning_severities = [w["severity"] for w in correlation_results["leakage_warnings"]]
        colors = ["red" if s == "HIGH" else "orange" for s in warning_severities]
        bars = ax.barh(range(len(warning_features)), warning_values, color=colors, alpha=0.8)
        ax.set_yticks(range(len(warning_features)))
        ax.set_yticklabels(warning_features)
        ax.set_xlabel("Absolute Correlation")
        ax.set_title("Data Leakage Warnings", color="red", fontweight="bold")
        ax.axvline(x=0.95, color="red", linestyle="--", alpha=0.7, label="Threshold (0.95)")
        ax.legend()
        ax.grid(axis="x", alpha=0.3)
        for i, (val, sev) in enumerate(zip(warning_values, warning_severities)):
            ax.text(val + 0.01, i, sev, va="center", fontweight="bold", color="red" if sev == "HIGH" else "orange")
    else:
        ax.text(
            0.5,
            0.5,
            " No Data Leakage\nWarnings Detected!",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
            color="green",
            fontweight="bold",
        )
        ax.set_title("Data Leakage Assessment", color="green", fontweight="bold")
    plt.tight_layout()
    save_path = Path(RESULTS / domain / "figures" / f"correlation_analysis_{target_column}.png")
    save_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Correlation visualization saved to: {save_path}")
