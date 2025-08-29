from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fft import fft2, fftshift
from scipy import signal
import os
import hashlib
import joblib
from utils.mmisc import resolve_path, RESULTS, CACHE
class SignalProcessingAnalyzer:
    """Advanced signal processing analysis for MRI images"""
    @staticmethod
    def compute_frequency_features(img):
        """Comprehensive frequency domain analysis"""
        windowed = (
            img.astype(np.float32)
            * signal.windows.hann(img.shape[0])[:, np.newaxis]
            * signal.windows.hann(img.shape[1])
        )
        f_transform = fft2(windowed)
        f_shift = fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        phase_spectrum = np.angle(f_shift)
        h, w = img.shape
        center = (h // 2, w // 2)
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
        r_max = min(center)
        n_bins = min(50, r_max // 2)
        radial_profile = []
        freqs = []
        for i in range(1, n_bins):
            r_inner = i * r_max / n_bins
            r_outer = (i + 1) * r_max / n_bins
            mask = (r >= r_inner) & (r < r_outer)
            if mask.sum() > 0:
                radial_profile.append(np.mean(magnitude_spectrum[mask]))
                freqs.append(r_inner / r_max)
        total_energy = np.sum(magnitude_spectrum**2)
        if len(radial_profile) > 0:
            spectral_centroid = np.sum(np.array(freqs) * np.array(radial_profile)) / np.sum(radial_profile)
        else:
            spectral_centroid = 0
        cumulative_energy = np.cumsum(radial_profile)
        if len(cumulative_energy) > 0:
            rolloff_threshold = 0.85 * cumulative_energy[-1]
            rolloff_idx = np.where(cumulative_energy >= rolloff_threshold)[0]
            spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
        else:
            spectral_rolloff = 0
        if len(radial_profile) > 0:
            geometric_mean = np.exp(np.mean(np.log(np.array(radial_profile) + 1e-10)))
            arithmetic_mean = np.mean(radial_profile)
            spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)
        else:
            spectral_flatness = 0
        phase_coherence = np.abs(np.mean(np.exp(1j * phase_spectrum)))
        features = {
            "dc_component": float(magnitude_spectrum[center]),
            "high_freq_energy": float(np.sum(magnitude_spectrum[center[0] + r_max // 4 :, center[1] + r_max // 4 :])),
            "low_freq_energy": float(
                np.sum(
                    magnitude_spectrum[
                        center[0] - r_max // 8 : center[0] + r_max // 8, center[1] - r_max // 8 : center[1] + r_max // 8
                    ]
                )
            ),
            "spectral_centroid": float(spectral_centroid),
            "spectral_rolloff": float(spectral_rolloff),
            "spectral_flatness": float(spectral_flatness),
            "phase_coherence": float(phase_coherence),
            "total_energy": float(total_energy),
            "energy_concentration": float(
                np.sum(magnitude_spectrum[center[0] - 5 : center[0] + 5, center[1] - 5 : center[1] + 5]) / total_energy
            ),
        }
        return features, radial_profile, freqs
    @staticmethod
    def compute_enhanced_snr_metrics(img, mask=None):
        """MRI-specific SNR analysis with multiple metrics"""
        img_float = img.astype(np.float32)
        if mask is not None and mask.sum() > 0:
            signal_region = img_float[mask > 0]
            background_region = img_float[mask == 0]
            if len(signal_region) > 0 and len(background_region) > 0:
                signal_mean = np.mean(signal_region)
                noise_std = np.std(background_region)
                traditional_snr = 20 * np.log10(signal_mean / (noise_std + 1e-10))
                cnr = abs(signal_mean - np.mean(background_region)) / (noise_std + 1e-10)
            else:
                traditional_snr = 0
                cnr = 0
        else:
            signal_mean = np.mean(img_float)
            noise_std = np.std(img_float)
            traditional_snr = 20 * np.log10(signal_mean / (noise_std + 1e-10))
            cnr = 0
        rose_snr = signal_mean / (noise_std + 1e-10) if "signal_mean" in locals() else 0
        peak_val = np.max(img_float)
        noise_power = np.var(img_float)
        peak_snr = 10 * np.log10(peak_val**2 / (noise_power + 1e-10))
        p95 = np.percentile(img_float, 95)
        p5 = np.percentile(img_float, 5)
        dynamic_range = p95 - p5
        adjusted_snr = 20 * np.log10(dynamic_range / (noise_std + 1e-10)) if "noise_std" in locals() else 0
        signal_variance = (
            np.var(signal_region) if "signal_region" in locals() and len(signal_region) > 0 else np.var(img_float)
        )
        noise_variance = (
            np.var(background_region) if "background_region" in locals() and len(background_region) > 0 else noise_power
        )
        return {
            "traditional_snr": traditional_snr,
            "rose_snr": rose_snr,
            "peak_snr": peak_snr,
            "adjusted_snr": adjusted_snr,
            "cnr": cnr,
            "dynamic_range": dynamic_range,
            "signal_variance": float(signal_variance),
            "noise_variance": float(noise_variance),
        }
    @staticmethod
    def analyze_distortion_transfer_functions(orig_img, dist_img):
        """Analyze frequency domain effects of distortions"""
        orig_fft = fft2(orig_img.astype(np.float32))
        dist_fft = fft2(dist_img.astype(np.float32))
        transfer_func = np.divide(
            dist_fft, orig_fft + 1e-10, out=np.zeros_like(dist_fft), where=(np.abs(orig_fft) > 1e-6)
        )
        transfer_magnitude = np.abs(transfer_func)
        transfer_phase = np.angle(transfer_func)
        h, w = transfer_magnitude.shape
        center = (h // 2, w // 2)
        low_freq_mask = np.zeros_like(transfer_magnitude)
        mid_freq_mask = np.zeros_like(transfer_magnitude)
        high_freq_mask = np.zeros_like(transfer_magnitude)
        r_low = min(center) // 8
        r_mid = min(center) // 4
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
        low_freq_mask[r <= r_low] = 1
        mid_freq_mask[(r > r_low) & (r <= r_mid)] = 1
        high_freq_mask[r > r_mid] = 1
        return {
            "low_freq_attenuation": float(np.mean(transfer_magnitude[low_freq_mask > 0])),
            "mid_freq_attenuation": float(np.mean(transfer_magnitude[mid_freq_mask > 0])),
            "high_freq_attenuation": float(np.mean(transfer_magnitude[high_freq_mask > 0])),
            "overall_attenuation": float(np.mean(transfer_magnitude)),
            "phase_distortion": float(np.std(transfer_phase)),
            "transfer_smoothness": float(np.mean(np.abs(np.diff(transfer_magnitude.flatten())))),
        }
def get_signal_processing_cache_key(results_df, domain):
    """Generate unique cache key for signal processing analysis"""
    content_str = f"{len(results_df)}_{results_df.columns.tolist()}"
    if "path_img" in results_df.columns:
        sample_paths = results_df["path_img"].dropna().head(10).tolist()
        content_str += f"_{sample_paths}"
    if "modality" in results_df.columns:
        content_str += f"_{results_df['modality'].value_counts().to_dict()}"
    if "distortion" in results_df.columns:
        content_str += f"_{results_df['distortion'].value_counts().to_dict()}"
    return hashlib.md5(content_str.encode()).hexdigest()
def cache_signal_processing_results(freq_df, transfer_df, cache_key, domain):
    """Cache signal processing analysis results"""
    cache_data = {"frequency_analysis": freq_df, "transfer_analysis": transfer_df, "cache_key": cache_key}
    cache_path = os.path.join(Path(RESULTS / domain / "cache"), f"signal_processing_{cache_key[:8]}.joblib")
    joblib.dump(cache_data, cache_path)
    print(f"? Signal processing results cached at {cache_path}")
def load_cached_signal_processing_results(cache_key, domain):
    """Load cached signal processing results if available"""
    cache_path = os.path.join(Path(RESULTS / domain / "cache"), f"signal_processing_{cache_key[:8]}.joblib")
    if os.path.exists(cache_path):
        try:
            cached_data = joblib.load(cache_path)
            if cached_data.get("cache_key") == cache_key:
                print("? Loaded cached signal processing results.")
                return cached_data["frequency_analysis"], cached_data["transfer_analysis"]
            else:
                print("! Cache key mismatch, will recompute.")
                return None, None
        except Exception as e:
            print(f"! Failed to load cached signal processing results: {e}")
            return None, None
    else:
        print("! No cached signal processing results found.")
        return None, None
def run_signal_processing_analysis(results_df, domain="medical"):
    """Comprehensive signal processing analysis pipeline with caching"""
    print("\n" + "=" * 60)
    print("SIGNAL PROCESSING ANALYSIS")
    print("=" * 60)
    cache_key = get_signal_processing_cache_key(results_df, domain)
    freq_df, transfer_df = load_cached_signal_processing_results(cache_key, domain)
    if freq_df is not None and transfer_df is not None and CACHE:
        print("? Using cached signal processing results.")
        create_signal_processing_figure(freq_df, transfer_df, domain)
        print_comprehensive_signal_processing_results(freq_df, transfer_df, domain)
        return freq_df, transfer_df
    print("Computing signal processing analysis from scratch...")
    analyzer = SignalProcessingAnalyzer()
    sample_df = results_df.copy()
    print("\n1. ANALYZING FREQUENCY CHARACTERISTICS...")
    print("-" * 50)
    freq_analysis = []
    processed_count = 0
    failed_count = 0
    for idx, row in sample_df.iterrows():
        try:
            img_path = resolve_path(row["path_img"])
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                failed_count += 1
                continue
            mask_path = resolve_path(row["path_mask_original"])
            mask = None
            if Path(mask_path).exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    mask = (mask > 0).astype(np.uint8)
            freq_features, _, _ = analyzer.compute_frequency_features(img)
            snr_features = analyzer.compute_enhanced_snr_metrics(img, mask)
            analysis_row = {
                "image_idx": idx,
                "modality": row["modality"],
                "distortion": row["distortion"],
                "severity": row.get("severity", 0),
                "health": row["health"],
                **freq_features,
                **snr_features,
            }
            freq_analysis.append(analysis_row)
            processed_count += 1
            if processed_count % 50 == 0:
                print(f"  Processed {processed_count} images...")
        except Exception as e:
            failed_count += 1
            print(f"  ! Failed to process image {idx}: {str(e)[:50]}...")
            continue
    freq_df = pd.DataFrame(freq_analysis)
    print(f"? Frequency analysis completed:")
    print(f"  Successfully processed: {processed_count} images")
    print(f"  Failed to process: {failed_count} images")
    print(f"  Success rate: {processed_count/(processed_count + failed_count)*100:.1f}%")
    print("\n2. ANALYZING DISTORTION TRANSFER FUNCTIONS...")
    print("-" * 50)
    transfer_analysis = []
    modalities = ["T1", "T2", "FLAIR"]
    distortions = ["blur", "noise", "motion", "rician", "bias", "ghosting"]
    transfer_processed = 0
    transfer_failed = 0
    for modality in modalities:
        print(f"  Processing {modality} modality...")
        for distortion in distortions:
            orig_subset = results_df[(results_df["modality"] == modality) & (results_df["distortion"] == "original")]
            dist_subset = results_df[(results_df["modality"] == modality) & (results_df["distortion"] == distortion)]
            if len(orig_subset) > 0 and len(dist_subset) > 0:
                try:
                    orig_path = resolve_path(orig_subset.iloc[0]["path_img"])
                    dist_path = resolve_path(dist_subset.iloc[0]["path_img"])
                    orig_img = cv2.imread(str(orig_path), cv2.IMREAD_GRAYSCALE)
                    dist_img = cv2.imread(str(dist_path), cv2.IMREAD_GRAYSCALE)
                    if orig_img is not None and dist_img is not None:
                        if orig_img.shape != dist_img.shape:
                            dist_img = cv2.resize(dist_img, (orig_img.shape[1], orig_img.shape[0]))
                        transfer_features = analyzer.analyze_distortion_transfer_functions(orig_img, dist_img)
                        transfer_row = {
                            "modality": modality,
                            "distortion": distortion,
                            "orig_samples": len(orig_subset),
                            "dist_samples": len(dist_subset),
                            **transfer_features,
                        }
                        transfer_analysis.append(transfer_row)
                        transfer_processed += 1
                        print(f"    ? {modality}-{distortion}: Transfer function computed")
                    else:
                        transfer_failed += 1
                        print(f"    ! {modality}-{distortion}: Failed to load images")
                except Exception as e:
                    transfer_failed += 1
                    print(f"    ! {modality}-{distortion}: Error - {str(e)[:40]}...")
                    continue
            else:
                print(
                    f"    - {modality}-{distortion}: Insufficient data (orig:{len(orig_subset)}, dist:{len(dist_subset)})"
                )
    transfer_df = pd.DataFrame(transfer_analysis)
    print(f"\n? Transfer function analysis completed:")
    print(f"  Successfully processed: {transfer_processed} combinations")
    print(f"  Failed to process: {transfer_failed} combinations")
    print(
        f"  Success rate: {transfer_processed/(transfer_processed + transfer_failed)*100:.1f}%"
        if (transfer_processed + transfer_failed) > 0
        else "  No combinations processed"
    )
    if CACHE:
        cache_signal_processing_results(freq_df, transfer_df, cache_key, domain)
    create_signal_processing_figure(freq_df, transfer_df, domain)
    print_comprehensive_signal_processing_results(freq_df, transfer_df, domain)
    return freq_df, transfer_df
def print_comprehensive_signal_processing_results(freq_df, transfer_df, domain):
    """Print comprehensive and correctly formatted signal processing results"""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE SIGNAL PROCESSING RESULTS")
    print("=" * 80)
    if not freq_df.empty:
        print("\n1. FREQUENCY DOMAIN ANALYSIS SUMMARY")
        print("-" * 50)
        print(f"Total samples analyzed: {len(freq_df)}")
        print(f"Modalities covered: {freq_df['modality'].nunique()}")
        print(f"Distortion types: {freq_df['distortion'].nunique()}")
        print(f"Health conditions: {freq_df['health'].nunique()}")
        print("\nFREQUENCY CHARACTERISTICS BY MODALITY:")
        print("-" * 40)
        freq_metrics = ["spectral_centroid", "spectral_rolloff", "spectral_flatness", "phase_coherence"]
        for modality in freq_df["modality"].unique():
            mod_data = freq_df[freq_df["modality"] == modality]
            print(f"\n{modality.upper()} MODALITY (n={len(mod_data)}):")
            for metric in freq_metrics:
                if metric in mod_data.columns:
                    mean_val = mod_data[metric].mean()
                    std_val = mod_data[metric].std()
                    min_val = mod_data[metric].min()
                    max_val = mod_data[metric].max()
                    print(f"  {metric.replace('_', ' ').title()}:")
                    print(f"    Mean  Std: {mean_val:.8f}  {std_val:.8f}")
                    print(f"    Range: [{min_val:.8f}, {max_val:.8f}]")
        print("\nENERGY DISTRIBUTION ANALYSIS:")
        print("-" * 30)
        energy_metrics = ["dc_component", "low_freq_energy", "high_freq_energy", "total_energy", "energy_concentration"]
        for modality in freq_df["modality"].unique():
            mod_data = freq_df[freq_df["modality"] == modality]
            print(f"\n{modality.upper()} ENERGY CHARACTERISTICS:")
            for metric in energy_metrics:
                if metric in mod_data.columns:
                    mean_val = mod_data[metric].mean()
                    std_val = mod_data[metric].std()
                    print(f"  {metric.replace('_', ' ').title()}: {mean_val:.8e}  {std_val:.8e}")
        print("\nSIGNAL-TO-NOISE RATIO ANALYSIS:")
        print("-" * 35)
        snr_metrics = ["traditional_snr", "rose_snr", "peak_snr", "adjusted_snr", "cnr"]
        for modality in freq_df["modality"].unique():
            mod_data = freq_df[freq_df["modality"] == modality]
            print(f"\n{modality.upper()} SNR CHARACTERISTICS:")
            for metric in snr_metrics:
                if metric in mod_data.columns:
                    mean_val = mod_data[metric].mean()
                    std_val = mod_data[metric].std()
                    median_val = mod_data[metric].median()
                    print(f"  {metric.replace('_', ' ').upper()}:")
                    print(f"    Mean  Std: {mean_val:.6f}  {std_val:.6f}")
                    print(f"    Median: {median_val:.6f}")
        print("\nDISTORTION IMPACT ON FREQUENCY CHARACTERISTICS:")
        print("-" * 50)
        for distortion in freq_df["distortion"].unique():
            if distortion != "original":
                dist_data = freq_df[freq_df["distortion"] == distortion]
                orig_data = freq_df[freq_df["distortion"] == "original"]
                if len(dist_data) > 0 and len(orig_data) > 0:
                    print(f"\n{distortion.upper()} DISTORTION (n={len(dist_data)}):")
                    for metric in ["spectral_centroid", "phase_coherence", "traditional_snr"]:
                        if metric in dist_data.columns and metric in orig_data.columns:
                            dist_mean = dist_data[metric].mean()
                            orig_mean = orig_data[metric].mean()
                            change_pct = ((dist_mean - orig_mean) / orig_mean * 100) if orig_mean != 0 else 0
                            print(f"  {metric.replace('_', ' ').title()}:")
                            print(f"    Original: {orig_mean:.8f}")
                            print(f"    Distorted: {dist_mean:.8f}")
                            print(f"    Change: {change_pct:+.2f}%")
    if not transfer_df.empty:
        print("\n" + "=" * 80)
        print("2. TRANSFER FUNCTION ANALYSIS SUMMARY")
        print("=" * 80)
        print(f"\nTotal transfer functions computed: {len(transfer_df)}")
        print(f"Modality-distortion combinations: {len(transfer_df)}")
        transfer_metrics = [
            "low_freq_attenuation",
            "mid_freq_attenuation",
            "high_freq_attenuation",
            "overall_attenuation",
            "phase_distortion",
            "transfer_smoothness",
        ]
        print("\nTRANSFER FUNCTION CHARACTERISTICS BY DISTORTION:")
        print("-" * 55)
        print(
            f"{'Distortion':<12} {'Low Freq':<12} {'Mid Freq':<12} {'High Freq':<12} {'Overall':<12} {'Phase Dist':<12} {'Smoothness':<12}"
        )
        print("-" * 84)
        for distortion in transfer_df["distortion"].unique():
            dist_data = transfer_df[transfer_df["distortion"] == distortion]
            row_values = [distortion[:11]]
            for metric in transfer_metrics:
                if metric in dist_data.columns:
                    mean_val = dist_data[metric].mean()
                    row_values.append(f"{mean_val:.6f}"[:11])
                else:
                    row_values.append("N/A")
            print(
                f"{row_values[0]:<12} {row_values[1]:<12} {row_values[2]:<12} {row_values[3]:<12} {row_values[4]:<12} {row_values[5]:<12} {row_values[6]:<12}"
            )
        print("\nMODALITY-SPECIFIC TRANSFER FUNCTION ANALYSIS:")
        print("-" * 45)
        for modality in transfer_df["modality"].unique():
            mod_data = transfer_df[transfer_df["modality"] == modality]
            print(f"\n{modality.upper()} MODALITY TRANSFER CHARACTERISTICS:")
            for metric in transfer_metrics:
                if metric in mod_data.columns:
                    mean_val = mod_data[metric].mean()
                    std_val = mod_data[metric].std()
                    min_val = mod_data[metric].min()
                    max_val = mod_data[metric].max()
                    print(f"  {metric.replace('_', ' ').title()}:")
                    print(f"    Mean  Std: {mean_val:.8f}  {std_val:.8f}")
                    print(f"    Range: [{min_val:.8f}, {max_val:.8f}]")
        print("\nCROSS-MODALITY TRANSFER FUNCTION COMPARISON:")
        print("-" * 45)
        for metric in ["overall_attenuation", "phase_distortion"]:
            if metric in transfer_df.columns:
                print(f"\n{metric.replace('_', ' ').upper()}:")
                pivot_table = transfer_df.pivot_table(
                    values=metric, index="distortion", columns="modality", aggfunc="mean"
                )
                print(pivot_table.to_string(float_format=lambda x: f"{x:.6f}"))
    print("\n" + "=" * 80)
    print("3. DATA QUALITY AND COVERAGE ASSESSMENT")
    print("=" * 80)
    if not freq_df.empty:
        print(f"\nFREQUENCY ANALYSIS COVERAGE:")
        print(f"  Total samples: {len(freq_df)}")
        print(f"  Modalities: {list(freq_df['modality'].unique())}")
        print(f"  Distortions: {list(freq_df['distortion'].unique())}")
        missing_freq = freq_df.isnull().sum()
        if missing_freq.any():
            print(f"\nMissing values in frequency analysis:")
            for col, count in missing_freq[missing_freq > 0].items():
                print(f"  {col}: {count} ({count/len(freq_df)*100:.1f}%)")
        else:
            print(f"? No missing values in frequency analysis")
    if not transfer_df.empty:
        print(f"\nTRANSFER FUNCTION ANALYSIS COVERAGE:")
        print(f"  Total combinations: {len(transfer_df)}")
        print(f"  Modalities: {list(transfer_df['modality'].unique())}")
        print(f"  Distortions: {list(transfer_df['distortion'].unique())}")
        missing_transfer = transfer_df.isnull().sum()
        if missing_transfer.any():
            print(f"\nMissing values in transfer function analysis:")
            for col, count in missing_transfer[missing_transfer > 0].items():
                print(f"  {col}: {count} ({count/len(transfer_df)*100:.1f}%)")
        else:
            print(f"? No missing values in transfer function analysis")
    print("\n" + "=" * 80)
    print("SIGNAL PROCESSING ANALYSIS COMPLETED!")
    print("=" * 80)
def create_signal_processing_figure(freq_df, transfer_df, domain):
    """Create comprehensive signal processing visualization"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Signal Processing Analysis - MRI Modalities", fontsize=16, fontweight="bold")
    if not freq_df.empty:
        ax = axes[0, 0]
        sns.boxplot(data=freq_df, x="modality", y="spectral_centroid", ax=ax)
        ax.set_title("Spectral Centroid Distribution")
        ax.set_ylabel("Normalized Frequency")
        ax = axes[0, 1]
        sns.boxplot(data=freq_df, x="modality", y="traditional_snr", ax=ax)
        ax.set_title("Signal-to-Noise Ratio")
        ax.set_ylabel("SNR (dB)")
        ax = axes[0, 2]
        sns.boxplot(data=freq_df, x="modality", y="phase_coherence", ax=ax)
        ax.set_title("Phase Coherence")
        ax.set_ylabel("Coherence")
    if not transfer_df.empty:
        ax = axes[1, 0]
        pivot_data = transfer_df.pivot_table(
            values="overall_attenuation", index="distortion", columns="modality", fill_value=0
        )
        sns.heatmap(pivot_data, annot=True, fmt=".3f", cmap="viridis", ax=ax)
        ax.set_title("Overall Frequency Attenuation")
        ax = axes[1, 1]
        pivot_high = transfer_df.pivot_table(
            values="high_freq_attenuation", index="distortion", columns="modality", fill_value=0
        )
        sns.heatmap(pivot_high, annot=True, fmt=".3f", cmap="plasma", ax=ax)
        ax.set_title("High Frequency Attenuation")
        ax = axes[1, 2]
        sns.barplot(data=transfer_df, x="distortion", y="phase_distortion", ax=ax)
        ax.set_title("Phase Distortion by Artifact Type")
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(RESULTS / domain / "figures/signal_processing_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()
