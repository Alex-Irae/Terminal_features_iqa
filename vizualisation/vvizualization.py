from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, pearsonr
import os, pickle
from mmisc import pick_first_present
from ccalibration import evaluate_real_segmentation
from mmisc import RESULTS, CACHE
def get_all_metrics():
    """Centralized function to return all relevant IQA metrics in a standardized format."""
    metrics = [
        "psnr",
        "ssim",
        "mse",
        "mae",
        "quality_score",
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
    return metrics
def main_visualization_pipeline(results_df, domain="medical"):
    """Phase 2 driver (stats + figures) - now takes results_df as argument."""
    create_publication_figures(results_df, domain=domain)
    create_statistical_table(results_df, domain=domain)
    analyze_segmentation_correlation(results_df, domain=domain)
    return results_df
def create_publication_figures(results_df, domain):
    """Wrapper to generate Figure 1 & 2; Figure 4 & 5 are produced in Phase 3."""
    print("\n" + "=" * 50)
    print("CREATING PUBLICATION FIGURES")
    print("=" * 50)
    create_modality_bias_figure(results_df, domain)
    create_distortion_impact_figure(results_df, domain)
    print(" Publication figures created in 'Results_/figures/' directory")
def create_statistical_table(results_df, domain):
    """Create comprehensive statistical analysis table and ANOVA file + LaTeX table."""
    print("\n" + "=" * 50)
    print("STATISTICAL ANALYSIS TABLE")
    print("=" * 50)
    metrics = get_all_metrics()
    modalities = ["T1", "T2", "FLAIR"]
    stats_data = []
    for metric in metrics:
        for modality in modalities:
            data = results_df[results_df["modality"] == modality][metric].dropna()
            if len(data) > 0:
                stats_data.append(
                    {
                        "Metric": metric.upper(),
                        "Modality": modality,
                        "Mean": np.mean(data),
                        "Std": np.std(data),
                        "Min": np.min(data),
                        "Max": np.max(data),
                        "Median": np.median(data),
                        "Count": len(data),
                    }
                )
    stats_df = pd.DataFrame(stats_data)
    anova_rows = []
    for metric in metrics:
        groups = [results_df[results_df["modality"] == m][metric].dropna() for m in modalities]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) >= 2:
            try:
                f_stat, p_val = f_oneway(*groups)
                all_vals = np.concatenate(groups)
                grand_mean = np.mean(all_vals)
                ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
                ss_total = np.sum((all_vals - grand_mean) ** 2)
                eta_sq = ss_between / ss_total if ss_total > 0 else 0.0
                anova_rows.append(
                    {
                        "Metric": metric.upper(),
                        "F_statistic": f_stat,
                        "p_value": p_val,
                        "eta_squared": eta_sq,
                        "significance": (
                            "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                        ),
                    }
                )
            except Exception as e:
                print(f"ANOVA error for {metric}: {e}")
    if CACHE:
        anova_df = pd.DataFrame(anova_rows)
        Path(RESULTS / domain / "analysis_logs").mkdir(exist_ok=True)
        stats_df.to_csv(RESULTS / domain / "analysis_logs/descriptive_statistics.csv", index=False)
        anova_df.to_csv(RESULTS / domain / "analysis_logs/anova_results.csv", index=False)
        create_paper_table(stats_df, anova_df, domain)
        print(" Statistical tables saved")
    return stats_df, anova_df
def analyze_segmentation_correlation(results_df, cache_file=None, domain="medical"):
    print("\n" + "=" * 50)
    print("SEGMENTATION CORRELATION ANALYSIS")
    print("=" * 50)
    if cache_file is None:
        cache_file = Path(RESULTS / domain / "cache" / "segmentation_correlation.pkl")
        cache_file.parent.mkdir(parents=True, exist_ok=True)
    if os.path.exists(cache_file) and CACHE:
        print(f" Using cached segmentation correlation analysis results: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    if results_df["dice_score"].isna().all():
        print("No segmentation metrics found; running real segmentation evaluation now...")
        results_df = evaluate_real_segmentation(results_df)
    df = results_df.copy()
    print(f" Total rows before filtering: {len(df)}")
    mask_col = pick_first_present(
        df, ["path_mask_original", "mask_path", "path_mask", "gt_path", "path_gt"], required=False
    )
    if mask_col:
        n_missing = df[mask_col].isna().sum()
        print(f" - Missing masks: {n_missing} rows")
    iqa_cols = get_all_metrics()
    for col in iqa_cols:
        if col in df:
            n_nan = df[col].isna().sum()
            n_inf = np.isinf(pd.to_numeric(df[col], errors="coerce")).sum()
            print(f" - IQA {col:12s}: {n_nan:4d} NaNs, {n_inf:4d} Infs")
    if "dice_score" in df:
        n_nan = df["dice_score"].isna().sum()
        n_inf = np.isinf(pd.to_numeric(df["dice_score"], errors="coerce")).sum()
        print(f" - Segmentation dice_score: {n_nan:4d} NaNs, {n_inf:4d} Infs")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(
        subset=[
            "dice_score",
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
        ]
    )
    print(f" Rows after dropping invalid: {len(df)}")
    valid_data = df.copy()
    valid_data = valid_data.replace([np.inf, -np.inf], np.nan)
    valid_data = valid_data.dropna(subset=["dice_score", "psnr"])
    if len(valid_data) < 10:
        print("! Insufficient valid data for correlation analysis.")
        return {}
    print(f"? Analyzing correlations with {len(valid_data)} valid samples")
    print("\n" + "=" * 60)
    print("FIGURE 3: IQA-SEGMENTATION CORRELATION RESULTS")
    print("=" * 60)
    metrics = [m for m in get_all_metrics() if m in valid_data.columns]
    pearsons = {}
    min_pairs = 10
    print("Pearson correlations between IQA metrics and dice_score scores:")
    print("-" * 60)
    for m in metrics:
        mask = np.isfinite(valid_data[m]) & np.isfinite(valid_data["dice_score"])
        if mask.sum() >= min_pairs:
            r, p = pearsonr(valid_data.loc[mask, m], valid_data.loc[mask, "dice_score"])
            pearsons[m] = (r, p)
            sig_stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  {m:15s}: r={r:+.4f}, p={p:.6f} ({sig_stars}), n={mask.sum()}")
    if pearsons:
        print(f"\nStrongest correlations (absolute value):")
        sorted_correlations = sorted(pearsons.items(), key=lambda x: abs(x[1][0]), reverse=True)
        for i, (metric, (r, p)) in enumerate(sorted_correlations[:5]):
            sig_stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  {i+1}. {metric}: r={r:+.4f} ({sig_stars})")
        if CACHE:
            with open(cache_file, "wb") as f:
                pickle.dump(pearsons, f)
            print(f" Cached segmentation correlation analysis to {cache_file}")
        preferred = metrics
        available = [m for m in preferred if m in pearsons]
        if available:
            key_metric = available[0]
            plt.figure(figsize=(6, 5))
            mask = np.isfinite(valid_data[key_metric]) & np.isfinite(valid_data["dice_score"])
            x_vals = valid_data.loc[mask, key_metric]
            y_vals = valid_data.loc[mask, "dice_score"]
            print(f"\nFigure 3 scatter plot statistics ({key_metric} vs dice_score):")
            print(f"  Sample size: {len(x_vals)}")
            print(f"  {key_metric} range: {x_vals.min():.4f} to {x_vals.max():.4f}")
            print(f"  dice_score range: {y_vals.min():.4f} to {y_vals.max():.4f}")
            print(f"  {key_metric} mean+/-std: {x_vals.mean():.4f}+/-{x_vals.std():.4f}")
            print(f"  dice_score mean+/-std: {y_vals.mean():.4f}+/-{y_vals.std():.4f}")
            plt.scatter(x_vals, y_vals, alpha=0.4)
            plt.xlabel(key_metric.upper())
            plt.ylabel("dice_score (real)")
            plt.title(f"IQA-Segmentation Correlation (real)\nr={pearsons[key_metric][0]:.4f}")
            plt.grid(True, alpha=0.3)
            z = np.polyfit(x_vals, y_vals, 1)
            p = np.poly1d(z)
            plt.plot(x_vals, p(x_vals), "r--", alpha=0.8)
            Path(RESULTS / domain / "figures").mkdir(exist_ok=True)
            plt.savefig(RESULTS / domain / "figures/figure3_correlation_analysis.png", dpi=300, bbox_inches="tight")
            plt.close()
            print(" Figure 3: Correlation analysis (real) saved")
    else:
        print("! No valid correlations found")
    return pearsons
def create_modality_bias_figure(results_df, domain):
    key_metrics = get_all_metrics()
    key_metrics.remove("spearman")
    key_metrics.remove("quality_score")
    fig, axes = plt.subplots(4, 4, figsize=(15, 10))
    fig.suptitle("IQA Metric Distributions Across MRI Modalities", fontsize=16, fontweight="bold")
    axes = axes.flatten()
    print("\n" + "=" * 60)
    print("FIGURE 1: MODALITY BIAS ANALYSIS RESULTS")
    print("=" * 60)
    anova_results = []
    modality_stats = {}
    for idx, metric in enumerate(key_metrics):
        ax = axes[idx]
        metric_data = results_df[results_df[metric].notna()]
        if len(metric_data) == 0:
            print(f"\n{metric.upper()}: No valid data")
            continue
        print(f"\n{metric.upper()} Statistics:")
        print("-" * 30)
        modality_stats[metric] = {}
        for modality in ["T1", "T2", "FLAIR"]:
            mod_data = metric_data[metric_data["modality"] == modality][metric]
            if len(mod_data) > 0:
                stats = {
                    "mean": mod_data.mean(),
                    "std": mod_data.std(),
                    "count": len(mod_data),
                    "median": mod_data.median(),
                    "min": mod_data.min(),
                    "max": mod_data.max(),
                }
                modality_stats[metric][modality] = stats
                print(
                    f"  {modality}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
                    f"n={stats['count']}, median={stats['median']:.4f}"
                )
        sns.violinplot(data=results_df, x="modality", y=metric, ax=ax)
        ax.set_title(metric.upper(), fontweight="bold")
        ax.set_xlabel("MRI Modality")
        ax.set_ylabel(f"{metric.upper()} Value")
        modalities = ["T1", "T2", "FLAIR"]
        groups = [results_df[results_df["modality"] == m][metric].dropna() for m in modalities]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) >= 2:
            try:
                f_stat, p_val = f_oneway(*groups)
                star = "***" if p_val < 1e-3 else "**" if p_val < 1e-2 else "*" if p_val < 5e-2 else "ns"
                all_vals = np.concatenate(groups)
                grand_mean = np.mean(all_vals)
                ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
                ss_total = np.sum((all_vals - grand_mean) ** 2)
                eta_squared = ss_between / ss_total if ss_total > 0 else 0.0
                anova_results.append(
                    {
                        "metric": metric,
                        "f_stat": f_stat,
                        "p_val": p_val,
                        "eta_squared": eta_squared,
                        "significance": star,
                    }
                )
                print(f"  ANOVA: F={f_stat:.4f}, p={p_val:.6f}, ^2={eta_squared:.4f} ({star})")
                ax.text(
                    0.02,
                    0.98,
                    f"ANOVA p={p_val:.3f} {star}",
                    transform=ax.transAxes,
                    va="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                )
            except Exception as e:
                print(f"  ANOVA failed: {e}")
    significant_metrics = [r for r in anova_results if r["significance"] != "ns"]
    print(f"\n=== MODALITY BIAS SUMMARY ===")
    print(f"Total metrics analyzed: {len(anova_results)}")
    print(f"Metrics showing significant modality differences: {len(significant_metrics)}")
    if significant_metrics:
        print("\nSignificant findings (ranked by effect size):")
        for r in sorted(significant_metrics, key=lambda x: x["eta_squared"], reverse=True):
            print(f"  - {r['metric']}: ^2={r['eta_squared']:.4f}, p={r['p_val']:.6f} {r['significance']}")
    plt.tight_layout()
    plt.savefig(RESULTS / domain / "figures/figure1_modality_bias.png", dpi=300, bbox_inches="tight")
    plt.savefig(RESULTS / domain / "figures/figure1_modality_bias.pdf", bbox_inches="tight")
    plt.close()
    print(" Figure 1: Modality bias analysis saved")
def create_distortion_impact_figure(results_df, domain):
    distorted_df = results_df[results_df["distortion"] != "original"].copy()
    metrics = get_all_metrics()
    distorted_df = distorted_df.replace([np.inf, -np.inf], np.nan)
    print("\n" + "=" * 60)
    print("FIGURE 2: DISTORTION IMPACT ANALYSIS RESULTS")
    print("=" * 60)
    for m in metrics:
        if m in distorted_df.columns:
            distorted_df[m] = pd.to_numeric(distorted_df[m], errors="coerce")
    if not any(c in distorted_df.columns for c in metrics):
        print("! No metric columns found for distortion impact.")
        return
    distortion_impact = distorted_df.groupby("distortion")[metrics].agg(["mean", "std", "count"]).round(4)
    print("\nFIGURE 2A: DISTORTION TYPE IMPACT")
    print("-" * 40)
    distortion_means = distorted_df.groupby("distortion")[metrics].mean(numeric_only=True)
    if distortion_means.empty or distortion_means[metrics].isna().all().all():
        print("! No data available to analyze distortion impact.")
        return
    print("Mean IQA values by distortion type:")
    for distortion in distortion_means.index:
        print(f"\n{distortion}:")
        for metric in metrics:
            if metric in distortion_means.columns:
                mean_val = distortion_means.loc[distortion, metric]
                if pd.notna(mean_val):
                    metric_data = distorted_df[distorted_df["distortion"] == distortion][metric].dropna()
                    std_val = metric_data.std()
                    count_val = len(metric_data)
                    print(f"  {metric}: {mean_val:.4f} +/- {std_val:.4f} (n={count_val})")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    heat = distortion_means.T.astype(float)
    sns.heatmap(
        heat.values,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        ax=ax1,
        xticklabels=list(heat.columns),
        yticklabels=list(heat.index),
    )
    ax1.set_title("Mean IQA Metrics by Distortion Type", fontweight="bold")
    ax1.set_xlabel("Distortion Type")
    ax1.set_ylabel("IQA Metrics")
    sev_col = "severity"
    severity_analysis_done = False
    if sev_col in distorted_df.columns:
        if distorted_df[sev_col].dtype == object:
            sev_numeric = pd.to_numeric(distorted_df[sev_col], errors="coerce")
            if sev_numeric.isna().all():
                sev_numeric = distorted_df[sev_col].astype(str).str.extract(r"(\d+)")[0].astype(float)
            distorted_df[sev_col + "_num"] = sev_numeric
            sev_col = sev_col + "_num"
        if sev_col in distorted_df.columns and distorted_df[sev_col].notna().any():
            print(f"\nFIGURE 2B: SEVERITY IMPACT ANALYSIS")
            print("-" * 40)
            severity_impact = (
                distorted_df.dropna(subset=[sev_col])
                .groupby(sev_col)[metrics]
                .agg(["mean", "std", "count"])
                .sort_index()
            )
            print("IQA metrics vs severity levels:")
            severity_means = (
                distorted_df.dropna(subset=[sev_col]).groupby(sev_col)[metrics].mean(numeric_only=True).sort_index()
            )
            for severity_level in severity_means.index:
                print(f"\nSeverity {severity_level}:")
                for metric in metrics:
                    if metric in severity_means.columns:
                        mean_val = severity_means.loc[severity_level, metric]
                        if pd.notna(mean_val):
                            sev_data = distorted_df[distorted_df[sev_col] == severity_level][metric].dropna()
                            std_val = sev_data.std()
                            count_val = len(sev_data)
                            print(f"  {metric}: {mean_val:.4f} +/- {std_val:.4f} (n={count_val})")
            print(f"\nCorrelations between severity and IQA metrics:")
            for metric in metrics:
                if metric in distorted_df.columns:
                    corr_data = distorted_df[[sev_col, metric]].dropna()
                    if len(corr_data) > 3:
                        correlation = corr_data[sev_col].corr(corr_data[metric])
                        if pd.notna(correlation):
                            print(f"  {metric} vs severity: r={correlation:.4f}")
            for m in metrics:
                if m == "laplacian_var":
                    continue
                if m in severity_means.columns:
                    ax2.plot(severity_means.index, severity_means[m], marker="o", label=m.upper(), linewidth=2)
            ax2.set_title("IQA Metrics vs Distortion Severity", fontweight="bold")
            ax2.set_xlabel("Distortion Severity Level")
            ax2.set_ylabel("IQA Metric Value")
            ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            ax2.grid(True, alpha=0.3)
            severity_analysis_done = True
    if not severity_analysis_done:
        ax2.text(0.5, 0.5, "No numeric severity data available", ha="center", va="center", transform=ax2.transAxes)
        ax2.axis("off")
        print("\nFIGURE 2B: No numeric severity data available for analysis")
    plt.tight_layout()
    fig.savefig(RESULTS / domain / "figures/figure2a_matrix_and_combined_severity.png", dpi=300, bbox_inches="tight")
    fig.savefig(RESULTS / domain / "figures/figure2a_matrix_and_combined_severity.pdf", bbox_inches="tight")
    plt.close(fig)
    print(" Figure 2a: Matrix + combined severity saved")
    if severity_analysis_done and sev_col in distorted_df.columns and distorted_df[sev_col].notna().any():
        severity_means = (
            distorted_df.dropna(subset=[sev_col]).groupby(sev_col)[metrics].mean(numeric_only=True).sort_index()
        )
        metrics = get_all_metrics()
        for i in ["quality_score", "rmse", "spearman", "iou", "dice_score"] :
            metrics.remove(i)
        n = len(metrics)
        cols = 6
        rows = (n + cols - 1) // cols
        fig2b, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), sharex=True)
        axes = axes.flatten()
        print(f"\nFIGURE 2B: Individual metric severity trends:")
        for idx, m in enumerate(metrics):
            axm = axes[idx]
            if m in severity_means.columns:
                y_values = severity_means[m].values
                x_values = severity_means.index.values
                mask = np.isfinite(x_values) & np.isfinite(y_values)
                x_values = x_values[mask]
                y_values = y_values[mask]
                if len(np.unique(x_values)) > 1 and len(x_values) >= 2:
                    try:
                        slope = np.polyfit(x_values, y_values, 1)[0]
                    except np.linalg.LinAlgError:
                        print(f"! Polyfit failed for {m}, skipping slope calculation")
                        slope = np.nan
                    r_squared = np.corrcoef(x_values, y_values)[0, 1] ** 2 if len(x_values) > 1 else 0
                    print(f"  {m}: slope={slope:.6f}, R^2={r_squared:.4f}")
                    axm.plot(x_values, y_values, marker="o", linewidth=2)
                    axm.set_title(f"{m.upper()} (slope={slope:.4f})", fontweight="bold")
                else:
                    slope = np.nan
                    axm.set_title(m.upper(), fontweight="bold")
            axm.set_xlabel("Severity")
            axm.set_ylabel("Value")
            axm.grid(True, alpha=0.3)
        for j in range(len(metrics), len(axes)):
            fig2b.delaxes(axes[j])
        fig2b.tight_layout()
        fig2b.savefig(RESULTS / domain / "figures/figure2b_severity_by_metric.png", dpi=300, bbox_inches="tight")
        fig2b.savefig(RESULTS / domain / "figures/figure2b_severity_by_metric.pdf", bbox_inches="tight")
        plt.close(fig2b)
        print(" Figure 2b: Individual severity trends per metric saved")
    else:
        print("! Figure 2b skipped: no numeric severity data available")
    print(f" Figure 2 analysis completed")
def create_paper_table(stats_df, anova_df, domain):
    """Create LaTeX-formatted table for paper"""
    metrics = ["PSNR", "SSIM", "SNR", "CNR"]
    modalities = ["T1", "T2", "FLAIR"]
    table_lines = []
    table_lines.append("\\begin{table}[htbp]")
    table_lines.append("\\centering")
    table_lines.append("\\caption{IQA Metric Statistics Across MRI Modalities}")
    table_lines.append("\\label{tab:modality_stats}")
    table_lines.append("\\begin{tabular}{l|ccc|c}")
    table_lines.append("\\hline")
    table_lines.append("Metric & T1 & T2 & FLAIR & ANOVA p-value \\\\")
    table_lines.append("\\hline")
    for metric in metrics:
        line_parts = [metric]
        for modality in modalities:
            subset = stats_df[(stats_df["Metric"] == metric) & (stats_df["Modality"] == modality)]
            if len(subset) > 0:
                mean_val = subset["Mean"].iloc[0]
                std_val = subset["Std"].iloc[0]
                line_parts.append(f"{mean_val:.3f} +- {std_val:.3f}")
            else:
                line_parts.append("N/A")
        anova_subset = anova_df[anova_df["Metric"] == metric]
        if len(anova_subset) > 0:
            p_val = anova_subset["p_value"].iloc[0]
            significance = anova_subset["significance"].iloc[0]
            if p_val < 0.001:
                line_parts.append(f"< 0.001{significance}")
            else:
                line_parts.append(f"{p_val:.3f}{significance}")
        else:
            line_parts.append("N/A")
        table_lines.append(" & ".join(line_parts) + " \\\\")
    table_lines.append("\\hline")
    table_lines.append("\\end{tabular}")
    table_lines.append("\\end{table}")
    with open(RESULTS / domain / "analysis_logs/paper_table.tex", "w") as f:
        f.write("\n".join(table_lines))
    print("? LaTeX table saved to Results_/analysis_logs/paper_table.tex")
def create_natural_distortion_impact_figure(results_df, domain="natural"):
    """Create distortion impact analysis figure for natural images (similar to Figure 2b for medical)"""
    print("\n" + "=" * 60)
    print("NATURAL IMAGES: DISTORTION IMPACT ANALYSIS")
    print("=" * 60)
    distorted_df = results_df[results_df["distortion"] != "original"].copy()
    metrics = ["psnr", "ssim", "mse", "mae", "snr", "cnr", "gradient_mag", "laplacian_var",
              "brisque_approx", "entropy", "edge_density", "quality_score"]
    distorted_df = distorted_df.replace([np.inf, -np.inf], np.nan)
    for m in metrics:
        if m in distorted_df.columns:
            distorted_df[m] = pd.to_numeric(distorted_df[m], errors="coerce")
    severity_col = None
    for col in ["severity", "level", "distortion_level", "intensity"]:
        if col in distorted_df.columns:
            severity_col = col
            break
    if severity_col is None:
        print("! No severity/level column found. Creating basic distortion type analysis.")
        _create_basic_natural_distortion_plot(distorted_df, metrics, domain)
        return
    if distorted_df[severity_col].dtype == object:
        severity_numeric = pd.to_numeric(distorted_df[severity_col], errors="coerce")
        if severity_numeric.isna().all():
            severity_numeric = distorted_df[severity_col].astype(str).str.extract(r"(\d+\.?\d*)")[0].astype(float)
        distorted_df[severity_col + "_num"] = severity_numeric
        severity_col = severity_col + "_num"
    if severity_col not in distorted_df.columns or distorted_df[severity_col].isna().all():
        print("! Could not convert severity to numeric. Creating basic analysis.")
        _create_basic_natural_distortion_plot(distorted_df, metrics, domain)
        return
    print(f"NATURAL IMAGES: SEVERITY LEVEL ANALYSIS using column '{severity_col}'")
    print("-" * 50)
    severity_means = distorted_df.dropna(subset=[severity_col]).groupby(severity_col)[metrics].mean().sort_index()
    if severity_means.empty:
        print("! No valid severity data for analysis")
        return
    print("IQA metrics vs severity levels (Natural Images):")
    for severity_level in severity_means.index:
        print(f"\nSeverity {severity_level}:")
        for metric in metrics:
            if metric in severity_means.columns:
                mean_val = severity_means.loc[severity_level, metric]
                if pd.notna(mean_val):
                    sev_data = distorted_df[distorted_df[severity_col] == severity_level][metric].dropna()
                    std_val = sev_data.std() if len(sev_data) > 1 else 0
                    count_val = len(sev_data)
                    print(f"  {metric}: {mean_val:.4f} +/- {std_val:.4f} (n={count_val})")
    print(f"\nCorrelations between severity and IQA metrics (Natural Images):")
    for metric in metrics:
        if metric in distorted_df.columns:
            corr_data = distorted_df[[severity_col, metric]].dropna()
            if len(corr_data) > 3:
                correlation = corr_data[severity_col].corr(corr_data[metric])
                if pd.notna(correlation):
                    direction = "" if correlation > 0 else ""
                    strength = "strong" if abs(correlation) > 0.7 else "moderate" if abs(correlation) > 0.4 else "weak"
                    print(f"  {metric} vs severity: r={correlation:.4f} ({strength} {direction})")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    distortion_means = distorted_df.groupby("distortion")[metrics].mean()
    if not distortion_means.empty:
        key_metrics = ["psnr", "ssim", "brisque_approx", "entropy", "quality_score"]
        available_key_metrics = [m for m in key_metrics if m in distortion_means.columns]
        if available_key_metrics:
            heat_data = distortion_means[available_key_metrics].T
            sns.heatmap(
                heat_data.values,
                annot=True,
                fmt=".3f",
                cmap="viridis",
                ax=ax1,
                xticklabels=[x.title() for x in heat_data.columns],
                yticklabels=[x.upper() for x in heat_data.index]
            )
            ax1.set_title("Mean IQA Metrics by Distortion Type\n(Natural Images)", fontweight="bold")
            ax1.set_xlabel("Distortion Type")
            ax1.set_ylabel("IQA Metrics")
    key_metrics_for_plot = ["psnr", "ssim", "quality_score", "brisque_approx", "entropy"]
    available_metrics = [m for m in key_metrics_for_plot if m in severity_means.columns]
    for m in available_metrics:
        y_values = severity_means[m].values
        x_values = severity_means.index.values
        mask = np.isfinite(x_values) & np.isfinite(y_values)
        if mask.sum() > 1:
            ax2.plot(x_values[mask], y_values[mask], marker="o", label=m.upper(), linewidth=2, markersize=6)
    ax2.set_title("IQA Metrics vs Distortion Severity\n(Natural Images)", fontweight="bold")
    ax2.set_xlabel("Distortion Severity Level")
    ax2.set_ylabel("IQA Metric Value")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(Path(RESULTS / domain / "figures/natural_distortion_combined.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(" Natural images combined distortion analysis saved")
    plot_metrics = [m for m in metrics if m in severity_means.columns]
    plot_metrics = [m for m in plot_metrics if m not in ["quality_score"]]
    n = len(plot_metrics)
    if n == 0:
        print("! No valid metrics for individual plotting")
        return
    cols = 4
    rows = (n + cols - 1) // cols
    fig_individual, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    print(f"\nNATURAL IMAGES: Individual metric severity trends:")
    for idx, m in enumerate(plot_metrics):
        ax = axes[idx]
        y_values = severity_means[m].values
        x_values = severity_means.index.values
        mask = np.isfinite(x_values) & np.isfinite(y_values)
        x_clean = x_values[mask]
        y_clean = y_values[mask]
        if len(x_clean) > 1:
            try:
                slope = np.polyfit(x_clean, y_clean, 1)[0]
                r_squared = np.corrcoef(x_clean, y_clean)[0, 1] ** 2 if len(x_clean) > 1 else 0
                ax.plot(x_clean, y_clean, marker="o", linewidth=2, markersize=6, color='tab:blue')
                trend_line = np.poly1d(np.polyfit(x_clean, y_clean, 1))
                ax.plot(x_clean, trend_line(x_clean), "--", alpha=0.7, color='red')
                ax.set_title(f"{m.upper()}\n(slope={slope:.4f}, R^2={r_squared:.3f})", fontweight="bold")
                print(f"  {m}: slope={slope:.6f}, R^2={r_squared:.4f}")
            except (np.linalg.LinAlgError, ValueError) as e:
                ax.plot(x_clean, y_clean, marker="o", linewidth=2, markersize=6)
                ax.set_title(f"{m.upper()}", fontweight="bold")
                print(f"  {m}: Could not compute trend ({e})")
        else:
            ax.set_title(f"{m.upper()}\n(insufficient data)", fontweight="bold")
            print(f"  {m}: Insufficient data for trend analysis")
        ax.set_xlabel("Severity Level")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
    for j in range(len(plot_metrics), len(axes)):
        fig_individual.delaxes(axes[j])
    fig_individual.suptitle("Natural Images: IQA Metrics vs Distortion Severity", fontsize=14, fontweight="bold")
    fig_individual.tight_layout()
    fig_individual.savefig(Path(RESULTS / domain / "figures/natural_distortion_individual.png"), dpi=300, bbox_inches="tight")
    plt.close(fig_individual)
    print(" Natural images individual metric severity plots saved")
def _create_basic_natural_distortion_plot(distorted_df, metrics, domain):
    """Create basic distortion analysis when no severity levels are available"""
    print("Creating basic distortion type analysis for natural images...")
    distortion_means = distorted_df.groupby("distortion")[metrics].mean()
    if distortion_means.empty:
        print("! No distortion data available for plotting")
        return
    key_metrics = ["psnr", "ssim", "brisque_approx", "entropy", "quality_score"]
    available_metrics = [m for m in key_metrics if m in distortion_means.columns]
    if not available_metrics:
        print("! No key metrics available for basic plotting")
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(distortion_means.index))
    width = 0.15
    for i, metric in enumerate(available_metrics):
        offset = (i - len(available_metrics)/2) * width
        values = distortion_means[metric].values
        ax.bar(x + offset, values, width, label=metric.upper(), alpha=0.8)
    ax.set_xlabel("Distortion Type")
    ax.set_ylabel("IQA Metric Value")
    ax.set_title("Natural Images: IQA Metrics by Distortion Type")
    ax.set_xticks(x)
    ax.set_xticklabels([d.title() for d in distortion_means.index], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(RESULTS / domain / "figures/natural_distortion_basic.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(" Basic natural images distortion analysis saved")
