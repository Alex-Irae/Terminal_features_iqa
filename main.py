import os
from utils.mmisc import utf8_
utf8_(os.path.dirname(os.path.abspath(__file__)))
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from datetime import datetime
from iqa_analysis import (
    run_iqa_analysis,
    export_natural_analysis_report,
    validate_natural_image_setup,
    natural_image_quality_analysis,
)
from ssignal import SignalProcessingAnalyzer
from aablation import run_comprehensive_ablation_study
from utils.ssummary import generate_comparative_summary
from utils.vvizualization import main_visualization_pipeline
from ccalibration import main_calibration_pipeline
from ssignal import run_signal_processing_analysis
from ccross_modality import run_cross_modality_generalization_test
from ccomplexity import run_computational_complexity_analysis
from ddomain_comparison import run_domain_comparison, prepare_mpd_dataset
from utils.mmisc import RESULTS, CACHE
from Terminal_features_iqa.terminal_features import *
class OutputCapture:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()
output_file = f"terminal_output_{RESULTS}.txt"
sys.stdout = OutputCapture(output_file)
print(f"Terminal output now being saved to: {output_file}")
warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")
MDP_P = Path("PHD/mdp_csv.csv")
MRI_P = Path("PHD/mri_csv.csv")
print("CACHE ON : ", CACHE)
print("MRI PATH : ", MRI_P)
print("MDP PATH : ", MDP_P)
def setup_environment(domain):
    """Setup the analysis environment and check required files."""
    dirs = [
        Path(RESULTS / domain),
        Path(RESULTS / domain / "figures"),
        Path(RESULTS / domain / "models"),
        Path(RESULTS / domain / "results"),
        Path(RESULTS / domain / "domain_comparison"),
        Path(RESULTS / domain / "analysis_logs"),
        Path(RESULTS / domain / "cache"),
    ]
    for d in dirs:
        Path(d).mkdir(
            exist_ok=True,
            parents=True,
        )
        print(f" Created/verified directory: {d}")
    required_files = [MRI_P, MDP_P]
    missing = [p for p in required_files if not Path(p).exists()]
    if missing:
        print("\n MISSING REQUIRED FILES:")
        for m in missing:
            print(f"   - {m}")
        if MDP_P in missing:
            print("! MPD dataset missing - will run medical-only analysis")
        return True
    print("\n All required files found")
    return True
def main():
    """FIXED VERSION with proper error handling"""
    print("\n" + "=" * 80)
    print("PHASE 0: DATASET INITIALIZATION")
    print("=" * 80)
    try:
        medical_df = pd.read_csv(MRI_P)
        print(f" Medical dataset: {len(medical_df)} samples")
    except Exception as e:
        print(f" Failed to load medical dataset: {e}")
        return
    mpd_df = None
    if Path(MDP_P).exists():
        try:
            mpd_df = prepare_mpd_dataset(MDP_P)
            print(f" Natural images dataset: {len(mpd_df)} samples")
        except Exception as e:
            print(f"  Failed to load MPD dataset: {e}")
            print("! Proceeding with medical-only analysis")
            mpd_df = None
    else:
        print("! MPD dataset not found - proceeding with medical-only analysis")
    print("\n" + "=" * 80)
    print("PHASE 1: MEDICAL DOMAIN ANALYSIS (MRI)")
    print("=" * 80)
    try:
        medical_results = run_medical_pipeline(medical_df)
        print(" Medical pipeline completed successfully")
    except Exception as e:
        print(f" Medical pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return
    natural_results = None
    if mpd_df is not None:
        print("\n" + "=" * 80)
        print("PHASE 2: NATURAL DOMAIN ANALYSIS (MPD)")
        print("=" * 80)
        try:
            natural_results = run_natural_pipeline(mpd_df)
            if natural_results is not None:
                print(" Natural pipeline completed successfully")
            else:
                print("  Natural pipeline validation failed")
        except Exception as e:
            print(f" Natural pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            natural_results = None
    if natural_results is not None:
        print("\n" + "=" * 80)
        print("PHASE 3: CROSS-DOMAIN COMPARATIVE ANALYSIS")
        print("=" * 80)
        try:
            run_comp(medical_results, natural_results, medical_df, mpd_df)
            print(" Domain comparison completed successfully")
        except Exception as e:
            print(f" Domain comparison failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n! Skipping domain comparison - natural pipeline not available")
    res = run_paper_critical_analysis(medical_df, mpd_df, medical_results, natural_results)
    print("\n" + "=" * 80)
    print("ANALYSIS PIPELINE COMPLETED")
    print("=" * 80)
def run_medical_pipeline(medical_df):
    """Run complete pipeline for medical images - FIXED VERSION"""
    domain = "medical"
    if not setup_environment(domain):
        print(" Environment setup failed - exiting")
        return
    results = {}
    medical_df = run_iqa_analysis(
        medical_df,
        domain=domain,
        csv_path=MRI_P,
        force_cache=False,
        epochs=65,
        skip_train=False,
        sample_size=None,
    )
    results["iqa"] = medical_df.copy()
    from Terminal_features_iqa.hp_tuning.hyperparameter_tuning import find_best_medical_params
    find_best_medical_params(medical_df)
    medical_df = main_visualization_pipeline(medical_df, domain=domain)
    results["viz"] = medical_df.copy()
    calibration_results = main_calibration_pipeline(
        medical_df,
        methods=["random_forest", "xgboost", "linear", "lightweight_cnn"],
        domain=domain,
    )
    results["calibration"] = calibration_results
    results["signal"] = run_signal_processing_analysis(medical_df, domain=domain)
    results["ablation"] = run_comprehensive_ablation_study(medical_df, domain=domain)
    results["complexity"] = run_computational_complexity_analysis(domain=domain)
    results["cross_modality"] = run_cross_modality_generalization_test(medical_df, domain=domain)
    return results
def run_natural_pipeline(mdp_df):
    """Run complete pipeline for natural images - FIXED VERSION"""
    domain = "natural"
    if not setup_environment(domain):
        print(" Environment setup failed - exiting")
        return
    results = {}
    print("Validating natural image dataset structure...")
    validation_results = validate_natural_image_setup(mdp_df, domain)
    if not validation_results["valid"]:
        print(" Dataset validation failed! Aborting natural pipeline.")
        return None
    mdp_df = run_iqa_analysis(
        mdp_df,
        domain=domain,
        csv_path=MDP_P,
        force_cache=False,
        epochs=0,
        skip_train=True,
        sample_size=None,
    )
    natural_image_quality_analysis(mdp_df, domain)
    export_natural_analysis_report(mdp_df, domain=domain)
    results["iqa"] = mdp_df.copy()
    calibration_results = main_calibration_pipeline(
        mdp_df, methods=["random_forest", "xgboost", "linear", "lightweight_cnn"], domain=domain
    )
    results["calibration"] = calibration_results
    return results
def run_comp(medical_results, natural_results, medical_df, mpd_df):
    domain = "comp"
    if not setup_environment(domain):
        print(" Environment setup failed - exiting")
        return
    comparison_results = run_domain_comparison(medical_results, natural_results, medical_df, mpd_df, domain)
    generate_comparative_summary(medical_results, natural_results, comparison_results, domain)
    return
def run_paper_critical_analysis(medical_df, natural_df, medical_results, natural_results):
    """
    Run all critical analyses for the paper in one go.
    """
    print("\n" + "=" * 80)
    print("RUNNING CRITICAL PAPER ANALYSES")
    print("=" * 80)
    run_full_analysis(medical_df, natural_df, medical_results, natural_results)
    return
    print("\n" + "=" * 80)
    print("PAPER CRITICAL ANALYSES COMPLETE!")
    print("All results saved to Results_/comp/paper_critical_results.json")
    print("=" * 80)
if __name__ == "__main__":
    main()
