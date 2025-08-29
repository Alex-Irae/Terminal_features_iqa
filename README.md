# Terminal Features in Domain-Specific IQA Calibration


[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)

> **A Signal Processing Framework for Identifying Task-Directed Metrics**

This repository implements the terminal features framework for domain-specific Image Quality Assessment (IQA), enabling radical simplification from multi-metric ensembles to single task-aligned metrics while preserving performance.

## Key Contributions

- **Terminal Features Theory**: Mathematical framework for identifying single metrics that proxy task performance via `τ(x) = f(ϕ(x)) + ε`
- **Dramatic Domain Asymmetry**: CNR captures 87% of MRI segmentation performance while failing completely on natural images (where SSIM dominates)
- **75% Computational Reduction**: From O(n) multi-metric calibration to O(1) terminal feature evaluation
- **Terminal Feature Index (TFI)**: Quantitative criterion for identifying terminal features in new domains
- **Mechanistic Validation**: Mathematical proof that CNR directly encodes tissue separability for segmentation tasks

## Abstract

Image quality assessment in specialized domains faces a critical disconnect: perceptual metrics fail to predict task utility. We introduce **terminal features**—single metrics ϕ where task performance τ satisfies τ(x) = f(ϕ(x)) + ε for monotonic f—enabling radical calibration simplification. 

**Key Finding**: CNR combined with SNR captures 87% of segmentation performance in medical imaging while contributing <1% in natural images where SSIM dominates, revealing fundamental domain asymmetry in IQA requirements.

## Quick Start

### Installation

```bash
git clone https://github.com/Alex-Irae/terminal_feayures_iqa.git
cd terminal-iqa
pip install -r requirements.txt
```


## Core Results

### Domain Asymmetry
| Domain | Best Model | R² | Terminal Feature | Importance |
|--------|------------|----|--------------------|------------|
| **Medical MRI** | Random Forest | 0.714 | **CNR** | 69.2% |
| **Natural Images** | XGBoost | 0.973 | **SSIM** | 85.8%* |

*Note: SSIM achieves only 83% retention as single metric, confirming natural image quality requires feature interactions

### Computational Performance
- **Traditional Pipeline**: O(n·m) + O(h(n)) ≈ 8 metrics × model complexity  
- **Terminal Features**: O(1·m') + O(1) ≈ 2 metrics (CNR+SNR)
- **Speedup**: 5-10× practical improvement with 87% performance retention

### Terminal Feature Index Results
```
CNR (Medical):     TFI = 0.82 ✓ (Terminal)
SSIM (Natural):    TFI = 0.52 ✗ (Non-terminal)  
SNR (Medical):     TFI = 0.31 ✗ (Supplementary)
```

## Theoretical Framework

### Terminal Feature Definition
A metric ϕ is **terminal** if:
1. **Sufficiency**: R² > 0.7 for task prediction
2. **Non-redundancy**: Adding other metrics improves R² by <0.1  
3. **Mechanistic alignment**: Direct physical relationship to task success

### Mathematical Formalization
```
E[(τ(x) - f(ϕ(x)))²] < E[(τ(x) - g(Φ(x)))²] + ξ
```
Where Φ represents any multi-metric ensemble and ξ is small tolerance.

### CNR Mechanistic Basis
For medical segmentation, CNR directly measures tissue separability:
```
CNR = |μ_fg - μ_bg| / σ_bg
Dice(I) = σ(α · CNR(I) - β) + ε
```

This isn't correlation—it's mathematical necessity for boundary detection algorithms.


## Experimental Reproduction

### Dataset Requirements
- **Medical**: BraTS2020 (300 reference images, T1/T2/FLAIR modalities)
- **Natural**: MDP Dataset (100 reference images, RGB)
- **Distortions**: 6 types × 5 severity levels = 30 conditions each

### Expected Runtime
- Medical analysis: ~45 minutes
- Natural analysis: ~20 minutes  
- Cross-domain transfer: ~30 minutes
- Figure generation: ~10 minutes

## Applications & Use Cases

### When Terminal Features Apply
✅ **Medical imaging**: Tissue contrast determines diagnostic utility  
✅ **Industrial inspection**: Defect size drives quality decisions  
✅ **Spectral imaging**: Absorption peaks define chemical identification  
✅ **Satellite monitoring**: Vegetation indices predict agricultural yield

### When They Don't Apply
❌ **Natural images**: Perceptual quality emerges from feature interactions  
❌ **Artistic evaluation**: Aesthetic judgment requires holistic assessment  
❌ **Complex scene understanding**: Multiple competing objectives

### Decision Framework
1. Does task success depend on a **single dominant physical property**?
2. Is that property **directly measurable** (not emergent)?
3. Is the measurement→performance mapping **monotonic and low-noise**?

If yes to all three, terminal features likely exist.

## Cross-Domain Transfer Analysis

### Key Limitation Discovered
Terminal features are **distribution-specific**:
- Cross-modality transfer: R² < 0.44 (T1↔T2↔FLAIR)
- Root cause: 80% variance in CNR distributions across MRI sequences
- Physics basis: Different tissue contrasts (T1: fat-water, T2: fluid-tissue, FLAIR: CSF suppression)

### Mitigation Strategies
```python
# Modality-aware normalization
cnr_normalized = (cnr - modality_mean) / modality_std

# Few-shot adaptation (5-50 samples)
adapted_model = analyzer.few_shot_adapt(source_model, target_samples)
```

## Citation

```bibtex
@inproceedings{carminot2026terminal,
  title={Terminal Features in Domain-Specific IQA Calibration: A Signal Processing Framework for Identifying Task-Directed Metrics},
  author={Carminot, A.},
  booktitle={ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2026},
  organization={IEEE}
}
```

## Contributing

We welcome contributions! Areas of particular interest:
- **New domains**: Apply terminal features to your specialized field
- **Distribution adaptation**: Improve cross-modality transfer  
- **Theoretical extensions**: Conditions for terminal feature existence
- **Computational optimizations**: Real-time implementations

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Known Issues & Limitations

1. **Cross-modality transfer**: Poor generalization across MRI sequences (R² < 0.44)
2. **Natural image ceiling**: SSIM achieves only 83% as single metric
3. **Pathology sensitivity**: Terminal features may vary with disease types
4. **Limited validation**: Current work focuses on segmentation tasks

## Support

- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: Join our discussions for theoretical questions
- **Email**: [alexandre.camrinot@outlook.com] for collaboration inquiries

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**"When task success depends on a quantifiable physical property directly measurable through a single metric, that metric becomes terminal, enabling radical simplification without sacrificing utility."**
