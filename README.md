# BENCHMARKING SYNTHETIC TABULAR DATA GENERATORS FOR STRUCTURAL COHERENCE OF BEHAVIORAL AND EDUCATIONAL DATASETS

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19759033.svg)](https://doi.org/10.5281/zenodo.19759033)

## 📌 Overview

Synthetic tabular data is increasingly used to enable data sharing in privacy-sensitive behavioral and educational research contexts. However, its suitability for **Structural Equation Modeling (SEM)** remains to be fully understood. This study benchmarks LLM-based and diffusion-based synthetic data generators for their ability to preserve **structural coherence**—including measurement properties, causal path directions, and global model fit—required for valid SEM-based analysis.

Using data from **PISA 2022** and **TIMSS 2023** (Singapore samples), this work evaluates generators across distributional fidelity, measurement reliability, discriminant validity, structural path preservation, global model fit, and privacy risk.

## 🔬 Key Findings

| Finding | Details |
|---------|---------|
| **Diffusion models excel** | TabDiff and TabSyn achieve strong preservation of structural relationships and global SEM fit |
| **LLM sensitivity** | LLM-based generators exhibit greater sensitivity to model size and hyperparameter configuration |
| **Privacy trade-offs** | In small-sample settings, certain diffusion models show substantial privacy leakage; well-tuned LLMs demonstrate more balanced performance |
| **Model capacity matters** | Increasing LLM model capacity (GReaT-Lrg) substantially improves measurement fidelity, rivaling diffusion-based approaches |

## 📊 Generators Benchmarked

| Category | Generators |
|----------|------------|
| **Diffusion-based** | TabDiff, TabSyn |
| **LLM-based** | GReaT, PredLLM, TapTap, REaLTabFormer, TabuLa |
| **Classical** | CTGAN |

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| Languages | Python, R |
| Deep Learning | PyTorch, HuggingFace Transformers |
| Generation Models | TabSyn, TabDiff, GReaT, PredLLM, TapTap, REaLTabFormer, TabuLa, CTGAN |
| SEM Evaluation | SEMinR (PLS-SEM), lavaan (CB-SEM) |
| Infrastructure | Slurm, NeSI HPC, Raapoi Cluster |
| Documentation | LaTeX |

## 📊 End-to-End Evaluation Pipeline

This repository implements a fully automated SEM-oriented evaluation pipeline:

1. **Auto-preprocessing** — PISA 2022 and TIMSS 2023 datasets
2. **Generator Training** — LLM fine-tuning + diffusion model training on HPC cluster
3. **Structural Evaluation** — SEM fit indices (CFI, TLI, RMSEA, SRMR), directional consistency, rank preservation, measurement reliability, discriminant validity (HTMT)
4. **Privacy Assessment** — Exact match rate, nearest-neighbor distance ratio (NNDR), membership inference risk, distance to closest record (DCR)
5. **Automated Reporting** — Tabular reports + data visualization outputs

## 📄 License

This project is licensed under the **GNU General Public License v3.0**.
See the [LICENSE](LICENSE) file for details.

For questions, collaborations, or code access:

mithun.thakkar8@gmail.com


Associated Paper: arXiv preprint (forthcoming)

Note: This work represents pioneering results at the intersection of generative AI and Structural Equation Modeling. If you build upon this work, please maintain the citation and license terms.

## 📝 Citation

If you use this software or build upon these methods in your research, please cite:

**Mithun Thakkar, "Benchmarking Synthetic Tabular Data Generators for Structural Coherence of Behavioral and Educational Datasets", arXiv preprint, 2026**

```bibtex
@software{Thakkar_Benchmarking_Synthetic_Tabular_2026,
  author = {Mithun Thakkar},
  title = {Benchmarking Synthetic Tabular Data Generators for Structural Coherence of Behavioral and Educational Datasets},
  url = {https://github.com/mithunthakkar8/Summer-Scholarship-Project},
  doi = {10.5281/zenodo.19759033},
  version = {1.0.0},
  year = {2026}
}







⚠️ Disclaimer

This repository contains a preliminary research report generated with the assistance of AI tools. 
The content is under active validation and should not be considered a finalized or peer-reviewed academic work.
