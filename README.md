# BENCHMARKING SYNTHETIC TABULAR DATA GENERATORS FOR STRUCTURAL COHERENCE OF BEHAVIORAL AND EDUCATIONAL DATASETS

Overview:
This repository benchmarks synthetic tabular data generators—specifically **LLM-based** (fine-tuned) 
and **diffusion-based** models—for their ability to preserve **structural coherence** in behavioral 
and educational datasets (TIMSS and PISA). Unlike traditional ML-focused evaluations, 
this work assesses generators through the lens of **Structural Equation Modeling (SEM)** .

## 🔬 Key Contributions

This project achieves **pioneering results** at the intersection of generative AI and structural equation modeling:

| Contribution | Details |
|--------------|---------|
| **Expanded Scope** | Extended from LLM-only to include diffusion-based generators (TabSyn, TabDiff), achieving pioneering results in SEM-oriented synthetic data generation |
| **Novel Application** | First successful mapping of synthetic data generators (built for ML) to SEM applications |
| **New Evaluation Metrics** | Proposed and implemented structural coherence metrics beyond standard RMSE, including **directional consistency** and **rank preservation** |
| **Code-Based Analytics** | Replaced SPSS-dependent workflows with reproducible, production-style Python/R pipeline |

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| Languages | Python, R |
| Deep Learning | PyTorch, HuggingFace Transformers |
| Generation Models | TabSyn, TabDiff, Fine-tuned LLMs |
| Evaluation | SEM frameworks, Custom structural coherence metrics |
| Infrastructure | Slurm, Git Bash |
| Documentation | LaTeX |

## 📊 End-to-End Analytics Pipeline

This repository implements a fully automated evaluation pipeline:

1. **Auto-preprocessing** — TIMSS and PISA datasets
2. **Generator Training** — LLM fine-tuning + diffusion model training
3. **Structural Evaluation** — SEM fit indices, directional consistency, rank preservation
4. **Automated Reporting** — Tabular reports + data visualization outputs

## 📄 License

This project is licensed under the **GNU General Public License v3.0**.
See the [LICENSE](LICENSE) file for details.

## 📝 Citation

If you use this software in your research, please cite:

**Mithun Thakkar, "Benchmarking Synthetic Tabular Data Generators for Structural Coherence of Behavioral and Educational Datasets", forthcoming on arXiv, 2026**
