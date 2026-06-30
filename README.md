BENCHMARKING SYNTHETIC TABULAR DATA GENERATORS FOR STRUCTURAL COHERENCE OF BEHAVIORAL AND EDUCATIONAL DATASETS
DOI

📌 Overview
Synthetic tabular data is increasingly used to enable data sharing in privacy-sensitive behavioral and educational research contexts. However, its suitability for Structural Equation Modeling (SEM) remains to be fully understood. This study benchmarks LLM-based and diffusion-based synthetic data generators for their ability to preserve structural coherence—including measurement properties, causal path directions, and global model fit—required for valid SEM-based analysis.

Using data from PISA 2022 and TIMSS 2023 (Singapore samples), this work evaluates generators across distributional fidelity, measurement reliability, discriminant validity, structural path preservation, global model fit, and privacy risk.

🔬 Key Findings
Finding	Details
Diffusion models excel	TabDiff and TabSyn achieve strong preservation of structural relationships and global SEM fit
LLM sensitivity	LLM-based generators exhibit greater sensitivity to model size and hyperparameter configuration
Privacy trade-offs	In small-sample settings, certain diffusion models show substantial privacy leakage; well-tuned LLMs demonstrate more balanced performance
Model capacity matters	Increasing LLM model capacity (GReaT-Lrg) substantially improves measurement fidelity, rivaling diffusion-based approaches

📊 Generators Benchmarked
Category	Generators
Diffusion-based	TabDiff, TabSyn
LLM-based	GReaT, PredLLM, TapTap, REaLTabFormer, TabuLa
Classical	CTGAN

🛠️ Tech Stack
Category	Technologies
Languages	Python, R
Deep Learning	PyTorch, HuggingFace Transformers
Generation Models	TabSyn, TabDiff, GReaT, PredLLM, TapTap, REaLTabFormer, TabuLa, CTGAN
SEM Evaluation	SEMinR (PLS-SEM), lavaan (CB-SEM)
Infrastructure	Slurm, NeSI HPC, Raapoi Cluster
Documentation	LaTeX

📊 End-to-End Evaluation Pipeline
This repository implements a fully automated SEM-oriented evaluation pipeline:

Auto-preprocessing — PISA 2022 and TIMSS 2023 datasets
Generator Training — LLM fine-tuning + diffusion model training on HPC cluster
Structural Evaluation — SEM fit indices (CFI, TLI, RMSEA, SRMR), directional consistency, rank preservation, measurement reliability, discriminant validity (HTMT)
Privacy Assessment — Exact match rate, nearest-neighbor distance ratio (NNDR), membership inference risk, distance to closest record (DCR)
Automated Reporting — Tabular reports + data visualization outputs

📈 Output Visualizations

All figures and interactive tables referenced below are in the [`Outputs/`](./Outputs) folder of this repo. Static `.png` figures render directly below; interactive `.html` reports are linked (GitHub renders HTML as source code, not as a live page — see note at the bottom).

### Interactive Reports (HTML)

These are best viewed live via GitHub Pages rather than opened as raw source on GitHub. Once Pages is enabled for this repo (see note below), each link will render as an interactive table/chart:

**Summary Tables**
- [Final ranking](./Outputs/table_00_final_ranking.html)
- [Boundary adherence](./Outputs/table_boundary_adherence.html)
- [Boundary adherence — diff vs real](./Outputs/table_boundary_adherence_diff_vs_real.html)
- [Category adherence](./Outputs/table_category_adherence.html)
- [Continuous summary](./Outputs/table_continuous_summary.html)
- [Covariate correlation error](./Outputs/table_covariate_corr_error.html)
- [Covariate correlation ranges](./Outputs/table_covariate_corr_ranges.html)
- [Discriminant margin](./Outputs/table_discriminant_margin.html)
- [Distribution summary — categorical](./Outputs/table_distribution_summary_categorical.html)
- [Distribution summary — continuous](./Outputs/table_distribution_summary_continuous.html)
- [Fornell-Larcker / HTMT RMSE](./Outputs/table_fl_htmt_rmse.html)
- [Fornell-Larcker diagonal](./Outputs/table_fornell_larcker_diag.html)
- [HTMT stability](./Outputs/table_htmt_stability.html)
- [HTMT summary](./Outputs/table_htmt_summary.html)
- [Indirect effects comparison](./Outputs/table_indirect_effects_comparison.html)
- [Latent correlation stability](./Outputs/table_latent_correlation_stability.html)
- [Missingness diagnostics](./Outputs/table_missingness_diagnostics.html)
- [Range sanity vs real](./Outputs/table_range_sanity_vs_real.html)
- [Sample size](./Outputs/table_sample_size.html)


**Global Model Fit (CB-SEM)**
- [Global fit — mean](./Outputs/cbsem_global_fit_mean.html)
- [Global fit — stability](./Outputs/cbsem_global_fit_stability.html)
- [R² summary](./Outputs/cbsem_r2_summary.html)

**Measurement / Loadings**
- [Loading delta matrix](./Outputs/loading_delta_matrix.html)
- [Loading range width](./Outputs/loading_range_width.html)
- [Loading sign flip](./Outputs/loading_sign_flip.html)
- [Mean absolute loading error](./Outputs/mean_absolute_loading_error.html)

**Structural Paths**
- [Path direction rank summary](./Outputs/path_direction_rank_summary.html)
- [Path range overlap](./Outputs/path_range_overlap.html)
- [Standardized path betas](./Outputs/standardized_path_betas.html)

**Reliability**
- [Reliability — MAD vs real](./Outputs/reliability_mad_vs_real.html)
- [Reliability — stability](./Outputs/reliability_stability.html)
- [Reliability — thresholds](./Outputs/reliability_thresholds.html)

**Privacy**
- [Privacy — k-anonymity mean](./Outputs/privacy_k_anonymity_mean.html)
- [Privacy — mean](./Outputs/privacy_mean.html)

> **Note:** GitHub does not render `.html` files inline — clicking a link above shows the raw source code. To view these as live, interactive pages, enable **GitHub Pages** for this repo (Settings → Pages → Deploy from branch → `main` → `/root`), then replace the relative links above with `https://mithunthakkar8.github.io/Summer-Scholarship-Project/Outputs/<filename>.html`.

### Structural Fidelity

Heatmaps comparing latent and observed correlation deltas between real and synthetic data (CB-SEM and PLS-SEM):

![Latent correlation delta (CB-SEM)](./Outputs/heatmap_latent_delta_cb.png)
![Latent correlation delta (PLS-SEM)](./Outputs/heatmap_latent_delta_pls.png)
![Mean absolute delta heatmap](./Outputs/heatmap_mean_abs_delta.png)

Standardized structural path comparisons across all generators:

![Path coefficients — bar grid](./Outputs/paths_bar_grid.png)
![Path coefficients — interval grid](./Outputs/paths_interval_grid.png)

### Measurement Fidelity

Factor loading deltas and reliability ranges across generators:

![Loading delta heatmap](./Outputs/loading_delta_heatmap.png)
![Loading range heatmap](./Outputs/loading_range_heatmap.png)

### Covariate Stability (Interval Dot Plots)

To check whether relationships between key covariates (ESCS, age, class size, gender, grade, immigration status, mother's education, school size) and the latent outcomes (SMP, SMS, SPI) hold up under synthetic generation, interval dot plots were produced for every covariate × outcome pair (27 total). Two representative examples:

![ESCS vs SMP interval dot plot](./Outputs/interval_dot_ESCS_SMP.png)
![Age vs SPI interval dot plot](./Outputs/interval_dot_age_SPI.png)

The full set of 27 covariate plots (all combinations of ESCS, age, class size, gender, grade, immigration status, mother's education, school size × SMP/SMS/SPI) is available in [`Outputs/`](./Outputs).



📄 License
This project is licensed under the GNU General Public License v3.0. See the LICENSE file for details.

For questions, collaborations, or code access:

mithun.thakkar8@gmail.com

Associated Paper: arXiv preprint (forthcoming)

Note: This work represents pioneering results at the intersection of generative AI and Structural Equation Modeling. If you build upon this work, please maintain the citation and license terms.

📝 Citation
If you use this software or build upon these methods in your research, please cite:

Mithun Thakkar, "Benchmarking Synthetic Tabular Data Generators for Structural Coherence of Behavioral and Educational Datasets", arXiv preprint, 2026

@software{Thakkar_Benchmarking_Synthetic_Tabular_2026,
  author = {Mithun Thakkar},
  title = {Benchmarking Synthetic Tabular Data Generators for Structural Coherence of Behavioral and Educational Datasets},
  url = {https://github.com/mithunthakkar8/Summer-Scholarship-Project},
  doi = {10.5281/zenodo.19759033},
  version = {1.0.0},
  year = {2026}
}

