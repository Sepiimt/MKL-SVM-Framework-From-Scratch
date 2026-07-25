# MKL-SVM Framework — From Scratch

**A ground-up implementation of Principal Component Analysis, Multiple Kernel Learning, and Support Vector Machines, unified under a single alternating-optimization pipeline with seven interchangeable kernel-weight optimization strategies.**

| **Author**            | Sepanta Metanat                                                |
| --------------------- | -------------------------------------------------------------- |
| **Version**           | 1.0.0                                                          |
| **License**           | GNU General Public License v3.0                                |
| **Core dependencies** | NumPy, pandas (scikit-learn used only for parity benchmarking) |
| **First / last edit** | 2026-06-30 / 2026-07-24                                        |

---

## Table of Contents

1. [Overview](#1-overview)
2. [Mathematical Formulation](#2-mathematical-formulation)
3. [Package Architecture](#3-package-architecture)
4. [Repository Layout](#4-repository-layout)
5. [Installation](#5-installation)
6. [Quickstart](#6-quickstart)
7. [Kernel Library](#7-kernel-library)
8. [Beta Optimization Methods](#8-beta-optimization-methods)
9. [Data Pipeline and Datasets](#9-data-pipeline-and-datasets)
10. [Empirical Results](#10-empirical-results)
11. [Model Persistence](#11-model-persistence)
12. [Known Limitations](#12-known-limitations)
13. [Known Issues](#13-known-issues)
14. [Notebooks and Further Documentation](#14-notebooks-and-further-documentation)
15. [License](#15-license)

---
## 1. Overview

This repository implements a **Multiple Kernel Learning Support Vector Machine (MKL-SVM)** entirely from first principles: no scikit-learn, LIBSVM, or third-party optimization library is used anywhere in the learning algorithm itself. scikit-learn appears only as an external reference point in the evaluation harness (`metrics.py`), where a precomputed-kernel `SVC` is trained on the same combined Gram matrix produced by this framework, to verify that the custom dual solver converges to the same decision boundary as a mature, independently engineered implementation.

The project combines four components that are each implemented independently and then composed:
	
- A **kernel library** (`src/mklsvm/kernels/kernels.py`) providing six kernel families with consistent centering and diagonal-normalization semantics for both training and out-of-sample evaluation.
- A **Sequential Minimal Optimization (SMO) solver** (`src/mklsvm/smo/smo.py`) implementing Platt's decomposition method with LIBSVM-style working-set alternation, warm-starting, and an O(1)-amortized non-bound index cache.
- A **Multiple Kernel Learning orchestrator** (`src/mklsvm/mkl/mkl.py`), which alternates between solving the SVM dual for a fixed kernel mixture and updating the mixture weights (**β**) via one of seven pluggable optimization strategies (`src/mklsvm/mkl/optimizers.py`).
- A **from-scratch Principal Component Analysis** module (`src/mklsvm/pca/pca.py`), used as an upstream dimensionality-reduction stage in every experiment in this repository.

These are exposed through a single public-facing class, `SVM`, which hides the internal MKL/SMO/kernel composition behind a `select_kernels()` (_optional usage_) / `set_beta_optimizer()` (_optional usage_) / `fit()` / `predict()` interface modeled loosely on the scikit-learn estimator convention.

Beyond the implementation itself, the repository includes a substantial empirical study — `docs/bom_and_mst_documentation.md` — comparing all seven β-optimization strategies against one another on a common stress-test benchmark, along with four additional per-dataset validation reports cross-checking the custom solver against scikit-learn. This `README.md` summarizes both.

**Note:** the full technical documentation should be consulted for derivations, convergence traces, and the complete limitations discussion: "[BOM Documentation](https://github.com/Sepiimt/MKL-SVM-Framework-From-Scratch/blob/main/docs/bom_and_mst_documentation.md)"

---
## 2. Mathematical Formulation

Given a set of $m$ base kernels $\{K_1, \dots, K_m\}$, the framework learns a convex combination

$$K(x_i, x_j) = \sum_{k=1}^{m} \beta_k K_k(x_i, x_j), \qquad \beta_k \ge 0, \quad \sum_{k=1}^{m} \beta_k = 1$$

so that $\beta$ lies on the probability simplex. The combined kernel is then passed to a standard soft-margin SVM dual:

$$\max_{\alpha} \; \sum_i \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j K(x_i, x_j) \quad \text{s.t.} \quad 0 \le \alpha_i \le C, \;\; \sum_i \alpha_i y_i = 0$$

Because $\alpha$ and $\beta$ depend on each other, the two are not solved jointly in one step. The implementation follows the standard MKL alternating scheme:

```text
Initialize β uniformly over the selected kernels
repeat
    1. Train the SVM dual (via SMO) on the current combined kernel  Σ βₖ Kₖ
    2. Compute each kernel's contribution to the current SVM solution
    3. Update β using the selected optimization method
until ‖β_new − β_old‖∞ < tolerance, or max_bo_iter is reached
Run one final SMO fit on the converged β
```

The final re-fit is necessary because the last β update changes the combined kernel; without it, the retained $\alpha$/$b$ would correspond to an already-stale kernel mixture. One optimizer (**Kernel Target Alignment**) departs from this loop structure: it computes β analytically from label alignment alone, without ever inspecting SMO's dual variables, and therefore always converges in a single outer (MKL) iteration.

Every base kernel is, by default, **diagonally normalized** before combination,

$$K'(x_i, x_j) = \frac{K(x_i, x_j)}{\sqrt{K(x_i, x_i)\, K(x_j, x_j)}}$$

which fixes every kernel's diagonal — and hence its trace — to a common scale, so that no kernel can dominate the mixture purely by virtue of an unbounded raw numeric range (an unnormalized Linear or Polynomial kernel against a bounded RBF or Laplacian kernel, for instance). **Centering** in feature space, $K_c = HKH$ with $H = I - \tfrac{1}{n}\mathbf{1}\mathbf{1}^\top$, is supported independently but is disabled in the headline stress-test benchmark (see §12).

---

## 3. Package Architecture

```txt
                              ┌───────────────────────┐
                              │          `SVM`         │  <- public facade
                              │  select_kernels()       │
                              │  set_kernels_arguments() │
                              │  set_beta_optimizer()     │
                              │  fit() / predict()         │
                              │  save_model() / load_model()│
                              └────────────┬──────────────┘
                                           │ delegates to
                              ┌────────────▼──────────────┐
                              │           MKL               │  <- alternating optimizer
                              │  select_kernels()             │
                              │  fit()  (outer β loop)          │
                              │  predict()                        │
                              └───┬────────────────┬─────────────┘
                                  │                │
                     ┌────────────▼───┐  ┌─────────▼──────────┐
                     │     Kernels     │  │        SMO          │  <- inner dual QP solver
                     │  (6 subclasses)  │  │  fit() (LIBSVM-style │
                     │  fit_K()          │  │  working-set alternation,│
                     │  compute_cross_K() │  │  warm-starting)            │
                     └───────────────────┘    └─────────────────────────────┘
                                  ▲
                     ┌────────────┴───────────────┐
                     │   optimizers.py (7 classes)  │  <- β-update strategies
                     │   BaseBetaOptimizer            │
                     └─────────────────────────────────┘

               ┌────────────┐            ┌───────────────┐           ┌───────────┐
               │     PCA      │          │    metrics      │         │   utils     │
               │ (standalone,  │         │ mkl_svm_evaluation│       │ train_test_split │
               │  SVD-based)     │       │ sklearn_semi_mkl_  │      │ timer               │
               └────────────────┘        │ evaluation          │     └───────────────────┘
                                         └───────────────────┘
```

**Design notes on cross-cutting mechanisms:**

- **Kernel registration is metaclass-driven.** `Kernels` uses a custom metaclass (`IterativeMeta`) so that `for kernel in Kernels: ...` iterates its own subclasses in definition order. `MKL._beta_and_kernels_instances_matrix_creator` relies on this to build `kernels_instances_matrix` in lockstep with the fixed six-element `selected_kernels` boolean mask (`[Linear, Polynomial, RBF, Laplacian, Rational Quadratic, Sigmoid]`), rather than hardcoding a kernel list.
- **Warm-starting.** `SMO.fit()` accepts `initial_alpha` / `initial_b`. Inside `MKL`'s outer β loop, each successive SMO call is warm-started from the previous call's solution (`cache_initial_alpha`, `cache_initial_b`), since the combined kernel typically shifts only slightly between consecutive β updates — this materially reduces the number of SMO iterations required on later outer iterations.
- **LIBSVM-style working-set alternation.** `SMO.fit()` alternates between scanning the full dataset (`examine_all=True`) and scanning only the non-bound support vectors ($0 < \alpha_i < C$), falling back to a full scan whenever the restricted scan makes no progress. Non-bound indices are cached and only recomputed when the mask actually changes (a dirty-flag optimization), avoiding an $O(N)$ `np.where` scan on every candidate index.
- **Support-vector pruning after convergence.** After `MKL.fit()` converges, only support-vector rows/columns are retained (`_extract_support_vectors`); dense $N \times N$ training kernel matrices and centering/normalization statistics are pruned to the SV subset, and the full training set is discarded from memory.

---

## 4. Repository Layout

```
MKL-SVM-Framework-From-Scratch-1.0.0/
├── src/mklsvm/
│   ├── kernels/          # Kernels base class + 6 kernel implementations
│   ├── smo/               # Sequential Minimal Optimization dual solver
│   ├── mkl/                 # MKL orchestrator (mkl.py) + 7 β-optimizers (optimizers.py)
│   ├── svm/                   # Public-facing SVM facade
│   ├── pca/                     # From-scratch PCA (SVD-based)
│   ├── metrics/                   # Evaluation harness + scikit-learn parity checks
│   └── utils/                       # train_test_split, timer
├── data/
│   ├── raw/                # Original CSV datasets (5)
│   ├── processed/            # Cleaned NumPy arrays (post data_cleaning_and_engineering.ipynb)
│   └── splits/                  # PCA-projected train/test splits, ready for SVM.fit()
├── artifacts/
│   ├── pca/                # Saved PCA models (.npz: eigenvectors, eigenvalues, scaler stats)
│   ├── svm/                  # Saved SVM/MKL/SMO/kernel model layers (.npy), one subtree per dataset
│   └── predictions/            # Saved raw test-set predictions (.npy)
├── notebooks/                    # 6 Jupyter notebooks, one per pipeline stage (§14)
├── results/
│   ├── reports/                # 4 per-dataset Markdown validation reports (custom vs. scikit-learn)
│   └── benchmarks/                # Raw metrics .txt backing the 7-optimizer stress test
├── docs/
│   └── bom_and_mst_documentation.md   # Full technical paper on different beta optimization methods — this README summarizes it
├── requirements.txt
├── LICENSE                       # GNU GPLv3
└── .gitignore
```

Under `artifacts/svm/telescope/` and `results/benchmarks/telescope/`, results are further split by β-optimizer (`ed/`, `fista/`, `fw/`, `hybrid/`, `kta/`, `l2mkl/`, `simplemkl/`), one subfolder per method compared in the stress test.

---

## 5. Installation

No `pyproject.toml` or `setup.py` is included in this snapshot; the package is currently consumed by adding `src/` to `sys.path` at runtime — the pattern used throughout every notebook in this repository:

```python
import sys
from pathlib import Path

src_path = (Path.cwd() / ".." / "src").resolve()   # adjust relative to your working directory
sys.path.insert(0, str(src_path))

from mklsvm.svm import SVM
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

```text
numpy==2.5.1
pandas==3.0.3
scikit-learn==1.9.0
```

`scikit-learn` is used exclusively by `metrics.py` for the precomputed-kernel parity harness (`SVC(kernel='precomputed')`) and by nothing in the core `kernels` / `smo` / `mkl` / `svm` / `pca` modules.

---

## 6. Quickstart

```python
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str((Path.cwd() / "src").resolve()))

from mklsvm.svm import SVM
from mklsvm.utils import train_test_split

# --- Load data (labels must ultimately be encoded as +1 / -1) ---
X = np.load("data/processed/cancer/processed_cancer_x_array.npy", allow_pickle=True)
Y = np.load("data/processed/cancer/processed_cancer_y_array.npy")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=True, random_state=42)

# --- Configure the model ---
model = SVM()
model.set_beta_optimizer("Entropic Descent")   # optional; this is also the default
model.select_kernels()                          # optional; defaults to
                                                  # [Linear, Polynomial, RBF, Laplacian, Rational Quadratic]
                                                  # with Sigmoid disabled

# --- Train ---
model.fit(
    X_train, Y_train,
    max_bo_iter=20, bo_tolerance=1e-3,   # outer β-optimization budget
    smo_c=1, max_smo_iter=10**5, smo_tolerance=1e-5,  # inner SMO budget
    detailed_info=True,                   # prints per-iteration β vectors and timings
)

# --- Predict & persist ---
predictions = model.predict(X_test, strict_binary_result=True)
model.save_model("artifacts/svm/cancer/")

# --- Reload later ---
reloaded = SVM()
reloaded.load_model("artifacts/svm/cancer/")
```

`SVM.select_kernels()` and `SVM.set_kernels_arguments()` accept explicit overrides when called with arguments; consult their docstrings (`help(SVM.select_kernels)`) for the boolean-mask and hyperparameter-vector conventions. `select_kernels()` accepts an optional length-6 boolean array corresponding, in order, to `[Linear, Polynomial, RBF, Laplacian, Rational Quadratic, Sigmoid]`.

---

## 7. Kernel Library

| Kernel | Formula | Hyperparameters | Notes |
|---|---|---|---|
| Linear | $K(x_i,x_j) = x_i^\top x_j$ | — | |
| Polynomial | $K(x_i,x_j) = (x_i^\top x_j + c)^d$ | $c$, $d$ | |
| RBF | $K(x_i,x_j) = \exp(-\gamma \lVert x_i-x_j\rVert_2^2)$ | $\gamma$ | |
| Laplacian | $K(x_i,x_j) = \exp(-\gamma \lVert x_i-x_j\rVert_1)$ | $\gamma$ | Uses $L_1$ distance; consequently *not* rotation-invariant, which interacts nontrivially with a PCA preprocessing step (§9.2 of the full documentation). |
| Rational Quadratic | $K(x_i,x_j) = \left(1+\dfrac{\lVert x_i-x_j\rVert_2^2}{2\alpha \ell^2}\right)^{-\alpha}$ | $\alpha$, $\ell$ | Equivalent to an infinite mixture of RBF kernels over a Gamma distribution of length scales. |
| Sigmoid | $K(x_i,x_j) = \tanh(\gamma\, x_i^\top x_j + c)$ | $\gamma$, $c$ | Not guaranteed positive semi-definite for arbitrary $\gamma,c$; `fit_K` checks the minimum eigenvalue and raises `ValueError` if it falls below $-10^{-5}$. Disabled by default in `SVM`. |

Every kernel supports independent **centering** and **diagonal normalization** (§2), applied consistently to the training Gram matrix and to out-of-sample cross-kernel matrices via cached training-set statistics (`post_process_fit` / `post_process_cross`), so that test-time kernel evaluations remain exactly consistent with the transform learned at training time.

**Default hyperparameter heuristics** (`SVM._arguments_default_values`) scale with the data rather than using fixed constants: RBF and Laplacian $\gamma$ are set to $1/(n_{\text{features}} \cdot \mathrm{Var}(X))$, the Rational Quadratic length scale grows as $\sqrt{n_{\text{features}} \cdot \mathrm{Var}(X)}$, and the Sigmoid $\gamma$ is derived from $\mathbb{E}[X^2]$ (i.e. variance plus squared mean) rather than variance alone, so that pre-tanh magnitudes stay in a well-conditioned range regardless of dimensionality — a fixed-scale default would otherwise either collapse off-diagonal similarity as dimensionality grows or saturate the Sigmoid kernel.

---

## 8. Beta Optimization Methods

Seven interchangeable strategies (`src/mklsvm/mkl/optimizers.py`) update the kernel-weight vector β given the current SMO dual solution:

| Method key                         | Also known as                                                   | Update mechanism                                                                                                                                                          | Typical kernel distribution                                                                 |
| ---------------------------------- | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `SimpleMKL`                        | Active-set / reduced-gradient / steepest descent on the simplex | Reduced-gradient descent relative to the active kernel with the smallest gradient, with a KKT boundary lock preventing re-entry of zeroed kernels below a step-size limit | Sparse-adaptive: one dominant kernel with several smaller supporting kernels                |
| `L2-MKL`                           | Analytic L2 / Cauchy–Schwarz update                             | Closed-form: normalizes each kernel's raw alignment score by the Euclidean norm of all scores; no step size                                                               | Dense, balanced across all active kernels                                                   |
| `Entropic Descent` **_(default)_** | Mirror descent / multiplicative weights update                  | Multiplicative exponentiated-gradient step on the simplex; β can never become negative by construction                                                                    | Adaptive, intermediate between sparse and dense                                             |
| `Hybrid L1/L2`                     | Proximal gradient                                               | Gradient step (with an L1/L2 mixing ratio $\rho$) followed by Euclidean projection onto the simplex                                                                       | Frequently collapses to a near-single-kernel solution in practice                           |
| `KTA`                              | Kernel Target Alignment                                         | Closed-form alignment of each kernel's Gram matrix with $yy^\top$; ignores the SMO dual variables entirely; converges in exactly one outer iteration                      | Dense; by far the fastest method                                                            |
| `Frank-Wolfe`                      | Conditional gradient                                            | Steps toward the simplex vertex indicated by the current gradient, with a local quadratic line search for the step size                                                   | Vertex solutions — a single active kernel                                                   |
| `FISTA`                            | Accelerated projected gradient (Nesterov)                       | Momentum-accelerated proximal gradient with adaptive restart, followed by simplex projection                                                                              | Sparse; in this benchmark converges to the same fixed point as Frank-Wolfe and Hybrid L1/L2 |

All seven share the `BaseBetaOptimizer` interface (`update(support_vector_weights, Y, kernels_instances_matrix, current_beta_array)`), so adding an eighth strategy requires implementing only that one method and registering it in `MKL.available_bom`.

---

## 9. Data Pipeline and Datasets

The end-to-end pipeline, reflected in the `notebooks/` directory, proceeds:

`data_cleaning_and_engineering` → `pca_model_train` → `svm_model_train` → `model_evaluation`,

with a parallel branch, `bom_stress_test` → `stress_test_results`, dedicated specifically to the seven-optimizer comparison described in §10.2.

| Dataset                 |     Samples | Raw features (post-cleaning) | PCA reduction (component range, variance retained) | Headline β-optimizer used           | Markdown Report/Documentation ⁽¹⁾                                                                                                      |
| ----------------------- | ----------: | ---------------------------: | -------------------------------------------------- | ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Breast Cancer Wisconsin |       `568` |                         `30` | `0–29` → `0–10` (96.15%)                           | Entropic Descent                    | [Breast Cancer Report](https://github.com/Sepiimt/MKL-SVM-Framework-From-Scratch/blob/main/results/reports/cancer_dataset_report.md)   |
| Ionosphere              |       `350` |                         `32` | `0–31` →` 0–22` (95.86%)                           | Hybrid L1/L2                        | [Banknote Report](https://github.com/Sepiimt/MKL-SVM-Framework-From-Scratch/blob/main/results/reports/banknote_dataset_report.md)      |
| Pima Indians Diabetes   |       `767` |                          `8` | `0–7` → `0–7` (100.0%, no reduction)               | KTA                                 | [Pima Diabetes Report](https://github.com/Sepiimt/MKL-SVM-Framework-From-Scratch/blob/main/results/reports/diabetes_dataset_report.md) |
| Banknote Authentication |     `1,371` |                          `4` | `0–3` → `0–3` (100.0%, no reduction)               | FISTA                               | [Banknote Report](https://github.com/Sepiimt/MKL-SVM-Framework-From-Scratch/blob/main/results/reports/ionosphere_dataset_report.md)    |
| MAGIC Gamma Telescope   | `4,000` ⁽²⁾ |                         `10` | `0–9 `→ `0–6 `(96.1%)                              | All seven (comparative stress test) | [BOM Documentation](https://github.com/Sepiimt/MKL-SVM-Framework-From-Scratch/blob/main/docs/bom_and_mst_documentation.md)             |
⁽¹⁾ See dedicated report/documentation for detailed observation, hypothesis, and conclusions.
⁽²⁾ A stratified subsample of the full 19,020-row dataset, drawn with a fixed random seed (`42`) to keep the seven-way optimizer comparison computationally tractable and computationally light (in comparison to full kernel matrices with the size of ≈ **2.9 GB** per kernel and ≈**17 GB** at maximum) while preserving class balance.

`PCA.scaler()` z-score standardizes each feature using training-set statistics only (test data reuses the stored means/stds via `fit=False`), and `PCA.fit()` / `PCA.transform()` follow a standard SVD-based decomposition, retaining components by explicit count rather than a fixed variance threshold — the variance-retained figures above are consequently reported, not targeted.

---

## 10. Empirical Results

### 10.1 Per-Dataset Validation Against scikit-learn

For each of the four smaller datasets, `results/reports/*.md` documents a full training run of the custom MKL-SVM against a scikit-learn `SVC(kernel='precomputed')` trained on the identical, final combined kernel matrix — isolating the comparison to the dual-solver implementation itself rather than kernel construction.

| Dataset                 | Accuracy (custom / sklearn) | F1 (custom / sklearn) | Support Vectors (custom / sklearn) | Active Kernels | Weight Entropy | Dominant Kernel                         |
| ----------------------- | --------------------------: | --------------------: | ---------------------------------: | -------------: | -------------: | --------------------------------------- |
| Banknote Authentication |         `96.71%` / `96.71%` |   `96.44%` / `96.41%` |                      `257` / `276` |            `1` |           ≈`0` | Rational Quadratic                      |
| Breast Cancer Wisconsin |         `97.35%` / `97.35%` |   `98.04%` / `98.04%` |                        `77` / `77` |            `5` |        `0.783` | Polynomial (`0.442`) / Linear (`0.357`) |
| Pima Indians Diabetes   |         `71.24%` / `69.93%` |   `62.71%` / `60.34%` |                      `340` / `340` |            `5` |        `0.988` | Polynomial (`0.278`), near-uniform      |
| Ionosphere              |         `95.71%` / `95.71%` |   `96.70%` / `96.70%` |                      `109` / `109` |            `1` |           ≈`0` | Laplacian                               |

Across all four datasets, accuracy and F1 agree with scikit-learn to within noise, and support-vector counts are frequently identical — evidence that the custom SMO solver converges to the same (or an equivalent) dual optimum as LIBSVM under an identical kernel matrix. Full per-dataset training logs, β trajectories, and discussion are in the corresponding files under `results/reports/`.

### 10.2 β-Optimizer Stress Test — MAGIC Gamma Telescope

All seven optimizers were run under an otherwise identical configuration (kernel set, PCA reduction, SMO penalty $C=1$, diagonal normalization on, centering off — see `docs/bom_and_mst_documentation.md` §4 for the full experimental protocol) on the 4,000-sample MAGIC Gamma Telescope subsample.

**Overall results:**

| Optimizer        |       Accuracy |      Precision |         Recall |             F1 |      Time | Active Kernels | Support Vectors |
| ---------------- | -------------: | -------------: | -------------: | -------------: | --------: | -------------: | --------------: |
| SimpleMKL        | _**`82.13%`**_ | _**`81.01%`**_ |       `94.83%` | _**`87.38%`**_ |    2m 53s |              4 |            1504 |
| L2-MKL           |       `82.00%` |       `80.78%` |       `95.02%` |       `87.32%` |    1m 53s |              5 |            1470 |
| Entropic Descent |       `82.00%` |       `80.78%` |       `95.02%` |       `87.32%` |    2m 30s |              4 |            1539 |
| Hybrid L1/L2     |       `81.38%` |       `80.23%` |       `94.83%` |       `86.92%` |    2m 23s |              1 |            1553 |
| KTA              |       `81.88%` |       `80.55%` | _**`95.21%`**_ |       `87.27%` | _**52s**_ |              5 |      _**1453**_ |
| Frank-Wolfe      |       `81.38%` |       `80.23%` |       `94.83%` |       `86.92%` |    2m 02s |              1 |            1553 |
| FISTA            |       `81.38%` |       `80.23%` |       `94.83%` |       `86.92%` |    1m 23s |              1 |            1553 |

**Final kernel weights (β):**

| Optimizer | Linear | Polynomial | RBF | Laplacian | Rational Quadratic |
|---|--:|--:|--:|--:|--:|
| SimpleMKL | 0.000 | 0.110 | 0.163 | **0.661** | 0.066 |
| L2-MKL | 0.050 | 0.203 | 0.277 | **0.329** | 0.141 |
| Entropic Descent | 0.000 | 0.004 | 0.099 | **0.897** | 0.000 |
| Hybrid L1/L2 | 0.000 | 0.000 | 0.000 | **1.000** | 0.000 |
| KTA | 0.173 | 0.156 | **0.255** | 0.212 | 0.204 |
| Frank-Wolfe | 0.000 | 0.000 | 0.000 | **1.000** | 0.000 |
| FISTA | 0.000 | 0.000 | 0.000 | **1.000** | 0.000 |

**Key findings** (full discussion in §5–§7 of the technical "[BOM Documentation](https://github.com/Sepiimt/MKL-SVM-Framework-From-Scratch/blob/main/docs/bom_and_mst_documentation.md)"):

- Predictive performance is remarkably insensitive to the choice of β-optimizer: the spread between the highest and lowest accuracy is under one percentage point, despite substantially different kernel distributions.
- Three qualitatively distinct regimes emerge: **sparse** optimizers (Hybrid L1/L2, Frank-Wolfe, FISTA) collapse to a single Laplacian kernel; **balanced** optimizers (L2-MKL, KTA) retain meaningful weight on every kernel; **adaptive** optimizers (SimpleMKL, Entropic Descent) land between the two, with one dominant kernel supported by several minor ones.
- KTA is markedly the fastest method (52 s vs. 1–3 minutes for the iterative methods) and converges in a single outer iteration, since its β update is a closed-form alignment score independent of SMO's dual variables — making it a reasonable low-cost baseline or warm-start heuristic for larger datasets.
- KTA is also the only method whose largest weight does not fall on the Laplacian kernel (0.255 on RBF), which the documentation treats as an open, only partially resolved question about how much of Laplacian's benchmark-wide dominance reflects genuine predictive value versus a PCA-induced geometric artifact (the PCA rotation step is not distance-preserving under the Laplacian kernel's $L_1$ metric — see §9.2 of `docs/bom_and_mst_documentation.md`).

**Practical recommendations** (condensed from §8 of the full "[BOM Documentation](https://github.com/Sepiimt/MKL-SVM-Framework-From-Scratch/blob/main/docs/bom_and_mst_documentation.md)"):

| Objective | Recommended optimizer | Rationale |
|---|---|---|
| Highest predictive performance | SimpleMKL | Highest benchmark accuracy with an interpretable, adaptive kernel mixture |
| Fastest optimization | KTA | Single-iteration convergence; competitive accuracy; smallest SV count |
| Balanced kernel utilization | L2-MKL | Retains meaningful weight on every candidate kernel |
| Maximum interpretability | Hybrid L1/L2, Frank-Wolfe, or FISTA | Converge to a single dominant kernel |
| Adaptive kernel selection | Entropic Descent | Gradual, multiplicative redistribution; compromise between sparse and dense |

The full documentation additionally covers per-optimizer theoretical background, convergence traces, computational-efficiency ranking, and a substantially more detailed limitations discussion (§9), including the fixed-$C$ / diagonal-normalization interaction and why the KTA Score column in Table 3 of that document should not be read as an independent metric for the KTA method itself.

---
## 11. Model Persistence

Model state is saved and loaded in three layers, mirroring the architecture in §3:

```python
model.save_model("artifacts/svm/cancer/")
```

writes, under `artifacts/svm/cancer/`:

| File                        | Written by                                     | Contents                                                                                      |
| --------------------------- | ---------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `model_data_SVM_layer.npy`  | `SVM.save_model`                               | Selected kernels, kernel hyperparameters                                                      |
| `model_data_MKL_layer.npy`  | `MKL.save_model`                               | Final β vector, kernel selection/centering/normalization flags, support-vector masked $X$/$Y$ |
| `model_data_SMO_layer.npy`  | `SMO.save_model`                               | Bias $b$, support-vector dual weights $\alpha$, support labels, support indices               |
| `model_data_{i}_kernel.npy` | `Kernels.save_model` _(one per active kernel)_ | Cached centering/normalization statistics for that kernel                                     |

`SVM.load_model(filepath)` reconstructs all three layers and re-derives the kernel instance list from the saved `selected_kernels` mask, so that a reloaded model can call `.predict()` without repeating any part of training.

---
## 12. Known Limitations

Condensed from §9 of "[BOM Documentation](https://github.com/Sepiimt/MKL-SVM-Framework-From-Scratch/blob/main/docs/bom_and_mst_documentation.md)", which should be consulted for the full argument behind each point:
	
- **Dataset scope:** All benchmarks are binary-classification, tabular datasets of small-to-medium size; findings should not be assumed to generalize to high-dimensional, multi-class, text, or image settings.
	
- **PCA / Laplacian-kernel interaction is an open question:** PCA's rotation step is not distance-preserving under the $L_1$ metric the Laplacian kernel uses, which is a documented, mechanism-level reason to be cautious about attributing Laplacian's benchmark-wide dominance purely to its predictive merit rather than partly to post-PCA geometry. The repository does not include the ablation (repeating the comparison without PCA, or with an $L_1$-preserving reduction) that would settle this.
	
- **Kernel hyperparameters are held fixed** within any given optimizer comparison, isolating the effect of β-optimization but leaving joint optimization of kernel hyperparameters and kernel weights out of scope.
	
- **Memory and compute scale quadratically** with the number of training samples (dense $N \times N$ kernel matrices); the framework targets medium-scale tabular problems rather than large-scale learning.
	
- **A fixed SVM penalty ($C=1$)** is used across all optimizer comparisons. Diagonal normalization (§2) closes the sharpest version of this concern — no kernel can dominate purely from a larger raw numeric scale — but does not guarantee identical off-diagonal / spectral structure across kernels, so a fixed $C$ is defensible rather than fully rigorous.
	
- **The KTA Score metric in the stress-test benchmark is definitionally biased** toward the KTA optimizer, since it is exactly the quantity that optimizer maximizes; KTA's training-time and support-vector-count results carry the independent evidential weight in that comparison, not its KTA Score.
	
- **Engineering scope:** The implementation prioritizes clarity, reproducibility, and mathematical transparency over the decades of low-level engineering (advanced caching strategies, further working-set refinements, GPU based calculations, numerical micro-optimizations) found in mature libraries such as LIBSVM, even though it already implements LIBSVM-style working-set alternation and warm-starting and various optimizations under the scope of the determined goal.

---
## 13. Known Issues/limitation

**No packaging metadata.** The absence of `pyproject.toml` / `setup.py` means the package cannot currently be `pip install`-ed; see §5 for the `sys.path` workaround used throughout the notebooks.

---
## 14. Notebooks and Further Documentation

| Notebook | Role |
|---|---|
| `data_cleaning_and_engineering.ipynb` | Cleans and engineers features for all five raw datasets |
| `pca_model_train.ipynb` | Fits and saves a `PCA` model per dataset; produces the PCA-projected train/test splits under `data/splits/` |
| `svm_model_train.ipynb` | Trains the `SVM` model per dataset and saves the resulting model artifacts and predictions |
| `model_evaluation.ipynb` | Computes and saves the metric comparisons behind `results/reports/*.md` |
| `bom_stress_test.ipynb` | Runs all seven β-optimizers on the MAGIC Gamma Telescope subsample |
| `stress_test_results.ipynb` | Aggregates the stress-test output into the tables reproduced in §10.2 |

For the complete treatment — theoretical background and behavior observations for each of the seven β-optimizers, per-optimizer convergence traces, a full computational-efficiency ranking, and the complete limitations discussion — see **"[BOM Documentation](https://github.com/Sepiimt/MKL-SVM-Framework-From-Scratch/blob/main/docs/bom_and_mst_documentation.md)"** (or at `docs/bom_and_mst_documentation.md`).

---
## 15. License

Distributed under the **GNU General Public License v3.0**. See [LICENSE](https://github.com/Sepiimt/MKL-SVM-Framework-From-Scratch/blob/main/LICENSE) for the full text.

```
PCA and MKL-SVM framework: Custom machine learning algorithm implementations from scratch.
Copyright (C) 2026  Sepanta Metanat

This program is free software: you can redistribute it and/or modify it under the 
terms of the GNU General Public License as published by the Free Software Foundation, 
either version 3 of the License, or any later version.
```

---
