
---
---
---
# **Beta Optimization Methods & Model Stress Test Documentation**
---
---
---
# 1. Introduction

"Multiple Kernel Learning" (MKL) extends standard kernel-based classification by allowing several kernels to contribute to the final decision function simultaneously. Instead of committing to a single fixed kernel, MKL learns a convex combination of kernels, enabling the model to adapt to different similarity structures in the data. This is particularly useful when no single kernel family is sufficient to capture all relevant patterns.

In this repository, the kernel combination is controlled by the coefficient vector **β**. Each component of **β** determines the contribution of one kernel to the final combined kernel. 
For a set of kernels (${K_1, K_2, \dots, K_m}$), the combined kernel is written as: $[  K(x_i, x_j) = \sum_{k=1}^{m} \beta_k K_k(x_i, x_j)  ]$  with the constraints: $[  \beta_k \ge 0, \qquad \sum_{k=1}^{m} \beta_k = 1  ]$

These constraints force the learned kernel weights to lie on the simplex, making the optimization well-defined and interpretable.

The purpose of β optimization is not only to improve classification performance, but also to determine which kernel families are most informative for a given dataset. Some optimizers tend to produce sparse solutions, concentrating most of the weight on a small subset of kernels, while others distribute the weights more evenly. This difference matters because it affects both the model interpretation and the computational cost of the learning process.

The implementation in this project was designed to compare multiple β optimization strategies under the same "SVM" training pipeline. The goal is to study how different update rules influence kernel selection, convergence behavior, support vector count, and overall classification performance.

---
---
---
# 2. Optimization Framework

The "MKL-SVM" training procedure follows an alternating optimization scheme. Since the "SVM" dual variables ($\alpha$) and the kernel weights ($\beta$) depend on each other, they are not optimized simultaneously in a single step. Instead, the algorithm alternates between solving the "SVM" problem for fixed kernel weights and updating the kernel weights using the current solution of the "SVM".

A simplified view of the procedure is:
```text
Initialize β
repeat
    1. Train SVM with the current combined kernel
    2. Compute the contribution of each kernel
    3. Update β using the selected optimization method
until convergence
Run a final SVM fit using the converged β
```

The first step solves the SVM subproblem while keeping the current kernel mixture fixed. The second step estimates how each kernel contributes to the current objective. The third step updates β according to one of the supported optimization strategies:
- `SimpleMKL`
    
- `L2-MKL`
    
- `Entropic Descent`
    
- `Hybrid L1/L2`
    
- `KTA`
    
- `Frank-Wolfe`
    
- `FISTA`

Each method follows the same overall alternating structure, but differs in how the β update is computed and how strongly it encourages sparsity or smoothness in the kernel mixture.

A final "SVM" fit is performed after β converges. This step is necessary because the last β update changes the combined kernel, and the final classifier should be trained on the converged kernel mixture rather than on an intermediate one. Without this final pass, the learned "SVM" parameters may correspond to an earlier kernel state and the model would no longer be fully synchronized.

Convergence is controlled by two criteria: the tolerance on β updates and the maximum number of outer iterations. In practice, the outer "MKL" loop and the inner "SMO" solver may converge at different rates. For that reason, the implementation reports both the β optimization progress and the "SVM" training behavior, so that the effect of each optimization method can be evaluated separately.

This nested structure is the central object of comparison in the repository. Different β optimizers can lead to similar classification accuracy while producing very different kernel distributions, convergence speeds, and support vector counts. That makes the optimization method itself an important experimental variable rather than a purely technical detail.

---
---
---
# 3. β Optimization Methods

Although every optimization method implemented in this repository seeks the same objective, they differ significantly in how they search the kernel-weight space. Some methods encourage sparse kernel selection, while others intentionally distribute weights across multiple kernels. Consequently, different optimization strategies may produce similar classification accuracy while exhibiting substantially different optimization dynamics, computational costs, and model complexity.

The following sections summarize the theoretical motivation of each optimizer together with the practical behavior observed during benchmarking.

---
## 3.1 SimpleMKL

#### Theoretical Background
"SimpleMKL", introduced by "Rakotomamonjy et al". (2008), is one of the earliest and most influential optimization methods for Multiple Kernel Learning. It formulates MKL as a joint optimization problem in which the Support Vector Machine parameters and kernel coefficients are alternately optimized until convergence.

Unlike heuristic kernel selection techniques, "SimpleMKL" directly optimizes the kernel mixture while preserving the convexity of the overall optimization problem.

#### Optimization Philosophy
"SimpleMKL" encourages sparse kernel combinations through gradient-based optimization over the simplex. Rather than averaging every available kernel equally, it gradually increases the weight of kernels that contribute most to the margin while suppressing less informative ones.

The resulting β vector typically contains one dominant kernel accompanied by a small number of supporting kernels.

#### Advantages
- Convex optimization with theoretical convergence guarantees.
    
- Produces interpretable kernel combinations.
    
- Naturally performs implicit kernel selection.
    
- Well-established baseline for "MKL" research.

#### Limitations
- Requires repeated "SVM" optimization during training.
    
- Computational cost increases with the number of kernels.
    
- Convergence speed depends on the quality of the inner "SVM" solution.

#### Behavior Observation
Within this implementation, "SimpleMKL" consistently converged toward sparse kernel mixtures. During experiments, the "Laplacian" kernel received the largest weight while "Polynomial" and "RBF" kernels remained active with considerably smaller contributions. The "Linear" kernel was excluded entirely (weight reduced to zero), while "Rational Quadratic" retained a small but nonzero weight.

Compared with the other optimization methods, SimpleMKL achieved the highest classification accuracy on the benchmark dataset while maintaining moderate kernel sparsity. The optimization trajectory was smooth and monotonic, with kernel weights stabilizing after only a small number of outer iterations.

---
## 3.2 L2-MKL

#### Theoretical Background
"L2-MKL" modifies the "MKL" objective by introducing an L2 regularization term over the kernel coefficients. Instead of encouraging sparsity, the optimization penalizes large deviations among kernel weights, promoting smoother kernel combinations.

#### Optimization Philosophy
Unlike "SimpleMKL", "L2-MKL" assumes that several kernels may simultaneously contribute meaningful information. Consequently, the optimizer distributes weight across multiple kernels instead of forcing a small subset to dominate.

This philosophy resembles ridge regularization in linear models, where coefficients are shrunk toward one another rather than eliminated.

#### Advantages
- Stable optimization behavior.
    
- Resistant to abrupt kernel elimination.
    
- Produces balanced kernel mixtures.
    
- Less sensitive to noisy gradient estimates.

#### Limitations
- Reduced interpretability due to dense kernel combinations.
    
- Higher computational cost.
    
- May retain kernels with only marginal predictive value.

#### Behavior Observation
Experimental results clearly reflected the expected theoretical behavior. Every kernel remained active throughout optimization, with no coefficient collapsing to zero. The final β distribution was relatively balanced, assigning moderate weight to all available kernels while giving a slight preference to the "Laplacian" and "RBF" kernels.

Compared with sparse optimization methods, "L2-MKL" required longer training time but produced one of the smoothest convergence trajectories observed during benchmarking.

---
## 3.3 Entropic Descent

#### Theoretical Background
"Entropic Descent" performs optimization directly on the probability simplex using entropy as the underlying geometry. Instead of additive updates, kernel weights are modified multiplicatively and normalized after every iteration.

This approach naturally preserves the simplex constraints without requiring explicit projection.

#### Optimization Philosophy
The multiplicative update mechanism strongly rewards kernels that consistently improve the objective while rapidly reducing the influence of poorly performing kernels.

Unlike hard thresholding methods, "Entropic Descent" allows suppressed kernels to recover if later optimization steps reveal useful information.

#### Advantages
- Naturally maintains valid probability distributions.
    
- Smooth optimization over the simplex.
    
- Stable multiplicative updates.
    
- Well suited for probability-constrained optimization.

#### Limitations
- Sensitive to learning-rate selection.
    
- May initially overemphasize dominant kernels.
    
- Interpretation of multiplicative updates is less intuitive than additive optimization.

#### Behavior Observation
This implementation's optimization trajectory looks different in this run than it has in earlier ones. The optimizer's first update, from the uniform initialization, moved directly to a β vector assigning most of the weight to the "Laplacian" kernel while already establishing a smaller "RBF" contribution — both in the same step. The second and final outer iteration reproduced that vector unchanged (§7.2, §7.3).

This is a change worth noting explicitly: in earlier runs, this method's RBF contribution built up gradually over several later iterations, and that gradual redistribution was one of the clearest examples of meaningful β evolution observed during benchmarking. That is no longer what the current logs show — convergence is now direct rather than gradual, consistent with the broader shift to two-iteration convergence across the benchmark (§7.2.1). The destination (a Laplacian/RBF mixture) is unchanged; only the path to it is.

---
## 3.4 Hybrid L1/L2

#### Theoretical Background
"Hybrid L1/L2" optimization combines sparsity-inducing L1 regularization with the smoothing properties of L2 regularization. The objective attempts to balance aggressive kernel selection with numerical stability.

#### Optimization Philosophy
The optimizer seeks compact kernel representations without producing excessively unstable updates. Depending on the optimization landscape, either the sparse or smooth component may dominate.

#### Advantages
- Combines feature selection with regularization.
    
- Can produce compact kernel mixtures.
    
- Often converges rapidly.

#### Limitations
- Balance depends on regularization parameters.
    
- Behavior may resemble pure L1 optimization when one kernel clearly dominates.

#### Behavior Observation
On the benchmark dataset, the optimizer converged to an extremely sparse solution, assigning essentially all weight to the "Laplacian" kernel while eliminating every remaining kernel. This suggests that the sparse component of the objective dominated under the selected hyperparameters and dataset characteristics.

Although the resulting classifier achieved competitive predictive performance, the optimization produced the simplest kernel representation among all tested methods.

---
## 3.5 Kernel Target Alignment (KTA)

#### Theoretical Background
"Kernel Target Alignment" differs fundamentally from iterative "MKL" optimization methods. Rather than repeatedly alternating between β updates and SVM optimization, "KTA" estimates kernel importance by measuring how strongly each kernel aligns with the target labels.

The alignment score estimates the similarity between the kernel matrix and the ideal target kernel induced by class labels.

#### Optimization Philosophy
Instead of directly minimizing the "SVM" objective, "KTA" assumes that kernels exhibiting stronger alignment with the target labels should receive larger weights.

The resulting β vector serves as a data-driven estimate of kernel quality before or alongside classifier optimization.

#### Advantages
- Computationally efficient.
    
- Requires fewer optimization iterations.
    
- Produces stable and reproducible kernel weights.
    
- Easy to interpret.

#### Limitations
- Alignment does not directly optimize classification margin.
    
- May overlook interactions between kernels.
    
- Performance depends on the quality of the alignment measure.

#### Behavior Observation
Among all evaluated methods, "KTA" demonstrated the lowest computational cost while maintaining classification performance comparable to iterative optimizers. The learned β distribution remained balanced across all kernels, producing the highest weight entropy and one of the smallest support vector counts observed during benchmarking. Its largest weight fell on the "RBF" kernel rather than "Laplacian" — the only method in this benchmark for which that is true (§6.4).

These results suggest that kernel alignment serves as an effective initialization and selection heuristic despite not directly optimizing the "SVM" objective.

---
## 3.6 Frank-Wolfe

#### Theoretical Background 
The "Frank-Wolfe" algorithm is a projection-free first-order optimization method for constrained convex optimization. Instead of projecting onto the simplex after every gradient step, the algorithm moves toward an extreme point determined by the current gradient.

#### Optimization Philosophy
Because every iteration moves toward a simplex vertex, "Frank-Wolfe" naturally encourages sparse solutions. Only kernels with strong gradient information continue to receive weight as optimization progresses.

#### Advantages
- Projection-free optimization.
    
- Low memory requirements.
    
- Naturally sparse solutions.
    
- Suitable for high-dimensional constrained optimization.

#### Limitations
- May converge more slowly near the optimum.
    
- Frequently produces vertex solutions.
    
- Sensitive to gradient accuracy.

#### Behavior Observation
Within this implementation, "Frank-Wolfe" consistently converged to a single-kernel solution dominated entirely by the "Laplacian" kernel. This behavior agrees with the theoretical tendency of the algorithm to approach simplex vertices. Although predictive performance remained competitive, the resulting kernel mixture was considerably sparser than those produced by "L2-MKL" or "KTA".

---
## 3.7 FISTA

#### Theoretical Background
The "Fast Iterative Shrinkage-Thresholding Algorithm" (FISTA) accelerates first-order optimization by introducing a momentum term into the standard proximal gradient method. Under convex objectives, "FISTA" achieves substantially faster theoretical convergence than ordinary gradient descent.

#### Optimization Philosophy
Momentum allows the optimizer to exploit information from previous iterations, reducing zigzag behavior and accelerating convergence toward the optimum.

When combined with sparsity-promoting regularization, "FISTA" often produces compact solutions while requiring relatively few iterations.

#### Advantages
- Accelerated convergence.
    
- Efficient first-order optimization.
    
- Suitable for large optimization problems.
    
- Naturally compatible with sparse regularization.

#### Limitations
- May overshoot if poorly parameterized.
    
- Performance depends on step-size estimation.
    
- Can converge to highly sparse solutions.

#### Behavior Observation
The implementation consistently converged to the same sparse kernel configuration obtained by "Frank-Wolfe" and the "Hybrid L1/L2" optimizer. Nearly the entire kernel weight was assigned to the "Laplacian" kernel, indicating that momentum acceleration did not alter the final optimization landscape under the evaluated conditions. Instead, "FISTA" primarily influenced the optimization process rather than the final kernel distribution.

---
## 3.8 Comparative Discussion

The benchmark results demonstrate that the choice of β optimizer influences **how** the optimization proceeds far more than **what** predictive accuracy is ultimately achieved. Across the evaluated datasets, classification performance remained remarkably consistent despite substantial differences in kernel distributions, convergence dynamics, computational cost, and model complexity.

Three distinct optimization behaviors emerged:
- **Sparse optimizers** ("Hybrid L1/L2", "Frank-Wolfe", and "FISTA") converged toward single-kernel solutions dominated by the "Laplacian" kernel.
    
- **Balanced optimizers** ("L2-MKL" and "KTA") maintained meaningful contributions from every kernel throughout optimization.
    
- **Adaptive optimizers** ("SimpleMKL" and "Entropic Descent") produced intermediate solutions in which one dominant kernel was supported by a small number of secondary kernels.

These observations reinforce an important conclusion: β optimization should not be evaluated solely by classification accuracy. Optimization behavior, kernel interpretability, convergence stability, computational efficiency, and model complexity provide a more complete picture of the strengths and limitations of each method.

---
---
---
# 4. Benchmark Methodology

To ensure a fair comparison between β optimization strategies, every optimizer was evaluated under an identical experimental configuration. Only the β optimization method was changed between experiments, while the dataset, preprocessing pipeline, kernel configuration, and "SVM" hyperparameters remained unchanged.

---
## 4.1 Dataset
The benchmark was performed using the "**MAGIC Gamma Telescope**" dataset, a binary classification problem commonly used for evaluating machine learning algorithms.

To maintain manageable computational requirements while preserving the statistical characteristics of the original dataset, a stratified random subset containing **`4,000` samples** was selected using a fixed random seed.

The resulting dataset was divided into training and testing subsets using an `80/20` split.

| Property          |                   Value |
| ----------------- | ----------------------: |
| Dataset           | `MAGIC Gamma Telescope` |
| Samples           |                 `4,000` |
| Original Features |                    `10` |
| Classification    |                `Binary` |
| Train/Test Split  |             `80% / 20%` |
| Random Seed       |                    `42` |

The selected benchmark is substantially larger than the remaining datasets included in this repository, making it suitable for evaluating the computational behavior of "Multiple Kernel Learning" algorithms.

---
## 4.2 Dimensionality Reduction
"Principal Component Analysis" (PCA) was applied before training the "MKL-SVM" model.

Rather than specifying the number of retained components manually, the "PCA" implementation preserved approximately **`96.1%`** of the total variance, reducing the feature space from ten original variables to seven principal components.

This preprocessing step reduces redundancy while preserving nearly all informative variance contained in the original data.

| PCA Property        |   Value |
| ------------------- | ------: |
| Original Dimensions |    `10` |
| Reduced Dimensions  |     `7` |
| Variance Retained   | `96.1%` |

---
## 4.3 Kernel Configuration
Every optimization method was evaluated using the same candidate kernel set.
	
- "Linear"
    
- "Polynomial"
    
- "Radial Basis Function" (RBF)
    
- "Laplacian"
    
- "Rational Quadratic"

The "Sigmoid" kernel was intentionally excluded from the benchmark because preliminary experiments consistently demonstrated inferior performance and unstable optimization behavior relative to the remaining kernels.

Every kernel matrix returned by these implementations is diagonally normalized before use: $K'(x_i, x_j) = K(x_i, x_j) / \sqrt{K(x_i, x_i) \, K(x_j, x_j)}$. This rescales each base kernel to a unit diagonal, so no single kernel can dominate the combined mixture purely as an artifact of operating on a larger intrinsic numeric scale than the others — a point that becomes relevant when interpreting the fixed "SVM" penalty $C$ in §9.6.

The implementation also supports kernel centering (projecting out the mean in feature space, $K_c = HKH$ with $H = I - \tfrac{1}{n}\mathbf{1}\mathbf{1}^\top$), applied independently of diagonal normalization. It is switched **off** for this stress test: empirically, leaving it on both degrades classification performance and causes training time to spike. A plausible mechanism, offered as a hypothesis rather than a verified explanation, is that centering can introduce negative entries into kernels that were originally non-negative ("RBF" and "Laplacian", in particular), which changes the conditioning of the "SMO" subproblem without changing its validity — and given that the "SMO" iteration cap is also removed for this stress test (§4.4), any centering-induced increase in per-call "SMO" cost would compound directly into wall-clock time rather than being absorbed by a ceiling. This has not been isolated experimentally and would be a reasonable target for a dedicated ablation.

Default kernel hyperparameters were used throughout all experiments in order to isolate the effect of β optimization itself.

---
## 4.4 Training Configuration
The "MKL" optimization followed an alternating optimization framework in which each β update was followed by an "SVM" optimization using "Sequential Minimal Optimization" (SMO).

The benchmark employed the following hyperparameters:

| Parameter              |                              Value |
| ---------------------- | ---------------------------------: |
| β Optimizer Iterations |                           `50` ⁽¹⁾ |
| β Tolerance            |                         `1 × 10⁻⁷` |
| SMO Penalty (C)        |                                `1` |
| Maximum SMO Iterations |      *removed* (previously `1000`) |
| SMO Tolerance          | `1 × 10⁻³` (previously `1 × 10⁻⁷`) |
| Kernel Centering       |                              `Off` |
| Kernel Normalization   |                               `On` |

Several of these settings were deliberately chosen for this stress test. The SMO tolerance was relaxed from `1×10⁻⁷` to `1×10⁻³`, and the hard cap on SMO iterations was removed entirely, so that computational effort scales with how hard each β update actually is for the inner SVM to resolve, rather than being truncated by an arbitrary ceiling that becomes an increasingly expensive bottleneck as the dataset grows. Kernel centering is switched off for the same underlying reason (§4.3): with "SMO" now uncapped, any centering-induced increase in per-call "SMO" cost would compound directly into wall-clock time. Collectively, these choices shift the benchmark's cost profile toward β optimization itself — the quantity actually under comparison — rather than toward the inner "SMO" solver's own convergence properties.

Using identical hyperparameters across all optimization methods ensures that observed differences originate from the optimization strategy itself rather than changes in the training procedure.

---
### 4.5 Evaluation Metrics
The following performance metrics were recorded for every optimization method.

#### Classification Performance
- Accuracy
    
- Precision
    
- Recall
    
- F1-score

#### MKL Characteristics
- Final β vector
    
- Number of active kernels
    
- Kernel Target Alignment (KTA)
    
- Weight entropy

#### Model Complexity
- Number of support vectors

#### Computational Performance
- Total training time
    
- Outer MKL iterations
    
- Memory usage during optimization

Collecting both predictive and optimization-related metrics enables comparison beyond classification accuracy alone. In particular, the benchmark evaluates how efficiently each optimization strategy reaches its solution and how that solution influences kernel selection and model complexity.

---
---
---
# 5. Benchmark Results

Reviewing stress-test benchmarks.

### Table 1. Overall Benchmark Results
Table 1 summarizes the overall performance of every implemented β optimization method.

| Optimizer        |   Accuracy |  Precision |     Recall |         F1 |    Time | Active Kernels | Support Vectors |
| ---------------- | ---------: | ---------: | ---------: | ---------: | ------: | -------------: | --------------: |
| SimpleMKL        | **82.13%** | **81.01%** |     94.83% | **87.38%** |  2m 53s |              4 |            1504 |
| L2-MKL           |     82.00% |     80.78% |     95.02% |     87.32% |  1m 53s |              5 |            1470 |
| Entropic Descent |     82.00% |     80.78% |     95.02% |     87.32% |  2m 30s |              4 |            1539 |
| Hybrid L1/L2     |     81.38% |     80.23% |     94.83% |     86.92% |  2m 23s |              1 |            1553 |
| KTA              |     81.88% |     80.55% | **95.21%** |     87.27% | **52s** |              5 |        **1453** |
| Frank-Wolfe      |     81.38% |     80.23% |     94.83% |     86.92% |  2m 02s |              1 |            1553 |
| FISTA            |     81.38% |     80.23% |     94.83% |     86.92% |  1m 23s |              1 |            1553 |

---
### Table 2. Final Kernel Weights (β)
Table 2 represent calculated β weights with respect to optimization method.

|Optimizer|Linear|Polynomial|RBF|Laplacian|Rational Quadratic|
|---|--:|--:|--:|--:|--:|
|SimpleMKL|0.000|0.110|0.163|**0.661**|0.066|
|L2-MKL|0.050|0.203|0.277|**0.329**|0.141|
|Entropic Descent|0.000|0.004|0.099|**0.897**|0.000|
|Hybrid L1/L2|0.000|0.000|0.000|**1.000**|0.000|
|KTA|0.173|0.156|**0.255**|0.212|0.204|
|Frank-Wolfe|0.000|0.000|0.000|**1.000**|0.000|
|FISTA|0.000|0.000|0.000|**1.000**|0.000|

*Note: weights above are rounded to three decimals for readability. A value shown as `0.000` may be exactly zero (a kernel pruned from the mixture, as with "Linear" under "SimpleMKL"/"Hybrid L1/L2"/"Frank-Wolfe"/"FISTA") or merely very small but still nonzero (e.g. Entropic Descent's Rational Quadratic weight is ≈0.0003, not exactly zero). This is why the "Active Kernels" counts in Table 1 do not always match what Table 2 appears to show at a glance — they are computed from the unrounded coefficients.*

---
### Table 3. Optimization Characteristics
Table 3 shows the details related to training process and results.

| Optimizer        |  KTA Score | Weight Entropy |       SMO b | MKL Iterations |
| ---------------- | ---------: | -------------: | ----------: | -------------- |
| SimpleMKL        |     0.1548 |          0.616 |     -0.3864 | 2              |
| L2-MKL           |     0.1644 |          0.914 |     -0.2788 | 2              |
| Entropic Descent |     0.1495 |          0.218 |     -0.5223 | 2              |
| Hybrid L1/L2     |     0.1460 |             ≈0 |     -0.4848 | 2              |
| KTA              | **0.1680** |      **0.991** | **-0.1740** | **1**          |
| Frank-Wolfe      |     0.1460 |             ≈0 |     -0.4850 | 2              |
| FISTA            |     0.1460 |             ≈0 |     -0.4853 | 2              |

*Note: the "KTA Score" column should be read differently for the "KTA" optimizer than for the others. KTA's β update directly maximizes kernel-target alignment, so its topping this column is closer to a tautology than an independent finding — it is by construction the method that scores best here. Its wins on training time and support-vector count are the metrics that carry independent evidential weight. See §9.6 for a discussion of alternative, optimizer-agnostic metrics for a future revision.*
*"MKL" iterations may also differ based on your training settings*

---
## 5.1 Summary of Experimental Findings
Several observations can be made immediately from the benchmark results.

First, all optimization methods achieved remarkably similar predictive performance. The difference between the highest and lowest classification accuracy was less than one percentage point, suggesting that the choice of β optimizer has only a limited influence on the final decision boundary for this dataset.

Second, despite similar predictive performance, the optimization strategies produced substantially different kernel distributions. Sparse optimization methods converged almost exclusively to the "Laplacian" kernel, whereas "L2-MKL" and "KTA" maintained meaningful contributions from every available kernel.

Third, computational efficiency varied considerably. "KTA" completed training in under a minute while simultaneously producing the smallest support vector set among all evaluated methods. In contrast, "L2-MKL" required roughly twice as long despite achieving almost identical classification performance — though "L2-MKL" is neither the fastest nor the slowest iterative method in this run; see §6.6 for the complete ranking and why it has not held stable across successive runs of this benchmark.

Finally, the benchmark demonstrates that evaluating β optimization solely by prediction accuracy provides an incomplete picture. Kernel sparsity, convergence behavior, computational cost, and model complexity exhibit far greater variation than the classification metrics themselves, indicating that these characteristics are more informative for distinguishing "MKL" optimization strategies.

---
---
---
# 6. Behavior Analysis

Building on §3.8 and §5.1, this section examines *how* each optimizer arrived at its solution — kernel selection, evolution over iterations, model complexity, and computational cost.

---
## 6.1 Predictive Performance
Across all experiments, the classification accuracy varied by less than one percentage point. Similar consistency was observed for "Precision", "Recall", and "F1-score".

Such a small performance gap indicates that all evaluated optimization methods successfully identified kernel combinations capable of separating the classes with comparable effectiveness.

This result should not be interpreted as evidence that every optimizer is equivalent. Rather, it suggests that the selected dataset contains several kernel combinations capable of producing similarly effective decision boundaries.

Consequently, predictive performance alone is insufficient for distinguishing "MKL" optimization strategies.

---
## 6.2 Kernel Selection Behavior
The benchmark reveals three distinct optimization philosophies.

#### Sparse Optimization
"Hybrid L1/L2", "Frank-Wolfe", and "FISTA" converged to nearly identical kernel distributions in which the "Laplacian" kernel received essentially the entire kernel weight.

The resulting β vectors contained a single active kernel, producing the sparsest solutions observed during benchmarking.

This behavior is consistent with optimization methods designed to approach vertices of the simplex or explicitly encourage sparse solutions.

Although these optimizers sacrificed a small amount of predictive performance, they produced the simplest kernel representations and therefore the highest level of interpretability.

**Why three different formulations land on the same vertex:** "Frank-Wolfe", "FISTA", and "Hybrid L1/L2" are not the same algorithm — they encourage sparsity through three distinct mechanisms (an explicit linear-minimization step onto a simplex vertex, momentum-accelerated proximal shrinkage, and a mixed L1/L2 penalty, respectively). That three different mechanisms converge to an identical one-hot solution is more informative than it looks: momentum and acceleration change the *path and speed* toward an optimum, not *which* optimum is reached. If the gradient of the underlying objective favors the "Laplacian" kernel by a wide enough margin over every other kernel — plausible here, given how convincingly "Laplacian" dominates even a comparatively balanced optimizer like "L2-MKL" in Table 2 — then Frank-Wolfe's vertex selection, FISTA's shrinkage, and Hybrid L1/L2's thresholding are all being pulled toward the same corner of the simplex from the first iteration onward. Acceleration (FISTA) only matters when there is genuine competition between candidate directions to navigate around; with one direction this dominant, there is no zigzag for momentum to correct. This is consistent with — and arguably a downstream symptom of — the open question raised in §9.2 about whether PCA's interaction with $L_1$-based kernel geometry is itself responsible for making Laplacian's gradient signal this dominant in the first place. Notably, "KTA" — the one method in this benchmark that does not route through that same iterative "SVM" gradient at all — does *not* show this dominance (§6.4), which is itself a data point worth weighing when assessing that hypothesis.

#### Balanced Optimization
"L2-MKL" and "Kernel Target Alignment" (KTA) produced fundamentally different kernel distributions.

Rather than suppressing weaker kernels, both methods maintained meaningful contributions from every available kernel.

This behavior is particularly evident in the weight entropy values, which approached the theoretical maximum among all evaluated methods.

The resulting kernel mixtures suggest that these optimization strategies favor cooperation among kernels instead of competition.

From an optimization perspective, this behavior reflects the influence of L2 regularization and alignment-based weighting, both of which discourage aggressive kernel elimination.

#### Adaptive Optimization
"SimpleMKL" and "Entropic Descent" occupy an intermediate position between sparse and dense optimization.

Both methods produced one dominant kernel accompanied by a small number of supporting kernels.

Instead of forcing a single winner or averaging every kernel equally, these optimizers reach a compromise configuration directly rather than gradually — like every other iterative method in this run, both converge within two outer iterations (§7.2.1, §6.3), so "gradually refined" no longer describes how this compromise is reached, only what it looks like once reached.

The resulting β vectors provide a compromise between interpretability and flexibility, retaining only kernels that continue to provide measurable benefit during optimization.

---
## 6.3 Kernel Evolution Patterns
The evolution of kernel coefficients throughout the optimization process provides additional insight into the behavior of each β optimization strategy.

While the final β vector describes the resulting kernel combination, the optimization trajectory reveals how the algorithm arrived at that solution. Different optimization methods may reach similar predictive performance while following substantially different paths through the kernel-weight space.

Three major patterns were observed during benchmarking.

#### Sparse Kernel Convergence
Sparse optimization methods, including "Hybrid L1/L2", "Frank-Wolfe", and "FISTA", demonstrated rapid concentration toward a single dominant kernel.

During optimization, the "Laplacian" kernel progressively increased its contribution until it represented almost the entire kernel mixture.

This behavior reflects the underlying objective of these methods: reducing unnecessary kernel contributions and producing compact representations.

The resulting models provide high interpretability because the selected kernel directly identifies the dominant similarity measure for the dataset.

However, kernel sparsity should not be confused with classifier simplicity. Although fewer kernels are used, the resulting "SVM" may still require a considerable number of support vectors.

#### Distributed Kernel Optimization
"L2-MKL" exhibited a fundamentally different trajectory from the sparse methods, though its own trajectory looks different in this run than it has previously.

Rather than suppressing kernels, the optimizer moved directly from the uniform initialization to a solution retaining meaningful weight on every kernel — "Linear", "Polynomial", "RBF", "Laplacian", and "Rational Quadratic" — in its very first update; the second and final outer iteration then reproduced that same β vector exactly, confirming it as the converged solution rather than an intermediate step (§7.2). With convergence now occurring in two outer iterations rather than the six to seven seen in earlier runs (§7.2.1), "gradual" is no longer the right description of how this β vector is reached — the smoothing effect of L2 regularization is visible in the *shape* of the final distribution (no coefficient near zero), not in a multi-step trajectory toward it.

The resulting kernel representation is less compact than the sparse methods' but may better capture complementary information from different similarity functions.

#### Adaptive Kernel Redistribution
"Entropic Descent" produced a similarly direct trajectory in this run.

Starting from a uniform initialization, $\beta=[0.2,0.2,0.2,0.2,0.2]$, the optimizer's first update moved almost the entire weight onto the "Laplacian" kernel while also establishing a smaller but real "RBF" contribution — both in the same update, not across a sequence of subsequent ones. The second and final outer iteration reproduced this β vector unchanged.

This is a genuine change from earlier runs of this benchmark, where the same method's RBF contribution was described as being "gradually restored" across several later iterations. That multi-step trajectory is not what the current logs show; the destination is the same (a Laplacian/RBF mixture), but it is now reached in a single update, then confirmed rather than refined.

---
## 6.4 Dominance of the Laplacian Kernel
One of the most consistent observations throughout the benchmark is the prominence of the "Laplacian" kernel — though it is no longer, with this run's updated results, a completely universal one.

Six of the seven optimization strategies assigned the largest kernel weight to the "Laplacian" kernel, although the magnitude of this preference varied considerably.

For sparse optimization methods, the "Laplacian" kernel completely dominated the optimization process.

"SimpleMKL" and "Entropic Descent" retained secondary contributions from the "RBF" and "Polynomial" kernels, while "L2-MKL" distributed weight more evenly across all kernels while still favoring "Laplacian" (0.329, versus 0.277 for its next-largest, "RBF").

"KTA" is the sole exception in this run: its weights are the most balanced of any method ("Linear" 0.173, "Polynomial" 0.156, "RBF" **0.255**, "Laplacian" 0.212, "Rational Quadratic" 0.204), and its largest weight now falls on "RBF" rather than "Laplacian" — a narrow margin, but a real one. This wasn't the case in earlier runs of this benchmark, where KTA's weights (like every other method's) peaked on "Laplacian"; the shift reflects the underlying implementation changes behind this run rather than a change in the dataset.

The consistency of Laplacian's preference across the six mathematically different optimization algorithms that alternate through the "SVM" objective still suggests that this preference originates from the characteristics of the dataset (or its post-"PCA" geometry, §9.2) rather than from any single optimization strategy. KTA's exception is informative precisely because it is the one method that does *not* alternate through the "SVM" objective — it estimates kernel importance directly from alignment scores — which raises the question of whether Laplacian's dominance is a property of the data alone, or specifically a property of how the iterative "SVM"-embedded optimizers respond to it. See §9.2 for further discussion of this distinction.

While this benchmark alone cannot establish a universal preference for "Laplacian" kernels — and KTA's result is direct evidence that it isn't universal — it still provides strong evidence that the reduced "MAGIC Gamma Telescope" dataset is particularly well represented by "Laplacian" similarity for the six iterative methods evaluated here.

---
## 6.5 Model Complexity
Although predictive performance remained nearly constant, the complexity of the resulting "SVM" models differed noticeably.

"Kernel Target Alignment" consistently produced the smallest support vector count while maintaining competitive predictive performance.

Conversely, sparse optimization methods generally required a larger number of support vectors despite relying on only a single kernel.

This observation highlights an important distinction between kernel sparsity and model sparsity.

Reducing the number of active kernels does not necessarily reduce the complexity of the resulting classifier.

Instead, different optimization strategies may shift complexity from the kernel representation to the support vector representation.

---
## 6.6 Computational Efficiency
The benchmark continues to show considerable variation in computational cost, and — with this run's updated implementation — the ranking has reshuffled again.

"Kernel Target Alignment" remains substantially faster than every iterative optimization method (52s), because kernel weights are estimated directly from alignment scores rather than through repeated alternating optimization. This ranking has been stable across every run of this benchmark, since "KTA" barely exercises the iterative "SMO" loop at all.

Among the iterative methods, however, the current ranking is: "FISTA" (1m 23s) < "L2-MKL" (1m 53s) < "Frank-Wolfe" (2m 02s) < "Hybrid L1/L2" (2m 23s) < "Entropic Descent" (2m 30s) < "SimpleMKL" (2m 53s, now the slowest method in the benchmark). Neither the sparse/dense split nor the previous stress-test run's ranking survives this update: "FISTA" was one of the two slowest methods last run and is now the second-fastest overall, while "SimpleMKL" — not a sparse method at all — is now the single slowest.

This is worth stating plainly rather than explaining away: **across the three runs of this benchmark so far (the original capped-SMO configuration, the loosened-tolerance stress test, and this code-updated run), the relative speed ranking of the iterative optimizers has not been stable.** The previous run's hypothesis — that collapsing to a single dominant kernel makes the "SMO" subproblem harder to converge — does not hold up against this run's numbers and should be treated as unsupported going forward rather than carried into future revisions. What *has* stayed stable across all three runs is each method's learned β vector (Table 2) for every optimizer except "KTA" (§6.4) — the kernel-selection findings in §6.2–§6.4 are, on that basis, the more trustworthy output of this benchmark, and timing comparisons between optimizer categories should be read with real caution until they've settled across further runs.

These results still demonstrate that optimization strategy influences computational efficiency far more strongly than classification accuracy (see §8 for application-specific recommendations) — that part has held across all three runs — but *which* strategies are fast or slow, specifically, has not.

---
---
---
# 7. Convergence Analysis
Beyond the final benchmark metrics, the optimization logs provide insight into the convergence behavior of the implemented algorithms.

Rather than evaluating only the final β vector, the evolution of kernel weights throughout training reveals whether an optimization method converges smoothly, oscillates, or prematurely terminates.

---
---
## 7.1 Stability of β Optimization
Across the evaluated iterative optimization methods, the β coefficients evolved in a stable and monotonic manner.

No optimizer exhibited large oscillations or unstable behavior after the initial iterations.

Instead, successive β updates became progressively smaller until convergence was achieved.

This behavior indicates that the implemented optimization framework successfully maintains numerical stability while alternately optimizing the "SVM" parameters and kernel coefficients.

---
---
## 7.2 Optimization Trace Analysis
The recorded optimization logs provide direct evidence of convergence behavior beyond the final benchmark metrics.

For every "MKL" iteration, the following information was collected:
- Current β vector
- SMO iteration count
- SMO execution time
- Number of support vectors
- Final convergence state

This information allows the optimization process to be analyzed as a sequence of kernel updates rather than as a single final result.


### Example:
Three "MKL-SVM" logs (out of seven) with different beta optimization methods are mentioned for better demonstration and understanding. 
All benchmark runs used identical hyperparameters to ensure a fair comparison (`max_bo_iter = 50`, `max_smo_iter = 10**5`, `bo_tolerance = 1e-7`, `smo_tolerance = 1e-3`).

##### "L2-MKL"
```txt
Active Kernels: ['Linear' 'Polynomial' 'RBF' 'Laplacian' 'Rational Quadratic']
MKL.fit() Init Beta Array: [0.2 0.2 0.2 0.2 0.2]

SMO.fit() Elapsed Time Is 0:00:50.005740 On 1601 Iteration With 1462 Support Vectors.
SMO.fit() New b: -0.12780359025305346
MKL.fit() New Beta Array: [0.04955272 0.20333831 0.27701903 0.32864134 0.14144859]

SMO.fit() Elapsed Time Is 0:00:03.601668 On 1 Iteration With 1462 Support Vectors.
SMO.fit() New b: -0.12780359025305346
MKL.fit() New Beta Array: [0.04955272 0.20333831 0.27701903 0.32864134 0.14144859]

SMO.fit() Elapsed Time Is 0:00:52.332690 On 488 Iteration With 1470 Support Vectors.
SMO.fit() New b: -0.27880816458699875
MKL.fit() Elapsed Time Is 0:01:53.491560 On 2 Iteration.

Dataset's Length: 3200
Model's Support Vector Count: 1470
MKL.fit() Final Beta Array: [0.04955272 0.20333831 0.27701903 0.32864134 0.14144859]
```

##### "Entropic Descent"
```txt
Active Kernels: ['Linear' 'Polynomial' 'RBF' 'Laplacian' 'Rational Quadratic']
MKL.fit() Init Beta Array: [0.2 0.2 0.2 0.2 0.2]

SMO.fit() Elapsed Time Is 0:01:42.668951 On 4335 Iteration With 1464 Support Vectors.
SMO.fit() New b: -0.12773026749927519
MKL.fit() New Beta Array: [5.97029907e-06 4.25915349e-03 9.87161833e-02 8.96716641e-01 3.02051665e-04]

SMO.fit() Elapsed Time Is 0:00:06.881922 On 1 Iteration With 1464 Support Vectors.
SMO.fit() New b: -0.12773026749927519
MKL.fit() New Beta Array: [5.97029907e-06 4.25915349e-03 9.87161833e-02 8.96716641e-01 3.02051665e-04]

SMO.fit() Elapsed Time Is 0:00:33.401195 On 533 Iteration With 1539 Support Vectors.
SMO.fit() New b: -0.5223811051387652
MKL.fit() Elapsed Time Is 0:02:30.026356 On 2 Iteration.

Dataset's Length: 3200
Model's Support Vector Count: 1539
MKL.fit() Final Beta Array: [5.97029907e-06 4.25915349e-03 9.87161833e-02 8.96716641e-01 3.02051665e-04]
```

##### "KTA"
```txt
Active Kernels: ['Linear' 'Polynomial' 'RBF' 'Laplacian' 'Rational Quadratic']
MKL.fit() Init Beta Array: [0.2 0.2 0.2 0.2 0.2]

SMO.fit() Elapsed Time Is 0:00:48.519111 On 1941 Iteration With 1453 Support Vectors.
SMO.fit() New b: -0.17407451307283972
MKL.fit() New Beta Array: [0.17302218 0.15621028 0.2551901  0.21179633 0.20378111]

MKL.fit() Elapsed Time Is 0:00:52.398803 On 1 Iteration.

Dataset's Length: 3200
Model's Support Vector Count: 1453
MKL.fit() Final Beta Array: [0.17302218 0.15621028 0.2551901  0.21179633 0.20378111]
```

---
---
### 7.2.1 Outer Optimization Convergence
Across all evaluated optimization strategies, the "MKL" outer loop converged within a small number of iterations.

Convergence now occurs after just two β updates for every iterative method, and after a single alignment computation for "KTA" (Table 3) — down substantially from the six to seven iterations typical of earlier runs of this benchmark.

This is consistent with the earlier-diagnosed fix to the β convergence stopping criterion, which restricted the stationarity check to the active (non-floor-pinned) kernel set rather than computing the alignment spread across every kernel including ones already pinned at the epsilon floor. Under the old criterion, that spread could remain artificially large for several extra iterations even after the active kernels had genuinely settled, forcing the optimizer to keep going past the point of real convergence. A corrected criterion would be expected to detect convergence sooner, which matches what these logs show. This is offered as the most likely explanation given the earlier diagnosis, not a certainty confirmed within this document.

This consistency indicates that the stopping behavior is now primarily determined by the "MKL" convergence criterion rather than by instability in the individual optimization methods.

The reduction of β changes between consecutive iterations demonstrates that every optimizer gradually approached a stable kernel configuration.

---
### 7.2.2 Interaction Between β Updates and SMO Optimization
The logs reveal an important characteristic of alternating "MKL" optimization.

During early iterations, "SMO" required substantially more work to converge under the loosened `smo_tolerance = 1e-3` and the now-uncapped `max_smo_iter` — the "L2-MKL" log in §7.2 needed 1,601 iterations on its very first call, for instance, well beyond what the previous `max_smo_iter = 1000` cap would ever have permitted it to reach.

This behavior is expected because each β update modifies the combined kernel matrix, requiring the SVM solver to adapt to a changing optimization landscape. Removing the hard iteration cap means this cost is now visible directly in the logs rather than being silently truncated.

As β approaches convergence, fewer "SMO" iterations were required.

For example, the final β-update iteration *within* the outer loop frequently required only one "SMO" iteration before reaching the stopping condition — not to be confused with the separate final re-fit performed after the loop exits, which is a substantially larger cost (§7.6).

This indicates that once the kernel combination stabilizes, the corresponding "SVM" solution also becomes stable — and that the expensive iterations are concentrated where they belong, in the early, genuinely unsettled part of the optimization, rather than being an artifact of a cap being hit repeatedly.

---
### 7.2.3 Support Vector Evolution
The number of support vectors was recorded after every "SMO" optimization step.

The experiments showed that support vector counts generally increased during early "MKL" iterations as the kernel mixture became more specialized.

The final "SVM" optimization also changes the support vector count relative to the last intermediate solve — in this run's logs, it *increased* it in both examples shown (§7.2: "L2-MKL" from 1,462 to 1,470; "Entropic Descent" from 1,464 to 1,539, the larger jump of the two). This is direct evidence for the final synchronization step described in §2: the classifier is retrained on the converged kernel mixture rather than inherited from an intermediate one, and that re-fit is not a formality — it can move the support vector count by a non-trivial amount in either direction depending on how the intermediate solve compares to genuine convergence at the loosened `smo_tolerance`.

---
### 7.2.4 Computational Behavior
The optimization logs also provide insight into computational cost.

Every iterative method now converges in two outer "MKL" iterations (§7.2.1), so the differences in total training time seen in Table 1 come almost entirely from how expensive each method's individual "SMO" calls are, not from how many outer iterations it takes.

That per-call cost has not tracked any single property of the optimizer consistently across runs of this benchmark — sparsity, kernel count, and regularization type have each looked predictive at one point and stopped being predictive the next (§6.6). This section previously offered a conditioning-based hypothesis for why sparse methods were slow; that hypothesis does not hold up against this run's numbers and has been withdrawn rather than carried forward. The safest conclusion at this point is that per-call "SMO" cost depends on implementation details this benchmark does not yet isolate, and further hypotheses should wait for the ranking to prove stable across at least one more run before being written up as an explanation rather than an observation.

"KTA" remains the one clear exception: it estimates kernel importance directly rather than repeatedly alternating between β updates and "SVM" optimization, and its timing has been the most stable result in the benchmark across every run so far (§6.6).

---
### 7.2.5 Importance of Optimization History
The benchmark demonstrates that the final β vector alone does not fully describe "MKL" behavior.

Two optimizers may produce similar predictive performance while differing substantially in:
- convergence speed,
- kernel redistribution strategy,
- support vector evolution,
- computational requirements.

Therefore, recording the optimization trajectory provides a more complete understanding of "MKL" behavior than evaluating final classification metrics alone.

The optimization history transforms the benchmark from a simple performance comparison into an analysis of algorithmic behavior.

---
---
## 7.3 Example: Entropic Descent
The updated trajectory described in §6.3 ("Adaptive Kernel Redistribution") is visible directly in the log above: the β vector reaches its "Laplacian"/"RBF" mixture in a single update from the uniform initialization, and the second outer iteration reproduces that same vector exactly. What the log adds beyond that description is the convergence signature — the two consecutive β vectors being numerically indistinguishable is what confirms this is genuine convergence rather than an interrupted trajectory.

---
---
## 7.4 Example: L2-MKL
The direct convergence described in §6.3 ("Distributed Kernel Optimization") shows up in the log above as a single large update from the uniform initialization straight to the final β vector, with the second outer iteration reproducing it exactly — every coefficient stays comfortably away from zero throughout, which is where the smoothing effect of L2 regularization is visible now, rather than in a multi-step trajectory.

---
---
## 7.5 Outer and Inner Optimization
The mismatch between outer and inner convergence rates, noted in §2, is visible directly in the logs, though it now shows up differently than in earlier runs. With convergence occurring in just two outer iterations (§7.2.1), there's no longer a multi-iteration trend to point to; the clearest evidence is the final re-fit itself (§7.6), where "SMO" still needs several hundred iterations to resolve even though β is no longer moving at all — the outer loop has fully converged, but the inner solver has real work left to do.

---
---
## 7.6 Final SVM Optimization
The final SVM re-fit (rationale in §2, evidence in §7.2.3) is not the trivial step this document previously described it as. In this run's logs, it required 488 "SMO" iterations for "L2-MKL" and 533 for "Entropic Descent" — smaller than each method's first, fully cold-started call (1,601 and 4,335 iterations respectively), but far from the single-iteration confirmation pass seen for the *intermediate* β-update checks in §7.2 (block 2 of each log). The final re-fit is evidently not simply re-confirming an already-settled solution; it is doing enough real work to be a meaningful contributor to each method's total training time in Table 1, and should be accounted for as such rather than assumed negligible.

---
---
## 7.7 Discussion
Despite employing mathematically different update rules, every iterative optimizer converged toward a stable kernel distribution without numerical instability or oscillation. This matters methodologically, though with a caveat this document did not need until repeated runs of the benchmark accumulated: it means the *kernel-selection* differences catalogued in §6.2–§6.4 arise from optimization philosophy rather than from implementation artifacts — those results have reproduced consistently across three separate runs of this benchmark (excluding "KTA", §6.4). The *timing* differences catalogued in §6.6 do not carry the same guarantee; they have changed materially with each run's configuration and code changes, and should be read as reflecting this implementation's current state rather than a stable property of each optimizer's underlying philosophy.

---
---
---
# 8. Practical Recommendations

Since accuracy alone doesn't meaningfully separate these methods (§5.1, §6), the recommendations below are organized by the dimensions that do: optimization behavior, computational efficiency, kernel interpretability, and model complexity.

| Objective                       | Recommended Optimizer                              | Rationale                                                                                                                                          |
| ------------------------------- | -------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Highest predictive performance  | **"SimpleMKL"**                                    | Achieved the highest benchmark accuracy while maintaining an adaptive kernel combination — one dominant kernel supported by several secondary ones. |
| Fastest optimization            | **"Kernel Target Alignment "(KTA)**                | Required only a single outer optimization step while producing competitive classification performance and the lowest support vector count.         |
| Balanced kernel utilization     | **L2-MKL**                                         | Distributed weight across every kernel, making it appropriate when multiple similarity measures are expected to contribute meaningful information. |
| Maximum kernel interpretability | **"Hybrid L1/L2"**, **"Frank-Wolfe"**, **"FISTA"** | Produced single-kernel solutions that clearly identify the dominant kernel for the evaluated dataset.                                              |
| Adaptive kernel selection       | **"Entropic Descent"**                             | Gradually refined kernel weights throughout optimization, producing a compromise between sparse and dense kernel mixtures.                         |

These recommendations should not be interpreted as universally optimal choices.

Instead, they reflect the observed behavior of the implemented optimization methods under a common experimental configuration. Different datasets, kernel parameters, or preprocessing strategies may alter the relative advantages of each optimizer.

---
## 8.1 Choosing an Optimizer
From a practical perspective, the benchmark suggests three broad categories of use.

#### Research and Comparative Studies
When the objective is to investigate kernel interactions or compare optimization strategies, **"SimpleMKL"**, **"L2-MKL"**, and **"Entropic Descent"** provide the richest information because they retain multiple active kernels throughout optimization.

These methods allow the evolution of kernel weights to be studied directly, making them particularly suitable for experimental analysis.

---
## 8.2 Computational Efficiency
When computational resources are limited, **"Kernel Target Alignment"** provides an attractive alternative.

The benchmark demonstrated that "KTA" required substantially less training time than iterative optimization methods while maintaining comparable classification performance.

Consequently, "KTA" may serve as an effective baseline or initialization strategy for larger datasets where repeated "MKL" iterations become computationally expensive.

---
## 8.3 Model Interpretation
When identifying the most informative kernel is more important than preserving multiple kernel interactions, sparse optimization methods such as **"Frank-Wolfe"**, **"FISTA"**, and **"Hybrid L1/L2"** provide highly interpretable solutions.

Because these optimizers frequently converge toward a single dominant kernel, they simplify analysis by reducing the final kernel representation to its most influential component.

---
## 8.4 General Recommendation
No single optimization strategy consistently dominated every evaluation criterion.

Instead, the benchmark illustrates an important practical observation:

> **β optimization methods primarily differ in optimization behavior rather than predictive accuracy.**

Therefore, optimizer selection should consider computational efficiency, desired kernel sparsity, convergence characteristics, and interpretability alongside classification performance.

---
---
---
# 9. Limitations

Although the benchmark provides a systematic comparison of several β optimization strategies implemented from scratch, its conclusions should be interpreted within the scope of the conducted experiments.

The following limitations identify aspects intentionally left outside the current implementation and suggest opportunities for future investigation.

---
## 9.1 Dataset Scope
The presented experiments evaluate optimization behavior using several binary classification datasets, with the "MAGIC Gamma Telescope" dataset serving as the primary stress benchmark.

While these datasets cover a range of sample sizes and feature characteristics, they do not represent every class of machine learning problem.

Consequently, the observed optimization behavior should not be assumed to generalize automatically to high-dimensional image datasets, text classification, multi-class learning, or regression tasks.

---
## 9.2 Dimensionality Reduction and Kernel Geometry
"Principal Component Analysis" was applied prior to MKL training (§4.2), reducing the feature space from ten dimensions to seven while retaining 96.1% of total variance. This interacts with the candidate kernels in ways worth stating explicitly rather than leaving implicit.

"PCA" is an orthogonal rotation of the coordinate axes, followed — since components were dropped — by a projection onto a lower-dimensional subspace. The rotation step, on its own, is an *isometry* under the $L_2$ (Euclidean) norm: it cannot change any pairwise Euclidean distance. The projection step, in contrast, can only shrink Euclidean distances, and by construction shrinks them by an amount tied to the 3.9% of variance not retained — a modest, roughly uniform distortion across the dataset, not one specific to any kernel.

The $L_1$ (Manhattan) norm, however, is *not* rotation-invariant. If this implementation's "Laplacian" kernel computes $L_1$ distance (the conventional definition), then the rotation step of "PCA" — independent of any dimensionality dropped — changes Laplacian-kernel distances in a way it does not change "RBF" or "Linear-kernel" distances, simply because the new coordinate axes are no longer aligned with the original features. This is a real, mechanism-level difference between how "PCA" interacts with $L_1$-based kernels versus $L_2$-based ones, and it is not addressed anywhere else in this document.

What this does *not* establish is a proven causal link to the two headline observations — the sub-1% accuracy spread across optimizers (§5.1) and the near-universal "Laplacian" dominance (§6.4). Both are plausible downstream effects of a PCA-reshaped geometry that happens to favor $L_1$-style similarity on this dataset, but confirming that would require an ablation this benchmark does not include: for instance, repeating the comparison without "PCA", or with a dimensionality-reduction method that preserves $L_1$ geometry instead of $L_2$ geometry, and checking whether Laplacian's dominance and the tight accuracy spread persist. Absent that ablation, this section records the mechanism as a documented open question rather than a settled explanation.

This run adds one relevant, if inconclusive, data point. "KTA" computes its weights directly from each kernel's target-alignment score against the same PCA-projected data every other method uses — no iterative "SVM" optimization involved — and it is now the one method whose top kernel is not "Laplacian" (§6.4). If the PCA-geometry hypothesis above were the whole story, a direct, static alignment measurement over the same reshaped feature space would be expected to reflect it too. That it doesn't suggests the observed Laplacian dominance may depend specifically on the iterative "SVM" margin-based gradient signal that six of the seven methods share, rather than being a static property of the post-PCA kernel geometry alone. This narrows, but does not resolve, the open question — it would still benefit from the same ablation suggested above, ideally run against "KTA" as well as the iterative methods.

---
## 9.3 Kernel Hyperparameters
All benchmark experiments employed a common set of kernel hyperparameters throughout the optimization process.

This design choice intentionally isolates the influence of β optimization by preventing simultaneous optimization of kernel-specific parameters.

However, in practical applications, kernel parameter selection and kernel weight optimization often influence one another.

Joint optimization of these quantities remains outside the scope of the current implementation.

---
## 9.4 Computational Scalability
"Multiple Kernel Learning" requires repeated construction and manipulation of kernel matrices.

Although the implementation incorporates several engineering optimizations to reduce computational overhead, the memory requirements of kernel-based methods continue to increase approximately quadratically with the number of training samples.

As a result, the current implementation is primarily intended for medium-scale datasets rather than very large learning problems.

Future extensions incorporating kernel approximations or distributed computation may substantially improve scalability.

---
## 9.5 Optimization Algorithms
The repository currently implements seven representative β optimization strategies covering gradient-based adaptive, sparse, dense, alignment-based, entropy-based, and accelerated first-order optimization methods.

This collection is not exhaustive.

Alternative optimization techniques, including second-order methods, stochastic optimization, Bayesian optimization, and adaptive gradient algorithms, were intentionally omitted to maintain a focused comparison among widely recognized "MKL" optimization approaches.

---
## 9.6 Benchmark Configuration
Every optimizer was evaluated using identical preprocessing, kernel configuration, and training hyperparameters, including a fixed "SVM" penalty ($C = 1$).

While this ensures a fair comparison, it also means that each optimization method operated under the same configuration rather than under individually tuned settings.

Allowing optimizer-specific hyperparameter tuning could potentially alter the relative ranking observed in the benchmark.

The purpose of this study, however, is comparative evaluation under controlled experimental conditions rather than maximizing the absolute performance of individual optimizers.

**On the fixed penalty $C$.** A more specific version of this limitation deserves separate mention. Because $\Sigma_k \beta_k = 1$, every combined kernel is a convex combination of the same five base kernels — but a convex combination does not, in general, guarantee a *constant* matrix norm, since it depends on how the base kernels themselves are scaled. This benchmark's kernels are diagonally normalized to a unit diagonal (§4.3), which directly addresses the sharpest version of this concern: no base kernel can dominate the combined mixture purely because it operates on a larger raw numeric scale (an unbounded "Linear" or "Polynomial" kernel versus a bounded "RBF" or "Laplacian" kernel, for instance). Since every base kernel's diagonal — and therefore its trace — is fixed at the same value regardless of which optimizer is running, a fixed $C$ across optimizers is a substantially more defensible choice than it would be without this normalization.

What diagonal normalization does *not* guarantee is that every base kernel has the same off-diagonal, or spectral, structure. Two kernels can share a unit diagonal while still differing in how quickly similarity decays with distance, which affects the shape (not just the scale) of the resulting Gram matrix and, in principle, the effective margin. This is a materially smaller concern than the one originally raised — the risk of a raw magnitude mismatch is closed — but it is not identical to guaranteeing a fixed effective regularization strength across optimizers. Given that, the earlier suggestion to confirm this with a discrete follow-up (e.g., logging the realized matrix norm or condition number of the combined kernel per optimizer) remains a reasonable, if now lower-priority, item for a future revision.

**On the "KTA" Score metric.** Table 3's "KTA Score" column is, by construction, the exact quantity the KTA optimizer maximizes, so its win there carries little independent evidential weight (see the caveat under Table 3). A more informative addition for a future revision would be a metric that is not the direct objective of any single optimizer — for example, cross-validation stability of the learned β vector across resampled folds, or a theoretical generalization bound on the combined kernel. Either would let all seven methods be judged on common, optimizer-agnostic ground.

---
## 9.7 Implementation Scope
This project focuses on implementing "Multiple Kernel Learning" and "Support Vector Machines" entirely from scratch for educational, experimental, and analytical purposes.

Accordingly, the implementation prioritizes transparency, readability, and reproducibility over highly specialized engineering optimizations found in mature production libraries.

Industrial implementations such as "LIBSVM" and related optimized frameworks employ decades of accumulated engineering improvements, including advanced working-set selection, sophisticated caching strategies, and low-level numerical optimizations.

Replicating these implementation-specific optimizations was not an objective of the present work.

Instead, the goal was to provide a clear and extensible framework for studying "MKL" optimization algorithms while preserving the mathematical structure of the underlying methods and being optimized to run to the extent of determined goal.

---
## 9.8 Concluding Remarks
The limitations described above should not be viewed as shortcomings of the presented framework, but rather as clearly defined boundaries of the current investigation.

By explicitly separating algorithmic evaluation from implementation-specific optimizations and by maintaining identical experimental conditions across all benchmarked methods, the presented comparison aims to emphasize the behavior of β optimization strategies themselves rather than the influence of unrelated experimental variables.

---
---
---