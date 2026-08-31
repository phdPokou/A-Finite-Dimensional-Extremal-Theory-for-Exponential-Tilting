# Score-Gap Constrained Log-Sum-Exp Optimization and Sharp Gibbs Bounds

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
![Reproducibility](https://img.shields.io/badge/Reproducibility-Computational%20Results-success)
![Numerical Verification](https://img.shields.io/badge/Numerical%20Verification-97.2M%20Vectors-blue)
![License](https://img.shields.io/badge/License-See%20Repository-lightgrey)

## Overview

This repository contains the computational material accompanying the paper

**“Score-Gap Constrained Log-Sum-Exp Optimization and Sharp Gibbs Bounds.”**

The paper studies finite-dimensional log-sum-exp optimization over score
vectors subject to a prescribed positive gap between their largest and
second-largest components.

Because the admissible score set is invariant under additive translations
whereas the log-sum-exp functional is translation equivariant, the
unnormalized optimization problem is unbounded. The analysis therefore uses a
translation-normalized formulation.

The paper establishes an exact extremal characterization, identifies the
complete maximizing set, and derives an exact optimality-deficit
representation together with quantitative stability bounds. The resulting
score-space structure yields sharp Gibbs concentration bounds and, through
majorization, sharp bounds for Shannon entropy, Kullback--Leibler divergence,
and effective sample size, with corresponding Rényi and Tsallis entropy
bounds.

The computational material in this repository is used only to verify the
numerical implementation of the proved inequalities. The analytical results
do not depend on numerical experiments.

---

## Mathematical Framework

Let the ordered components of a score vector satisfy

```text
h_(1) >= h_(2) >= ... >= h_(n)
