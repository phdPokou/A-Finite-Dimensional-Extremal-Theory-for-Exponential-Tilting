# Score-Gap Constrained Log-Sum-Exp Optimization and Sharp Gibbs Bounds

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
![Reproducibility](https://img.shields.io/badge/Reproducibility-Computational%20Results-success)
![Numerical Verification](https://img.shields.io/badge/Numerical%20Verification-97.2M%20Vectors-blue)

## Overview

This repository contains the computational material accompanying the paper

**“Score-Gap Constrained Log-Sum-Exp Optimization and Sharp Gibbs Bounds.”**

The paper studies finite-dimensional log-sum-exp optimization over score
vectors subject to a prescribed positive gap between their largest and
second-largest ordered components.

The admissible score set is invariant under additive translations, whereas
the log-sum-exp functional is translation equivariant. Consequently, the
unnormalized optimization problem is unbounded, and the analysis is formulated
using a translation-normalized objective.

The paper establishes an exact extremal characterization, determines the
complete set of maximizing score vectors, and derives an exact representation
of the optimality deficit together with quantitative stability bounds. The
resulting score-space structure yields sharp Gibbs concentration bounds and,
through a majorization relation, sharp bounds for Shannon entropy,
Kullback--Leibler divergence, and effective sample size, with corresponding
Rényi and Tsallis entropy bounds.

The numerical calculations contained in this repository are used to verify
the implementation of the principal proved inequalities over a prescribed
finite computational design. They do not enter the analytical proofs.

---

## Mathematical Framework

Let $h_{(1)} >= h_{(2)} >= ... >= h_{(n)}$
denote the ordered components of a finite-dimensional score vector. The
admissible set is defined by the prescribed positive gap
$h_{(1)} - h_{(2)} = \Delta > 0.$

The analysis considers the scaled log-sum-exp functional under this order
statistic constraint. Because additive translations preserve the score gap
but shift the log-sum-exp objective, the finite extremal problem is formulated
after removing this translation degree of freedom.

The principal analytical results are:

- an exact extremal value for the translation-normalized objective;
- a complete characterization of the maximizing score vectors;
- uniqueness of the maximizing configuration up to additive translations and
  coordinate permutations;
- an exact representation of the optimality deficit;
- two-sided quantitative stability bounds;
- an equivalence between near optimality and collapse of the lower-score
  departures toward the maximizing configuration;
- sharp bounds for the dominant and residual Gibbs masses;
- a majorization relation for every admissible Gibbs vector;
- sharp Shannon entropy, Kullback--Leibler divergence, and effective
  sample-size bounds;
- corresponding Rényi and Tsallis entropy bounds;
- an exact monotone parametrization of the extremal Gibbs and information
  quantities.

Up to additive translations and coordinate permutations, the maximizing score
configuration has two levels: one maximal component and all remaining
components exactly one prescribed score gap below it.

---

## Repository Structure

The main computational script is

```text
run_extremal_gibbs_q1pp_v3_final_memorysafe.py
```

It implements the finite numerical verification associated with the sharp
inequalities studied in the paper.

The `memorysafe` implementation is intended for the large final computational
design and avoids the need to retain the complete collection of generated
score vectors simultaneously in memory.

The final numerical experiment is controlled through three command-line
arguments:

```text
--profile q1pp
--seeds 30
--num-random 3000
```

These options specify the full computational profile, the number of
distinct random seeds, and the number of random score vectors generated per
configuration.

---

## Computational Requirements

The experiments require a working Python 3 installation together with the
Python packages imported by the main script.

Check the installed Python version with

```bash
python --version
```

or, on systems where Python 3 is exposed separately,

```bash
python3 --version
```

A dedicated virtual environment is recommended.

### Windows

Create the environment with

```bash
python -m venv .venv
```

and activate it with

```bash
.venv\Scripts\activate
```

### Linux / macOS

Create the environment with

```bash
python3 -m venv .venv
```

and activate it with

```bash
source .venv/bin/activate
```

Install the Python dependencies required by the imports in
`run_extremal_gibbs_q1pp_v3_final_memorysafe.py` before executing the
experiment.

For exact computational reproducibility, the Python and package versions used
for the reported final run should be retained with the repository whenever
available.

---

## Reproducing the Numerical Experiments

Clone the repository:

```bash
git clone https://github.com/phdPokou/A-Finite-Dimensional-Extremal-Theory-for-Exponential-Tilting.git
```

Enter the repository:

```bash
cd A-Finite-Dimensional-Extremal-Theory-for-Exponential-Tilting
```

Activate the Python environment containing the required dependencies.

### Full Final Run

The numerical experiment reported in the final computational design is
reproduced with:

```bash
python run_extremal_gibbs_q1pp_v3_final_memorysafe.py --profile q1pp --seeds 30 --num-random 3000
```

This is the reference command for reproducing the full numerical verification.

The arguments have the following roles:

| Argument | Final value | Role |
|---|---:|---|
| `--profile` | `q1pp` | Selects the full  computational profile |
| `--seeds` | `30` | Uses 30 distinct random seeds |
| `--num-random` | `3000` | Generates 3000 random admissible vectors per configuration |

Depending on the local Python installation, the same command may need to be
executed with `python3`:

```bash
python3 run_extremal_gibbs_q1pp_v3_final_memorysafe.py --profile q1pp --seeds 30 --num-random 3000
```

---

## Final Computational Design

The full numerical verification reported in the paper covers the following
parameter grid:

| Component | Specification |
|---|---|
| Dimension | `5, 10, 25, 50, 100, 250` |
| Temperature | `0.05, 0.10, 0.25, 0.50, 1.00` |
| Score gap | `0.05, 0.10, 0.25, 0.50, 1.00, 2.00` |
| Score generators | Boundary, Exponential, Laplace, Student t, Two scale, Uniform |
| Distinct random seeds | `30` |
| Vectors per configuration | `3000` |
| Violation tolerance | `1e-12` |

The corresponding Cartesian design contains

```text
6 × 5 × 6 × 6 × 3000 × 30 = 97,200,000
```

evaluated admissible score vectors.

The large number of evaluated vectors is a finite numerical verification
design. It is not a substitute for the analytical proofs.

---

## Inequalities Verified Numerically

The computational experiment evaluates six principal sharp inequalities
established analytically in the paper:

1. the upper bound for the normalized log-sum-exp objective;
2. the lower bound for the dominant Gibbs probability;
3. the upper bound for the residual Gibbs mass;
4. the upper bound for Shannon entropy;
5. the lower bound for Kullback--Leibler divergence from the uniform
   distribution;
6. the upper bound for effective sample size.

All benchmark quantities are evaluated from the corresponding closed-form
analytical expressions.

The numerical experiment therefore compares admissible generated score
vectors against theoretical quantities that have already been established
analytically.

---

## Numerical Violation Criterion

Each inequality is evaluated in its proved direction using a normalized signed
slack.

For an analytical upper bound, the signed slack is constructed so that a
nonnegative value is consistent with the proved inequality. The same
orientation is used for analytical lower bounds.

A numerical violation is recorded only when

```text
signed normalized slack < -1e-12.
```

Thus, the numerical violation tolerance is

```text
1e-12.
```

Small negative residuals between `-1e-12` and zero are retained in the
numerical diagnostics but are not classified as violations.

This tolerance is exclusively a numerical decision rule. It has no role in
the analytical statements or proofs.

---

## Numerical Verification Results

For the final computational design, the numerical verification covers

```text
97,200,000 admissible score vectors.
```

The reported minimum signed normalized slacks are:

| Quantity | Minimum signed slack | Mean absolute slack | Violations |
|---|---:|---:|---:|
| Normalized objective | `-9.313e-16` | `9.021e-02` | `0` |
| Dominant Gibbs probability | `-2.961e-16` | `4.299e-02` | `0` |
| Residual Gibbs mass | `-9.127e-16` | `3.681e-02` | `0` |
| Shannon entropy | `-4.078e-15` | `7.408e-02` | `0` |
| Kullback--Leibler divergence | `-3.485e-15` | `1.614e-01` | `0` |
| Effective sample size | `-7.075e-16` | `1.714e-01` | `0` |

No evaluated configuration is classified as violating any of the six
inequalities at the prescribed tolerance of `1e-12`.

The small negative minimum signed slacks have magnitudes consistent with
floating-point numerical effects relative to the prescribed tolerance. The
nonzero mean absolute slacks reflect that sampled configurations need not lie
on the analytical equality set.

These results are numerical consistency checks of the implementation only.
Validity, sharpness, equality characterization, and quantitative stability
follow from the analytical results of the paper.

---

## Generated Figures and Tables

The paper distinguishes two types of computational output.

### Analytical Figures

Figures describing the extremal Gibbs path, the cross-dimensional
representation, and the quantitative stability envelopes are obtained by
direct evaluation of closed-form expressions established analytically.

These figures do not rely on statistical estimation or simulation to establish
the corresponding mathematical results.

### Numerical Verification

The numerical verification figure and table summarize the signed-slack
diagnostics obtained from the finite computational experiment.

In particular, the verification output reports:

- the absolute value of the minimum signed normalized slack for visualization;
- the mean absolute normalized slack;
- the prescribed numerical tolerance;
- the number of evaluated admissible score vectors;
- the number of classified violations.

Violation classification is always based on the original signed slack, not on
the absolute value displayed on a logarithmic axis.

---

## Reproducibility

The final computational experiment is specified by the reference command

```bash
python run_extremal_gibbs_q1pp_v3_final_memorysafe.py --profile q1pp --seeds 30 --num-random 3000
```

and uses:

- a fixed  computational profile;
- explicitly specified dimensions;
- explicitly specified temperature values;
- explicitly specified score-gap values;
- six score-generation mechanisms;
- 30 distinct random seeds;
- 3000 generated vectors per configuration;
- double-precision numerical calculations;
- a signed-slack violation tolerance of `1e-12`;
- closed-form analytical benchmarks.

The full reported design evaluates `97,200,000` admissible score vectors.

The distinction between analytical proof and numerical verification is
essential:

> **The numerical experiment checks the computational implementation of the
> proved inequalities. It is not used to establish their validity, sharpness,
> equality cases, or stability properties.**

This separation is maintained throughout the accompanying paper.

---

## Recommended Reproducibility Workflow

For an independent reproduction of the reported numerical experiment:

1. Clone the repository.
2. Record the local Python version.
3. Create and activate an isolated Python environment.
4. Install the dependencies imported by the main script.
5. Execute the full final run:

```bash
python run_extremal_gibbs_q1pp_v3_final_memorysafe.py --profile q1pp --seeds 30 --num-random 3000
```

6. Inspect the generated numerical summaries and figures.
7. Verify that violation counts are computed from the signed normalized slacks
   using the `1e-12` tolerance.
8. Compare the resulting summary statistics with those reported in the paper.

Because the final experiment is intentionally large, its execution time and
memory consumption depend on the local hardware and software environment.

---

## Paper and Code Correspondence

The repository is intended to make the computational component of the paper
auditable.

The correspondence is:

| Paper component | Computational role |
|---|---|
| Sharp extremal characterization | Closed-form benchmark |
| Exact optimality deficit | Analytical result |
| Quantitative stability bounds | Closed-form analytical envelopes |
| Gibbs concentration bounds | Closed-form benchmark |
| Majorization relation | Analytical result |
| Shannon entropy bound | Numerically checked |
| Kullback--Leibler bound | Numerically checked |
| Effective sample-size bound | Numerically checked |
| Extremal parameter paths | Direct analytical evaluation |
| Numerical verification | Finite signed-slack experiment |

The code should therefore be interpreted as a reproducibility companion to the
analytical paper rather than as evidence on which the mathematical results
depend.

---


## Reproducibility Statement

The analytical results in the accompanying paper are proved independently of
the numerical experiments. The repository provides the computational material
used to evaluate the principal sharp inequalities over the prescribed finite
design and to reproduce the corresponding numerical diagnostics.

The reference full-run command is:

```bash
python run_extremal_gibbs_q1pp_v3_final_memorysafe.py --profile q1pp --seeds 30 --num-random 3000
```
