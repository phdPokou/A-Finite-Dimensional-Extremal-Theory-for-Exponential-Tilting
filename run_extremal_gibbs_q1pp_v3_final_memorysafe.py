#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_extremal_gibbs_q1pp_v3_final_memorysafe.py

Final V3 memory-safe validation platform for
"A Unified Extremal Theory for the Logarithmic Partition Function under Score Separation".

Purpose
-------
This version is designed to be the final journal-facing numerical companion:
- exactly 8 publication-oriented figures;
- exactly 4 compact Q1++ tables;
- one CSV file behind every figure;
- memory-safe streaming for all empirical simulations;
- figures can be regenerated from CSV files without rerunning simulations.

Recommended commands
--------------------
Smoke test:
    python run_extremal_gibbs_q1pp_v3_final_memorysafe.py --profile smoke

Full final run:
    python run_extremal_gibbs_q1pp_v3_final_memorysafe.py --profile q1pp --seeds 30 --num-random 3000

Regenerate figures only from existing CSV files:
    python run_extremal_gibbs_q1pp_v3_final_memorysafe.py --figures-only --output-dir Results_Extremal_Gibbs_Q1PP_V3_Final

Outputs
-------
Results_Extremal_Gibbs_Q1PP_V3_Final/
    figures/       PNG, PDF, SVG
    figure_data/   CSV inputs for each figure
    tables/        CSV and LaTeX compact tables
    logs/          JSON runtime manifests
    certificates/  reproducibility certificate
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    profile: str = "medium"
    output_dir: str = "Results_Extremal_Gibbs_Q1PP_V3_Final"
    seeds: int = 10
    num_random: int = 1000
    n_grid: Tuple[int, ...] = (5, 10, 25, 50, 100)
    eta_grid: Tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 1.0)
    delta_grid: Tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 1.0, 2.0)
    generators: Tuple[str, ...] = ("boundary", "exponential", "laplace", "student", "two_scale", "uniform")
    lambda_points: int = 700
    phase_n_points: int = 100
    phase_rho_points: int = 320
    asymptotic_points: int = 420
    empirical_bins: int = 26
    n_reference: int = 25
    empirical_sample_size: int = 80000
    empirical_bin_sample_size: int = 2500
    skip_figures: bool = False
    figures_only: bool = False


def apply_profile(cfg: Config) -> Config:
    if cfg.profile == "smoke":
        cfg.seeds = min(cfg.seeds, 2)
        cfg.num_random = min(cfg.num_random, 100)
        cfg.n_grid = (5, 10, 25)
        cfg.eta_grid = (0.10, 0.50)
        cfg.delta_grid = (0.10, 1.0)
        cfg.generators = ("boundary", "exponential", "student")
        cfg.lambda_points = 160
        cfg.phase_n_points = 25
        cfg.phase_rho_points = 90
        cfg.asymptotic_points = 140
        cfg.empirical_bins = 12
        cfg.empirical_sample_size = min(cfg.empirical_sample_size, 8000)
        cfg.empirical_bin_sample_size = min(cfg.empirical_bin_sample_size, 350)
    elif cfg.profile == "medium":
        pass
    elif cfg.profile == "q1pp":
        cfg.seeds = max(cfg.seeds, 30)
        cfg.num_random = max(cfg.num_random, 3000)
        cfg.n_grid = (5, 10, 25, 50, 100, 250)
        cfg.lambda_points = max(cfg.lambda_points, 1200)
        cfg.phase_n_points = max(cfg.phase_n_points, 150)
        cfg.phase_rho_points = max(cfg.phase_rho_points, 500)
        cfg.asymptotic_points = max(cfg.asymptotic_points, 700)
        cfg.empirical_bins = max(cfg.empirical_bins, 32)
        cfg.empirical_sample_size = max(cfg.empirical_sample_size, 120000)
        cfg.empirical_bin_sample_size = max(cfg.empirical_bin_sample_size, 3000)
    else:
        raise ValueError("profile must be smoke, medium, or q1pp")
    return cfg


# =============================================================================
# Helpers
# =============================================================================

def make_dirs(root: Path) -> Dict[str, Path]:
    out = {"root": root}
    for name in ["figures", "figure_data", "tables", "certificates", "logs"]:
        p = root / name
        p.mkdir(parents=True, exist_ok=True)
        out[name] = p
    return out


def savefig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=330)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def write_table(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path.with_suffix(".csv"), index=False)
    df.to_latex(path.with_suffix(".tex"), index=False, float_format="%.6g")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_log10(x: np.ndarray | float, floor: float = 1e-300):
    return np.log10(np.clip(x, floor, None))


# =============================================================================
# Mathematical layer
# =============================================================================

def logsumexp(x: np.ndarray) -> float:
    m = float(np.max(x))
    return m + float(np.log(np.sum(np.exp(x - m))))


def Lambda(n: int, eta: float, delta: float) -> float:
    return float((n - 1) * math.exp(-delta / eta))


def q1_star(lam):
    return 1.0 / (1.0 + np.asarray(lam, dtype=float))


def R_star(lam):
    arr = np.asarray(lam, dtype=float)
    return arr / (1.0 + arr)


def H_star(n: int, lam):
    arr = np.asarray(lam, dtype=float)
    q1 = 1.0 / (1.0 + arr)
    r = 1.0 - q1
    out = np.zeros_like(arr, dtype=float)
    mask = r > 0
    out[mask] = -q1[mask] * np.log(q1[mask]) - r[mask] * np.log(r[mask] / (n - 1))
    return float(out) if np.isscalar(lam) else out


def KL_star(n: int, lam):
    return math.log(n) - H_star(n, lam)


def ESS_star(n: int, lam):
    arr = np.asarray(lam, dtype=float)
    q1 = 1.0 / (1.0 + arr)
    r = arr / (1.0 + arr)
    out = 1.0 / (q1 * q1 + (r * r) / (n - 1))
    return float(out) if np.isscalar(lam) else out


def binary_entropy(r):
    arr = np.asarray(r, dtype=float)
    out = np.zeros_like(arr, dtype=float)
    mask = (arr > 0) & (arr < 1)
    out[mask] = -arr[mask] * np.log(arr[mask]) - (1.0 - arr[mask]) * np.log(1.0 - arr[mask])
    return float(out) if np.isscalar(r) else out


def h_extremizer(n: int, delta: float) -> np.ndarray:
    h = np.full(n, -delta, dtype=float)
    h[0] = 0.0
    return h


def distance_to_extremizer(h: np.ndarray, delta: float) -> float:
    hs = np.sort(h)[::-1]
    return float(np.linalg.norm(hs[1:] - hs[0] + delta))


def random_score(n: int, delta: float, eta: float, rng: np.random.Generator, generator: str) -> np.ndarray:
    h = np.empty(n, dtype=float)
    h[0] = 0.0
    h[1] = -delta
    scale = max(delta, eta, 1e-12)
    if n > 2:
        if generator == "boundary":
            extras = rng.uniform(0.0, 1e-7 * scale, size=n - 2)
        elif generator == "exponential":
            extras = rng.exponential(scale=scale, size=n - 2)
        elif generator == "laplace":
            extras = np.abs(rng.laplace(loc=0.0, scale=scale, size=n - 2))
        elif generator == "student":
            extras = np.abs(rng.standard_t(df=3, size=n - 2)) * scale
        elif generator == "two_scale":
            mask = rng.random(n - 2) < 0.55
            extras = np.where(
                mask,
                rng.exponential(scale=1e-3 * scale, size=n - 2),
                rng.exponential(scale=8.0 * scale, size=n - 2),
            )
        elif generator == "uniform":
            extras = rng.uniform(0.0, 4.0 * scale, size=n - 2)
        else:
            raise ValueError(generator)
        h[2:] = -delta - extras
    rng.shuffle(h)
    return h


def gibbs(h: np.ndarray, eta: float) -> np.ndarray:
    z = h / eta
    z -= np.max(z)
    e = np.exp(z)
    return e / e.sum()


def entropy(q: np.ndarray) -> float:
    q = np.clip(q, 1e-300, 1.0)
    return float(-np.sum(q * np.log(q)))


def kl_to_uniform(q: np.ndarray) -> float:
    return float(math.log(len(q)) - entropy(q))


def ess(q: np.ndarray) -> float:
    return float(1.0 / np.sum(q * q))


def vbar(h: np.ndarray, eta: float) -> float:
    return logsumexp(h / eta) - float(np.max(h)) / eta


def stable_rel_slack(slack: float, observed: float, bound: float) -> float:
    return float(slack / (1.0 + abs(observed) + abs(bound)))


def evaluate_case(n: int, eta: float, delta: float, h: np.ndarray, seed: int, generator: str, case: str) -> Dict[str, float]:
    lam = Lambda(n, eta, delta)
    q = gibbs(h, eta)
    qs = np.sort(q)[::-1]
    q1 = float(qs[0])
    R = 1.0 - q1
    H = entropy(q)
    KL = kl_to_uniform(q)
    ESS = ess(q)
    V = vbar(h, eta)
    bounds = {
        "vbar": math.log1p(lam),
        "q1": float(q1_star(lam)),
        "R": float(R_star(lam)),
        "H": float(H_star(n, lam)),
        "KL": math.log(n) - float(H_star(n, lam)),
        "ESS": float(ESS_star(n, lam)),
    }
    obs = {"vbar": V, "q1": q1, "R": R, "H": H, "KL": KL, "ESS": ESS}
    slacks = {
        "vbar_slack": bounds["vbar"] - V,
        "q1_slack": q1 - bounds["q1"],
        "R_slack": bounds["R"] - R,
        "H_slack": bounds["H"] - H,
        "KL_slack": KL - bounds["KL"],
        "ESS_slack": bounds["ESS"] - ESS,
    }
    rel = {
        "vbar_rel_slack": stable_rel_slack(slacks["vbar_slack"], V, bounds["vbar"]),
        "q1_rel_slack": stable_rel_slack(slacks["q1_slack"], q1, bounds["q1"]),
        "R_rel_slack": stable_rel_slack(slacks["R_slack"], R, bounds["R"]),
        "H_rel_slack": stable_rel_slack(slacks["H_slack"], H, bounds["H"]),
        "KL_rel_slack": stable_rel_slack(slacks["KL_slack"], KL, bounds["KL"]),
        "ESS_rel_slack": stable_rel_slack(slacks["ESS_slack"], ESS, bounds["ESS"]),
    }
    universal_score = float(sum(max(x, 0.0) ** 2 for x in rel.values()))
    return {
        "seed": seed, "case": case, "generator": generator,
        "n": n, "eta": eta, "delta": delta, "lambda": lam,
        "distance_to_hstar": distance_to_extremizer(h, delta),
        **{f"{k}_obs": v for k, v in obs.items()},
        **{f"{k}_bound": v for k, v in bounds.items()},
        **slacks, **rel,
        "universal_score": universal_score,
    }


# =============================================================================
# Data generation
# =============================================================================

def make_theory_grid(cfg: Config, dirs: Dict[str, Path]) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for n in cfg.n_grid:
        for lam in np.linspace(1e-10, n - 1, cfg.lambda_points):
            R = float(R_star(lam))
            H = float(H_star(n, lam))
            KL = math.log(n) - H
            ESSv = float(ESS_star(n, lam))
            rows.append({
                "n": n, "lambda": lam, "rho": lam / (n - 1),
                "vbar": math.log1p(lam), "q1": float(q1_star(lam)), "R": R,
                "H": H, "KL": KL, "ESS": ESSv,
                "H_over_logn": H / math.log(n),
                "KL_over_logn": KL / math.log(n),
                "ESS_over_n": ESSv / n,
                "entropy_binary_part": H - R * math.log(n - 1),
                "binary_entropy_R": float(binary_entropy(R)),
            })
    df = pd.DataFrame(rows)
    df.to_csv(dirs["figure_data"] / "v3_theory_grid.csv", index=False)
    return df


def make_phase_grid(cfg: Config, dirs: Dict[str, Path]) -> pd.DataFrame:
    n_values = np.unique(np.round(np.exp(np.linspace(math.log(3), math.log(1000), cfg.phase_n_points))).astype(int))
    rho_values = np.linspace(1e-5, 1.0, cfg.phase_rho_points)
    rows: List[Dict[str, float]] = []
    for n in n_values:
        for rho in rho_values:
            lam = rho * (n - 1)
            Hn = float(H_star(int(n), lam)) / math.log(n)
            rows.append({
                "n": int(n), "rho": rho, "lambda": lam,
                "q1": float(q1_star(lam)),
                "H_over_logn": Hn,
                "ESS_over_n": float(ESS_star(int(n), lam)) / n,
            })
    df = pd.DataFrame(rows)
    df.to_csv(dirs["figure_data"] / "v3_fig01_phase_diagram.csv", index=False)
    return df


def make_asymptotic_grid(cfg: Config, dirs: Dict[str, Path]) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for n in cfg.n_grid:
        for lam in np.logspace(-9, -2, cfg.asymptotic_points):
            H = float(H_star(n, lam))
            R = float(R_star(lam))
            ESSv = float(ESS_star(n, lam))
            rows.append({
                "n": n, "lambda": lam,
                "c_vbar": (math.log1p(lam) - lam) / (lam ** 2),
                "c_R": (lam - R) / (lam ** 2),
                "c_ESS": (ESSv - 1.0) / lam,
                "c_H": H / (lam * math.log((n - 1) / lam)),
                "relerr_vbar": abs(math.log1p(lam) - lam) / max(abs(math.log1p(lam)), 1e-300),
                "relerr_R": abs(R - lam) / max(abs(R), 1e-300),
                "relerr_H": abs(H - lam * math.log((n - 1) / lam)) / max(abs(H), 1e-300),
                "relerr_ESS": abs(ESSv - (1.0 + 2.0 * lam)) / max(abs(ESSv), 1e-300),
            })
    df = pd.DataFrame(rows)
    df.to_csv(dirs["figure_data"] / "v3_fig06_small_lambda_constants.csv", index=False)
    return df


class RunningStats:
    def __init__(self) -> None:
        self.count = 0
        self.sum = 0.0
        self.min = math.inf
        self.max = -math.inf
        self.viol_1e9 = 0
        self.viol_1e12 = 0

    def update(self, x: float) -> None:
        self.count += 1
        self.sum += float(x)
        self.min = min(self.min, float(x))
        self.max = max(self.max, float(x))
        self.viol_1e9 += int(x < -1e-9)
        self.viol_1e12 += int(x < -1e-12)

    def mean(self) -> float:
        return self.sum / max(self.count, 1)


def make_empirical_data(cfg: Config, dirs: Dict[str, Path]) -> pd.DataFrame:
    """Memory-safe empirical generation with bounded reservoir samples.

    The full random experiment is streamed. Only a bounded sample is stored for
    figure-level point clouds, while exact counts and extrema are accumulated for
    tables.
    """
    t0 = time.time()
    rng_reservoir = np.random.default_rng(24681357)
    sample_rows: List[Dict[str, float]] = []
    total_rows = 0
    total_random_rows = 0

    metrics = ["vbar_rel_slack", "q1_rel_slack", "R_rel_slack", "H_rel_slack", "KL_rel_slack", "ESS_rel_slack", "universal_score"]
    rel_metric_for_theorem = {
        "Theorem 4": "vbar_rel_slack",
        "Theorem 5": "q1_rel_slack",
        "Corollary 5": "R_rel_slack",
        "Theorem 6-H": "H_rel_slack",
        "Theorem 6-KL": "KL_rel_slack",
        "Theorem 6-ESS": "ESS_rel_slack",
    }
    gen_stats: Dict[str, Dict[str, RunningStats]] = {}
    thm_stats: Dict[str, RunningStats] = {name: RunningStats() for name in rel_metric_for_theorem}
    sat_stats: Dict[str, Dict[str, Any]] = {}

    log_edges = np.linspace(-16.0, 3.0, cfg.empirical_bins + 1)
    bin_reservoirs: Dict[Tuple[str, int], Dict[str, Any]] = {}

    def reservoir_add(row: Dict[str, float]) -> None:
        nonlocal total_rows, sample_rows
        total_rows += 1
        if len(sample_rows) < cfg.empirical_sample_size:
            sample_rows.append(dict(row))
        else:
            j = int(rng_reservoir.integers(0, total_rows))
            if j < cfg.empirical_sample_size:
                sample_rows[j] = dict(row)

    def fixed_bin_add(key: Tuple[str, int], value: float) -> None:
        if key not in bin_reservoirs:
            bin_reservoirs[key] = {"count": 0, "values": []}
        obj = bin_reservoirs[key]
        obj["count"] = int(obj["count"]) + 1
        vals: List[float] = obj["values"]
        if len(vals) < cfg.empirical_bin_sample_size:
            vals.append(float(value))
        else:
            j = int(rng_reservoir.integers(0, int(obj["count"])))
            if j < cfg.empirical_bin_sample_size:
                vals[j] = float(value)

    def update_saturation(row: Dict[str, float]) -> None:
        case = str(row["case"])
        if case not in sat_stats:
            sat_stats[case] = {"count": 0, "score": RunningStats(), "distance": RunningStats()}
        sat_stats[case]["count"] += 1
        sat_stats[case]["score"].update(float(row["universal_score"]))
        sat_stats[case]["distance"].update(float(row["distance_to_hstar"]))

    def update_tables(row: Dict[str, float]) -> None:
        nonlocal total_random_rows
        update_saturation(row)
        for name, metric in rel_metric_for_theorem.items():
            thm_stats[name].update(float(row[metric]))
        if str(row["case"]) != "random":
            return
        total_random_rows += 1
        gen = str(row["generator"])
        if gen not in gen_stats:
            gen_stats[gen] = {m: RunningStats() for m in metrics}
        for m in metrics:
            gen_stats[gen][m].update(float(row[m]))
        logd = math.log10(max(float(row["distance_to_hstar"]), 1e-16))
        idx = int(np.searchsorted(log_edges, logd, side="right") - 1)
        idx = max(0, min(cfg.empirical_bins - 1, idx))
        y = math.log10(max(float(row["universal_score"]), 1e-300))
        fixed_bin_add((gen, idx), y)

    for seed in range(cfg.seeds):
        rng = np.random.default_rng(10000 + seed)
        for n in cfg.n_grid:
            for eta in cfg.eta_grid:
                for delta in cfg.delta_grid:
                    base = h_extremizer(n, delta)
                    row = evaluate_case(n, eta, delta, base, seed, "extremizer", "extremizer")
                    reservoir_add(row); update_tables(row)

                    for eps in np.logspace(-14, 0, 25):
                        h = base.copy()
                        if n > 2:
                            h[2:] -= np.linspace(0.0, eps, n - 2)
                        row = evaluate_case(n, eta, delta, h, seed, "path", "near_extremizer")
                        reservoir_add(row); update_tables(row)

                    for gen in cfg.generators:
                        for _ in range(cfg.num_random):
                            h = random_score(n, delta, eta, rng, gen)
                            row = evaluate_case(n, eta, delta, h, seed, gen, "random")
                            reservoir_add(row); update_tables(row)
        print(f"streaming seed {seed + 1}/{cfg.seeds}; rows={total_rows:,}; saved={len(sample_rows):,}", flush=True)

    df = pd.DataFrame(sample_rows)
    df.to_csv(dirs["figure_data"] / "v3_empirical_reservoir_sample.csv", index=False)

    band_rows: List[Dict[str, float]] = []
    for (gen, idx), obj in sorted(bin_reservoirs.items()):
        vals = np.asarray(obj["values"], dtype=float)
        if len(vals) < 5:
            continue
        left, right = log_edges[idx], log_edges[idx + 1]
        band_rows.append({
            "generator": gen,
            "bin": idx,
            "log_distance_left": float(left),
            "log_distance_right": float(right),
            "distance_mid": float(10 ** ((left + right) / 2.0)),
            "median_log10_score": float(np.median(vals)),
            "q25_log10_score": float(np.quantile(vals, 0.25)),
            "q75_log10_score": float(np.quantile(vals, 0.75)),
            "count_exact_bin": int(obj["count"]),
            "count_sample_bin": int(len(vals)),
        })
    pd.DataFrame(band_rows).to_csv(dirs["figure_data"] / "v3_fig07_generator_universality.csv", index=False)

    # Compact Table 2: generator certificate.
    gen_rows: List[Dict[str, Any]] = []
    for gen, st in sorted(gen_stats.items()):
        max_viol = max(int(st[m].viol_1e9) for m in metrics)
        gen_rows.append({
            "generator": gen,
            "tests": int(next(iter(st.values())).count),
            "mean_universal_score": st["universal_score"].mean(),
            "min_universal_score": st["universal_score"].min,
            "max_universal_score": st["universal_score"].max,
            "worst_rel_slack_min": min(st[m].min for m in metrics if m != "universal_score"),
            "violations_tol_1e_minus_9": max_viol,
            "status": "PASS" if max_viol == 0 else "CHECK",
        })
    write_table(pd.DataFrame(gen_rows), dirs["tables"] / "Table02_GeneratorCertificate")

    # Compact Table 4: theorem verification.
    thm_rows: List[Dict[str, Any]] = []
    for theorem, st in thm_stats.items():
        thm_rows.append({
            "theorem": theorem,
            "validated_quantity": rel_metric_for_theorem[theorem],
            "tests": int(st.count),
            "mean_relative_slack": st.mean(),
            "minimum_relative_slack": st.min,
            "violations_tol_1e_minus_9": int(st.viol_1e9),
            "violations_tol_1e_minus_12": int(st.viol_1e12),
            "status": "PASS" if st.viol_1e9 == 0 else "CHECK",
        })
    write_table(pd.DataFrame(thm_rows), dirs["tables"] / "Table04_TheoremVerification")

    # Compact saturation table used for Figure 4.
    sat_rows: List[Dict[str, Any]] = []
    for case, st in sorted(sat_stats.items()):
        sat_rows.append({
            "class": case,
            "tests": int(st["count"]),
            "mean_distance_to_hstar": st["distance"].mean(),
            "min_distance_to_hstar": st["distance"].min,
            "max_distance_to_hstar": st["distance"].max,
            "mean_universal_score": st["score"].mean(),
            "min_universal_score": st["score"].min,
            "max_universal_score": st["score"].max,
        })
    pd.DataFrame(sat_rows).to_csv(dirs["figure_data"] / "v3_fig04_saturation_gap_summary.csv", index=False)

    manifest = {
        "mode": "memory_safe_streaming",
        "rows_streamed": int(total_rows),
        "random_rows_streamed": int(total_random_rows),
        "reservoir_sample_rows_saved": int(len(df)),
        "empirical_sample_size": int(cfg.empirical_sample_size),
        "empirical_bin_sample_size": int(cfg.empirical_bin_sample_size),
        "note": "Full empirical rows are streamed and not materialized; figure CSV files contain bounded samples and exact compact tables.",
    }
    (dirs["logs"] / "empirical_streaming_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (dirs["logs"] / "runtime_empirical.json").write_text(json.dumps({"seconds": time.time() - t0, **manifest}, indent=2), encoding="utf-8")
    return df


# =============================================================================
# Tables
# =============================================================================

def make_compact_tables(cfg: Config, dirs: Dict[str, Path]) -> None:
    asym = pd.read_csv(dirs["figure_data"] / "v3_fig06_small_lambda_constants.csv")
    stable = asym[(asym["lambda"] >= 1e-7) & (asym["lambda"] <= 1e-4)]
    rows = []
    for quantity, exact, col in [
        (r"$(\log(1+\Lambda)-\Lambda)/\Lambda^2$", 0.5, "c_vbar"),
        (r"$(\Lambda-R^*)/\Lambda^2$", 1.0, "c_R"),
        (r"$(\mathrm{ESS}^*-1)/\Lambda$", 2.0, "c_ESS"),
        (r"$H^*/[\Lambda\log((n-1)/\Lambda)]$", 1.0, "c_H"),
    ]:
        est = float(stable[col].median())
        rows.append({
            "quantity": quantity,
            "theoretical_constant": exact,
            "median_estimate": est,
            "absolute_error": abs(est - exact),
            "relative_error": abs(est - exact) / max(abs(exact), 1e-300),
        })
    write_table(pd.DataFrame(rows), dirs["tables"] / "Table01_UniversalConstants")

    # Table 3: structural regimes.
    n = cfg.n_reference
    regimes = [
        ("concentrated", 1e-4, "dominant state; low entropy; ESS close to one"),
        ("transition", 1.0, "intermediate mass transfer between dominant and residual states"),
        ("diffuse", n - 1.0, "near-uniform extremal Gibbs measure"),
    ]
    reg_rows = []
    for name, lam, desc in regimes:
        H = float(H_star(n, lam)); KL = math.log(n) - H; ESSv = float(ESS_star(n, lam))
        reg_rows.append({
            "regime": name,
            "lambda": lam,
            "q1_star": float(q1_star(lam)),
            "R_star": float(R_star(lam)),
            "H_star_over_log_n": H / math.log(n),
            "KL_star_over_log_n": KL / math.log(n),
            "ESS_star_over_n": ESSv / n,
            "geometric_interpretation": desc,
        })
    write_table(pd.DataFrame(reg_rows), dirs["tables"] / "Table03_StructuralRegimes")


# =============================================================================
# Figure CSV preparation and figures
# =============================================================================

def prepare_figure_csvs(cfg: Config, dirs: Dict[str, Path]) -> None:
    theory = pd.read_csv(dirs["figure_data"] / "v3_theory_grid.csv")
    emp = pd.read_csv(dirs["figure_data"] / "v3_empirical_reservoir_sample.csv")
    asym = pd.read_csv(dirs["figure_data"] / "v3_fig06_small_lambda_constants.csv")
    # Fig 02: exact Gibbs/information path.
    path = theory[theory["n"] == cfg.n_reference].copy()
    path[["lambda", "q1", "R", "H_over_logn", "KL_over_logn", "ESS_over_n"]].to_csv(
        dirs["figure_data"] / "v3_fig02_exact_gibbs_information_path.csv", index=False
    )
    # Fig 03: information frontier.
    ran = emp[(emp["case"] == "random") & (emp["n"] == cfg.n_reference)].copy()
    if len(ran) > 8000:
        ran = ran.sample(8000, random_state=123)
    fig3_random = pd.DataFrame({
        "type": "random",
        "H_over_logn": ran["H_obs"].to_numpy() / math.log(cfg.n_reference),
        "ESS_over_n": ran["ESS_obs"].to_numpy() / cfg.n_reference,
    })
    fig3_path = pd.DataFrame({
        "type": "boundary",
        "H_over_logn": path["H_over_logn"].to_numpy(),
        "ESS_over_n": path["ESS_over_n"].to_numpy(),
    })
    pd.concat([fig3_random, fig3_path], ignore_index=True).to_csv(dirs["figure_data"] / "v3_fig03_information_frontier.csv", index=False)
    # Fig 04: saturation gap from reservoir.
    sat = emp[["case", "distance_to_hstar", "universal_score"]].copy()
    sat["log10_universal_score"] = safe_log10(sat["universal_score"].to_numpy())
    sat.to_csv(dirs["figure_data"] / "v3_fig04_saturation_gap.csv", index=False)
    # Fig 05: structural collapse.
    theory[["n", "R", "entropy_binary_part", "binary_entropy_R"]].to_csv(dirs["figure_data"] / "v3_fig05_entropy_decomposition_collapse.csv", index=False)
    # Fig 06 is already prepared; copy stable subset for figure.
    small = asym[(asym["n"] == cfg.n_reference) & (asym["lambda"] >= 1e-7) & (asym["lambda"] <= 1e-2)].copy()
    small.to_csv(dirs["figure_data"] / "v3_fig06_small_lambda_constants_plot.csv", index=False)
    # Fig 08: theorem dashboard from compact table.
    thm = pd.read_csv(dirs["tables"] / "Table04_TheoremVerification.csv")
    thm.to_csv(dirs["figure_data"] / "v3_fig08_theorem_dashboard.csv", index=False)


def make_figures(cfg: Config, dirs: Dict[str, Path]) -> None:
    F = dirs["figures"]

    # Fig 01. Phase diagram.
    phase = pd.read_csv(dirs["figure_data"] / "v3_fig01_phase_diagram.csv")
    pivot = phase.pivot_table(index="n", columns="rho", values="H_over_logn")
    q_piv = phase.pivot_table(index="n", columns="rho", values="q1")
    X, Y = np.meshgrid(q_piv.columns.values, q_piv.index.values)
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    im = ax.imshow(pivot.values, aspect="auto", origin="lower", extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()])
    c1 = ax.contour(X, Y, q_piv.values, levels=[0.2, 0.5, 0.8], linewidths=1.1)
    c2 = ax.contour(X, Y, pivot.values, levels=[0.5, 0.9], linestyles="dashed", linewidths=1.1)
    ax.clabel(c1, inline=True, fontsize=7, fmt={0.8:r"$q_1=0.8$",0.5:r"$q_1=0.5$",0.2:r"$q_1=0.2$"})
    ax.clabel(c2, inline=True, fontsize=7, fmt={0.5:r"$H/\log n=0.5$",0.9:r"$H/\log n=0.9$"})
    ax.set_yscale("log")
    ax.set_xlabel(r"Normalized structural parameter $\rho=\Lambda/(n-1)$")
    ax.set_ylabel(r"Dimension $n$")
    ax.set_title(r"Universal Phase Diagram Generated by $\Lambda$")
    fig.colorbar(im, ax=ax, label=r"Normalized entropy $H^*/\log n$")
    savefig(fig, F / "Fig01_UniversalPhaseDiagram")

    # Fig 02. Exact Gibbs/information path.
    path = pd.read_csv(dirs["figure_data"] / "v3_fig02_exact_gibbs_information_path.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    ax.plot(path["lambda"], path["q1"], linewidth=2.2, label=r"$q_{(1)}^*$")
    ax.plot(path["lambda"], path["R"], linewidth=2.2, label=r"$R^*$")
    ax.plot(path["lambda"], path["H_over_logn"], linewidth=2.2, label=r"$H^*/\log n$")
    ax.plot(path["lambda"], path["KL_over_logn"], linewidth=2.2, label=r"$D_{\mathrm{KL}}^*/\log n$")
    ax.plot(path["lambda"], path["ESS_over_n"], linewidth=2.2, label=r"$\mathrm{ESS}^*/n$")
    ax.set_xlabel(r"Structural parameter $\Lambda$")
    ax.set_ylabel("Normalized value")
    ax.set_title(r"Exact Gibbs--Information Path Generated by $\Lambda$")
    ax.legend(fontsize=8, ncol=2)
    savefig(fig, F / "Fig02_ExactGibbsInformationPath")

    # Fig 03. Information frontier.
    f3 = pd.read_csv(dirs["figure_data"] / "v3_fig03_information_frontier.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    rnd = f3[f3["type"] == "random"]
    bnd = f3[f3["type"] == "boundary"]
    ax.scatter(rnd["H_over_logn"], rnd["ESS_over_n"], s=8, alpha=0.12, label="random admissible scores")
    ax.plot(bnd["H_over_logn"], bnd["ESS_over_n"], linewidth=2.8, label="extremal Gibbs boundary")
    ax.set_xlabel(r"Normalized entropy $H/\log n$")
    ax.set_ylabel(r"Normalized effective sample size $\mathrm{ESS}/n$")
    ax.set_title(r"Extremal Information Frontier in the $(H,\mathrm{ESS})$ Plane")
    ax.legend(fontsize=8)
    savefig(fig, F / "Fig03_InformationFrontier")

    # Fig 04. Saturation gap.
    sat = pd.read_csv(dirs["figure_data"] / "v3_fig04_saturation_gap.csv")
    order = ["extremizer", "near_extremizer", "random"]
    data = [sat[sat["case"] == c]["log10_universal_score"].to_numpy(dtype=float) for c in order]
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.boxplot(data, labels=order, showfliers=False)
    for i, vals in enumerate(data, start=1):
        if len(vals):
            ax.text(i, np.nanmedian(vals), f"median={np.nanmedian(vals):.1f}\nN={len(vals):,}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel(r"$\log_{10} S_{\mathrm{univ}}$")
    ax.set_title("Universal Saturation Gap")
    savefig(fig, F / "Fig04_SaturationGap")

    # Fig 05. Structural entropy collapse.
    f5 = pd.read_csv(dirs["figure_data"] / "v3_fig05_entropy_decomposition_collapse.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    r_grid = np.linspace(1e-6, 1 - 1e-6, 500)
    ax.plot(r_grid, binary_entropy(r_grid), linewidth=2.8, label=r"$h_{\mathrm{bin}}(R)$")
    for n, g in f5.groupby("n"):
        ax.scatter(g["R"], g["entropy_binary_part"], s=6, alpha=0.22, label=rf"$n={int(n)}$")
    ax.set_xlabel(r"Residual mass $R^*$")
    ax.set_ylabel(r"$H^*-R^*\log(n-1)$")
    ax.set_title("Exact Finite-Size Entropy Decomposition Collapse")
    ax.legend(fontsize=7, ncol=2)
    savefig(fig, F / "Fig05_StructuralEntropyCollapse")

    # Fig 06. Universal constants.
    small = pd.read_csv(dirs["figure_data"] / "v3_fig06_small_lambda_constants_plot.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    ax.plot(small["lambda"], small["c_vbar"], linewidth=2.0, label=r"$(\log(1+\Lambda)-\Lambda)/\Lambda^2 \to 1/2$")
    ax.plot(small["lambda"], small["c_R"], linewidth=2.0, label=r"$(\Lambda-R^*)/\Lambda^2 \to 1$")
    ax.plot(small["lambda"], small["c_ESS"], linewidth=2.0, label=r"$(\mathrm{ESS}^*-1)/\Lambda \to 2$")
    ax.plot(small["lambda"], small["c_H"], linewidth=2.0, label=r"$H^*/[\Lambda\log((n-1)/\Lambda)] \to 1$")
    for y in [0.5, 1.0, 2.0]:
        ax.axhline(y, linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\Lambda$")
    ax.set_ylabel("Normalized asymptotic constant")
    ax.set_title(r"Universal Small-$\Lambda$ Constants")
    ax.legend(fontsize=7)
    savefig(fig, F / "Fig06_UniversalConstants")

    # Fig 07. Generator universality.
    bands = pd.read_csv(dirs["figure_data"] / "v3_fig07_generator_universality.csv")
    fig, ax = plt.subplots(figsize=(7.3, 4.7))
    for gen, g in bands.groupby("generator"):
        ax.plot(g["distance_mid"], g["median_log10_score"], marker="o", linewidth=1.7, label=gen)
        ax.fill_between(g["distance_mid"], g["q25_log10_score"], g["q75_log10_score"], alpha=0.14)
    ax.set_xscale("log")
    ax.set_xlabel(r"Distance to $h^*$")
    ax.set_ylabel(r"$\log_{10} S_{\mathrm{univ}}$")
    ax.set_title("Generator Universality of the Saturation Certificate")
    ax.legend(fontsize=7, ncol=2)
    savefig(fig, F / "Fig07_GeneratorUniversality")

    # Fig 08. Theorem dashboard.
    thm = pd.read_csv(dirs["figure_data"] / "v3_fig08_theorem_dashboard.csv")
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    y = np.clip(thm["minimum_relative_slack"].to_numpy(dtype=float), 1e-16, None)
    x = np.arange(len(thm))
    ax.bar(x, y)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(thm["theorem"], rotation=25, ha="right")
    ax.set_ylabel("Minimum relative slack")
    ax.set_title("Complete Theorem Verification")
    for i, row in thm.iterrows():
        ax.text(i, y[i] * 1.2, str(row["status"]), ha="center", va="bottom", fontsize=8)
    savefig(fig, F / "Fig08_TheoremVerification")


# =============================================================================
# Certificate and CLI
# =============================================================================

def write_certificate(cfg: Config, dirs: Dict[str, Path]) -> None:
    cert = {
        "script": Path(__file__).name,
        "sha256": sha256(Path(__file__)),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": plt.matplotlib.__version__,
        "config": asdict(cfg),
        "generated_at_unix": time.time(),
    }
    (dirs["certificates"] / "V3_reproducibility_certificate.json").write_text(json.dumps(cert, indent=2, default=str), encoding="utf-8")


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Final V3 memory-safe validation for extremal Gibbs theory.")
    p.add_argument("--profile", choices=["smoke", "medium", "q1pp"], default="medium")
    p.add_argument("--output-dir", default="Results_Extremal_Gibbs_Q1PP_V3_Final")
    p.add_argument("--seeds", type=int, default=None)
    p.add_argument("--num-random", type=int, default=None)
    p.add_argument("--skip-figures", action="store_true")
    p.add_argument("--figures-only", action="store_true", help="Regenerate figures from existing CSV files without rerunning simulations.")
    args = p.parse_args()
    cfg = Config(profile=args.profile, output_dir=args.output_dir)
    cfg = apply_profile(cfg)
    if args.seeds is not None:
        cfg.seeds = int(args.seeds)
    if args.num_random is not None:
        cfg.num_random = int(args.num_random)
    cfg.skip_figures = bool(args.skip_figures)
    cfg.figures_only = bool(args.figures_only)
    return cfg


def main() -> None:
    start = time.time()
    cfg = parse_args()
    dirs = make_dirs(Path(cfg.output_dir))
    print("Configuration:")
    print(json.dumps(asdict(cfg), indent=2, default=str))

    if cfg.figures_only:
        make_figures(cfg, dirs)
        print(f"Figures regenerated from existing CSV files in: {dirs['root'].resolve()}")
        return

    make_theory_grid(cfg, dirs)
    make_phase_grid(cfg, dirs)
    make_asymptotic_grid(cfg, dirs)
    make_empirical_data(cfg, dirs)
    make_compact_tables(cfg, dirs)
    prepare_figure_csvs(cfg, dirs)

    if not cfg.skip_figures:
        make_figures(cfg, dirs)

    write_certificate(cfg, dirs)
    (dirs["logs"] / "runtime_total.json").write_text(json.dumps({"seconds": time.time() - start}, indent=2), encoding="utf-8")
    print(f"Done. Results saved in: {dirs['root'].resolve()}")


if __name__ == "__main__":
    main()
