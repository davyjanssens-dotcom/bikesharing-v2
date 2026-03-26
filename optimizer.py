"""
Maximal Coverage Location Problem (MCLP) solvers.

Greedy:  O(p * n_candidates) – fast, ~(1-1/e) ≈ 63 % of optimal.
ILP:     Exact via PuLP + CBC solver.
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class OptimizationResult:
    selected_sites: List[int]           # indices into candidates dataframe
    covered_demand_ids: List[int]       # indices into demand dataframe
    total_demand_covered: float
    total_demand: float
    coverage_pct: float
    n_demand_covered: int
    n_demand_total: int
    runtime_seconds: float
    method: str
    coverage_by_step: List[float]       # cumulative coverage % as stations added


# ---------------------------------------------------------------------------
# Greedy solver
# ---------------------------------------------------------------------------

def solve_greedy(
    demand_weights: np.ndarray,
    coverage_matrix: np.ndarray,
    n_stations: int,
) -> OptimizationResult:
    """Greedy maximum weighted coverage."""
    t0 = time.perf_counter()
    n_demand, n_sites = coverage_matrix.shape
    covered = np.zeros(n_demand, dtype=bool)
    selected: List[int] = []
    coverage_by_step: List[float] = []
    total_demand = demand_weights.sum()

    for _ in range(n_stations):
        best_site, best_gain = -1, -1.0
        for j in range(n_sites):
            if j in selected:
                continue
            new_covered = coverage_matrix[:, j].astype(bool) & ~covered
            gain = demand_weights[new_covered].sum()
            if gain > best_gain:
                best_gain = gain
                best_site = j
        if best_site == -1 or best_gain == 0:
            break
        selected.append(best_site)
        covered |= coverage_matrix[:, best_site].astype(bool)
        coverage_by_step.append(float(demand_weights[covered].sum() / total_demand * 100))

    covered_ids = list(np.where(covered)[0])
    total_covered = float(demand_weights[covered].sum())

    return OptimizationResult(
        selected_sites=selected,
        covered_demand_ids=covered_ids,
        total_demand_covered=total_covered,
        total_demand=float(total_demand),
        coverage_pct=total_covered / total_demand * 100,
        n_demand_covered=int(covered.sum()),
        n_demand_total=n_demand,
        runtime_seconds=time.perf_counter() - t0,
        method="Greedy",
        coverage_by_step=coverage_by_step,
    )


# ---------------------------------------------------------------------------
# ILP solver (PuLP + CBC)
# ---------------------------------------------------------------------------

def solve_ilp(
    demand_weights: np.ndarray,
    coverage_matrix: np.ndarray,
    n_stations: int,
    time_limit: int = 60,
) -> OptimizationResult:
    """Exact MCLP via Integer Linear Programming."""
    try:
        import pulp
    except ImportError:
        raise ImportError("PuLP is required for the ILP solver. Run: pip install pulp")

    t0 = time.perf_counter()
    n_demand, n_sites = coverage_matrix.shape
    total_demand = demand_weights.sum()

    prob = pulp.LpProblem("MCLP", pulp.LpMaximize)

    y = [pulp.LpVariable(f"y_{j}", cat="Binary") for j in range(n_sites)]
    z = [pulp.LpVariable(f"z_{i}", cat="Binary") for i in range(n_demand)]

    # Objective: maximise weighted covered demand
    prob += pulp.lpSum(demand_weights[i] * z[i] for i in range(n_demand))

    # Budget constraint
    prob += pulp.lpSum(y) <= n_stations

    # Coverage constraint: demand i covered only if at least one covering site open
    for i in range(n_demand):
        covering_sites = np.where(coverage_matrix[i] == 1)[0]
        if len(covering_sites) == 0:
            prob += z[i] == 0
        else:
            prob += pulp.lpSum(y[j] for j in covering_sites) >= z[i]

    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit)
    prob.solve(solver)

    selected = [j for j in range(n_sites) if pulp.value(y[j]) and pulp.value(y[j]) > 0.5]
    covered_mask = np.zeros(n_demand, dtype=bool)
    for j in selected:
        covered_mask |= coverage_matrix[:, j].astype(bool)

    # Build step-by-step coverage (greedy order of selected sites for the chart)
    coverage_by_step: List[float] = []
    incremental_covered = np.zeros(n_demand, dtype=bool)
    remaining = list(selected)
    while remaining:
        best = max(remaining, key=lambda j: demand_weights[coverage_matrix[:, j].astype(bool) & ~incremental_covered].sum())
        remaining.remove(best)
        incremental_covered |= coverage_matrix[:, best].astype(bool)
        coverage_by_step.append(float(demand_weights[incremental_covered].sum() / total_demand * 100))

    covered_ids = list(np.where(covered_mask)[0])
    total_covered = float(demand_weights[covered_mask].sum())

    return OptimizationResult(
        selected_sites=selected,
        covered_demand_ids=covered_ids,
        total_demand_covered=total_covered,
        total_demand=float(total_demand),
        coverage_pct=total_covered / total_demand * 100,
        n_demand_covered=int(covered_mask.sum()),
        n_demand_total=n_demand,
        runtime_seconds=time.perf_counter() - t0,
        method="ILP (Exact)",
        coverage_by_step=coverage_by_step,
    )


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

def sensitivity_analysis(
    demand_weights: np.ndarray,
    coverage_matrix: np.ndarray,
    max_stations: int,
    method: str = "greedy",
) -> pd.DataFrame:
    """Run optimization for p = 1..max_stations and return coverage curve."""
    rows = []
    for p in range(1, max_stations + 1):
        if method == "ilp":
            res = solve_ilp(demand_weights, coverage_matrix, p, time_limit=30)
        else:
            res = solve_greedy(demand_weights, coverage_matrix, p)
        rows.append({
            "n_stations": p,
            "coverage_pct": round(res.coverage_pct, 2),
            "demand_covered": round(res.total_demand_covered, 2),
            "n_points_covered": res.n_demand_covered,
        })
    return pd.DataFrame(rows)
