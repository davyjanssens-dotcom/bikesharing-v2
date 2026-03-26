# 🚲 Bike-Sharing Station Optimizer

An interactive tool for solving the **Maximal Coverage Location Problem (MCLP)** — placing *p* bike-sharing stations to maximize coverage of potential demand.

## Problem Statement

Given:
- A set of **demand points** (potential users, weighted by proximity to POIs)
- A set of **candidate station locations**
- A **coverage radius** *r* (a station covers all demand within *r* km)
- A **budget** of *p* stations

**Objective:** Select *p* candidate sites to maximize the total weighted demand covered.

## Algorithms

| Method | Guarantee | Speed |
|---|---|---|
| **Greedy** | ≥ (1−1/e) ≈ 63% of optimal | Very fast |
| **ILP (Exact)** | Global optimum | Slower for large instances |

The ILP formulation:

```
Maximize:  Σ d_i · z_i
Subject to:
  Σ_{j ∈ N_i} y_j ≥ z_i    ∀i  (covered only if a station is nearby)
  Σ_j y_j ≤ p               (station budget)
  y_j, z_i ∈ {0,1}
```

Where `N_i` is the set of candidate sites within radius *r* of demand point *i*.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Features

- **Interactive coverage map** — demand points coloured by coverage status, stations shown as stars, coverage circles rendered
- **Incremental coverage chart** — marginal gain of each added station
- **Sensitivity analysis** — coverage % vs. number of stations curve
- **Station details table** — ranked by marginal contribution
- Synthetic city with realistic demand clusters (CBD, university, transit hubs, residential areas)
