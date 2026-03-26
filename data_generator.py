import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class CityConfig:
    width: float = 12.0        # km (Brussels region width)
    height: float = 10.0       # km (Brussels region height)
    n_demand_points: int = 400
    n_candidates: int = 180
    seed: int = 42
    city_name: str = "Brussels"
    
    # Brussels coordinate reference (southwest corner)
    lat_origin: float = 50.810  # Latitude of origin (south)
    lon_origin: float = 4.290   # Longitude of origin (west)
    km_per_deg_lat: float = 111.0  # Approx km per degree latitude
    km_per_deg_lon: float = 73.0   # Approx km per degree longitude at Brussels latitude

    # POI cluster centres [x, y, weight_multiplier, spread]
    # Real Brussels locations mapped to local km grid
    poi_clusters: List[Tuple] = field(default_factory=lambda: [
        (6.0, 5.5, 3.5, 0.8),   # Grand Place / Pentagon (CBD)
        (7.5, 6.0, 3.0, 0.9),   # European Quarter (EU institutions)
        (5.0, 6.5, 2.2, 0.7),   # Gare du Midi (South Station)
        (6.5, 7.0, 2.0, 0.6),   # Gare Centrale (Central Station)
        (8.5, 6.5, 1.9, 0.7),   # Parc du Cinquantenaire / Schuman
        (3.5, 5.0, 1.8, 0.8),   # Anderlecht (residential)
        (9.5, 5.5, 1.7, 0.7),   # Etterbeek / ULB campus
        (5.5, 8.0, 1.8, 0.6),   # Gare du Nord (North Station)
        (4.0, 7.5, 1.6, 0.7),   # Molenbeek (residential)
        (8.0, 4.0, 1.5, 0.6),   # Ixelles / Flagey
        (7.0, 3.5, 1.4, 0.5),   # Uccle (residential south)
        (10.5, 6.0, 1.6, 0.6),  # Woluwe / Montgomery
        (2.5, 6.0, 1.3, 0.5),   # Koekelberg
        (6.0, 9.0, 1.5, 0.5),   # Schaerbeek (north residential)
    ])


def km_to_latlon(x_km: float, y_km: float, cfg: CityConfig) -> tuple[float, float]:
    """Convert local km coordinates to lat/lon."""
    lat = cfg.lat_origin + (y_km / cfg.km_per_deg_lat)
    lon = cfg.lon_origin + (x_km / cfg.km_per_deg_lon)
    return lat, lon


def generate_demand_points(cfg: CityConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    points = []

    per_cluster = cfg.n_demand_points // len(cfg.poi_clusters)
    remainder = cfg.n_demand_points - per_cluster * len(cfg.poi_clusters)

    for idx, (cx, cy, weight_mult, spread) in enumerate(cfg.poi_clusters):
        n = per_cluster + (1 if idx < remainder else 0)
        xs = rng.normal(cx, spread, n).clip(0.1, cfg.width - 0.1)
        ys = rng.normal(cy, spread, n).clip(0.1, cfg.height - 0.1)
        base_demand = rng.uniform(0.5, 1.5, n) * weight_mult
        for x, y, d in zip(xs, ys, base_demand):
            points.append({"x": round(x, 4), "y": round(y, 4), "demand": round(d, 3)})

    df = pd.DataFrame(points).reset_index(drop=True)
    df.index.name = "demand_id"
    return df


def generate_candidate_sites(cfg: CityConfig) -> pd.DataFrame:
    """Candidate station locations on a slightly perturbed grid + near POIs."""
    rng = np.random.default_rng(cfg.seed + 99)

    # Regular grid candidates
    side = int(np.ceil(np.sqrt(cfg.n_candidates * 0.6)))
    xs_g = np.linspace(0.5, cfg.width - 0.5, side)
    ys_g = np.linspace(0.5, cfg.height - 0.5, side)
    grid_pts = [(x + rng.uniform(-0.3, 0.3), y + rng.uniform(-0.3, 0.3))
                for x in xs_g for y in ys_g]

    # POI-adjacent candidates
    poi_pts = []
    for cx, cy, _, _ in cfg.poi_clusters:
        n_near = max(2, cfg.n_candidates // (len(cfg.poi_clusters) * 2))
        for _ in range(n_near):
            poi_pts.append((
                float(np.clip(cx + rng.uniform(-0.6, 0.6), 0.1, cfg.width - 0.1)),
                float(np.clip(cy + rng.uniform(-0.6, 0.6), 0.1, cfg.height - 0.1)),
            ))

    all_pts = grid_pts + poi_pts
    rng.shuffle(all_pts)
    all_pts = all_pts[:cfg.n_candidates]

    df = pd.DataFrame(all_pts, columns=["x", "y"])
    df["x"] = df["x"].clip(0.1, cfg.width - 0.1).round(4)
    df["y"] = df["y"].clip(0.1, cfg.height - 0.1).round(4)
    df.index.name = "site_id"
    return df.reset_index(drop=True)


def compute_coverage_matrix(
    demand: pd.DataFrame,
    candidates: pd.DataFrame,
    radius: float,
) -> np.ndarray:
    """Binary matrix [n_demand x n_candidates]: 1 if site j covers demand i."""
    dx = demand["x"].values[:, None] - candidates["x"].values[None, :]
    dy = demand["y"].values[:, None] - candidates["y"].values[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    return (dist <= radius).astype(np.int8)
