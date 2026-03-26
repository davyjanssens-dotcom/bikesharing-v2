from typing import Optional

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from data_generator import CityConfig, generate_demand_points, generate_candidate_sites, compute_coverage_matrix, km_to_latlon
from optimizer import solve_greedy, solve_ilp, sensitivity_analysis, OptimizationResult

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Bike-Sharing Station Optimizer",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1a73e8;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #5f6368;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #e8f0fe, #f8f9fa);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #1a73e8;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #202124;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #1a73e8 !important;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Sidebar – parameters
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚲 Configuration")
    st.markdown("---")

    st.markdown("### City Parameters")
    n_demand = st.slider("Demand points", 100, 600, 400, 50,
                         help="Number of demand locations across Brussels neighborhoods")
    n_candidates = st.slider("Candidate sites", 50, 250, 180, 10,
                              help="Number of potential station locations")
    seed = st.number_input("Random seed", 0, 9999, 42, 1)

    st.markdown("---")
    st.markdown("### Optimization Parameters")
    n_stations = st.slider("Number of stations (p)", 1, 30, 10, 1,
                            help="Budget: how many stations to place")
    coverage_radius = st.slider("Coverage radius (km)", 0.3, 2.5, 0.8, 0.1,
                                 help="A station covers all demand within this radius")
    method = st.radio("Solver", ["Greedy (fast)", "ILP – Exact"],
                      help="Greedy ≈63% of optimal, ILP is exact but slower")

    st.markdown("---")
    st.markdown("### Sensitivity Analysis")
    run_sensitivity = st.checkbox("Show coverage vs. # stations", value=True)
    max_p_sens = st.slider("Max stations for analysis", 5, 30, 20, 1)

    st.markdown("---")
    run_btn = st.button("▶  Run Optimization", type="primary", use_container_width=True)


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown('<div class="main-header">🚲 Bike-Sharing Station Optimizer — Brussels</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Maximal Coverage Location Problem (MCLP) — place <em>p</em> stations to cover the most potential demand across Brussels neighborhoods</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Session state cache
# ──────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None
if "demand_df" not in st.session_state:
    st.session_state.demand_df = None
if "cand_df" not in st.session_state:
    st.session_state.cand_df = None
if "cov_matrix" not in st.session_state:
    st.session_state.cov_matrix = None
if "sens_df" not in st.session_state:
    st.session_state.sens_df = None
if "last_params" not in st.session_state:
    st.session_state.last_params = {}

current_params = dict(
    n_demand=n_demand, n_candidates=n_candidates, seed=seed,
    n_stations=n_stations, coverage_radius=coverage_radius,
    method=method, run_sensitivity=run_sensitivity, max_p_sens=max_p_sens,
)

# Auto-run on first load
if st.session_state.result is None or run_btn:
    st.session_state.last_params = current_params

    with st.spinner("Generating city data…"):
        cfg = CityConfig(n_demand_points=n_demand, n_candidates=n_candidates, seed=seed)
        demand_df = generate_demand_points(cfg)
        cand_df = generate_candidate_sites(cfg)
        cov_matrix = compute_coverage_matrix(demand_df, cand_df, coverage_radius)

    st.session_state.demand_df = demand_df
    st.session_state.cand_df = cand_df
    st.session_state.cov_matrix = cov_matrix

    solver_label = "ILP" if "ILP" in method else "greedy"
    with st.spinner(f"Running {solver_label} optimizer…"):
        if "ILP" in method:
            result = solve_ilp(demand_df["demand"].values, cov_matrix, n_stations)
        else:
            result = solve_greedy(demand_df["demand"].values, cov_matrix, n_stations)
    st.session_state.result = result

    if run_sensitivity:
        with st.spinner("Running sensitivity analysis…"):
            sens_df = sensitivity_analysis(
                demand_df["demand"].values, cov_matrix, max_p_sens,
                method="ilp" if "ILP" in method else "greedy",
            )
        st.session_state.sens_df = sens_df
    else:
        st.session_state.sens_df = None

result: OptimizationResult = st.session_state.result
demand_df: pd.DataFrame = st.session_state.demand_df
cand_df: pd.DataFrame = st.session_state.cand_df
cov_matrix: np.ndarray = st.session_state.cov_matrix
sens_df: Optional[pd.DataFrame] = st.session_state.sens_df

# ──────────────────────────────────────────────
# KPI row
# ──────────────────────────────────────────────
st.markdown("---")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Stations placed", f"{len(result.selected_sites)}")
k2.metric("Demand points covered", f"{result.n_demand_covered:,} / {result.n_demand_total:,}")
k3.metric("Coverage", f"{result.coverage_pct:.1f}%")
k4.metric("Weighted demand covered", f"{result.total_demand_covered:.1f}")
k5.metric("Solver runtime", f"{result.runtime_seconds:.2f}s  ({result.method})")

st.markdown("---")

# ──────────────────────────────────────────────
# Main layout: map | charts
# ──────────────────────────────────────────────
col_map, col_charts = st.columns([3, 2], gap="large")

# ── MAP ──────────────────────────────────────
with col_map:
    st.markdown('<div class="section-title">📍 Coverage Map — Brussels</div>', unsafe_allow_html=True)

    selected_set = set(result.selected_sites)
    covered_set = set(result.covered_demand_ids)

    # Convert km to lat/lon for all points
    cfg_map = CityConfig()
    demand_df["covered"] = demand_df.index.isin(covered_set)
    demand_df["lat"] = demand_df["y"].apply(lambda y: km_to_latlon(0, y, cfg_map)[0])
    demand_df["lon"] = demand_df["x"].apply(lambda x: km_to_latlon(x, 0, cfg_map)[1])
    
    cand_df["lat"] = cand_df["y"].apply(lambda y: km_to_latlon(0, y, cfg_map)[0])
    cand_df["lon"] = cand_df["x"].apply(lambda x: km_to_latlon(x, 0, cfg_map)[1])

    fig_map = go.Figure()

    # Uncovered demand
    unc = demand_df[~demand_df["covered"]]
    if len(unc) > 0:
        fig_map.add_trace(go.Scattermapbox(
            lat=unc["lat"], lon=unc["lon"],
            mode="markers",
            name="Uncovered demand",
            marker=dict(size=8, color="#ea4335", opacity=0.7),
            hovertemplate="<b>Demand:</b> %{customdata:.2f}<extra>Uncovered</extra>",
            customdata=unc["demand"],
        ))

    # Covered demand
    cov = demand_df[demand_df["covered"]]
    if len(cov) > 0:
        fig_map.add_trace(go.Scattermapbox(
            lat=cov["lat"], lon=cov["lon"],
            mode="markers",
            name="Covered demand",
            marker=dict(size=8, color="#34a853", opacity=0.75),
            hovertemplate="<b>Demand:</b> %{customdata:.2f}<extra>Covered</extra>",
            customdata=cov["demand"],
        ))

    # Candidate sites (not selected)
    non_sel = cand_df[~cand_df.index.isin(selected_set)]
    if len(non_sel) > 0:
        fig_map.add_trace(go.Scattermapbox(
            lat=non_sel["lat"], lon=non_sel["lon"],
            mode="markers",
            name="Candidate site",
            marker=dict(size=10, color="#fbbc04", opacity=0.6),
            hovertemplate="<b>Candidate site</b><extra></extra>",
        ))

    # Selected stations
    sel = cand_df[cand_df.index.isin(selected_set)]
    if len(sel) > 0:
        fig_map.add_trace(go.Scattermapbox(
            lat=sel["lat"], lon=sel["lon"],
            mode="markers+text",
            name="Selected station",
            marker=dict(size=18, color="#1a73e8", opacity=1.0),
            text=["★"] * len(sel),
            textfont=dict(size=14, color="white"),
            hovertemplate="<b>Station %{customdata}</b><br>Lat: %{lat:.4f}, Lon: %{lon:.4f}<extra></extra>",
            customdata=[i+1 for i in range(len(sel))],
        ))

    # Center map on Brussels
    center_lat = cfg_map.lat_origin + (cfg_map.height / 2 / cfg_map.km_per_deg_lat)
    center_lon = cfg_map.lon_origin + (cfg_map.width / 2 / cfg_map.km_per_deg_lon)

    fig_map.update_layout(
        height=520,
        margin=dict(l=0, r=0, t=0, b=0),
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=11.5,
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=0.02, xanchor="left", x=0.02,
            font=dict(size=11), bgcolor="rgba(255,255,255,0.9)"
        ),
        showlegend=True,
    )
    st.plotly_chart(fig_map, use_container_width=True)

# ── CHARTS ────────────────────────────────────
with col_charts:
    # -- Incremental coverage bar
    st.markdown('<div class="section-title">📈 Incremental Coverage per Station</div>', unsafe_allow_html=True)
    steps = result.coverage_by_step
    prev = [0.0] + steps[:-1]
    increments = [s - p for s, p in zip(steps, prev)]
    bar_colors = px.colors.sequential.Blues[3:]
    bar_colors = (bar_colors * (len(increments) // len(bar_colors) + 1))[:len(increments)]

    fig_inc = go.Figure(go.Bar(
        x=[f"S{i+1}" for i in range(len(increments))],
        y=increments,
        marker_color=bar_colors,
        hovertemplate="Station %{x}<br>+%{y:.1f}%<extra></extra>",
    ))
    fig_inc.update_layout(
        height=210,
        margin=dict(l=10, r=10, t=10, b=30),
        yaxis_title="Added coverage (%)",
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="white",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#e0e0e0"),
    )
    st.plotly_chart(fig_inc, use_container_width=True)

    # -- Sensitivity analysis
    if sens_df is not None:
        st.markdown('<div class="section-title">🔍 Coverage vs. Number of Stations</div>', unsafe_allow_html=True)

        fig_sens = go.Figure()
        fig_sens.add_trace(go.Scatter(
            x=sens_df["n_stations"],
            y=sens_df["coverage_pct"],
            mode="lines+markers",
            line=dict(color="#1a73e8", width=2.5),
            marker=dict(size=7, color="#1a73e8"),
            hovertemplate="p=%{x}<br>Coverage=%{y:.1f}%<extra></extra>",
            fill="tozeroy",
            fillcolor="rgba(26,115,232,0.1)",
        ))
        # Mark current selection
        if n_stations <= max_p_sens:
            row = sens_df[sens_df["n_stations"] == n_stations]
            if not row.empty:
                fig_sens.add_trace(go.Scatter(
                    x=row["n_stations"],
                    y=row["coverage_pct"],
                    mode="markers",
                    marker=dict(size=13, color="#ea4335", symbol="diamond"),
                    name=f"Current p={n_stations}",
                    hovertemplate=f"p={n_stations}: %{{y:.1f}}%<extra></extra>",
                ))

        fig_sens.update_layout(
            height=210,
            margin=dict(l=10, r=10, t=10, b=30),
            xaxis_title="Number of stations (p)",
            yaxis_title="Coverage (%)",
            plot_bgcolor="#f8f9fa",
            paper_bgcolor="white",
            showlegend=False,
            yaxis=dict(range=[0, 105], showgrid=True, gridcolor="#e0e0e0"),
            xaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig_sens, use_container_width=True)

    # -- Demand distribution pie
    st.markdown('<div class="section-title">🥧 Demand Coverage Breakdown</div>', unsafe_allow_html=True)
    covered_w = result.total_demand_covered
    uncovered_w = result.total_demand - covered_w
    fig_pie = go.Figure(go.Pie(
        labels=["Covered", "Uncovered"],
        values=[covered_w, uncovered_w],
        marker_colors=["#34a853", "#ea4335"],
        hole=0.55,
        textinfo="percent+label",
        hovertemplate="%{label}: %{value:.1f} demand units<extra></extra>",
    ))
    fig_pie.update_layout(
        height=180,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        paper_bgcolor="white",
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# ──────────────────────────────────────────────
# Station details table
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-title">📋 Selected Station Details</div>', unsafe_allow_html=True)

rows = []
covered_so_far = np.zeros(len(demand_df), dtype=bool)
# Order by greedy marginal gain for display
remaining = list(result.selected_sites)
ordered = []
while remaining:
    best = max(remaining, key=lambda j: demand_df["demand"].values[
        cov_matrix[:, j].astype(bool) & ~covered_so_far].sum())
    remaining.remove(best)
    ordered.append(best)
    new_cov = cov_matrix[:, best].astype(bool) & ~covered_so_far
    covered_so_far |= cov_matrix[:, best].astype(bool)

covered_so_far2 = np.zeros(len(demand_df), dtype=bool)
for rank, j in enumerate(ordered, 1):
    new_cov = cov_matrix[:, j].astype(bool) & ~covered_so_far2
    gain = demand_df["demand"].values[new_cov].sum()
    covered_so_far2 |= cov_matrix[:, j].astype(bool)
    rows.append({
        "Rank": rank,
        "Site ID": j,
        "Location (x, y)": f"({cand_df.loc[j,'x']:.2f}, {cand_df.loc[j,'y']:.2f})",
        "Points in radius": int(cov_matrix[:, j].sum()),
        "New demand added": round(gain, 2),
        "Cumulative coverage %": round(demand_df["demand"].values[covered_so_far2].sum() /
                                       demand_df["demand"].values.sum() * 100, 1),
    })

tbl_df = pd.DataFrame(rows)
st.dataframe(tbl_df, use_container_width=True, hide_index=True,
             column_config={
                 "Rank": st.column_config.NumberColumn(width="small"),
                 "Coverage %": st.column_config.ProgressColumn(min_value=0, max_value=100),
             })

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown("---")
st.caption(
    "**MCLP** — Maximal Coverage Location Problem | "
    "Greedy achieves ≥ (1−1/e) ≈ 63 % of optimal; ILP finds the global optimum. "
    "Demand is weighted by proximity to points of interest (CBD, transit hubs, residential areas)."
)
