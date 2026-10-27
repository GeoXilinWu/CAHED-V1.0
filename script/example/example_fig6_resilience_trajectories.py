#!/usr/bin/env python3
"""
Example 1 — Reproduce Fig. 6 of the CAHED manuscript:
trajectories of the resilience indicators, 1985-2045.

Eight panels: (a-d) the five components R1_bal / R1_stab / R2_prof /
R2_stab / R3_wdf (first four shown), (e-g) the dimension indices
R_S / R_I / R_W, (h) the overall index R.  The model ensemble
(baseline member + N_MC stochastic members, each with its own weather
realization) is shown as median, p25-p75 and p05-p95 bands; the
observed-computed indicators are overlaid on the dimension and overall
panels for the 1989-2023 overlap.

Modes
-----
default        recompute the full paper ensemble (200 members) and plot
--quick        reduced ensemble (20 members) for a fast smoke test
--from-results plot directly from data/results/resilience_indicators.csv
               (the paper's archived output) without recomputation

Outputs (example/output/):
  Fig6_resilience_trajectories.png
  resilience_indicators_recomputed.csv   (recompute modes only)

Usage (from the repository root):
  python example/example_fig6_resilience_trajectories.py [--quick]
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from script import (  # noqa: E402
    Config, build_baseline_params, generate_precipitation_mmmpp,
    generate_pet_cosine, merge_weather, simulate_soil_moisture,
    run_coupled_simulation, compute_monthly_irrigation, growing_months,
    components_from_series, water_annual_score, demand_benchmark_tonnage,
    IND9)
from script import data_utils as du  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"
ARCHIVE = du.DATA_RESULTS / "resilience_indicators.csv"

C_BAND = "skyblue"      # p05-p95 band
C_IBAND = "skyblue"     # p25-p75 band (denser alpha)
C_MED = "deepskyblue"   # ensemble median
C_BASE = "darkred"      # deterministic baseline member


def panel_label(ax, s, x=0.95, y=0.06):
    """Bold panel tag, e.g. '(a)', bottom-right inside the panel."""
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom", ha="right")


PANELS = [("R1_bal", "Supply adequacy $R_S^{bal}$"),
          ("R1_stab", "Production stability $R_S^{stab}$"),
          ("R2_prof", "Profitability $R_I^{prof}$"),
          ("R2_stab", "Returns stability $R_I^{stab}$"),
          ("R_S", "Supply resilience $R_S$"),
          ("R_I", "Income resilience $R_I$"),
          ("R_W", "Water-resource resilience $R_W$"),
          ("R", "Agriculture system resilience $R$")]


def recompute(n_mc: int) -> pd.DataFrame:
    """Run the paper ensemble protocol and return the indicator table
    (same column layout as data/results/resilience_indicators.csv)."""
    rp = du.load_resilience_params()
    PAR, WINDOW = rp["PAR"], rp["WINDOW"]
    START, N_YEARS = 1985, 61
    YEARS = np.arange(START, START + N_YEARS)
    ANCHOR = rp["ANCHOR_YEAR"]

    cfg = du.build_run_config(N_YEARS, seed=rp["WEATHER_SEED_BASE"],
                              start_year=START)
    panel, d4 = du.load_panel(), du.load_hydro_cali()
    wa_m = du.monthly_availability(rp["WA_SCALE"])
    D_path = demand_benchmark_tonnage(cfg, YEARS, ANCHOR)
    bp = build_baseline_params()
    gm = growing_months(cfg)

    def make_weather(wseed):
        return merge_weather(
            generate_precipitation_mmmpp(cfg, bp, seed=wseed),
            generate_pet_cosine(cfg, seed=wseed))

    def run_member(mseed, wseed):
        soil = simulate_soil_moisture(cfg, make_weather(wseed))
        ir_m = compute_monthly_irrigation(soil)
        r = run_coupled_simulation(cfg, soil, start_year=START,
                                   policy=True, seed=mseed,
                                   anchor_year=ANCHOR)
        return r, ir_m

    # baseline member (defines the NBS yield basis ratio)
    r0, ir0 = run_member(rp["MARKET_SEED_BASE"], rp["WEATHER_SEED_BASE"])
    y14 = float(r0.loc[r0.Year == 2014, "Y_t_ha"].iloc[0])
    basis = float(panel[panel.year == 2014].yield_t_ha.iloc[0]) / y14
    print(f"basis_ratio (NBS 2014 / model 2014) = {basis:.3f}", flush=True)

    def comp_of(r, ir_m):
        A = r["A_ha"].values
        S = r["S_t"].values
        Yn = r["Y_t_ha"].values * basis
        m = (r["P"].values * Yn - r["C_ha"].values) / (r["P"].values * Yn)
        s_w = water_annual_score(A, A[:, None] * ir_m,
                                 np.tile(wa_m, (len(A), 1)), gm,
                                 M=rp["WEI_M"])
        return components_from_series(S, D_path, m, s_w, PAR, WINDOW)

    comp_base = comp_of(r0, ir0)
    ens = np.full((n_mc, N_YEARS, len(IND9)), np.nan)
    t0 = time.time()
    for i in range(n_mc):
        r_i, ir_i = run_member(rp["MARKET_SEED_M0"] + i,
                               rp["WEATHER_SEED_M0"] + i)
        c_i = comp_of(r_i, ir_i)
        ens[i] = np.column_stack([c_i[k] for k in IND9])
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{n_mc} members ({time.time() - t0:.0f}s)",
                  flush=True)

    # ---- observed side (same exogenous benchmarks as the model side) --
    S_o = panel["production_t"].values.astype(float)
    D_o = demand_benchmark_tonnage(cfg, panel["year"].values, ANCHOR)
    pmkt = panel["ccindex_sepnov"].values.astype(float)
    y_o = panel["yield_t_ha"].values.astype(float)
    c_o = panel["cost_cny_ha"].values.astype(float)
    m_o = np.where(np.isfinite(pmkt * y_o * c_o),
                   (pmkt * y_o - c_o) / (pmkt * y_o), np.nan)
    A_o = panel["area_ha"].values.astype(float)
    W_o = A_o * panel["year"].map(
        d4.set_index("year")["irr_mm"]).values * 10.0
    soil_real = du.real_weather_soil(cfg)
    ir_real = compute_monthly_irrigation(soil_real)
    mshare = ir_real.sum(axis=0) / ir_real.sum()
    w_o_m = W_o[:, None] * mshare[None, :]
    s_w_o = water_annual_score(A_o, w_o_m,
                               np.tile(wa_m, (len(A_o), 1)), gm,
                               M=rp["WEI_M"])
    comp_obs = components_from_series(S_o, D_o, m_o, s_w_o, PAR, WINDOW)

    out = pd.DataFrame({"Year": YEARS})
    obs_i = pd.DataFrame({"Year": panel["year"], **comp_obs}).set_index("Year")
    for j, name in enumerate(IND9):
        out[f"{name}_base"] = comp_base[name]
        for q in (5, 25, 50, 75, 95):
            out[f"{name}_p{q:02d}"] = np.nanpercentile(
                ens[:, :, j], q, axis=0)
        out[f"{name}_obs"] = obs_i.reindex(YEARS)[name].values
    return out


def plot(ind: pd.DataFrame, png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(12, 4), dpi=600,
                             sharey=True, sharex=True)
    for i, (ax, (name, title)) in enumerate(zip(axes.flat, PANELS)):
        x = ind["Year"].values
        ax.fill_between(x, ind[f"{name}_p05"], ind[f"{name}_p95"],
                        color=C_BAND, alpha=0.20, lw=0, label="p05-p95")
        ax.fill_between(x, ind[f"{name}_p25"], ind[f"{name}_p75"],
                        color=C_IBAND, alpha=0.50, lw=0, label="p25-p75")
        ax.plot(x, ind[f"{name}_p50"], color=C_MED, lw=1.5,
                label="ensemble median")
        ax.plot(x, ind[f"{name}_base"], color=C_BASE, lw=1.5, alpha=0.56,
                label="baseline")
        ax.set_title(title, loc="left", fontsize=8)
        ax.set_xlim(1988, 2046)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        panel_label(ax, f"({chr(97 + i)})")
    axes[0, 0].set_ylabel("resilience index")
    axes[1, 0].set_ylabel("resilience index")
    for c in range(4):
        axes[1, c].set_xlabel("year")
    axes[0, 0].legend(loc="center left", fontsize=8,
                      bbox_to_anchor=(0.0, 0.32), frameon=False)
    fig.tight_layout()
    fig.savefig(png, dpi=600)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="reduced ensemble (20 members)")
    ap.add_argument("--from-results", action="store_true",
                    help="plot the archived paper results without "
                         "recomputation")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    OUT.mkdir(exist_ok=True)

    if args.from_results:
        ind = pd.read_csv(ARCHIVE)
        print(f"loaded archived indicators: {ARCHIVE}")
    else:
        n_mc = 20 if args.quick else du.load_resilience_params()["N_MC"]
        print(f"recomputing Fig. 6 ensemble (n_members={n_mc}) ...")
        ind = recompute(n_mc)
        csv = OUT / "resilience_indicators_recomputed.csv"
        ind.to_csv(csv, index=False)
        print("indicators ->", csv)

    png = OUT / "Fig6_resilience_trajectories.png"
    plot(ind, png)
    cols = ["Year", "R_S_p50", "R_I_p50", "R_W_p50", "R_p50"]
    print(ind.loc[ind.Year.isin(range(1990, 2046, 10)), cols]
          .round(3).to_string(index=False))
    print("figure ->", png)


if __name__ == "__main__":
    main()
