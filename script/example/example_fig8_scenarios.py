#!/usr/bin/env python3
"""
Example 2 — Reproduce the 15-scenario resilience-indicator figure of
the CAHED manuscript (Fig. 8), arranged by driver family: climate (top row),
socio-economic (middle row), human intervention (bottom row).

The 14 one-way scenarios + baseline are defined in
config/scenarios.yaml.  Common random numbers (CRN) hold stochastic
realizations constant across scenarios, so ensemble differences are
attributable to the imposed lever alone.  Each panel shows the
scenario median (solid = high/stronger lever, dashed = low/weaker)
with an IQR ribbon, against the baseline (black).

Modes
-----
default        recompute the full paper ensemble (100 members per
               scenario; ~1500 coupled simulations, a few minutes)
--quick        reduced ensemble (4 weather x 4 market seeds) for a
               fast smoke test
--from-results plot directly from data/results/scenario_windowed.csv
               (the paper's archived output) without recomputation

Outputs (example/output/):
  Fig8_scenario_trajectories.png
  scenario_windowed_recomputed.csv   (recompute modes only)

Usage (from the repository root):
  python example/example_fig8_scenarios.py [--quick]
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from script import (  # noqa: E402
    Config, build_baseline_params, generate_precipitation_mmmpp,
    generate_pet_cosine, merge_weather, simulate_soil_moisture,
    run_coupled_simulation, compute_monthly_irrigation, growing_months,
    components_from_series, water_annual_score, demand_benchmark_tonnage)
from script import data_utils as du  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"
ARCHIVE = du.DATA_RESULTS / "scenario_windowed.csv"

DIMS = ["R_S", "R_I", "R_W", "R"]
COMPS = ["R1_bal", "R1_stab", "R2_prof", "R2_stab", "R3_wdf"]
LAB = {"R_S": "$R_S$ (supply)", "R_I": "$R_I$ (income)",
       "R_W": "$R_W$ (water)", "R": "$R$ (overall)"}
GROUPS = ["climate", "socioeconomic", "intervention"]
GTITLE = {"climate": "Climate change", "socioeconomic": "Socio-economic",
          "intervention": "Human intervention"}
LEVER_COLOR = {"rain_freq": "#4C7C9B", "pet": "#B08463", "wa": "#7C9B6E",
               "cost": "#4C7C9B", "demand": "#B08463",
               "irrig": "#4C7C9B", "subsidy": "#B08463"}
SHORT = {"rain_freq_high": "rain high", "rain_freq_low": "rain low",
         "pet_high": "PET high", "pet_low": "PET low",
         "wa_high": "WA high", "wa_low": "WA low",
         "cost_high": "cost high", "cost_low": "cost low",
         "demand_high": "demand high", "demand_low": "demand low",
         "irrig_high": "irrig. high", "irrig_low": "irrig. low",
         "subsidy_plus": "subsidy +15%", "subsidy_off": "subsidy off"}


def load_scenarios():
    d = yaml.safe_load(
        (du.CONFIG / "scenarios.yaml").read_text(encoding="utf-8"))
    return d["design"], d["scenarios"]


def build_cfg(spec, design):
    """Scenario configuration; economic overrides go through
    Config.from_yaml, weather/availability/policy levers are applied
    at build time."""
    ov = {}
    for key in ("c_trend", "g_d", "s_irrig"):
        if key in spec:
            ov[key] = spec[key]
    if "tp_scale" in spec:
        f = spec["tp_scale"]
        tp = {int(y): v * f for y, v in design["tp_schedule"].items()
              if y != "default"}
        ov["target_price"] = tp
        ov["tp_default"] = design["tp_schedule"]["default"] * f
    return Config.from_yaml(str(du.MODEL_YAML), N=design["N"], **ov)


def build_weather(cfg, spec, ws, design):
    bp = build_baseline_params()
    f = spec.get("rain_freq")
    if f is not None:
        bp["lambda_vals"] = bp["lambda_vals"] * f
        bp["trans_dw"] = bp["trans_dw"] * f      # pure frequency shift
    pet = generate_pet_cosine(cfg,
                              pet_mean=design["PET_mean"]
                              * spec.get("pet", 1.0), seed=ws)
    return merge_weather(generate_precipitation_mmmpp(cfg, bp, seed=ws),
                         pet)


def recompute(wx_seeds, econ_seeds) -> pd.DataFrame:
    rp = du.load_resilience_params()
    design, scenarios = load_scenarios()
    N_YEARS, START = int(design["N"]), int(design["start_year"])
    ANCHOR = int(design["anchor_year"])
    years = np.arange(START, START + N_YEARS)
    panel = du.load_panel()
    wa_base = du.monthly_availability(rp["WA_SCALE"])
    y_nbs_2014 = float(panel[panel.year == 2014].yield_t_ha.iloc[0])

    basis = None
    rows = []
    t0 = time.time()
    for sc in scenarios:
        group, label, direction, spec = (sc["group"], sc["name"],
                                         sc["direction"], sc["spec"])
        cfg = build_cfg(spec, design)
        policy = spec.get("policy", True)
        gm = growing_months(cfg)
        D = demand_benchmark_tonnage(cfg, years, ANCHOR)
        w_avail = wa_base * spec.get("wa", 1.0)
        for ws in wx_seeds:
            wx = build_weather(cfg, spec, ws, design)
            soil = simulate_soil_moisture(cfg, wx)
            ir_m = compute_monthly_irrigation(soil)          # m3/ha
            for sd in econ_seeds:
                r = run_coupled_simulation(cfg, soil,
                                           start_year=START, policy=policy,
                                           seed=sd, anchor_year=ANCHOR)
                if basis is None:               # baseline, first member
                    y_m14 = float(r.loc[r.Year == 2014, "Y_t_ha"].iloc[0])
                    basis = y_nbs_2014 / y_m14
                    print(f"basis_ratio = {basis:.3f}", flush=True)
                A = r["A_ha"].values
                S = r["S_t"].values
                Yn = r["Y_t_ha"].values * basis
                m = (r["P"].values * Yn - r["C_ha"].values) \
                    / (r["P"].values * Yn)
                s_w = water_annual_score(A, A[:, None] * ir_m,
                                         np.tile(w_avail, (len(A), 1)),
                                         gm, M=rp["WEI_M"])
                comp = components_from_series(S, D, m, s_w, rp["PAR"],
                                              int(design["window"]))
                for i, yr in enumerate(years):
                    row = {"group": group, "scenario": label,
                           "direction": direction, "member": f"{ws}-{sd}",
                           "Year": int(yr), "A_ha": A[i],
                           "P": r["P"].iloc[i],
                           "profit_CNY": r["profit_CNY"].iloc[i]}
                    for k in COMPS + DIMS:
                        row[k] = comp[k][i]
                    rows.append(row)
        print(f"  {group}/{label} done ({time.time() - t0:.0f}s)",
              flush=True)
    return pd.DataFrame(rows)


def plot(traj: pd.DataFrame, png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # map each scenario's single spec key to its lever colour
    SPEC_LEVER = {"rain_freq": "rain_freq", "pet": "pet", "wa": "wa",
                  "c_trend": "cost", "g_d": "demand",
                  "s_irrig": "irrig", "tp_scale": "subsidy",
                  "policy": "subsidy"}
    color_of = {}
    for s in load_scenarios()[1]:
        if s["spec"]:
            color_of[s["name"]] = LEVER_COLOR[SPEC_LEVER[
                list(s["spec"].keys())[0]]]

    fig, axes = plt.subplots(3, 4, figsize=(13.5, 8.2), sharex=True)

    # per-column y-range (shared across the three rows): central 98 %
    # of the pooled scenario values, padded by 10 %, clipped to [0, 1]
    # — zooms each resilience dimension to its actual variation band
    ylims = {}
    for out in DIMS:
        v = traj[out].dropna().values
        lo, hi = np.nanquantile(v, 0.01), np.nanquantile(v, 0.99)
        pad = 0.10 * (hi - lo)
        ylims[out] = (max(0.0, lo - pad), min(1.0, hi + pad))

    for r, group in enumerate(GROUPS):
        for c, out in enumerate(DIMS):
            ax = axes[r, c]
            b = traj[traj.scenario == "baseline"].groupby("Year")[out]
            med, q1, q3 = b.median(), b.quantile(.25), b.quantile(.75)
            ax.fill_between(med.index, q1, q3, color="#333333",
                            alpha=.12, lw=0)
            ax.plot(med.index, med, color="#333333", lw=2.2,
                    label="baseline")
            for label, color in color_of.items():
                g = traj[(traj.group == group) & (traj.scenario == label)]
                ls = "-" if label.endswith(("high", "plus")) else "--"
                gg = g.groupby("Year")[out]
                med, q1, q3 = gg.median(), gg.quantile(.25), gg.quantile(.75)
                ax.fill_between(med.index, q1, q3, color=color, alpha=.13,
                                lw=0)
                ax.plot(med.index, med, color=color, ls=ls, lw=1.5,
                        label=SHORT[label])
            if r == 0:
                ax.set_title(LAB[out], fontsize=11)
            if c == 0:
                ax.set_ylabel(GTITLE[group], fontsize=10)
            if r == 2:
                ax.set_xlabel("Year (final year of 5-yr window)")
            ax.set_ylim(ylims[out])
            ax.grid(alpha=.25, lw=0.5)
            ax.spines[["top", "right"]].set_visible(False)
            if r == 0 and c == 3:
                # legend outside the last panel, clear of the curves
                ax.legend(fontsize=7, frameon=False, loc="center left",
                          bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(png, dpi=300)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="reduced ensemble (4x4 members per scenario)")
    ap.add_argument("--from-results", action="store_true",
                    help="plot the archived paper results without "
                         "recomputation")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    OUT.mkdir(exist_ok=True)

    if args.from_results:
        traj = pd.read_csv(ARCHIVE)
        print(f"loaded archived scenario ensemble: {ARCHIVE}")
    else:
        design, _ = load_scenarios()
        if args.quick:
            wx, ec = design["wx_seeds"][:4], design["econ_seeds"][:4]
        else:
            wx, ec = design["wx_seeds"], design["econ_seeds"]
        print(f"recomputing scenario ensemble "
              f"({len(wx) * len(ec)} members x 15 scenarios) ...")
        traj = recompute(wx, ec)
        csv = OUT / "scenario_windowed_recomputed.csv"
        traj.to_csv(csv, index=False)
        print("scenario trajectories ->", csv)

    png = OUT / "Fig8_scenario_trajectories.png"
    plot(traj, png)
    print("figure ->", png)


if __name__ == "__main__":
    main()
