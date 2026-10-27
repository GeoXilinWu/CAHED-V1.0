"""
cahed.market
============

Market-behavioural module: supply-demand equilibrium, reduced-form
Nerlove planting decisions, target-price subsidy, and farm/fiscal
accounting (Section 2.2.2, Eqs. 11-17).

- ``demand``                  : isoelastic demand curve (Eq. 12)
- ``equilibrium_price``       : closed-form market-clearing price (Eq. 13)
- ``nerlove_area``            : reduced-form dynamic acreage response (Eq. 14)
- ``target_price_subsidy``    : price-contingent subsidy wedge (Eqs. 15-16)
- ``farm_profit``             : farm-income identity (Eq. 17)
- ``fiscal_expenditure``      : total fiscal expenditure (Eq. 17)
- ``run_coupled_simulation``  : full recursive coupled loop
  (Section 2.2.3, Eq. 18)
- ``compute_market_volatility``: realised volatility metrics (diagnostic)

Changes from v1.2 (V2 model description)
----------------------------------------
1. The OU daily price layer and the marketing-window averaging are
   removed: the annual market price IS the closed-form equilibrium
   price of the production cycle (Eq. 13).
2. The unobservable adaptive-expectation variable is removed: the
   reduced-form acreage equation responds directly to the LAGGED
   effective price (Eq. 14), an observable proxy that already embeds
   adaptive expectations and cobweb-type adjustment (Nerlove, 1958;
   Chavas and Holt, 1990).
3. The subsidy is a pure price-contingent transfer s = max(TP - P, 0)
   (Eq. 15); the effective price P_eff = P + s replaces the lagged
   price term in the acreage equation (Eq. 16).  A volume cap can
   still be imposed optionally for scenario analysis.
4. Farm profit is written as the income identity that separates
   market income from public transfers (Eq. 17).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import Config
from .soil_water import (
    compute_yearly_yield, compute_yearly_irrigation,
    compute_relative_yield, simulate_soil_moisture)


# --------------------------------------------------------------------------- #
#  Demand and market-clearing price (Eqs. 12-13)
# --------------------------------------------------------------------------- #

def demand(P: np.ndarray, kappa: float, eta: float) -> np.ndarray:
    """Isoelastic demand D = kappa * P^eta, eta < 0 (Eq. 12)."""
    return kappa * np.power(P, eta)


def equilibrium_price(S: float, kappa: float, eta: float) -> float:
    """Closed-form market-clearing price P = (S/kappa)^(1/eta) (Eq. 13).

    Parameters
    ----------
    S : float
        Aggregate supply (t), hydrologically determined (Eq. 11).
    kappa : float
        Demand-scale parameter (may grow at rate g_d, Eq. 12).
    eta : float
        Price elasticity of demand (< 0).
    """
    return float(np.exp((np.log(S) - np.log(kappa)) / eta))


# --------------------------------------------------------------------------- #
#  Planting decisions (Eq. 14)
# --------------------------------------------------------------------------- #

def nerlove_area(
    P_eff_lag: float,
    A_prev: float,
    cost_ratio: float,
    cfg: Config,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Reduced-form dynamic acreage-response equation (Eq. 14).

    ln A_t = b0 + b1 ln P_eff,t-1 + b2 ln A_{t-1} + b3 ln(C_t/C_0) + u_t

    The LAGGED effective price (subsidy-adjusted, Eq. 16) provides an
    observable reduced-form representation of farmers' adaptive
    response to past market conditions; the lagged-acreage term
    captures persistence in land allocation.  b0 is calibrated in
    ``Config`` so that A* = A0 at (P0, C0).
    """
    u = 0.0 if rng is None else rng.normal(0, cfg.area_noise_sd)
    lnA = (cfg.b0 + cfg.b1 * np.log(P_eff_lag) + cfg.b2 * np.log(A_prev)
           + cfg.b3 * np.log(cost_ratio) + u)
    return float(np.exp(lnA))


# --------------------------------------------------------------------------- #
#  Target price subsidy (Eqs. 15-16)
# --------------------------------------------------------------------------- #

def target_price_subsidy(
    P: float,
    S: float,
    TP: float,
    Q_cap: Optional[float] = None,
) -> tuple:
    """Price-contingent subsidy wedge (Eqs. 15-16).

    The per-unit subsidy is s = max(TP - P, 0) (Eq. 15) and the
    subsidy-adjusted effective price is P_eff = P + s (Eq. 16).

    Parameters
    ----------
    Q_cap : float, optional
        Subsidized volume cap (t).  ``None`` (default, V2 formulation)
        pays the wedge on the full crop; a finite cap dilutes the
        per-unit subsidy by min(1, Q_cap/S) for scenario analysis.

    Returns
    -------
    s_t : float
        Effective per-ton subsidy (CNY/t).
    P_eff : float
        Effective farm-gate price = P + s_t (Eq. 16).
    """
    wedge = max(TP - P, 0.0)
    if wedge > 0.0 and S > 0.0:
        s_t = wedge if Q_cap is None else wedge * min(1.0, Q_cap / S)
    else:
        s_t = 0.0
    return s_t, P + s_t


# --------------------------------------------------------------------------- #
#  Accounting: farm-income identity and fiscal expenditure (Eq. 17)
# --------------------------------------------------------------------------- #

def farm_profit(P: float, s_t: float, Y_t_ha: float, C: float,
                A: float) -> float:
    """Farm-income identity (Eq. 17).

    Pi_t = (P * Y - C) * A + s * S  [CNY]

    separating income generated through market transactions,
    (P * Y - C) * A, from public transfers, s * S.
    """
    S = A * Y_t_ha
    return (P * Y_t_ha - C) * A + s_t * S


def fiscal_expenditure(s_t: float, S: float) -> float:
    """Total fiscal expenditure F_t = s_t * S  [CNY] (Eq. 17)."""
    return s_t * S


# --------------------------------------------------------------------------- #
#  Full recursive coupled simulation (Section 2.2.3, Eq. 18)
# --------------------------------------------------------------------------- #

def run_coupled_simulation(
    cfg: Config,
    soil_df: pd.DataFrame,
    start_year: int = 2014,
    policy: bool = True,
    seed: Optional[int] = None,
    anchor_year: Optional[int] = None,
) -> pd.DataFrame:
    """Run the recursive coupled agro-hydro-economic simulation.

    Protocol (Section 2.2.3, Eq. 18): the sown area is decided at the
    beginning of each production cycle from the lagged effective price
    and lagged acreage (Eqs. 14, 16); hydroclimatic forcing then
    determines soil-moisture dynamics, yield and irrigation
    requirements during the growing season (Eqs. 1-10); after harvest,
    production clears the market (Eqs. 11-13) and the policy
    settlement follows (Eqs. 15-17); the resulting effective price
    informs land allocation in the subsequent cycle.

    Notes
    -----
    Under the regional-homogeneity setting the point-scale water
    balance is independent of area, so the hydrological module is run
    once up front; the market module scales its per-hectare outputs to
    the region through the endogenous sown area.  The irrigation
    withdrawal W_t = A_t * IR_t is returned for the external regional
    water-balance / groundwater assessment.

    Parameters
    ----------
    cfg : Config
    soil_df : pd.DataFrame
        Output of ``simulate_soil_moisture`` (multi-year).
    start_year : int
        Calendar year of the first simulation year (for TP lookup).
    policy : bool
        If False, the target price policy is switched off (TP = 0).
    seed : int, optional
    anchor_year : int, optional
        Calendar anchor for the demand scale kappa_t and the cost
        trend (EXP13 long-horizon extension).  When given, kappa is
        evaluated as ``cfg.demand_kappa(year - anchor_year)`` so that
        the calibrated demand level holds at ``anchor_year`` rather
        than at ``start_year``, and the cost recursion is initialised
        at ``C0 * (1 + c_trend) ** (start_year - anchor_year)`` with
        the Eq. 14 cost ratio taken relative to the deterministic
        trend path ``C0 * (1 + c_trend) ** (year - anchor_year)`` —
        preserving the original semantics (ratio = 1 at the anchor).
        ``None`` (default) reproduces the original t = 0 anchoring.

    Returns
    -------
    pd.DataFrame
        Annual trajectories: ``Year``, ``A_ha``, ``Y_t_ha``,
        ``IR_m3_ha``, ``S_t``, ``W_m3``, ``P``, ``TP``, ``s_t``,
        ``P_eff``, ``C_ha``, ``profit_CNY``, ``fiscal_CNY``.
    """
    rng = np.random.default_rng(seed if seed is not None else cfg.seed + 100)

    ydf = compute_yearly_yield(cfg, soil_df, start_year=start_year)
    idf = compute_yearly_irrigation(soil_df)
    ann = ydf.merge(idf, on='Sim_Year')
    n_years = len(ann)

    # Initial state = baseline calibration year.  With a calendar
    # anchor, the cost state is back-projected along the deterministic
    # trend so that E[C] = C0 at anchor_year.
    A_prev = cfg.A0
    P_eff_prev = cfg.P0
    if anchor_year is None:
        C_prev = cfg.C0
    else:
        C_prev = cfg.C0 * (1.0 + cfg.c_trend) ** (start_year
                                                  - anchor_year)

    rows = []
    for t in range(n_years):
        year = start_year + t

        # (1) Planting: reduced-form acreage response to the lagged
        #     effective price (Eqs. 14, 16)
        C = (C_prev * (1 + cfg.c_trend)
             * np.exp(rng.normal(0, cfg.c_cv)))
        if anchor_year is None:
            cost_ratio = C / cfg.C0
        else:
            cost_ratio = C / (cfg.C0 * (1.0 + cfg.c_trend)
                              ** (year - anchor_year))
        A = nerlove_area(P_eff_prev, A_prev, cost_ratio, cfg, rng)

        # (2) Growing season: hydrology already simulated (per-ha)
        Y_t_ha = ann['Absolute_Yield'].iloc[t] / 1000.0     # kg -> t
        IR_m3_ha = ann['IR_m3_ha'].iloc[t]

        # (3) Harvest and market clearing (Eqs. 11-13)
        S = A * Y_t_ha                                       # Eq. 11
        W = A * IR_m3_ha                                     # withdrawal
        kappa = cfg.demand_kappa(
            t if anchor_year is None else year - anchor_year)  # Eq. 12
        P = equilibrium_price(S, kappa, cfg.eta)             # Eq. 13

        # (4) Policy settlement (Eqs. 15-17)
        TP = cfg.get_target_price(year, policy=policy)
        s_t, P_eff = target_price_subsidy(P, S, TP, cfg.Q_cap)
        profit = farm_profit(P, s_t, Y_t_ha, C, A)           # Eq. 17
        fiscal = fiscal_expenditure(s_t, S)                  # Eq. 17

        rows.append({
            'Year': year, 'A_ha': A, 'Y_t_ha': Y_t_ha,
            'IR_m3_ha': IR_m3_ha, 'S_t': S, 'W_m3': W,
            'P': P, 'TP': TP, 's_t': s_t,
            'P_eff': P_eff, 'C_ha': C,
            'profit_CNY': profit, 'fiscal_CNY': fiscal,
        })

        # (5) State updating
        A_prev, P_eff_prev, C_prev = A, P_eff, C

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  Sequential coupled simulation with availability-constrained irrigation
#  (EXP13 v4 extension)
# --------------------------------------------------------------------------- #

def run_coupled_simulation_wq(
    cfg: Config,
    weather_df: pd.DataFrame,
    quota_fn=None,
    start_year: int = 2014,
    policy: bool = True,
    seed: Optional[int] = None,
    anchor_year: Optional[int] = None,
    collect_monthly: bool = False,
):
    """Coupled simulation with year-by-year soil dynamics and an optional
    monthly irrigation quota.

    Unlike :func:`run_coupled_simulation` (point-scale hydrology run once
    up front, valid when water is unlimited), here the soil is simulated
    one year at a time *after* the planting decision, so the irrigation
    quota can depend on the current-year sown area::

        quota_fn(year, A_ha) -> (12,) monthly quota in mm

    This closes the water-economy coupling: area expansion or a dry year
    tightens the per-hectare quota, Feddes stress develops, yield falls,
    and the yield shock propagates to supply, prices and income.  The
    soil-moisture state is carried across years via
    ``soil_df.attrs['s_final']``.

    Returns the same columns as :func:`run_coupled_simulation` plus
    ``IR_need_m3_ha`` (uncapped stress-avoidance requirement) and
    ``Rel_Yield`` (Jensen relative yield).  With
    ``collect_monthly=True`` returns ``(df, monthly)`` where
    ``monthly = {'ir_applied': (T,12), 'ir_need': (T,12)}`` in m3/ha.
    """
    from .soil_water import doy_to_month

    rng = np.random.default_rng(seed if seed is not None else cfg.seed + 100)
    sim_years = sorted(weather_df['Sim_Year'].unique())

    A_prev = cfg.A0
    P_eff_prev = cfg.P0
    if anchor_year is None:
        C_prev = cfg.C0
    else:
        C_prev = cfg.C0 * (1.0 + cfg.c_trend) ** (start_year
                                                  - anchor_year)

    s_state = None
    rows = []
    m_applied = np.zeros((len(sim_years), 12))
    m_need = np.zeros((len(sim_years), 12))
    for t, syr in enumerate(sim_years):
        year = start_year + t

        # (1) Planting: reduced-form acreage response (Eqs. 14, 16)
        C = (C_prev * (1 + cfg.c_trend)
             * np.exp(rng.normal(0, cfg.c_cv)))
        if anchor_year is None:
            cost_ratio = C / cfg.C0
        else:
            cost_ratio = C / (cfg.C0 * (1.0 + cfg.c_trend)
                              ** (year - anchor_year))
        A = nerlove_area(P_eff_prev, A_prev, cost_ratio, cfg, rng)

        # (2) Growing season: this year's soil under the quota that the
        #     planting decision implies
        q = quota_fn(year, A) if quota_fn is not None else None
        wy = weather_df[weather_df['Sim_Year'] == syr]
        soil_y = simulate_soil_moisture(
            cfg, wy, s0=s_state,
            quota_mm=None if q is None else np.asarray(q, float)[None, :])
        s_state = soil_y.attrs.get('s_final')
        rel = compute_relative_yield(cfg, soil_y)
        Y_t_ha = cfg.y_pot(year) * rel / 1000.0           # kg -> t
        IR_m3_ha = float(soil_y['Irrigation'].sum()) * 10.0
        IR_need = float(soil_y['Irrigation_Need'].sum()) * 10.0
        if collect_monthly:
            mth = doy_to_month(soil_y['Day_Of_Cycle'].values)
            for m in range(12):
                sel = mth == m
                m_applied[t, m] = soil_y['Irrigation'].values[sel].sum() * 10.0
                m_need[t, m] = soil_y['Irrigation_Need'].values[sel].sum() * 10.0

        # (3) Harvest and market clearing (Eqs. 11-13)
        S = A * Y_t_ha
        W = A * IR_m3_ha
        kappa = cfg.demand_kappa(
            t if anchor_year is None else year - anchor_year)
        P = equilibrium_price(S, kappa, cfg.eta)

        # (4) Policy settlement (Eqs. 15-17)
        TP = cfg.get_target_price(year, policy=policy)
        s_t, P_eff = target_price_subsidy(P, S, TP, cfg.Q_cap)
        profit = farm_profit(P, s_t, Y_t_ha, C, A)
        fiscal = fiscal_expenditure(s_t, S)

        rows.append({
            'Year': year, 'A_ha': A, 'Y_t_ha': Y_t_ha,
            'IR_m3_ha': IR_m3_ha, 'IR_need_m3_ha': IR_need,
            'Rel_Yield': rel, 'S_t': S, 'W_m3': W,
            'P': P, 'TP': TP, 's_t': s_t,
            'P_eff': P_eff, 'C_ha': C,
            'profit_CNY': profit, 'fiscal_CNY': fiscal,
        })

        # (5) State updating
        A_prev, P_eff_prev, C_prev = A, P_eff, C

    out = pd.DataFrame(rows)
    if collect_monthly:
        return out, {'ir_applied': m_applied, 'ir_need': m_need}
    return out




# --------------------------------------------------------------------------- #
#  Volatility metrics (diagnostic, kept from earlier versions)
# --------------------------------------------------------------------------- #

def compute_market_volatility(
    prices: np.ndarray,
    window: int = 5,
) -> Dict[str, float]:
    """Compute realised volatility metrics from a price series."""
    log_ret = np.diff(np.log(np.asarray(prices) + 1e-6))
    total_vol = float(np.std(log_ret))
    if len(log_ret) >= window:
        local_vols = np.array([
            np.std(log_ret[i:i + window])
            for i in range(len(log_ret) - window + 1)])
        mean_local = float(np.mean(local_vols))
        max_local = float(np.max(local_vols))
    else:
        mean_local = total_vol
        max_local = total_vol
    return {
        'total_volatility': total_vol,
        'mean_local_volatility': mean_local,
        'max_local_volatility': max_local,
        'price_cv': float(np.std(prices) / (np.mean(prices) + 1e-6)),
    }
