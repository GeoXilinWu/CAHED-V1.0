"""
cahed.soil_water
================

Soil moisture simulation, irrigation, and yield computation (Eqs. 2-10).

- ``feddes_stress``            : Feddes piecewise-linear stress coefficient (Eq. 6)
- ``leakage_rate``             : gravity-driven deep percolation (Eq. 4)
- ``water_balance_step``       : single-day water balance update (Eqs. 2-3)
- ``simulate_soil_moisture``   : multi-year daily simulation with irrigation
- ``compute_relative_yield``   : Jensen moisture production function (Eq. 8)
- ``compute_yearly_yield``     : per-year yield
- ``compute_yearly_irrigation``: per-year irrigation requirement IR_t (Eq. 10)

Changes from v0.1
-----------------
1. Deep percolation is now zero at or below field capacity and follows
   the normalized Clapp-Hornberger form above it (Eq. 4).
2. Infiltration is truncated by available pore space; the excess is
   recorded as saturation-excess runoff (Eq. 3).
3. A stress-avoidance irrigation rule is added (Eq. 9): when soil
   moisture falls below ``s_irrig`` on a growth day, the soil is
   restored to ``s_target``; applied depths are recorded.
4. ``compute_yearly_irrigation`` aggregates daily applications to the
   annual irrigation requirement IR_t in m3/ha (Eq. 10).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import Config


# --------------------------------------------------------------------------- #
#  Feddes stress coefficient (Eq. 6)
# --------------------------------------------------------------------------- #

def feddes_stress(
    s: np.ndarray,
    sw: float,
    s_star: float,
    s_fc: float,
) -> np.ndarray:
    """Feddes piecewise-linear water-stress coefficient gamma(s) (Eq. 6)."""
    ks = np.zeros_like(s)
    mask1 = s > sw
    ks[mask1] = (s[mask1] - sw) / (s_star - sw)
    mask2 = s >= s_star
    ks[mask2] = 1.0
    return np.clip(ks, 0.0, 1.0)


# --------------------------------------------------------------------------- #
#  Deep percolation (Eq. 4)
# --------------------------------------------------------------------------- #

def leakage_rate(s: float, cfg: Config) -> float:
    """Gravity-driven deep percolation L(s) (Eq. 4).

    Zero at or below field capacity; above it, the normalized
    Clapp-Hornberger power form L = Ks * ((s - s_fc)/(porosity - s_fc))^omega
    (expressed as a relative saturation loss rate, 1/day).
    """
    if s <= cfg.s_fc:
        return 0.0
    excess = (s - cfg.s_fc) / (cfg.porosity - cfg.s_fc)
    return cfg.Ks * excess ** cfg.omega / (cfg.Zr * cfg.porosity)


# --------------------------------------------------------------------------- #
#  Single-day water balance (Eqs. 2-3)
# --------------------------------------------------------------------------- #

def water_balance_step(
    s_t: float,
    rain: float,
    pet: float,
    kc: float,
    cfg: Config,
    irrigation: float = 0.0,
) -> tuple:
    """Advance soil moisture by one day (Eqs. 2-3).

    Effective infiltration is truncated by the available pore space;
    the excess is runoff.  Returns
    ``(s_next, et_act, leakage, runoff)``.
    """
    # Infiltration truncation -> saturation-excess runoff
    avail_pore = max((cfg.porosity - s_t) * cfg.Zr * cfg.porosity, 0.0)
    rain_infil = min(rain, avail_pore)
    runoff = rain - rain_infil

    ks = feddes_stress(np.array([s_t]), cfg.sw, cfg.s_star, cfg.s_fc)[0]
    et_act = kc * ks * pet
    leak = leakage_rate(s_t, cfg) * cfg.Zr * cfg.porosity  # mm/day

    ds = (rain_infil + irrigation - et_act - leak) / (cfg.Zr * cfg.porosity)
    s_next = float(np.clip(s_t + ds, 0.0, cfg.porosity))
    return s_next, et_act, leak, runoff


# --------------------------------------------------------------------------- #
#  Multi-year simulation with irrigation (Eqs. 2-6, 9)
# --------------------------------------------------------------------------- #

def simulate_soil_moisture(
    cfg: Config,
    weather_df: pd.DataFrame,
    s0: Optional[float] = None,
    quota_mm: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Run the daily soil moisture simulation for *N* years.

    If ``cfg.irrigate`` is True, the stress-avoidance rule (Eq. 9) is
    applied at the start of each growth day: when s < s_irrig the soil
    is refilled toward s_target and the application depth is recorded.

    quota_mm : optional array of shape (n_years, 12) giving the monthly
        irrigation quota (mm).  When provided, each application is
        capped by the remaining monthly quota, so water-scarce
        months/years develop real Feddes stress and yield penalties
        (availability-constrained irrigation).  ``None`` (default)
        reproduces the unlimited-supply rule.

    Returns
    -------
    pd.DataFrame
        Columns: ``Sim_Year``, ``Day_Of_Cycle``, ``Stage``,
        ``Soil_Moisture``, ``Ks_theta``, ``ET_Act``, ``Leakage``,
        ``Rain``, ``PET``, ``Irrigation``, ``Irrigation_Need``,
        ``Runoff``, ``Jensen_Lambda``.  ``Irrigation_Need`` is the
        uncapped stress-avoidance requirement; ``Irrigation`` is the
        actually applied (possibly capped) depth.  The end-of-simulation
        soil moisture is exposed as ``df.attrs['s_final']`` for
        sequential year-by-year coupling.
    """
    s = cfg.s_fc if s0 is None else s0
    T_total = len(weather_df)
    records = []

    kc_arr = cfg.kc_annual
    jensen_arr = cfg.get_jensen_array()
    qrem = quota_mm.astype(float).copy() if quota_mm is not None else None
    y_idx, prev_yr = -1, None

    for t in range(T_total):
        row = weather_df.iloc[t]
        day = int(row['Day_Of_Cycle'])
        syr = int(row['Sim_Year'])
        if syr != prev_yr:
            y_idx, prev_yr = y_idx + 1, syr
        kc = kc_arr[day]
        rain = row['Rain_Intensity']
        pet = row['PET']

        # --- Irrigation rule (Eq. 9), start of day, growth days only ---
        irrig_mm, need_mm = 0.0, 0.0
        if cfg.irrigate and cfg.is_growth_day(day) and s < cfg.s_irrig:
            need_mm = (cfg.s_target - s) * cfg.Zr * cfg.porosity
            if qrem is None:
                irrig_mm = need_mm
            else:
                mth = int(doy_to_month(day))
                irrig_mm = float(min(need_mm, max(qrem[y_idx, mth], 0.0)))
                qrem[y_idx, mth] -= irrig_mm
            s += irrig_mm / (cfg.Zr * cfg.porosity)

        s_next, et_act, leak, runoff = water_balance_step(
            s, rain, pet, kc, cfg)

        records.append({
            'Sim_Year': syr,
            'Day_Of_Cycle': day,
            'Stage': row['Stage'],
            'Soil_Moisture': s,
            'Ks_theta': feddes_stress(np.array([s]), cfg.sw,
                                      cfg.s_star, cfg.s_fc)[0],
            'ET_Act': et_act,
            'Leakage': leak,
            'Rain': rain,
            'PET': pet,
            'Irrigation': irrig_mm,
            'Irrigation_Need': need_mm,
            'Runoff': runoff,
            'Jensen_Lambda': jensen_arr[day],
        })
        s = s_next

    df = pd.DataFrame(records)
    df.attrs['s_final'] = s
    return df


# --------------------------------------------------------------------------- #
#  Yield helpers (Eqs. 7-8)
# --------------------------------------------------------------------------- #

def compute_stage_stress(
    cfg: Config,
    soil_df: pd.DataFrame,
) -> Dict[str, float]:
    """Mean stress coefficient per growth stage (Eq. 7).

    The stage water stress of Eq. 7, zeta_i = integral of the
    instantaneous stress over the stage soil-moisture density, is
    evaluated by its discrete analog: the mean of the daily stress
    response gamma(s_t) = 1 - zeta(s_t) over the stage days.  The
    Jensen function (Eq. 8) then uses (1 - zeta_i) = mean gamma.
    """
    stress: Dict[str, float] = {}
    for sid, sc in cfg.stages_config.items():
        if sc['type'] == 'G' and sid in cfg._jensen_lambdas:
            mask = ((soil_df['Day_Of_Cycle'] >= sc['start_date'])
                    & (soil_df['Day_Of_Cycle'] < sc['end_date']))
            stress[sid] = soil_df.loc[mask, 'Ks_theta'].mean()
    return stress


def compute_relative_yield(cfg: Config, soil_df: pd.DataFrame) -> float:
    """Relative yield via the Jensen production function (Eq. 8).

    Ya/Ym = prod_k (AET_k / PET_k)^lambda_k , evaluated with the
    stage-mean stress coefficient AET_k/PET_k = mean gamma(s).
    """
    log_sum = 0.0
    for sid, sc in cfg.stages_config.items():
        lam = cfg._jensen_lambdas.get(sid, 0.0)
        if lam <= 0:
            continue
        mask = ((soil_df['Day_Of_Cycle'] >= sc['start_date'])
                & (soil_df['Day_Of_Cycle'] < sc['end_date']))
        ks_mean = soil_df.loc[mask, 'Ks_theta'].mean()
        log_sum += lam * np.log(np.clip(ks_mean, 1e-6, 1.0))
    return float(np.exp(log_sum))


def compute_absolute_yield(cfg: Config, soil_df: pd.DataFrame,
                           year: Optional[int] = None) -> float:
    """Absolute yield (kg ha-1).

    With the technology trend enabled, the potential yield is
    Y_pot,t = Omega_t * Y_max; ``year`` is the calendar year used for
    Omega_t (if None, the static Y_max is used).
    """
    y_pot = cfg.y_pot(year) if (year is not None) else cfg.Y_max
    return y_pot * compute_relative_yield(cfg, soil_df)


def compute_yearly_yield(cfg: Config, soil_df: pd.DataFrame,
                         start_year: Optional[int] = None) -> pd.DataFrame:
    """Yield for each simulation year.

    Parameters
    ----------
    start_year : int, optional
        Calendar year of the first simulation year.  Required to apply
        the technology trend (Y_pot,t = Omega_t * Y_max); when omitted
        the static Y_max is used even if the trend is enabled.
    """
    results = []
    for year, gdf in soil_df.groupby('Sim_Year'):
        rel = compute_relative_yield(cfg, gdf)
        cal = (start_year + int(year)) if start_year is not None else None
        y_pot = cfg.y_pot(cal) if cal is not None else cfg.Y_max
        results.append({
            'Sim_Year': year,
            'Relative_Yield': rel,
            'Absolute_Yield': y_pot * rel,   # kg/ha
        })
    return pd.DataFrame(results)


# --------------------------------------------------------------------------- #
#  Annual irrigation requirement (Eq. 10)
# --------------------------------------------------------------------------- #

def compute_yearly_irrigation(soil_df: pd.DataFrame) -> pd.DataFrame:
    """Annual irrigation requirement IR_t per unit area (Eq. 10).

    Returns
    -------
    pd.DataFrame
        Columns: ``Sim_Year``, ``Irrigation_mm``, ``IR_m3_ha``
        (1 mm over 1 ha = 10 m3).
    """
    g = soil_df.groupby('Sim_Year')['Irrigation'].sum().reset_index()
    g.columns = ['Sim_Year', 'Irrigation_mm']
    g['IR_m3_ha'] = g['Irrigation_mm'] * 10.0
    return g


# --------------------------------------------------------------------------- #
#  Monthly irrigation (input to the WEI+-style water resilience WR,
#  Eq. 21)
# --------------------------------------------------------------------------- #

# Cumulative day-of-year at the start of each month (non-leap year)
_MONTH_STARTS = np.array([0, 31, 59, 90, 120, 151,
                          181, 212, 243, 273, 304, 334])


def doy_to_month(day_of_cycle: np.ndarray) -> np.ndarray:
    """Map 0-based day-of-cycle to 0-based month (0 = January)."""
    d = np.asarray(day_of_cycle, dtype=int)
    return np.searchsorted(_MONTH_STARTS, d, side='right') - 1


def growing_months(cfg: Config) -> np.ndarray:
    """0-based months that overlap any growth (irrigable) stage."""
    months = set()
    for sc in cfg.stages_config.values():
        if sc['type'] != 'G':
            continue
        for d in range(sc['start_date'], min(sc['end_date'], cfg.T_n)):
            months.add(int(doy_to_month(d)))
    return np.array(sorted(months), dtype=int)


def compute_monthly_irrigation(soil_df: pd.DataFrame) -> np.ndarray:
    """Monthly irrigation requirement per unit area (Eq. 10 by month).

    Parameters
    ----------
    soil_df : pd.DataFrame
        Output of ``simulate_soil_moisture`` (columns ``Sim_Year``,
        ``Day_Of_Cycle``, ``Irrigation`` [mm]).

    Returns
    -------
    np.ndarray
        Array of shape (N_years, 12) with monthly irrigation in
        m3 ha-1 (1 mm over 1 ha = 10 m3), aligned with the sorted
        ``Sim_Year`` values.
    """
    df = soil_df.copy()
    df['Month'] = doy_to_month(df['Day_Of_Cycle'].values)
    years = sorted(df['Sim_Year'].unique())
    out = np.zeros((len(years), 12))
    g = df.groupby(['Sim_Year', 'Month'])['Irrigation'].sum()
    for yi, yr in enumerate(years):
        for m in range(12):
            out[yi, m] = g.get((yr, m), 0.0) * 10.0  # mm -> m3/ha
    return out


# --------------------------------------------------------------------------- #
#  Convenience: daily theta from simple arrays
# --------------------------------------------------------------------------- #

def compute_daily_theta(
    cfg: Config,
    rain_arr: np.ndarray,
    pet_arr: np.ndarray,
    kc_arr: Optional[np.ndarray] = None,
    s0: Optional[float] = None,
) -> np.ndarray:
    """Daily soil moisture from 1-D arrays (no DataFrame overhead)."""
    s = cfg.s_fc if s0 is None else s0
    kc = cfg.kc_annual if kc_arr is None else kc_arr
    theta = np.zeros(len(rain_arr))
    for t in range(len(rain_arr)):
        theta[t] = s
        ks = feddes_stress(np.array([s]), cfg.sw, cfg.s_star, cfg.s_fc)[0]
        et = kc[t] * ks * pet_arr[t]
        leak = leakage_rate(s, cfg) * cfg.Zr * cfg.porosity
        s = np.clip(s + (rain_arr[t] - et - leak) / (cfg.Zr * cfg.porosity),
                    0.0, cfg.porosity)
    return theta
