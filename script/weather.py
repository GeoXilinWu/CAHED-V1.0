"""
cahed.weather
=============

Stochastic weather generation module (Eq. 1; Supplementary Text S1.1).

- ``build_baseline_params`` : construct MMPP parameter arrays
- ``generate_precipitation_mmmpp`` : Markov-modulated marked Poisson process
- ``generate_pet_cosine`` : cosine-based PET with Gaussian noise
  (Supplementary Text S1.1)
- ``merge_weather`` : combine precipitation and PET into one DataFrame

Changes from v0.1
-----------------
- The 2-state Markov transition matrix (p_dw, p_wd; Eq. 1) is now an
  explicit, calibratable parameter instead of a hard-coded constant.

Changes from v1.3
-----------------
- Wet-day marks can be drawn from a Gamma distribution with
  per-stage shape ``kappa_mmpp`` (mean preserved at ``alpha_mmpp``);
  the station calibration shows Gamma(k ~ 0.3) systematically
  outperforming the exponential (AIC, quantile adequacy).
  ``kappa_mmpp = 1`` reproduces the exponential mark distribution
  exactly (default).
"""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import pandas as pd

from .config import Config

# Wet-day depth threshold (mm) used in the station calibration;
# Gamma marks are fitted on, and generated as, (depth - threshold).
WET_DAY_THRESHOLD = 0.1


def build_baseline_params(
    stage_params: Optional[Dict[str, Dict]] = None,
    stage_trans: Optional[Dict[str, tuple]] = None,
) -> Dict[str, np.ndarray]:
    """Build baseline MMPP parameter arrays.

    Parameters
    ----------
    stage_params : dict, optional
        Custom stage-level parameters with keys ``q01``, ``lambda_mmpp``,
        ``alpha_mmpp`` and, optionally, ``kappa_mmpp`` (Gamma shape of
        the wet-day depth distribution; default 1.0 = exponential).
        If *None*, sensible defaults are used.
    stage_trans : dict, optional
        Custom per-stage transition probabilities
        ``{stage: (p_dw, p_wd)}`` (Eq. 1).  If *None*, the pooled
        station MLEs documented below are used.

    Returns
    -------
    dict
        Keys: ``q01``, ``lambda_vals``, ``alpha_vals`` (arrays per stage),
        ``trans_dw`` / ``trans_wd`` (per-stage transition probabilities,
        Eq. 1), ``kappa_vals`` (per-stage Gamma shape of the wet-day
        depth distribution; 1.0 = exponential), and ``trans`` (pooled
        2x2 matrix kept for backward compatibility).  The per-stage
        arrays are the ones used by ``generate_precipitation_mmmpp``;
        their stationary wet probabilities reproduce the observed
        per-stage wet-day frequencies almost exactly, which a single
        pooled matrix cannot.
    """
    if stage_params is None:
        # Defaults calibrated from 5 Tarim-basin stations (Kuqa, Korla,
        # Bachu, Tikanlik, Shache), 1981-2020 daily observations, using
        # only near-complete station-years (>= 360 valid days; 132
        # station-years) so that missing observations do not bias the
        # wet-day statistics.  Pooled onto the 4-stage calendar
        # (S1: DOY 1-104, S2: 105-195, S3: 196-285, S4: 286-365).
        # Station-calibrated set (manuscript Table S3): per-stage
        # wet-day frequency lambda, shifted-Gamma marks with shape
        # kappa_mmpp = k and mean depth alpha_mmpp = h0 + k*theta
        # (h0 = WET_DAY_THRESHOLD = 0.1 mm), and per-stage transitions
        # (p_dw, p_wd) below.  q01 = 1 - lambda = stationary dry
        # probability.  Pooled annual precipitation = 66.8 mm;
        # growing season (S2+S3) = 53.3 mm.
        stage_params = {
            'S1': {'q01': 0.958, 'lambda_mmpp': 0.042,
                   'alpha_mmpp': 2.04, 'kappa_mmpp': 0.316},
            'S2': {'q01': 0.888, 'lambda_mmpp': 0.112,
                   'alpha_mmpp': 2.78, 'kappa_mmpp': 0.372},
            'S3': {'q01': 0.888, 'lambda_mmpp': 0.112,
                   'alpha_mmpp': 2.48, 'kappa_mmpp': 0.355},
            'S4': {'q01': 0.970, 'lambda_mmpp': 0.030,
                   'alpha_mmpp': 1.91, 'kappa_mmpp': 0.286},
        }

    sids = sorted(stage_params.keys())
    q01 = np.array([stage_params[s]['q01'] for s in sids])
    lam = np.array([stage_params[s]['lambda_mmpp'] for s in sids])
    alp = np.array([stage_params[s]['alpha_mmpp'] for s in sids])
    kap = np.array([stage_params[s].get('kappa_mmpp', 1.0)
                    for s in sids])

    # Default 2-state transition matrix; overridable via calibration.
    # Pooled MLE over the 132 complete station-years, 1981-2020:
    # p_dw = 0.0486 (dry spells persistent, mean ~21 d),
    # p_wd = 0.7230 (wet spells short, mean ~1.4 d) -- consistent with
    # the hyper-arid Tarim climate.
    trans = np.array([[0.9514, 0.0486],   # dry -> (dry, wet)
                      [0.7230, 0.2770]])  # wet -> (dry, wet)

    # Per-stage transition MLEs (manuscript Table S3; within-stage
    # transitions only).  Stationary wet probabilities:
    # S1 0.042, S2 0.112, S3 0.112, S4 0.030 -- matching the observed
    # stage wet-day frequencies (lambda above).
    if stage_trans is not None:
        trans_dw = np.array([stage_trans[s][0] for s in sids])
        trans_wd = np.array([stage_trans[s][1] for s in sids])
    else:
        trans_dw = np.array([0.027, 0.088, 0.092, 0.019])
        trans_wd = np.array([0.617, 0.697, 0.732, 0.619])

    return {'q01': q01, 'lambda_vals': lam, 'alpha_vals': alp,
            'kappa_vals': kap,
            'trans': trans, 'trans_dw': trans_dw, 'trans_wd': trans_wd}


def generate_precipitation_mmmpp(
    cfg: Config,
    baseline_params: Dict[str, np.ndarray],
    seed: Optional[int] = None,
    climate_volatility: float = 0.0,
) -> pd.DataFrame:
    """Generate precipitation using the MMPP model (Eq. 1).

    Wet-day depths are drawn from an exponential distribution by
    default; if ``baseline_params['kappa_vals']`` contains values
    != 1, depths follow ``WET_DAY_THRESHOLD + Gamma(kappa,
    (alpha - threshold)/kappa)`` — a shifted Gamma with unchanged mean
    ``alpha``, matching the station calibration in which Gamma marks
    with kappa ~ 0.3 (fitted on depths minus the 0.1 mm wet-day threshold)
    reproduce the observed depth distribution far better than the
    exponential.

    Parameters
    ----------
    cfg : Config
        Model configuration.
    baseline_params : dict
        Output from ``build_baseline_params`` (may contain a custom
        ``trans`` matrix).
    seed : int, optional
        Random seed.  If *None*, uses ``cfg.seed``.
    climate_volatility : float
        Additional volatility multiplier (0 = baseline).

    Returns
    -------
    pd.DataFrame
        Columns: ``Sim_Year``, ``Day_Of_Cycle``, ``Stage``,
        ``Rain_Intensity``, ``Wet_State``.
    """
    rng = np.random.default_rng(seed if seed is not None else cfg.seed)
    T_total = cfg.N * cfg.T_n

    q01 = baseline_params['q01'].copy()
    lam_arr = baseline_params['lambda_vals'].copy()
    alp_arr = baseline_params['alpha_vals'].copy()
    # Gamma shape of wet-day depth marks (1.0 = exponential, Eq. 1)
    kap_arr = np.asarray(baseline_params.get(
        'kappa_vals', np.ones(len(alp_arr))), dtype=float)
    trans = np.asarray(baseline_params.get(
        'trans', [[0.7, 0.3], [0.4, 0.6]]), dtype=float)
    # Per-stage transition probabilities (preferred when present)
    trans_dw = baseline_params.get('trans_dw')
    trans_wd = baseline_params.get('trans_wd')
    use_stage_trans = trans_dw is not None and trans_wd is not None
    if use_stage_trans:
        trans_dw = np.asarray(trans_dw, dtype=float)
        trans_wd = np.asarray(trans_wd, dtype=float)

    # Apply climate volatility (scenario knob)
    if climate_volatility > 0:
        lam_arr *= (1.0 + climate_volatility
                    * rng.normal(0, 0.1, size=len(lam_arr)))
        alp_arr *= (1.0 + climate_volatility
                    * rng.normal(0, 0.1, size=len(alp_arr)))

    # Stage lookup
    stage_ids = []
    for sid in sorted(cfg.stages_config.keys()):
        sc = cfg.stages_config[sid]
        stage_ids.extend([sid] * (sc['end_date'] - sc['start_date']))
    stage_ids = np.array(stage_ids * cfg.N)[:T_total]

    stage_idx = np.zeros(T_total, dtype=int)
    for i, sid in enumerate(sorted(cfg.stages_config.keys())):
        stage_idx[stage_ids == sid] = i

    wet_state = np.zeros(T_total, dtype=int)
    wet_state[0] = 1 if rng.random() < trans[0, 1] else 0
    rain_intensity = np.zeros(T_total)
    prev_stage = stage_idx[0]

    for t in range(T_total):
        sidx = stage_idx[t]
        # Stage transition -> re-draw state from stationary wet frequency
        if sidx != prev_stage:
            wet_state[t] = 1 if rng.random() < (1.0 - q01[sidx]) else 0
            prev_stage = sidx
        elif t > 0:
            if use_stage_trans:
                p_wet = (trans_dw[sidx] if wet_state[t - 1] == 0
                         else 1.0 - trans_wd[sidx])
                wet_state[t] = 1 if rng.random() < p_wet else 0
            else:
                wet_state[t] = rng.choice(2, p=trans[wet_state[t - 1]])

        if wet_state[t] == 1:
            if kap_arr[sidx] == 1.0:
                rain_intensity[t] = rng.exponential(alp_arr[sidx])
            else:
                # Shifted Gamma mark: WET_DAY_THRESHOLD + Gamma(kappa,
                # (alpha - threshold)/kappa), matching the calibration
                # (fit on depths >= threshold) and preserving the mean
                # wet-day depth alpha.
                rain_intensity[t] = WET_DAY_THRESHOLD + rng.gamma(
                    kap_arr[sidx],
                    max(alp_arr[sidx] - WET_DAY_THRESHOLD, 1e-6)
                    / kap_arr[sidx])

    years = np.repeat(np.arange(cfg.N), cfg.T_n)[:T_total]
    days = np.tile(np.arange(cfg.T_n), cfg.N)[:T_total]

    return pd.DataFrame({
        'Sim_Year': years,
        'Day_Of_Cycle': days,
        'Stage': stage_ids,
        'Rain_Intensity': rain_intensity,
        'Wet_State': wet_state,
    })


def generate_pet_cosine(
    cfg: Config,
    pet_mean: float = 3.37,
    pet_amp: float = 2.81,
    t_max: float = 183.7,
    noise_std: float = 0.59,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate PET: seasonal cosine + Gaussian noise (Suppl. S1.1).

    PET(t) = pet_mean + pet_amp * cos(2 pi (t - t_max) / T_n) + eps(t).

    Defaults fitted to Hargreaves ET0 computed from the 5 station
    records (1981-2020; station latitudes 38.4-41.7 N): pooled annual
    ET0 = 1230 mm, cosine peak at DOY 183.7 (early July), daily
    residual std = 0.59 mm/d (manuscript Table S4).
    Kc-weighted growing-season crop ET (Kc 0.65/0.95 over DOY 105-285)
    reproduces the 640-720 mm realized-ET benchmark of Jans et al.
    (2021) for China under soil-water stress.
    """
    rng = np.random.default_rng(seed if seed is not None else cfg.seed)
    T_total = cfg.N * cfg.T_n

    days = np.tile(np.arange(cfg.T_n), cfg.N)[:T_total]
    pet = (pet_mean
           + pet_amp * np.cos(2.0 * np.pi * (days - t_max) / cfg.T_n)
           + rng.normal(0, noise_std, size=T_total))
    pet = np.clip(pet, 0.5, None)

    years = np.repeat(np.arange(cfg.N), cfg.T_n)[:T_total]
    return pd.DataFrame({'Sim_Year': years, 'Day_Of_Cycle': days, 'PET': pet})


def merge_weather(
    df_precip: pd.DataFrame,
    df_pet: pd.DataFrame,
) -> pd.DataFrame:
    """Combine precipitation and PET into a single DataFrame."""
    return df_precip.merge(
        df_pet, on=['Sim_Year', 'Day_Of_Cycle'], how='outer',
        validate='one_to_one')
