"""
cahed.fast_sim
==============

Fast columnar re-implementation of the CAHED coupled simulation for
large Monte-Carlo ensembles (GLUE, sensitivity analysis).

The numerical protocol is identical to ``weather.generate_*``,
``soil_water.simulate_soil_moisture`` and ``market.run_coupled_simulation``
(the per-day irrigation rule sets s -> s_target and the balance step
receives irrigation=0, matching the reference implementation), but
works on plain numpy arrays and returns only annual aggregates, giving
a ~40x speed-up over the DataFrame-based reference.

New accounting option: gross irrigation withdrawal
    W_gross = A * IR_net / eta_irrig
is reported so that irrigation efficiency enters as an explicit
sensitivity parameter (Experiment 2).

Resilience metrics SR / IR / WR (Section 2.3, Eqs. 19-21) are
attached to the annual output of ``fast_coupled`` (window means over
the simulation) so that global sensitivity analysis can target them
directly; WR requires the monthly irrigation array.

Verification
------------
- fast_soil yields identical annual yield / irrigation series as the
  reference when fed the same rain/PET arrays (max abs diff = 0).
- fast_coupled reproduces run_coupled_simulation exactly when fed the
  same annual series and master seed (identical RNG call order in the
  market loop).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .config import Config
from .weather import build_baseline_params


# --------------------------------------------------------------------------- #
#  Day-index helpers
# --------------------------------------------------------------------------- #

def make_day_arrays(cfg: Config):
    """Per-day stage index and growth-day mask (length T_n)."""
    stage_idx = np.zeros(cfg.T_n, dtype=np.int64)
    growth = np.zeros(cfg.T_n, dtype=bool)
    for i, sid in enumerate(sorted(cfg.stages_config.keys())):
        sc = cfg.stages_config[sid]
        stage_idx[sc['start_date']:sc['end_date']] = i
        if sc['type'] == 'G':
            growth[sc['start_date']:sc['end_date']] = True
    return stage_idx, growth


# --------------------------------------------------------------------------- #
#  Fast weather generation (Eqs. 1-3)
# --------------------------------------------------------------------------- #

def gen_weather_fast(
    cfg: Config,
    baseline_params: Dict[str, np.ndarray],
    freq_scale: float = 1.0,
    alpha_scale: float = 1.0,
    pet_scale: float = 1.0,
    seed: Optional[int] = None,
):
    """Array-only MMPP precipitation + cosine PET.

    ``freq_scale`` scales the wet-day frequency (stationary wet
    probability via trans_dw and the stage-reset probability 1-q01);
    ``alpha_scale`` scales the mean wet-day depth (mm).
    """
    rng = np.random.default_rng(seed if seed is not None else cfg.seed)
    N, T_n = cfg.N, cfg.T_n
    T_total = N * T_n

    bp = baseline_params
    q01 = 1.0 - np.clip((1.0 - bp['q01']) * freq_scale, 0.0, 1.0)
    alp = bp['alpha_vals'] * alpha_scale
    tdw = np.clip(bp['trans_dw'] * freq_scale, 0.0, 1.0)
    twd = bp['trans_wd']

    stage_idx, _ = make_day_arrays(cfg)
    stage_full = np.tile(stage_idx, N)

    wet = np.zeros(T_total, dtype=np.int8)
    rain = np.zeros(T_total)
    wet[0] = 1 if rng.random() < 0.0486 else 0
    prev = stage_full[0]
    for t in range(T_total):
        sidx = stage_full[t]
        if sidx != prev:
            wet[t] = 1 if rng.random() < (1.0 - q01[sidx]) else 0
            prev = sidx
        elif t > 0:
            p = tdw[sidx] if wet[t - 1] == 0 else 1.0 - twd[sidx]
            wet[t] = 1 if rng.random() < p else 0
        if wet[t]:
            rain[t] = rng.exponential(alp[sidx])

    days = np.tile(np.arange(T_n), N)
    pet = ((3.37 + 3.04 * np.cos(2.0 * np.pi * (days - 182) / T_n))
           * pet_scale
           + rng.normal(0.0, 0.60, T_total))
    np.clip(pet, 0.5, None, out=pet)
    return rain, pet


# --------------------------------------------------------------------------- #
#  Fast soil-water balance (Eqs. 2-10)
# --------------------------------------------------------------------------- #

def fast_soil(
    cfg: Config,
    rain: np.ndarray,
    pet: np.ndarray,
    s_irrig: Optional[float] = None,
    s_target: Optional[float] = None,
    s0: Optional[float] = None,
    return_monthly: bool = False,
    return_et: bool = False,
    start_year: Optional[int] = None,
):
    """Annual yield (kg/ha) and net irrigation (m3/ha) from arrays.

    Identical update order to ``simulate_soil_moisture``: irrigation
    rule at start of day (sets s -> s_target), then the balance step
    with infiltration truncation.  Deep percolation is identically
    zero because s <= porosity < s_fc in the calibrated setup.

    Parameters
    ----------
    start_year : int, optional
        Calendar year of simulation year 0.  When the EXP2 v2
        technology trend is enabled (``cfg.tech_trend_enabled``), the
        potential yield of year ``y`` is Omega_t * Y_max with
        Omega_t = clip((intercept + slope * (start_year + y)) / Y_max,
        0, 1).  When omitted, the static Y_max is used even if the
        trend is enabled.
    return_monthly : bool
        If True, also return the (N, 12) monthly irrigation array
        [m3 ha-1] used by the WEI+-style water-use resilience metric
        WR (Eq. 21).
    return_et : bool
        If True, also return the annual actual-ET and rainfall-
        infiltration totals ``(ET_mm, RINF_mm)`` [mm], used to
        compute the green/blue water footprint in the crop
        calibration stage (green ET ~ min(ET, RINF)).

    Return combinations: default ``(Y_kg_ha, IR_m3_ha)``; monthly adds
    ``IR_monthly``; ET adds ``(ET_mm, RINF_mm)`` after all other items.
    """
    s_ir = cfg.s_irrig if s_irrig is None else s_irrig
    s_tg = cfg.s_target if s_target is None else s_target
    Zp = cfg.Zr * cfg.porosity
    por = cfg.porosity
    sw, s_star = cfg.sw, cfg.s_star
    kc = cfg.kc_annual
    stage_idx, growth = make_day_arrays(cfg)
    n_stages = len(cfg.stages_config)
    N, T_n = cfg.N, cfg.T_n
    month_of_day = _month_of_day(T_n)

    s = cfg.s_fc if s0 is None else s0
    irr_mm = np.zeros(N)
    et_mm = np.zeros(N)
    rinf_mm = np.zeros(N)
    irr_mm_monthly = np.zeros((N, 12)) if return_monthly else None
    ks_sum = np.zeros((N, n_stages))
    ks_cnt = np.zeros((N, n_stages))

    for y in range(N):
        base = y * T_n
        for d in range(T_n):
            t = base + d
            if growth[d] and s < s_ir:
                i_d = (s_tg - s) * Zp
                irr_mm[y] += i_d
                if return_monthly:
                    irr_mm_monthly[y, month_of_day[d]] += i_d
                s = s_tg
            ks = 0.0
            if s > sw:
                ks = (s - sw) / (s_star - sw)
                if ks > 1.0:
                    ks = 1.0
            et = kc[d] * ks * pet[t]
            et_mm[y] += et
            avail = (por - s) * Zp
            rinf = rain[t] if rain[t] < avail else avail
            rinf_mm[y] += rinf
            s = s + (rinf - et) / Zp
            if s < 0.0:
                s = 0.0
            elif s > por:
                s = por
            si = stage_idx[d]
            ks_sum[y, si] += ks
            ks_cnt[y, si] += 1.0

    # Jensen relative yield (Eq. 8) over growth stages with lambda > 0
    lam = [cfg._jensen_lambdas.get(sid, 0.0)
           for sid in sorted(cfg.stages_config.keys())]
    rel = np.ones(N)
    for k in range(n_stages):
        if lam[k] <= 0:
            continue
        ks_mean = ks_sum[:, k] / np.maximum(ks_cnt[:, k], 1.0)
        rel *= np.clip(ks_mean, 1e-6, 1.0) ** lam[k]

    # EXP2 v2 technology trend: Y_t = Omega_t * Y_max * rel_t
    if cfg.tech_trend_enabled and start_year is not None:
        omega_vec = np.clip(
            (cfg.tech_trend_intercept
             + cfg.tech_trend_slope * (start_year + np.arange(N)))
            / cfg.Y_max, 0.0, 1.0)
    else:
        omega_vec = np.ones(N)
    Y_kg_ha = cfg.Y_max * omega_vec * rel
    IR_m3_ha = irr_mm * 10.0
    out = [Y_kg_ha, IR_m3_ha]
    if return_monthly:
        out.append(irr_mm_monthly * 10.0)
    if return_et:
        out.extend([et_mm, rinf_mm])
    return tuple(out) if len(out) > 2 else (Y_kg_ha, IR_m3_ha)


def _month_of_day(T_n: int = 365) -> np.ndarray:
    """0-based month index for each day of the (non-leap) cycle."""
    starts = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
    return np.searchsorted(starts, np.arange(T_n), side='right') - 1


# --------------------------------------------------------------------------- #
#  Resilience window means (Section 2.3, Eqs. 19-21)
# --------------------------------------------------------------------------- #

def _inv_sq_cv(x: np.ndarray, window: int) -> float:
    """Mean of the trailing-window inverse squared CV (SR / IR form)."""
    x = np.asarray(x, dtype=float)
    vals = []
    for i in range(window - 1, len(x)):
        w = x[i - window + 1:i + 1]
        mu, sd = np.mean(w), np.std(w)
        vals.append(np.inf if sd == 0.0 else (mu / sd) ** 2)
    return float(np.mean(vals)) if vals else np.nan


def _wr_mean(ir_monthly: np.ndarray, area: np.ndarray,
             w_avail_monthly: np.ndarray, window: int,
             threshold: float) -> float:
    """Mean WR over trailing windows (Eq. 21).

    Growing-season months are taken as months with any irrigation.
    """
    ir = np.asarray(ir_monthly, dtype=float)
    gm = np.where(ir.sum(axis=0) > 0)[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (ir * np.asarray(area, dtype=float)[:, None]) / w_avail_monthly
    ok = ratio <= threshold
    vals = [float(ok[i - window + 1:i + 1][:, gm].mean())
            for i in range(window - 1, len(area))]
    return float(np.mean(vals)) if vals else np.nan


# --------------------------------------------------------------------------- #
#  Fast coupled simulation (Section 2.2.3, Eq. 18)
# --------------------------------------------------------------------------- #

def fast_coupled(
    cfg: Config,
    Y_kg_ha: np.ndarray,
    IR_m3_ha: np.ndarray,
    start_year: int = 2014,
    policy: bool = True,
    seed: Optional[int] = None,
    tp_scale: float = 1.0,
    a0_init: Optional[float] = None,
    pe0_init: Optional[float] = None,
    c0_init: Optional[float] = None,
    ir_monthly: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Market loop identical to ``run_coupled_simulation`` (same RNG
    call order) but array-based.  Returns annual trajectories, the
    gross withdrawal series W_gross = A * IR / eta_irrig, and the
    window-mean resilience metrics SR / IR (/ WR when ``ir_monthly``
    is given).

    ``a0_init`` / ``pe0_init`` / ``c0_init`` override the initial state
    (sown area, lagged effective price, unit cost) for initial-
    condition robustness tests; the calibration anchors (b0, kappa0)
    are left unchanged."""
    rng = np.random.default_rng(seed if seed is not None else cfg.seed + 100)
    n = len(Y_kg_ha)

    b0 = (1.0 - cfg.b2) * np.log(cfg.A0) - cfg.b1 * np.log(cfg.P0)
    ln_kappa0 = cfg.ln_kappa0
    log1g = np.log(1.0 + cfg.g_d)

    A_prev = cfg.A0 if a0_init is None else a0_init
    P_eff_prev = cfg.P0 if pe0_init is None else pe0_init
    C_prev = cfg.C0 if c0_init is None else c0_init

    out = {k: np.zeros(n) for k in
           ('A', 'Y', 'IR', 'S', 'W', 'W_gross', 'P', 'TP',
            's_t', 'P_eff', 'C', 'profit', 'fiscal')}

    for t in range(n):
        year = start_year + t
        # (1) Planting: reduced-form acreage response to the lagged
        #     effective price (Eqs. 14, 16)
        C = C_prev * (1.0 + cfg.c_trend) * np.exp(rng.normal(0.0, cfg.c_cv))
        lnA = (b0 + cfg.b1 * np.log(P_eff_prev) + cfg.b2 * np.log(A_prev)
               + cfg.b3 * np.log(C / cfg.C0)
               + rng.normal(0.0, cfg.area_noise_sd))
        A = float(np.exp(lnA))

        # (2-3) Growing season (precomputed) + market clearing (Eqs. 11-13)
        Y = Y_kg_ha[t] / 1000.0
        IR = IR_m3_ha[t]
        S = A * Y
        W = A * IR
        kappa = float(np.exp(ln_kappa0 + t * log1g))
        P = float(np.exp((np.log(S) - np.log(kappa)) / cfg.eta))

        # (4) Policy settlement (Eqs. 15-17)
        if policy:
            TP = float(cfg.target_price.get(year, cfg.tp_default)) * tp_scale
        else:
            TP = 0.0
        wedge = max(TP - P, 0.0)
        if wedge > 0.0 and S > 0.0:
            s_t = (wedge if cfg.Q_cap is None
                   else wedge * min(1.0, cfg.Q_cap / S))
        else:
            s_t = 0.0
        P_eff = P + s_t
        profit = (P * Y - C) * A + s_t * S                   # Eq. 17
        fiscal = s_t * S                                     # Eq. 17

        out['A'][t] = A; out['Y'][t] = Y; out['IR'][t] = IR
        out['S'][t] = S; out['W'][t] = W
        out['W_gross'][t] = A * IR / cfg.eta_irrig
        out['P'][t] = P; out['TP'][t] = TP
        out['s_t'][t] = s_t; out['P_eff'][t] = P_eff; out['C'][t] = C
        out['profit'][t] = profit; out['fiscal'][t] = fiscal

        # (5) State updating
        A_prev, P_eff_prev, C_prev = A, P_eff, C

    # Resilience metrics (Section 2.3, Eqs. 19-21): window means
    win = cfg.resilience_window
    out['SR'] = _inv_sq_cv(out['S'], win)
    out['IR'] = _inv_sq_cv(out['profit'], win)
    if ir_monthly is not None:
        ir = np.asarray(ir_monthly, dtype=float)
        gm = np.where(ir.sum(axis=0) > 0)[0]
        wa = np.zeros(12)
        wa[gm] = cfg.W_threshold / max(len(gm), 1)
        out['WR'] = _wr_mean(ir, out['A'], np.tile(wa, (n, 1)),
                             win, cfg.wei_threshold)
    return out


# --------------------------------------------------------------------------- #
#  One-call wrapper: parameter dict -> summary outputs (for SA)
# --------------------------------------------------------------------------- #

# Sensitivity-analysis output definitions: (key, unit, transform).
# SR / IR / WR are the window-mean resilience metrics of Section 2.3
# (Eqs. 19-21) computed by fast_coupled.
SA_OUTPUTS = [
    ('Y',        't/ha',      lambda o: o['Y'].mean()),
    ('A',        'Mha',       lambda o: o['A'].mean() / 1e6),
    ('W_gross',  'km3',       lambda o: o['W_gross'].mean() / 1e9),
    ('profit',   'bn CNY',    lambda o: o['profit'].mean() / 1e9),
    ('fiscal',   'bn CNY',    lambda o: o['fiscal'].mean() / 1e9),
    ('P',        'kCNY/t',    lambda o: o['P'].mean() / 1e3),
    ('SR',       '-',         lambda o: o['SR']),
    ('IR',       '-',         lambda o: o['IR']),
    ('WR',       '-',         lambda o: o['WR']),
    ('CV_S',     '-',         lambda o: o['S'].std() / o['S'].mean()),
    ('CV_profit', '-',        lambda o: o['profit'].std()
                                / max(abs(o['profit'].mean()), 1e-9)),
]


def run_sa_point(cfg_kwargs: Dict, freq_scale: float, alpha_scale: float,
                 seed: int, N: int = 30) -> np.ndarray:
    """Run one coupled simulation and return the SA output vector."""
    kw = dict(cfg_kwargs)
    tp_scale = kw.pop('_tp_scale', 1.0)
    cfg = Config(N=N, **kw)
    bp = build_baseline_params()
    rain, pet = gen_weather_fast(cfg, bp, freq_scale, alpha_scale,
                                 seed=seed)
    # start_year matches the fast_coupled default so the technology
    # trend (if enabled) is evaluated on the same calendar years.
    Y, IR, IRm = fast_soil(cfg, rain, pet, return_monthly=True,
                           start_year=2014)
    out = fast_coupled(cfg, Y, IR, seed=seed, tp_scale=tp_scale,
                       ir_monthly=IRm)
    return np.array([f(out) for _, _, f in SA_OUTPUTS])
