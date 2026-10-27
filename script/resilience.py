"""
Resilience indicator system (paper Section 2.3 and Supporting Text S5).

Five bounded components computed from annual model (or observed) series
over trailing 5-year windows, aggregated geometrically into three
dimension indices and an overall index.  All scores lie in (0, 1].

Components (annual scores; w = trailing window of length `window`):

  Production
      R1_bal  = exp( -ln^2(S_t / D_t) / (2 s_b^2) )      supply-demand
               alignment; D_t = kappa_t * P_ref**eta is the demand
               benchmark tonnage at the fixed reference price
               P_ref = kappa_anchor_price (calendar-anchored)
      R1_stab = 1 / (1 + SD_w(d ln S_t) / v_S)            production
               stability
      R_S     = sqrt(R1_bal * R1_stab)
  Income
      m_t     = (P_t * Y_nbs - C_t) / (P_t * Y_nbs)       pre-subsidy
               market margin at the NBS-equivalent yield basis
      R2_prof = mean_w[ 1 / (1 + exp(-m_t / m0)) ]        profitability
      R2_stab = 1 / (1 + SD_w(m_t) / v_I)                 margin stability
      R_I     = sqrt(R2_prof * R2_stab)
  Water
      R3_wdf  = mean over growing months of
               min(1, M * W_avail / W_req)                water-demand
               fulfilment (WEI+ style threshold M)
      R_W     = R3_wdf
  Overall
      R       = (R_S * R_I * R_W) ** (1/3)

The normalization parameters s_b, v_S, m0, v_I were calibrated on a
40-member diagnostic ensemble and are stored in
``config/resilience_params.yaml``.
"""
from __future__ import annotations

import numpy as np

# Calibrated normalization parameters (single source of truth:
# config/resilience_params.yaml; mirrored here for convenience).
PAR = dict(s_b=0.30, v_S=0.07, m0=0.10, v_I=0.05)
WINDOW = 5          # trailing-window length (years)
WEI_M = 0.2         # WEI+ water-stress threshold M (available-water share)
ANCHOR_YEAR = 2014  # calendar anchor for kappa_t and the cost trend

COMPS = ["R1_bal", "R1_stab", "R2_prof", "R2_stab", "R3_wdf"]
DIMS = ["R_S", "R_I", "R_W", "R"]
IND9 = ["R1_bal", "R1_stab", "R_S", "R2_prof", "R2_stab", "R_I",
        "R3_wdf", "R_W", "R"]


# ------------------------------------------------------------------ windows
def trailing_mean(x, w):
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    for i in range(w - 1, len(x)):
        out[i] = np.nanmean(x[i - w + 1:i + 1])
    return out


def trailing_sd(x, w):
    """SD over a trailing window of an already-differenced series."""
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    for i in range(w - 1, len(x)):
        out[i] = np.nanstd(x[i - w + 2:i + 1])
    return out


def window_sd_of_level(x, w):
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    for i in range(w - 1, len(x)):
        out[i] = np.nanstd(x[i - w + 1:i + 1])
    return out


def logistic(x):
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-x))


# --------------------------------------------------------------- indicators
def components_from_series(S, D, m, s_water, par=None, window=None):
    """Five components + three dimensions + overall index from annual
    series.

    Parameters
    ----------
    S : (T,) aggregate supply (t)
    D : (T,) demand benchmark tonnage (t)
    m : (T,) pre-subsidy market margin (-)
    s_water : (T,) annual water-demand fulfilment score (-)
    par : normalization parameters (default PAR)
    window : trailing-window length (default WINDOW)
    """
    par = PAR if par is None else par
    window = WINDOW if window is None else window
    B = np.exp(-np.log(S / D) ** 2 / (2 * par["s_b"] ** 2))
    R1_bal = trailing_mean(B, window)
    dlnS = np.concatenate([[np.nan], np.diff(np.log(S))])
    R1_stab = 1.0 / (1.0 + trailing_sd(dlnS, window) / par["v_S"])
    R_S = np.sqrt(R1_bal * R1_stab)
    R2_prof = trailing_mean(logistic(m / par["m0"]), window)
    R2_stab = 1.0 / (1.0 + window_sd_of_level(m, window) / par["v_I"])
    R_I = np.sqrt(R2_prof * R2_stab)
    R3_wdf = trailing_mean(s_water, window)
    R_W = R3_wdf.copy()
    with np.errstate(invalid="ignore"):
        R = (R_S * R_I * R_W) ** (1.0 / 3.0)
    return dict(R1_bal=R1_bal, R1_stab=R1_stab, R_S=R_S,
                R2_prof=R2_prof, R2_stab=R2_stab, R_I=R_I,
                R3_wdf=R3_wdf, R_W=R_W, R=R)


def water_annual_score(A, w_req_monthly, wa_monthly, gm, M=None):
    """Annual water-demand fulfilment score.

    Mean over growing months of min(1, M * W_avail / W_req), where
    W_req = A * irrigation requirement per hectare.

    Parameters
    ----------
    A : (T,) sown area (ha)
    w_req_monthly : (T, 12) irrigation requirement (m3/ha per month)
    wa_monthly : (T, 12) or (12,) available water (m3 per month)
    gm : growing-month indices (0-based)
    M : WEI+ threshold (default WEI_M)
    """
    M = WEI_M if M is None else M
    with np.errstate(divide="ignore", invalid="ignore"):
        Q = np.minimum(1.0, M / (w_req_monthly / wa_monthly))
    return np.nanmean(Q[:, gm], axis=1)


def demand_benchmark_tonnage(cfg, years, anchor_year=None):
    """Demand benchmark D_t = kappa_t * P_ref**eta at the fixed
    reference price, calendar-anchored at anchor_year."""
    anchor_year = ANCHOR_YEAR if anchor_year is None else anchor_year
    P_ref = float(cfg.kappa_anchor_price)
    kap = np.array([cfg.demand_kappa(int(y - anchor_year))
                    for y in years])
    return kap * P_ref ** cfg.eta
