"""
cahed.config
============

Configuration class for the CAHED (Coupled Agro-Hydro-Economic Dynamics)
model.

Provides ``Config`` -- a container for all model parameters:
stage definitions, soil properties, vegetation traits, irrigation rule,
market/demand parameters, Nerlove behavioural coefficients,
target-price policy settings, and resilience settings.

Equation numbers in the docstrings refer to the model description
(Section 2 of the manuscript, V2 numbering).
"""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np


class Config:
    """Configuration container for the coupled model.

    Parameters
    ----------
    N : int
        Number of simulation years (default: 30).
    T_n : int
        Days per year / cycle (default: 365).
    S_n : int
        Number of growth stages (default: 4).
    seed : int
        Random seed for reproducibility (default: 2026).

    Hydrological module (Eqs. 2-10)
    -------------------------------
    s_fc, Ks, omega, porosity, Zr, sw, s_star : soil/vegetation params.
        ``omega`` = 2b + 3 (Clapp-Hornberger exponent).
    s_irrig, s_target : float
        Irrigation trigger and target soil moisture (Eq. 9).
    Y_max : float
        Maximum potential yield (kg ha-1).

    Market-behavioural module (Eqs. 11-17)
    --------------------------------------
    eta : float
        Price elasticity of demand, < 0 (Eq. 12).
    g_d : float
        Demand-scale growth rate (Eq. 12).
    A0, P0, Y0_ref, C0 : float
        Baseline sown area (ha), price (CNY/t), reference yield (t/ha),
        and unit cost (CNY/ha) used to calibrate b0 and kappa0.
    b1, b2, b3 : float
        Nerlove reduced-form coefficients (Eq. 14): short-run price
        response, acreage persistence, cost sensitivity.
    area_noise_sd : float
        Standard deviation of the area-equation disturbance (Eq. 14).
    target_price : dict
        {year: target price CNY/t}; years absent use ``tp_default``.
    tp_default : float
        Default target price (18,600 CNY/t).
    Q_cap : float, optional
        Subsidized volume cap (t); ``None`` (default, V2 formulation)
        pays the wedge on the full crop (Eq. 15).
    c_trend, c_cv : float
        Cost growth rate and volatility.

    Resilience (Section 2.3, Eqs. 19-21)
    ------------------------------------
    resilience_window : int
        Moving-window length in years (default 5).
    W_threshold : float
        Annual withdrawal reference volume (m3), used as the annual
        available-water budget that ``compute_resilience_metrics``
        distributes over growing-season months when no monthly
        available-water series is provided (Eq. 21).
    wei_threshold : float
        Water-stress threshold M for WR (default 0.2; the conventional
        WEI+ criterion, Eurostat 2025): a growing-season month counts
        as compliant when irrigation withdrawal <= M * available water.
    """

    # ------------------------------------------------------------------ #
    #  Stage defaults (v0.3: aligned to Xinjiang cotton phenology,
    #  Chen et al. 2023 / Zhou et al. 2024: sowing mid-April ~ DOY 105,
    #  180-d growing season, harvest through December)
    # ------------------------------------------------------------------ #
    _DEFAULT_STAGE_LENGTHS: Dict[str, int] = {
        'S1': 104, 'S2': 91, 'S3': 90, 'S4': 80,
    }
    _DEFAULT_STAGE_TYPES: Dict[str, str] = {
        'S1': 'F', 'S2': 'G', 'S3': 'G', 'S4': 'H',
    }
    _DEFAULT_STAGE_LABELS: Dict[str, str] = {
        'S1': 'Fallow', 'S2': 'Growth-Early', 'S3': 'Growth-Late', 'S4': 'Harvest',
    }
    # FAO-56 style Kc for Xinjiang drip-irrigated cotton (Chen et al.
    # 2023; Chinese values take precedence over the LPJmL global setup
    # of Jans et al. 2021).  Kc-weighted seasonal PET with these values
    # reproduces the 640-720 mm crop-ET benchmark once soil-water
    # stress is accounted for.
    _DEFAULT_KC: Dict[str, float] = {
        'S1': 0.15, 'S2': 0.65, 'S3': 0.95, 'S4': 0.20,
    }
    # Jensen stage sensitivities -- calibrated:
    # lambda_S2 = 0.2082 (p05-p95 0.198-0.213), lambda_S3 = 0.4301
    # (0.415-0.437); the growth-late stage remains dominant
    # (lambda_S3 ~ 2x lambda_S2).
    _DEFAULT_JENSEN: Dict[str, float] = {
        'S2': 0.2082, 'S3': 0.4301,
    }

    def __init__(
        self,
        N: int = 30,
        T_n: int = 365,
        S_n: int = 4,
        seed: int = 2026,
        # Stage configuration
        stage_lengths: Optional[Dict[str, int]] = None,
        stage_types: Optional[Dict[str, str]] = None,
        stage_labels: Optional[Dict[str, str]] = None,
        kc_values: Optional[Dict[str, float]] = None,
        jensen_lambdas: Optional[Dict[str, float]] = None,
        # Soil parameters
        s_fc: float = 0.45,
        Ks: float = 35.0,
        omega: float = 4.5,
        porosity: float = 0.42,
        # Vegetation parameters
        Zr: float = 200.0,
        sw: float = 0.12,
        # Feddes stress threshold s* -- calibrated:
        # 0.2690 (behavioural p05-p95 0.265-0.271); coincides with the
        # calibrated refill target, i.e. drip scheduling keeps s at
        # the onset of stress.
        s_star: float = 0.2690,
        # Yield potential: Y_max is the fixed BIOPHYSICAL ceiling
        # (2400 kg/ha); the observed
        # technology trend is embedded via the time-varying potential
        # yield Y_pot,t = Omega_t * Y_max (see tech_trend_* below).
        Y_max: float = 2400.0,
        # --- Technology trend ------------------------------------------
        # Omega_t = clip((tech_trend_intercept + tech_trend_slope *
        # calendar_year) / Y_max, 0, 1), from the OLS trend of observed
        # Xinjiang lint yield 1987-2024 (+11.19 kg/ha/yr, R2 = 0.86).
        # Extrapolated linearly beyond 2024 until Omega = 1 (reached
        # around 2082), after which Y_pot = Y_max.  The trend applies
        # only when a calendar start_year is passed to the yield
        # functions (see soil_water.compute_yearly_yield /
        # fast_sim.fast_soil).
        tech_trend_enabled: bool = False,
        tech_trend_intercept: float = 0.0,   # kg/ha at calendar year 0
        tech_trend_slope: float = 0.0,       # kg/ha per year
        # Irrigation rule (Eq. 9) -- calibrated: KGE-based GLUE on raw
        # levels, 30 real-weather years of the Xinjiang cotton water-use
        # dataset, with the Omega_t technology path embedded.  Best set:
        # trigger 0.2075, refill
        # 0.2686 ~= s* -- frequent small drip irrigations holding the
        # root zone at the stress threshold.  Behavioural p05-p95
        # 0.204-0.209 / 0.261-0.272.
        s_irrig: float = 0.2075,
        s_target: float = 0.2686,
        eta_irrig: float = 0.70,
        irrigate: bool = True,
        # --- Demand (Eqs. 12-13) -- calibrated (Experiment 1). --------
        # eta is the EFFECTIVE price flexibility of regional supply in
        # the integrated national market (imports + national production
        # buffer Xinjiang supply shocks), not a textbook demand
        # elasticity: calibrated eta = -1.7, g_d = 6.5 %/yr (growth of
        # demand for Xinjiang lint as its national share rose from ~40 %
        # to >90 %).  The literature prior eta = -0.3 fails validation
        # (NSE < 0); the calibrated pair reproduces CCINDEX Sep-Nov
        # 2004-2024 with NSE = 0.14, RMSE = 2,885 CNY/t, PBIAS = -1.8 %
        # (policy window 2014-2024: NSE = 0.19, RMSE = 1,896).
        eta: float = -1.70,
        g_d: float = 0.065,
        # --- Baseline calibration anchors ----------------------------
        # C0 = 32,900 CNY/ha: Xinjiang cotton production cost in the
        # 2014 target-price base year (China cost-benefit survey data;
        # Xinjiang cost rose from 14,156 CNY/ha in 2004 to ~37,393 in
        # 2023).  Y0_ref = 1.90 t/ha lint confirmed by NBS data
        # (Xinjiang lint yield 1.88-2.32 t/ha over the record).
        A0: float = 1.8844e6,
        P0: float = 18000.0,
        Y0_ref: float = 1.90,
        C0: float = 32900.0,
        # --- Demand-scale anchor override ------------------------------
        # When BOTH are given, kappa_0 is anchored to the SIMULATED total
        # supply at the observed base-year price instead of A0*Y0_ref at
        # P0:  ln(kappa_0) = ln(kappa_anchor_supply)
        #                     - eta * ln(kappa_anchor_price).
        # This removes the cross-module yield-basis mismatch (hydro
        # module on the v2 water-dataset basis ~1.5 t/ha vs Y0_ref = 1.90
        # NBS lint basis): the demand scale is referenced to the model's
        # own simulated supply (A_sim x Y_sim), so the equilibrium price
        # starts from the observed base-year market price by construction.
        kappa_anchor_supply: Optional[float] = None,   # t (lint)
        kappa_anchor_price: Optional[float] = None,    # CNY/t
        # --- Nerlove behavioural block (Eq. 14) -- estimated ----------
        # OLS on the 2005-2023 Xinjiang panel (n = 17), speculative
        # spike years 2010-2011 excluded, lagged price floored at the
        # target price (P_eff, Eqs. 15-16; manuscript Table S9, model M1):
        # b1 = 0.875 (p = 0.016), b2 = 0.752 (p = 0.0006),
        # b3 = -0.499 (p = 0.098), R2 = 0.88, DW = 1.86.
        # Validation: in-sample NSE = 0.87; out-of-sample (expanding
        # window) R2 = 0.34, MAPE = 7.2 %.  Uncertainty ranges (95 %
        # CI): b1 [0.19, 1.56], b2 [0.39, 1.11], b3 [-1.10, 0.11],
        # noise sd [0.062, 0.139].
        b1: float = 0.875,
        b2: float = 0.752,
        b3: float = -0.499,
        area_noise_sd: float = 0.086,
        # --- Target price policy (Eqs. 15-16) -------------------------
        target_price: Optional[Dict[int, float]] = None,
        tp_default: float = 18600.0,
        Q_cap: Optional[float] = None,
        # --- Cost dynamics --------------------------------------------
        # c_trend = 2.1 %/yr from the post-2013 Xinjiang cost series
        # (the long-run 2004-2023 trend is ~5.2 %/yr but includes the
        # pre-subsidy inflation phase; post-2013 regime preferred).
        c_trend: float = 0.021,
        c_cv: float = 0.02,
        # --- Resilience (Section 2.3, Eqs. 19-21) ---------------------
        # W_threshold anchored to Li et al. (2023): Xinjiang cotton
        # total water footprint ~ 18e9 m3, blue share 80.5 %
        # (~14.5e9 m3); 12e9 m3 sits slightly below, +-25 % sensitivity
        # recommended.
        resilience_window: int = 5,
        W_threshold: float = 12.0e9,
        wei_threshold: float = 0.2,
    ) -> None:
        self.N = N
        self.T_n = T_n
        self.S_n = S_n
        self.seed = seed

        # Soil
        self.s_fc = s_fc
        self.Ks = Ks
        self.omega = omega
        self.porosity = porosity

        # Vegetation
        self.Zr = Zr
        self.sw = sw
        self.s_star = s_star

        # Yield
        self.Y_max = Y_max

        # Technology trend: Y_pot,t = Omega_t * Y_max
        self.tech_trend_enabled = bool(tech_trend_enabled)
        self.tech_trend_intercept = float(tech_trend_intercept)
        self.tech_trend_slope = float(tech_trend_slope)

        # Irrigation (Eq. 9)
        self.s_irrig = s_irrig
        self.s_target = s_target
        self.eta_irrig = eta_irrig
        self.irrigate = irrigate

        # Demand (Eqs. 12-13)
        self.eta = eta
        self.g_d = g_d

        # Baseline calibration
        self.A0 = A0
        self.P0 = P0
        self.Y0_ref = Y0_ref
        self.C0 = C0
        self.kappa_anchor_supply = kappa_anchor_supply
        self.kappa_anchor_price = kappa_anchor_price

        # Nerlove (Eq. 14)
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.area_noise_sd = area_noise_sd

        # Policy (Eqs. 15-16)
        self.target_price = target_price or {2014: 19800.0, 2015: 19100.0}
        self.tp_default = tp_default
        self.Q_cap = Q_cap

        # Cost dynamics
        self.c_trend = c_trend
        self.c_cv = c_cv

        # Resilience
        self.resilience_window = resilience_window
        self.W_threshold = W_threshold
        self.wei_threshold = wei_threshold

        # Derived steady-state parameters ---------------------------- #
        # b0 chosen so that A* = A0 at P_eff = P0, C = C0 (Eq. 14)
        self.b0 = ((1.0 - self.b2) * np.log(self.A0)
                   - self.b1 * np.log(self.P0))
        # ln(kappa_0) = ln(S_ref) - eta*ln(P_ref)  (Eqs. 12-13); default
        # anchor S_ref = A0*Y0_ref at P0, overridden by the simulated-
        # supply anchor when both kappa_anchor_* are provided.
        if (self.kappa_anchor_supply is not None
                and self.kappa_anchor_price is not None):
            self.ln_kappa0 = (np.log(self.kappa_anchor_supply)
                              - self.eta * np.log(self.kappa_anchor_price))
        else:
            self.ln_kappa0 = (np.log(self.A0 * self.Y0_ref)
                              - self.eta * np.log(self.P0))

        # Stage configuration
        self._stage_lengths = stage_lengths or dict(self._DEFAULT_STAGE_LENGTHS)
        self._stage_types = stage_types or dict(self._DEFAULT_STAGE_TYPES)
        self._stage_labels = stage_labels or dict(self._DEFAULT_STAGE_LABELS)
        self._kc_values = kc_values or dict(self._DEFAULT_KC)
        self._jensen_lambdas = jensen_lambdas or dict(self._DEFAULT_JENSEN)

        self._auto_complete_stages()
        self.kc_annual: np.ndarray = self._build_kc_annual()
        self.stages_config: Dict[str, Dict] = self._build_stages_config()

    # ------------------------------------------------------------------ #
    #  Internal helpers (unchanged)
    # ------------------------------------------------------------------ #

    def _auto_complete_stages(self) -> None:
        current_sids = sorted(self._stage_lengths.keys())
        n_provided = len(current_sids)
        if n_provided < self.S_n:
            for i in range(n_provided + 1, self.S_n + 1):
                sid = f'S{i}'
                if i == self.S_n:
                    used = sum(self._stage_lengths[s] for s in current_sids)
                    self._stage_lengths[sid] = self.T_n - used
                    self._stage_types[sid] = 'H'
                    self._stage_labels[sid] = f'Harvest-{i}'
                    self._kc_values[sid] = 0.20
                else:
                    self._stage_lengths[sid] = 90
                    self._stage_types[sid] = 'G'
                    self._stage_labels[sid] = f'Growth-{i}'
                    self._kc_values[sid] = 0.85 + 0.10 * (i - 2)
                    self._jensen_lambdas[sid] = 0.15 + 0.10 * (i - 2)
        total = sum(self._stage_lengths.values())
        if total < self.T_n:
            self._stage_lengths[f'S{self.S_n}'] += self.T_n - total

    def _build_kc_annual(self) -> np.ndarray:
        kc_arr = np.zeros(self.T_n)
        start = 0
        for sid in sorted(self._stage_lengths.keys()):
            length = self._stage_lengths[sid]
            kc_arr[start:start + length] = self._kc_values.get(sid, 0.0)
            start += length
        return kc_arr

    def _build_stages_config(self) -> Dict[str, Dict]:
        config = {}
        start = 0
        for sid in sorted(self._stage_lengths.keys()):
            length = self._stage_lengths[sid]
            config[sid] = {
                'start_date': start,
                'end_date': start + length,
                'length': length,
                'type': self._stage_types.get(sid, 'G'),
                'label': self._stage_labels.get(sid, sid),
                'kc': self._kc_values.get(sid, 0.0),
            }
            start += length
        return config

    # ------------------------------------------------------------------ #
    #  Public helpers
    # ------------------------------------------------------------------ #

    def get_stage_for_day(self, day: int) -> str:
        for sid, sc in self.stages_config.items():
            if sc['start_date'] <= day < sc['end_date']:
                return sid
        return f'S{self.S_n}'

    def is_growth_day(self, day: int) -> bool:
        """True if the day belongs to a growth stage (irrigable window)."""
        sid = self.get_stage_for_day(day)
        return self.stages_config[sid]['type'] == 'G'

    def get_jensen_array(self) -> np.ndarray:
        arr = np.zeros(self.T_n)
        for sid, sc in self.stages_config.items():
            lam = self._jensen_lambdas.get(sid, 0.0)
            arr[sc['start_date']:sc['end_date']] = lam
        return arr

    def get_target_price(self, year: int, policy: bool = True) -> float:
        """Target price for a calendar year; 0 when policy is off."""
        if not policy:
            return 0.0
        return float(self.target_price.get(year, self.tp_default))

    def omega_t(self, year: int) -> float:
        """Technology correction factor Omega_t in [0, 1].

        Omega_t = clip((tech_trend_intercept + tech_trend_slope * year)
        / Y_max, 0, 1): the observed lint-yield OLS trend relative to
        the fixed biophysical ceiling Y_max.  Linear extrapolation
        beyond the observed record continues until Omega = 1, after
        which Y_pot = Y_max.  Returns 1.0 when the trend option is
        disabled (backward-compatible static potential yield).
        """
        if not self.tech_trend_enabled:
            return 1.0
        trend = self.tech_trend_intercept + self.tech_trend_slope * year
        return float(np.clip(trend / self.Y_max, 0.0, 1.0))

    def y_pot(self, year: int) -> float:
        """Potential lint yield for a calendar year (kg/ha)."""
        return self.omega_t(year) * self.Y_max

    def demand_kappa(self, year_index: int) -> float:
        """Demand scale kappa_t = kappa_0 * (1 + g_d)^t (Eq. 12)."""
        return float(np.exp(self.ln_kappa0
                            + year_index * np.log(1.0 + self.g_d)))

    @classmethod
    def from_yaml(cls, path: str, **overrides) -> "Config":
        """Build a ``Config`` from a YAML parameter file.

        The YAML file should contain a flat mapping of parameter names
        to values (see ``config/model_params.yaml`` for the documented
        template).  Any keyword passed in ``overrides`` replaces the
        file value, which is convenient for scenario experiments::

            cfg = Config.from_yaml("config/model_params.yaml",
                                   s_irrig=0.20)

        Parameters
        ----------
        path : str
            Path to the YAML file.
        **overrides
            Parameter values overriding the file contents.

        Returns
        -------
        Config
        """
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            params = yaml.safe_load(f) or {}
        params.update(overrides)
        return cls(**params)

    # Elasticities derived from the acreage equation (Eq. 14)
    @property
    def elasticity_sr(self) -> float:
        return self.b1

    @property
    def elasticity_lr(self) -> float:
        return self.b1 / (1.0 - self.b2)

    @property
    def lambda_adj(self) -> float:
        return 1.0 - self.b2

    @property
    def jensen_lambdas(self) -> Dict[str, float]:
        return dict(self._jensen_lambdas)

    def __repr__(self) -> str:
        return (
            f"Config(N={self.N}, T_n={self.T_n}, S_n={self.S_n}, "
            f"s_fc={self.s_fc}, Ks={self.Ks}, Zr={self.Zr}, "
            f"Y_max={self.Y_max}, eta={self.eta}, b1={self.b1})"
        )

    def print_summary(self) -> None:
        print("=" * 60)
        print("CAHED Configuration Summary")
        print("=" * 60)
        print(f"  Simulation years (N) : {self.N}")
        print(f"  Days per cycle (T_n) : {self.T_n}")
        print(f"  Number of stages     : {self.S_n}")
        print(f"  Random seed          : {self.seed}")
        print(f"\n  Soil: s_fc={self.s_fc}, Ks={self.Ks}, omega={self.omega}")
        print(f"  Veg : Zr={self.Zr}, sw={self.sw}, s_star={self.s_star}")
        print(f"  Irrigation: s_irrig={self.s_irrig}, "
              f"s_target={self.s_target}, eta={self.eta_irrig}, "
              f"on={self.irrigate}")
        print(f"  Yield max: {self.Y_max} kg/ha")
        if self.tech_trend_enabled:
            yr_one = ((self.Y_max - self.tech_trend_intercept)
                      / self.tech_trend_slope)
            print(f"  Tech trend: ON, {self.tech_trend_slope:+.2f} kg/ha/yr"
                  f" (Omega=1 from {yr_one:.0f})")
        else:
            print("  Tech trend: OFF (static Y_max)")
        print(f"\n  Demand: eta={self.eta}, g_d={self.g_d}")
        print(f"  Nerlove: b1={self.b1}, b2={self.b2}, b3={self.b3}")
        print(f"  Elasticities: SR={self.elasticity_sr:.3f}, "
              f"LR={self.elasticity_lr:.3f}")
        print(f"  Policy: TP default={self.tp_default}, "
              f"Q_cap={self.Q_cap}")
        print("\n  Stages:")
        for sid in sorted(self.stages_config.keys()):
            sc = self.stages_config[sid]
            print(f"    {sid}: {sc['start_date']:3d}-{sc['end_date']:3d} "
                  f"({sc['type']}) {sc['label']:<20s} Kc={sc['kc']}")
        print("=" * 60)
