"""
CAHED: Coupled Agro-Hydro-Economic Dynamics Model
=================================================

A Python package for simulating the coupled dynamics of agricultural
production, soil hydrology, market economics, and price policy in
dryland agricultural systems.

Quick start
-----------
>>> from script import (Config, build_baseline_params,
...                      generate_precipitation_mmmpp,
...                      generate_pet_cosine, merge_weather,
...                      simulate_soil_moisture, run_coupled_simulation,
...                      components_from_series)
>>> cfg = Config(N=30)
>>> bp = build_baseline_params()
>>> weather = merge_weather(generate_precipitation_mmmpp(cfg, bp),
...                         generate_pet_cosine(cfg))
>>> soil = simulate_soil_moisture(cfg, weather)
>>> run = run_coupled_simulation(cfg, soil, start_year=2014, policy=True)
>>> res = components_from_series(run["S_t"], D, m, s_water)

Modules
-------
config
    ``Config`` — parameter container (hydrology, market, policy).
weather
    Agro-hydrological module: MMPP precipitation, cosine PET.
soil_water
    Daily water balance with stress-avoidance irrigation, stage water
    stress and Jensen yield, annual/monthly irrigation requirement.
market
    Market-behavioral module: isoelastic demand and equilibrium price,
    reduced-form Nerlove acreage response, target-price subsidy, farm
    and fiscal accounting, and the recursive coupled loop.
fast_sim
    Fast columnar implementation of the coupled loop for ensembles.
resilience
    Five-component resilience indicator system and aggregation into
    dimension and overall indices (trailing 5-yr windows).
data_utils
    Path-resolved loaders for the bundled raw data and calibrated
    configuration.
utils
    Plotting helpers, data export, run summaries.
"""
from __future__ import annotations

__version__ = "1.4.0"
__author__ = "Xilin Wu"
__license__ = "MIT"

from .config import Config
from .weather import (
    build_baseline_params,
    generate_precipitation_mmmpp,
    generate_pet_cosine,
    merge_weather,
)
from .soil_water import (
    feddes_stress,
    leakage_rate,
    water_balance_step,
    simulate_soil_moisture,
    compute_stage_stress,
    compute_relative_yield,
    compute_absolute_yield,
    compute_yearly_yield,
    compute_yearly_irrigation,
    compute_monthly_irrigation,
    growing_months,
    doy_to_month,
    compute_daily_theta,
)
from .market import (
    demand,
    equilibrium_price,
    nerlove_area,
    target_price_subsidy,
    farm_profit,
    fiscal_expenditure,
    run_coupled_simulation,
    compute_market_volatility,
)
from .fast_sim import fast_coupled, gen_weather_fast, fast_soil
from .resilience import (
    PAR,
    WINDOW,
    WEI_M,
    ANCHOR_YEAR,
    COMPS,
    DIMS,
    IND9,
    components_from_series,
    water_annual_score,
    demand_benchmark_tonnage,
)

try:
    from .utils import (
        plot_soil_moisture,
        plot_yield_trajectory,
        plot_price_dynamics,
        plot_coupled_dashboard,
        export_run,
        summarize_run,
    )
except Exception:  # pragma: no cover - matplotlib optional
    pass

__all__ = [
    "__version__",
    "Config",
    # weather
    "build_baseline_params", "generate_precipitation_mmmpp",
    "generate_pet_cosine", "merge_weather",
    # soil water
    "feddes_stress", "leakage_rate", "water_balance_step",
    "simulate_soil_moisture", "compute_stage_stress",
    "compute_relative_yield", "compute_absolute_yield",
    "compute_yearly_yield", "compute_yearly_irrigation",
    "compute_monthly_irrigation", "growing_months", "doy_to_month",
    "compute_daily_theta",
    # market
    "demand", "equilibrium_price", "nerlove_area",
    "target_price_subsidy", "farm_profit", "fiscal_expenditure",
    "run_coupled_simulation", "compute_market_volatility",
    # coupling
    "fast_coupled", "gen_weather_fast", "fast_soil",
    # resilience indicators
    "PAR", "WINDOW", "WEI_M", "ANCHOR_YEAR", "COMPS", "DIMS", "IND9",
    "components_from_series", "water_annual_score",
    "demand_benchmark_tonnage",
]
