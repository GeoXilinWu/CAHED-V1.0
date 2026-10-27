"""
Data and configuration helpers for the CAHED examples.

All paths resolve relative to the repository root, so the examples run
unchanged after a fresh clone.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_RESULTS = REPO_ROOT / "data" / "results"
CONFIG = REPO_ROOT / "config"

PANEL_CSV = DATA_RAW / "cahed_xinjiang_annual_panel.csv"
HYDRO_CALI_CSV = DATA_RAW / "data_4_hydro_cali.csv"
CNRD_CSV = DATA_RAW / "cnrd_monthly_available_water_40-45N_80-90E.csv"
DAILY_FORCING_CSV = DATA_RAW / "daily_weather_forcing.csv"
MODEL_YAML = CONFIG / "model_params.yaml"
RESILIENCE_YAML = CONFIG / "resilience_params.yaml"


def load_resilience_params():
    """Indicator protocol + normalization parameters from
    config/resilience_params.yaml (falls back to the calibrated
    defaults in script.resilience if PyYAML is unavailable)."""
    from . import resilience as res
    out = dict(PAR=dict(res.PAR), WINDOW=res.WINDOW, WEI_M=res.WEI_M,
               ANCHOR_YEAR=res.ANCHOR_YEAR,
               WA_SCALE=1.0, N_MC=200,
               WEATHER_SEED_BASE=2026, WEATHER_SEED_M0=3001,
               MARKET_SEED_BASE=7, MARKET_SEED_M0=1000)
    try:
        import yaml
        d = yaml.safe_load(RESILIENCE_YAML.read_text(encoding="utf-8"))
        out["PAR"] = dict(d["normalization"])
        out["WINDOW"] = int(d["window"])
        out["WEI_M"] = float(d["water"]["WEI_M"])
        out["WA_SCALE"] = float(d["water"]["WA_SCALE"])
        out["ANCHOR_YEAR"] = int(d["anchor_year"])
        p = d.get("protocol", {})
        out["N_MC"] = int(p.get("ensemble", {}).get("n_members", 200))
        out["WEATHER_SEED_BASE"] = int(p["baseline"]["weather_seed"])
        out["MARKET_SEED_BASE"] = int(p["baseline"]["market_seed"])
        out["WEATHER_SEED_M0"] = int(p["ensemble"]["weather_seed0"])
        out["MARKET_SEED_M0"] = int(p["ensemble"]["market_seed0"])
    except Exception:
        pass
    return out


def load_panel() -> pd.DataFrame:
    """Xinjiang annual panel (NBS / China Cotton Association sources,
    1978-2024)."""
    return pd.read_csv(PANEL_CSV)


def load_hydro_cali() -> pd.DataFrame:
    """Annual calibration series (1987-2024): observed irrigation
    (mm), regional available water (1e8 m3)."""
    return pd.read_csv(HYDRO_CALI_CSV)


def cnrd_monthly_shares() -> np.ndarray:
    """Monthly shares of available water from the CNRD monthly
    climatology (40-45 N, 80-90 E)."""
    cnrd = pd.read_csv(CNRD_CSV, encoding="utf-8-sig")
    cnrd.columns = ["month", "runoff_mm_clim", "WA_1e8m3"] + list(
        cnrd.columns[3:])
    wa = cnrd.sort_values("month")["WA_1e8m3"].values
    return wa / wa.sum()


def monthly_availability(wa_scale=1.0) -> np.ndarray:
    """Constant climatological monthly available water (m3): 1987-2023
    mean annual total x CNRD monthly shares."""
    d4 = load_hydro_cali()
    wa_clim = float(d4[d4.year <= 2023].water_avail_1e8m3.dropna().mean())
    return wa_clim * wa_scale * 1e8 * cnrd_monthly_shares()


def real_weather_soil(cfg) -> pd.DataFrame:
    """Soil-moisture simulation driven by the observed daily forcing
    (1987-2023), used for the observed-side monthly irrigation shape."""
    from .soil_water import simulate_soil_moisture
    wx = pd.read_csv(DAILY_FORCING_CSV)
    wdf = pd.DataFrame({
        "Sim_Year": wx["year"].values,
        "Day_Of_Cycle": wx["doy"].values - 1,
        "Stage": [cfg.get_stage_for_day(int(d - 1))
                  for d in wx["doy"].values],
        "Rain_Intensity": wx["P_mm"].values,
        "PET": wx["PET_mm"].values})
    return simulate_soil_moisture(cfg, wdf)


def build_run_config(n_years, seed=2026, start_year=1985):
    """Model configuration for a full 1985-anchored run: target price
    is zero before 2014, 19800 / 19100 CNY/t in 2014 / 2015, then the
    tp_default of model_params.yaml."""
    from .config import Config
    cfg = Config.from_yaml(str(MODEL_YAML), N=n_years, seed=seed)
    cfg.target_price = ({y: 0.0 for y in range(start_year, 2014)}
                        | {2014: 19800.0, 2015: 19100.0})
    return cfg
