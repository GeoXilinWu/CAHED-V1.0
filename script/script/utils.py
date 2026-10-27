"""
cahed.utils
===========

Plotting helpers, data exporters, and convenience utilities.
Updated for the revised market-module output columns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:  # pragma: no cover
    _HAS_MPL = False


def _require_mpl() -> None:
    if not _HAS_MPL:
        raise ImportError(
            "Plotting requires matplotlib. Install with: pip install matplotlib")


STAGE_COLOURS: Dict[str, str] = {
    'F': '#A8D0E6', 'G': '#F7DC6F', 'H': '#F1948A',
}


def plot_soil_moisture(
    soil_df: pd.DataFrame,
    cfg: Optional[Any] = None,
    figsize: Tuple[float, float] = (14, 5),
    title: str = "Soil Moisture Trajectory",
    ax: Optional[Any] = None,
    alpha_bands: float = 0.15,
) -> Any:
    """Plot soil moisture time-series with growth-stage background bands."""
    _require_mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(soil_df))
    ax.plot(x, soil_df['Soil_Moisture'], color='#2E86C1', lw=1.2,
            label='Soil moisture')
    if 'Irrigation' in soil_df.columns:
        irr_days = soil_df['Irrigation'] > 0
        ax.scatter(x[irr_days], soil_df.loc[irr_days, 'Soil_Moisture'],
                   s=4, color='#1b7837', alpha=0.5, label='Irrigation event')
    if cfg is not None:
        for sid, sc in cfg.stages_config.items():
            ax.axvspan(sc['start_date'], sc['end_date'],
                       color=STAGE_COLOURS.get(sc['type'], '#EEEEEE'),
                       alpha=alpha_bands, zorder=0)
    ax.set_xlabel('Day (all years)')
    ax.set_ylabel('Soil moisture (volumetric)')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.set_ylim(bottom=0)
    return ax


def plot_yield_trajectory(
    yield_df: pd.DataFrame,
    figsize: Tuple[float, float] = (10, 4),
    title: str = "Yearly Yield",
    ax: Optional[Any] = None,
    color: str = '#27AE60',
) -> Any:
    """Bar chart of yearly yield."""
    _require_mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.bar(yield_df['Sim_Year'], yield_df['Absolute_Yield'],
           color=color, edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Simulation Year')
    ax.set_ylabel('Yield (kg/ha)')
    ax.set_title(title)
    return ax


def plot_price_dynamics(
    run_df: pd.DataFrame,
    figsize: Tuple[float, float] = (12, 5),
    title: str = "Price Dynamics",
    ax: Optional[Any] = None,
) -> Any:
    """Plot market-clearing, effective, and target prices."""
    _require_mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.plot(run_df['Year'], run_df['P'], color='#9B59B6', lw=2,
            marker='o', ms=4, label='Equilibrium price $P$')
    ax.plot(run_df['Year'], run_df['P_eff'], color='#27AE60', lw=1.5,
            ls='-.', label='Effective price $P^{eff}$')
    if run_df['TP'].max() > 0:
        ax.step(run_df['Year'], run_df['TP'], color='#C0392B', lw=1.8,
                where='mid', label='Target price $TP$')
    ax.set_xlabel('Year')
    ax.set_ylabel('Price (CNY/t)')
    ax.set_title(title)
    ax.legend()
    return ax


def plot_coupled_dashboard(
    cfg: Any,
    soil_df: pd.DataFrame,
    yield_df: pd.DataFrame,
    run_df: pd.DataFrame,
    figsize: Tuple[float, float] = (16, 12),
    save_path: Optional[Union[str, Path]] = None,
) -> Any:
    """6-panel dashboard of a full coupled run."""
    _require_mpl()
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    plot_soil_moisture(soil_df, cfg=cfg, ax=axes[0, 0])
    plot_yield_trajectory(yield_df, ax=axes[0, 1])
    plot_price_dynamics(run_df, ax=axes[0, 2])

    axes[1, 0].plot(run_df['Year'], run_df['A_ha'] / 1e4,
                    color='#2E86C1', lw=2)
    axes[1, 0].set_title('Sown Area')
    axes[1, 0].set_ylabel('Area (10$^4$ ha)')

    axes[1, 1].plot(run_df['Year'], run_df['profit_CNY'] / 1e8,
                    color='#27AE60', lw=2, label='Farm profit $\\Pi_t$')
    axes[1, 1].plot(run_df['Year'], run_df['fiscal_CNY'] / 1e8,
                    color='#8E44AD', lw=2, ls='--',
                    label='Fiscal expenditure $F_t$')
    axes[1, 1].set_title('Accounting Outputs')
    axes[1, 1].set_ylabel('10$^8$ CNY')
    axes[1, 1].legend()

    axes[1, 2].plot(run_df['Year'], run_df['W_m3'] / 1e9,
                    color='#2166ac', lw=2)
    axes[1, 2].axhline(cfg.W_threshold / 1e9, color='#C0392B', ls='--',
                       lw=1.5, label='$\\overline{W}$ threshold')
    axes[1, 2].set_title('Irrigation Withdrawal $W_t$')
    axes[1, 2].set_ylabel('10$^9$ m$^3$')
    axes[1, 2].legend()

    fig.suptitle(f'CAHED Coupled Run Dashboard (N={cfg.N})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def export_run(
    output_dir: Union[str, Path],
    soil_df: Optional[pd.DataFrame] = None,
    yield_df: Optional[pd.DataFrame] = None,
    run_df: Optional[pd.DataFrame] = None,
    weather_df: Optional[pd.DataFrame] = None,
    prefix: str = "run",
    fmt: str = "csv",
) -> List[Path]:
    """Export DataFrames to disk (CSV or HDF5)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    mapping = {'soil': soil_df, 'yield': yield_df,
               'coupled': run_df, 'weather': weather_df}
    for name, df in mapping.items():
        if df is None:
            continue
        fname = f"{prefix}_{name}"
        if fmt == "csv":
            fpath = out / f"{fname}.csv"
            df.to_csv(fpath, index=False)
        elif fmt == "h5":
            fpath = out / f"{fname}.h5"
            df.to_hdf(fpath, key=name, mode='w')
        else:
            raise ValueError(f"Unsupported format: {fmt}")
        saved.append(fpath)
    return saved


def summarize_run(cfg: Any, run_df: pd.DataFrame) -> str:
    """Plain-text summary of a coupled run."""
    lines = ["=" * 60,
             "  CAHED Coupled Run Summary",
             "=" * 60,
             f"  Years               : {len(run_df)}",
             f"  Mean area           : {run_df['A_ha'].mean()/1e4:.1f} x10^4 ha",
             f"  Mean yield          : {run_df['Y_t_ha'].mean():.2f} t/ha",
             f"  Mean supply         : {run_df['S_t'].mean()/1e6:.2f} Mt",
             f"  Mean price P        : {run_df['P'].mean():.0f} CNY/t",
             f"  Mean subsidy s_t    : {run_df['s_t'].mean():.0f} CNY/t",
             f"  Mean profit         : {run_df['profit_CNY'].mean()/1e8:.1f} x10^8 CNY",
             f"  Mean fiscal cost    : {run_df['fiscal_CNY'].mean()/1e8:.1f} x10^8 CNY",
             f"  Mean withdrawal W   : {run_df['W_m3'].mean()/1e9:.2f} x10^9 m3",
             "=" * 60]
    return "\n".join(lines)
