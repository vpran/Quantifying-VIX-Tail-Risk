"""AR-GARCH style modeling utilities."""

from __future__ import annotations

from typing import Dict, Iterable, Literal, Tuple

import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats

from . import config, forecast_evaluation


Distribution = Literal["normal", "t", "ged"]


def build_garch_model(
    returns: pd.Series,
    distribution: Distribution = "normal",
):
    """Return an un-fitted AR(1)-GARCH(1,1) model for the provided series."""

    return arch_model(
        returns * 100,
        mean="AR",
        lags=1,
        vol="GARCH",
        p=1,
        o=0,
        q=1,
        dist=distribution,
    )


def build_garch_model(
    returns: pd.Series,
    distribution: Distribution = "normal",
):
    """Return an un-fitted AR(1)-GARCH(1,1) model for the provided series."""

    return arch_model(
        returns * 100,
        mean="AR",
        lags=1,
        vol="GARCH",
        p=1,
        o=0,
        q=1,
        dist=distribution,
    )


def fit_garch(
    returns: pd.Series,
    distribution: Distribution = "normal",
) -> Dict[str, object]:
    """Fit AR(1)-GARCH(1,1) and extract diagnostics."""

    model = build_garch_model(returns, distribution)
    res = model.fit(disp="off", show_warning=False)
    persistence = float(res.params.get("alpha[1]", 0.0) + res.params.get("beta[1]", 0.0))
    half_life = np.log(0.5) / np.log(persistence) if persistence < 1 else np.inf

    std_resid = res.std_resid
    lb_resid = acorr_ljungbox(std_resid, lags=[12], return_df=True)
    lb_sq = acorr_ljungbox(std_resid**2, lags=[12], return_df=True)

    return {
        "result": res,
        "persistence": persistence,
        "half_life": half_life,
        "lb_pvalue": lb_resid["lb_pvalue"].iloc[0],
        "lb_sq_pvalue": lb_sq["lb_pvalue"].iloc[0],
        "distribution": distribution,
    }


def _conditional_mean(result, returns: pd.Series) -> pd.Series:
    scaled = returns * 100
    mean = pd.Series(result.params.get("Const", 0.0), index=returns.index)
    for name, value in result.params.items():
        if name.startswith("ar["):
            lag = int(name.split("[")[1].rstrip("]"))
            mean += value * scaled.shift(lag)
    return mean / 100.0


def in_sample_forecast_frame(result, returns: pd.Series) -> pd.DataFrame:
    """Return aligned mean/variance series implied by a fitted model."""

    mean = _conditional_mean(result, returns)
    variance = (result.conditional_volatility / 100.0) ** 2
    frame = pd.concat([mean.rename("mean"), variance.rename("variance")], axis=1)
    return frame.dropna()


def pit_uniformity_stat(result, returns: pd.Series) -> float:
    """Compute KS statistic of PIT values versus Uniform(0,1)."""

    frame = in_sample_forecast_frame(result, returns)
    actual = returns.loc[frame.index]
    pits = forecast_evaluation.pit_values(actual, frame["mean"], frame["variance"])
    pits = pits.dropna()
    if pits.empty:
        return np.inf
    return float(stats.kstest(pits, "uniform").statistic)


def select_garch_distribution(
    returns: pd.Series,
    candidates: Iterable[Distribution] | None = None,
) -> Tuple[Distribution, Dict[str, object]]:
    """Pick the distribution that yields the most uniform PITs in-sample."""

    candidates = tuple(candidates) if candidates is not None else config.GARCH_CANDIDATE_DISTS
    best_dist: Distribution | None = None
    best_fit: Dict[str, object] | None = None
    best_stat = np.inf

    for dist in candidates:
        fit = fit_garch(returns, distribution=dist)  # type: ignore[arg-type]
        stat = pit_uniformity_stat(fit["result"], returns)
        if stat < best_stat:
            best_stat = stat
            best_fit = fit
            best_dist = dist  # type: ignore[assignment]

    if best_fit is None or best_dist is None:
        raise RuntimeError("Failed to select a GARCH distribution.")

    best_fit["pit_stat"] = best_stat
    return best_dist, best_fit


def fit_egarch(
    returns: pd.Series,
    distribution: Distribution = "normal",
) -> Dict[str, object]:
    """Fit AR(1)-EGARCH(1,1) and extract diagnostics."""

    model = arch_model(
        returns * 100,
        mean="AR",
        lags=1,
        vol="EGARCH",
        p=1,
        o=1,
        q=1,
        dist=distribution,
    )
    res = model.fit(disp="off", show_warning=False)
    beta = float(res.params.get("beta[1]", np.nan))
    half_life = np.log(0.5) / np.log(beta) if beta < 1 else np.inf

    std_resid = res.std_resid
    lb_resid = acorr_ljungbox(std_resid, lags=[12], return_df=True)
    lb_sq = acorr_ljungbox(std_resid**2, lags=[12], return_df=True)

    return {
        "result": res,
        "persistence": beta,
        "half_life": half_life,
        "lb_pvalue": lb_resid["lb_pvalue"].iloc[0],
        "lb_sq_pvalue": lb_sq["lb_pvalue"].iloc[0],
        "distribution": distribution,
    }


def summarize_fits(*fits: Dict[str, object]) -> pd.DataFrame:
    """Create a tidy summary table from multiple fit outputs."""

    rows = []
    for item in fits:
        res = item["result"]
        rows.append(
            {
                "model": res.model.volatility.__class__.__name__,
                "distribution": item["distribution"],
                "aic": res.aic,
                "bic": res.bic,
                "persistence": item["persistence"],
                "half_life_days": item["half_life"],
                "lb_pvalue": item["lb_pvalue"],
                "lb_sq_pvalue": item["lb_sq_pvalue"],
            }
        )
    return pd.DataFrame(rows)
