"""Shock identification and arrival-rate modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import minimize


@dataclass
class ShockDefinition:
    threshold: float
    indicator: pd.Series
    method: str = "quantile"  # "quantile" or "volatility_relative"


@dataclass
class HPPResult:
    rate_per_day: float
    rate_per_year: float
    ci_95: tuple


@dataclass
class NHPPResult:
    model: sm.GLM
    result: sm.GLMResultsWrapper
    design: pd.DataFrame


@dataclass
class HawkesResult:
    """Results from Hawkes self-exciting point process."""
    mu: float           # Baseline intensity
    alpha: float        # Excitation magnitude
    beta: float         # Decay rate
    branching_ratio: float  # alpha/beta - measures clustering
    log_likelihood: float
    half_life: float    # ln(2)/beta - time for excitation to halve


@dataclass
class CompoundPoissonResult:
    """Results from Compound Poisson Process modeling."""
    arrival_rate: float          # λ: shocks per day
    arrival_rate_annual: float   # λ × 252: shocks per year
    jump_distribution: str       # Name of fitted distribution
    jump_params: Dict            # Distribution parameters
    mean_jump: float             # E[J]: expected jump size
    std_jump: float              # Std[J]: jump size standard deviation
    expected_annual_impact: float  # E[S] = λ × E[J] × 252
    var_95: float                # 95% VaR from simulation
    cvar_95: float               # 95% CVaR (Expected Shortfall)
    aic: float                   # Akaike Information Criterion
    ks_statistic: float          # Kolmogorov-Smirnov test statistic
    ks_pvalue: float             # KS test p-value


def define_shocks(
    returns: pd.Series,
    quantile: float,
) -> ShockDefinition:
    threshold = returns.quantile(quantile)
    indicator = (returns >= threshold).astype(int)
    return ShockDefinition(threshold=threshold, indicator=indicator, method="quantile")


def define_shocks_volatility_relative(
    returns: pd.Series,
    conditional_vol: pd.Series,
    multiplier: float = 2.0,
) -> ShockDefinition:
    """Define shocks as |r_t| > k * sigma_t (relative to conditional volatility).

    This is more economically meaningful as a 'shock' is something unexpected
    given current volatility expectations, not just a large absolute return.

    Parameters
    ----------
    returns : pd.Series
        Return series (log returns or simple returns)
    conditional_vol : pd.Series
        Conditional volatility from GARCH or similar model
    multiplier : float
        Number of standard deviations to define a shock (default 2.0)

    Returns
    -------
    ShockDefinition with volatility-relative threshold
    """
    aligned = pd.concat([returns, conditional_vol], axis=1).dropna()
    aligned.columns = ["returns", "vol"]

    # Shock occurs when |return| exceeds multiplier * conditional vol
    threshold_series = multiplier * aligned["vol"]
    indicator = (np.abs(aligned["returns"]) > threshold_series).astype(int)

    # Store the average threshold for reporting
    avg_threshold = threshold_series.mean()

    return ShockDefinition(
        threshold=avg_threshold,
        indicator=indicator,
        method=f"volatility_relative_k={multiplier}",
    )


def fit_gpd(
    returns: pd.Series,
    threshold: Optional[float] = None,
) -> stats._distn_infrastructure.rv_frozen:
    """Fit a Generalized Pareto distribution to exceedances."""

    threshold = threshold or returns.quantile(0.95)
    exceedances = returns[returns > threshold] - threshold
    if exceedances.empty:
        raise ValueError("No exceedances above the specified threshold.")
    shape, loc, scale = stats.genpareto.fit(exceedances, floc=0)
    return stats.genpareto(c=shape, loc=0, scale=scale)


def interarrival_series(shock_indicator: pd.Series) -> pd.Series:
    """Compute inter-arrival times (in days) between shocks."""

    shock_dates = shock_indicator[shock_indicator == 1].index
    if len(shock_dates) < 2:
        raise ValueError("Need at least two shocks to compute inter-arrivals.")
    diffs = shock_dates.to_series().diff().dropna().dt.days
    return diffs


def fit_hpp(interarrivals: pd.Series) -> HPPResult:
    """Estimate homogeneous Poisson rate with confidence interval."""

    avg_days = interarrivals.mean()
    rate_per_day = 1.0 / avg_days
    rate_per_year = rate_per_day * 252
    n = len(interarrivals)
    ci_low = stats.chi2.ppf(0.025, 2 * n) / (2 * interarrivals.sum())
    ci_high = stats.chi2.ppf(0.975, 2 * (n + 1)) / (2 * interarrivals.sum())
    return HPPResult(rate_per_day, rate_per_year, (ci_low * 252, ci_high * 252))


# ---------------------------------------------------------------------------
# Hawkes Self-Exciting Point Process
# ---------------------------------------------------------------------------


def _hawkes_intensity(times: np.ndarray, mu: float, alpha: float, beta: float) -> np.ndarray:
    """Compute Hawkes intensity at each event time.

    λ(t) = μ + Σ_{t_i < t} α * exp(-β * (t - t_i))
    """
    n = len(times)
    intensities = np.zeros(n)

    for i in range(n):
        intensities[i] = mu
        for j in range(i):
            intensities[i] += alpha * np.exp(-beta * (times[i] - times[j]))

    return intensities


def _hawkes_log_likelihood(params: np.ndarray, times: np.ndarray, T: float) -> float:
    """Negative log-likelihood for Hawkes process (to minimize)."""
    mu, alpha, beta = params

    if mu <= 0 or alpha < 0 or beta <= 0 or alpha >= beta:
        return 1e10  # Invalid parameters

    n = len(times)
    if n == 0:
        return 1e10

    # Sum of log intensities at event times
    intensities = _hawkes_intensity(times, mu, alpha, beta)
    ll_events = np.sum(np.log(np.maximum(intensities, 1e-10)))

    # Integral of intensity over [0, T]
    integral = mu * T
    for ti in times:
        integral += (alpha / beta) * (1 - np.exp(-beta * (T - ti)))

    return -(ll_events - integral)


def fit_hawkes(shock_indicator: pd.Series) -> HawkesResult:
    """Fit a Hawkes self-exciting point process to shock arrivals.

    The Hawkes process models how past shocks increase the probability of
    future shocks, capturing the clustering behavior observed in financial markets.

    λ(t) = μ + Σ_{t_i < t} α * exp(-β * (t - t_i))

    where:
    - μ: baseline intensity (shocks per day without excitation)
    - α: excitation magnitude (how much each shock increases intensity)
    - β: decay rate (how quickly excitation dies down)
    - α/β: branching ratio (average number of 'child' events per 'parent')
    """
    shock_dates = shock_indicator[shock_indicator == 1].index
    if len(shock_dates) < 10:
        raise ValueError("Need at least 10 shocks to fit Hawkes process reliably.")

    # Convert to numeric times (days from start)
    t0 = shock_indicator.index[0]
    times = np.array([(t - t0).days for t in shock_dates], dtype=float)
    T = (shock_indicator.index[-1] - t0).days

    # Initial guess based on HPP rate
    n_events = len(times)
    mu_init = n_events / T * 0.5  # Assume half from baseline
    alpha_init = 0.3
    beta_init = 0.5

    # Optimize
    result = minimize(
        _hawkes_log_likelihood,
        x0=[mu_init, alpha_init, beta_init],
        args=(times, T),
        method="L-BFGS-B",
        bounds=[(1e-6, None), (1e-6, None), (1e-6, None)],
    )

    mu, alpha, beta = result.x
    branching_ratio = alpha / beta if beta > 0 else np.inf
    half_life = np.log(2) / beta if beta > 0 else np.inf
    log_likelihood = -result.fun

    return HawkesResult(
        mu=mu,
        alpha=alpha,
        beta=beta,
        branching_ratio=branching_ratio,
        log_likelihood=log_likelihood,
        half_life=half_life,
    )


def hawkes_simulate_intensity(
    shock_indicator: pd.Series,
    hawkes_result: HawkesResult,
) -> pd.Series:
    """Simulate the Hawkes intensity over time given fitted parameters."""
    shock_dates = shock_indicator[shock_indicator == 1].index
    t0 = shock_indicator.index[0]
    times = np.array([(t - t0).days for t in shock_dates], dtype=float)

    # Compute intensity at each day
    all_days = np.array([(t - t0).days for t in shock_indicator.index], dtype=float)
    mu, alpha, beta = hawkes_result.mu, hawkes_result.alpha, hawkes_result.beta

    intensities = np.full(len(all_days), mu)
    for ti in times:
        mask = all_days > ti
        intensities[mask] += alpha * np.exp(-beta * (all_days[mask] - ti))

    return pd.Series(intensities, index=shock_indicator.index, name="hawkes_intensity")


def monthly_counts(
    shock_indicator: pd.Series,
    log_vix: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Aggregate shocks by calendar month for NHPP modeling.

    Parameters
    ----------
    shock_indicator:
        Daily indicator (1/0) denoting whether a shock occurred.
    log_vix:
        Daily log VIX levels used to build *lagged* covariates that avoid
        incorporating information from the same month as the response.
    """

    if log_vix is None:
        raise ValueError("log_vix series is required to form lagged NHPP covariates.")

    df = shock_indicator.to_frame(name="shocks")
    monthly = df.resample("ME").sum()
    monthly["exposure"] = shock_indicator.resample("ME").size()
    monthly["time"] = np.arange(len(monthly))
    monthly["month"] = monthly.index.month

    avg_log_vix = log_vix.resample("ME").mean()
    monthly["lag_avg_log_vix"] = avg_log_vix.shift(1)
    monthly = monthly.dropna(subset=["lag_avg_log_vix"])
    return monthly


def fit_nhpp(counts: pd.DataFrame) -> NHPPResult:
    """Fit Poisson GLM with exposure to allow time-varying rates."""

    design = pd.get_dummies(
        counts[["time", "lag_avg_log_vix", "month"]],
        columns=["month"],
        drop_first=True,
    )
    design = sm.add_constant(design)
    design = design.astype(float)
    model = sm.GLM(
        counts["shocks"],
        design,
        family=sm.families.Poisson(),
        offset=np.log(counts["exposure"].clip(lower=1)),
    )
    result = model.fit()
    return NHPPResult(model=model, result=result, design=design)


# ---------------------------------------------------------------------------
# Regime Analysis
# ---------------------------------------------------------------------------


@dataclass
class RegimeAnalysis:
    """Results from subsample/regime analysis."""
    regime_name: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    n_obs: int
    n_shocks: int
    shock_rate: float  # per year
    avg_return: float
    volatility: float
    hpp_rate: float


def analyze_regime(
    returns: pd.Series,
    shock_indicator: pd.Series,
    regime_name: str,
    start_date: str,
    end_date: str,
) -> RegimeAnalysis:
    """Analyze shock characteristics within a specific time regime."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    mask = (returns.index >= start) & (returns.index <= end)
    regime_returns = returns[mask]
    regime_shocks = shock_indicator[mask]

    n_obs = len(regime_returns)
    n_shocks = int(regime_shocks.sum())
    years = n_obs / 252
    shock_rate = n_shocks / years if years > 0 else 0

    # Try to fit HPP if enough shocks
    try:
        interarrivals = interarrival_series(regime_shocks)
        hpp = fit_hpp(interarrivals)
        hpp_rate = hpp.rate_per_year
    except ValueError:
        hpp_rate = shock_rate

    return RegimeAnalysis(
        regime_name=regime_name,
        start_date=start,
        end_date=end,
        n_obs=n_obs,
        n_shocks=n_shocks,
        shock_rate=shock_rate,
        avg_return=float(regime_returns.mean()),
        volatility=float(regime_returns.std() * np.sqrt(252)),
        hpp_rate=hpp_rate,
    )


def run_regime_analysis(
    returns: pd.Series,
    shock_indicator: pd.Series,
) -> pd.DataFrame:
    """Analyze multiple market regimes for structural differences."""
    regimes = [
        ("Pre-Crisis", "2010-01-01", "2019-12-31"),
        ("COVID Crisis", "2020-01-01", "2020-12-31"),
        ("Post-COVID", "2021-01-01", "2023-12-31"),
        ("Recent", "2024-01-01", "2025-12-31"),
        ("Full Sample", returns.index.min().strftime("%Y-%m-%d"),
         returns.index.max().strftime("%Y-%m-%d")),
    ]

    results = []
    for name, start, end in regimes:
        try:
            analysis = analyze_regime(returns, shock_indicator, name, start, end)
            if analysis.n_obs > 0:
                results.append({
                    "Regime": analysis.regime_name,
                    "Start": analysis.start_date.strftime("%Y-%m-%d"),
                    "End": analysis.end_date.strftime("%Y-%m-%d"),
                    "Obs": analysis.n_obs,
                    "Shocks": analysis.n_shocks,
                    "Rate/Year": analysis.shock_rate,
                    "Ann. Vol": analysis.volatility,
                    "HPP Rate": analysis.hpp_rate,
                })
        except Exception:
            continue

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Compound Poisson Process
# ---------------------------------------------------------------------------


def _fit_jump_distribution(
    jump_sizes: np.ndarray,
    distribution: str,
) -> Tuple[Dict, float, float, float]:
    """Fit a specific distribution to jump sizes and return params + fit statistics."""
    
    n = len(jump_sizes)
    
    if distribution == "exponential":
        # MLE for exponential: scale = mean
        scale = np.mean(jump_sizes)
        params = {"scale": scale}
        log_likelihood = np.sum(stats.expon.logpdf(jump_sizes, scale=scale))
        n_params = 1
        
    elif distribution == "gamma":
        # Fit gamma distribution
        shape, loc, scale = stats.gamma.fit(jump_sizes, floc=0)
        params = {"shape": shape, "scale": scale}
        log_likelihood = np.sum(stats.gamma.logpdf(jump_sizes, shape, loc=0, scale=scale))
        n_params = 2
        
    elif distribution == "lognormal":
        # Fit lognormal distribution
        log_jumps = np.log(jump_sizes[jump_sizes > 0])
        if len(log_jumps) < 2:
            return {}, -np.inf, np.inf, np.inf
        mu = np.mean(log_jumps)
        sigma = np.std(log_jumps, ddof=1)
        params = {"mu": mu, "sigma": sigma}
        log_likelihood = np.sum(stats.lognorm.logpdf(jump_sizes, s=sigma, scale=np.exp(mu)))
        n_params = 2
        
    elif distribution == "pareto":
        # Fit Pareto distribution (for heavy tails)
        xmin = np.min(jump_sizes)
        if xmin <= 0:
            xmin = 1e-6
        # MLE for Pareto shape parameter
        alpha_pareto = n / np.sum(np.log(jump_sizes / xmin))
        params = {"alpha": alpha_pareto, "xmin": xmin}
        log_likelihood = np.sum(stats.pareto.logpdf(jump_sizes, alpha_pareto, scale=xmin))
        n_params = 2
        
    elif distribution == "weibull":
        # Fit Weibull distribution
        shape, loc, scale = stats.weibull_min.fit(jump_sizes, floc=0)
        params = {"shape": shape, "scale": scale}
        log_likelihood = np.sum(stats.weibull_min.logpdf(jump_sizes, shape, loc=0, scale=scale))
        n_params = 2
        
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    
    # Compute AIC
    aic = 2 * n_params - 2 * log_likelihood
    
    # Compute KS statistic
    if distribution == "exponential":
        ks_stat, ks_pval = stats.kstest(jump_sizes, "expon", args=(0, params["scale"]))
    elif distribution == "gamma":
        ks_stat, ks_pval = stats.kstest(jump_sizes, "gamma", args=(params["shape"], 0, params["scale"]))
    elif distribution == "lognormal":
        ks_stat, ks_pval = stats.kstest(jump_sizes, "lognorm", args=(params["sigma"], 0, np.exp(params["mu"])))
    elif distribution == "pareto":
        ks_stat, ks_pval = stats.kstest(jump_sizes, "pareto", args=(params["alpha"], 0, params["xmin"]))
    elif distribution == "weibull":
        ks_stat, ks_pval = stats.kstest(jump_sizes, "weibull_min", args=(params["shape"], 0, params["scale"]))
    else:
        ks_stat, ks_pval = np.nan, np.nan
    
    return params, log_likelihood, aic, ks_stat, ks_pval


def _compute_jump_moments(distribution: str, params: Dict) -> Tuple[float, float]:
    """Compute mean and standard deviation for a fitted jump distribution."""
    
    if distribution == "exponential":
        mean = params["scale"]
        std = params["scale"]
        
    elif distribution == "gamma":
        mean = params["shape"] * params["scale"]
        std = np.sqrt(params["shape"]) * params["scale"]
        
    elif distribution == "lognormal":
        mu, sigma = params["mu"], params["sigma"]
        mean = np.exp(mu + sigma**2 / 2)
        std = mean * np.sqrt(np.exp(sigma**2) - 1)
        
    elif distribution == "pareto":
        alpha = params["alpha"]
        xmin = params["xmin"]
        if alpha > 1:
            mean = (alpha * xmin) / (alpha - 1)
        else:
            mean = np.inf
        if alpha > 2:
            var = (xmin**2 * alpha) / ((alpha - 1)**2 * (alpha - 2))
            std = np.sqrt(var)
        else:
            std = np.inf
            
    elif distribution == "weibull":
        shape, scale = params["shape"], params["scale"]
        mean = scale * np.exp(np.math.lgamma(1 + 1/shape))
        var = scale**2 * (np.exp(np.math.lgamma(1 + 2/shape)) - np.exp(2*np.math.lgamma(1 + 1/shape)))
        std = np.sqrt(max(var, 0))
        
    else:
        mean, std = np.nan, np.nan
    
    return mean, std


def _sample_from_distribution(distribution: str, params: Dict, n_samples: int) -> np.ndarray:
    """Sample from a fitted jump distribution."""
    
    if distribution == "exponential":
        return np.random.exponential(scale=params["scale"], size=n_samples)
        
    elif distribution == "gamma":
        return np.random.gamma(shape=params["shape"], scale=params["scale"], size=n_samples)
        
    elif distribution == "lognormal":
        return np.random.lognormal(mean=params["mu"], sigma=params["sigma"], size=n_samples)
        
    elif distribution == "pareto":
        return (np.random.pareto(params["alpha"], size=n_samples) + 1) * params["xmin"]
        
    elif distribution == "weibull":
        return params["scale"] * np.random.weibull(params["shape"], size=n_samples)
        
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


def fit_compound_poisson(
    returns: pd.Series,
    shock_indicator: pd.Series,
    candidate_distributions: List[str] = None,
    n_simulations: int = 10000,
) -> CompoundPoissonResult:
    """Fit a Compound Poisson Process to shock arrivals and magnitudes.
    
    The Compound Poisson Process models:
    - N(t) ~ Poisson(λt): number of shocks by time t
    - J_i ~ F: jump sizes drawn from distribution F
    - S(t) = Σ_{i=1}^{N(t)} J_i: cumulative shock impact
    
    Parameters
    ----------
    returns : pd.Series
        Return series (log returns)
    shock_indicator : pd.Series
        Binary indicator of shock occurrences
    candidate_distributions : List[str]
        Distributions to try for jump sizes. Default: exponential, gamma, lognormal, pareto
    n_simulations : int
        Number of Monte Carlo simulations for VaR/CVaR
        
    Returns
    -------
    CompoundPoissonResult with fitted parameters and risk metrics
    """
    if candidate_distributions is None:
        candidate_distributions = ["exponential", "gamma", "lognormal", "pareto", "weibull"]
    
    # Extract shock times and magnitudes
    shock_dates = shock_indicator[shock_indicator == 1].index
    shock_magnitudes = np.abs(returns.loc[shock_dates].values)
    
    if len(shock_magnitudes) < 10:
        raise ValueError("Need at least 10 shocks to fit Compound Poisson Process")
    
    # Remove any zero or negative values (shouldn't happen but safety check)
    shock_magnitudes = shock_magnitudes[shock_magnitudes > 0]
    
    # Compute arrival rate (HPP)
    n_shocks = len(shock_magnitudes)
    n_days = len(shock_indicator)
    arrival_rate = n_shocks / n_days
    arrival_rate_annual = arrival_rate * 252
    
    # Fit candidate distributions and select best by AIC
    best_dist = None
    best_aic = np.inf
    best_params = {}
    best_ks_stat = np.nan
    best_ks_pval = np.nan
    
    for dist in candidate_distributions:
        try:
            params, ll, aic, ks_stat, ks_pval = _fit_jump_distribution(shock_magnitudes, dist)
            if aic < best_aic:
                best_aic = aic
                best_dist = dist
                best_params = params
                best_ks_stat = ks_stat
                best_ks_pval = ks_pval
        except Exception:
            continue
    
    if best_dist is None:
        raise ValueError("Could not fit any distribution to jump sizes")
    
    # Compute jump moments
    mean_jump, std_jump = _compute_jump_moments(best_dist, best_params)
    
    # Expected annual impact: E[S] = λ × E[J] × 252
    expected_annual_impact = arrival_rate * mean_jump * 252
    
    # Monte Carlo simulation for VaR/CVaR
    np.random.seed(42)  # For reproducibility
    annual_impacts = []
    
    for _ in range(n_simulations):
        # Simulate number of shocks in a year
        n_annual_shocks = np.random.poisson(arrival_rate_annual)
        
        if n_annual_shocks > 0:
            # Simulate jump sizes
            jumps = _sample_from_distribution(best_dist, best_params, n_annual_shocks)
            total_impact = np.sum(jumps)
        else:
            total_impact = 0
        
        annual_impacts.append(total_impact)
    
    annual_impacts = np.array(annual_impacts)
    
    # Compute VaR and CVaR at 95% level
    var_95 = np.percentile(annual_impacts, 95)
    cvar_95 = np.mean(annual_impacts[annual_impacts >= var_95])
    
    return CompoundPoissonResult(
        arrival_rate=arrival_rate,
        arrival_rate_annual=arrival_rate_annual,
        jump_distribution=best_dist,
        jump_params=best_params,
        mean_jump=mean_jump,
        std_jump=std_jump,
        expected_annual_impact=expected_annual_impact,
        var_95=var_95,
        cvar_95=cvar_95,
        aic=best_aic,
        ks_statistic=best_ks_stat,
        ks_pvalue=best_ks_pval,
    )


def compound_poisson_by_regime(
    returns: pd.Series,
    shock_indicator: pd.Series,
    regimes: List[Tuple[str, str, str]] = None,
) -> pd.DataFrame:
    """Fit Compound Poisson Process separately for each market regime.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    shock_indicator : pd.Series
        Binary shock indicator
    regimes : List of (name, start_date, end_date) tuples
    
    Returns
    -------
    DataFrame with CPP parameters for each regime
    """
    if regimes is None:
        regimes = [
            ("Pre-Crisis", "2010-01-01", "2019-12-31"),
            ("COVID", "2020-01-01", "2020-12-31"),
            ("Post-COVID", "2021-01-01", "2023-12-31"),
            ("Recent", "2024-01-01", "2025-12-31"),
        ]
    
    results = []
    for name, start, end in regimes:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        mask = (returns.index >= start_ts) & (returns.index <= end_ts)
        
        regime_returns = returns[mask]
        regime_shocks = shock_indicator[mask]
        
        if regime_shocks.sum() < 5:
            continue
        
        try:
            cpp = fit_compound_poisson(regime_returns, regime_shocks)
            results.append({
                "Regime": name,
                "λ/Year": cpp.arrival_rate_annual,
                "Jump Dist": cpp.jump_distribution,
                "E[J]": cpp.mean_jump,
                "Std[J]": cpp.std_jump,
                "E[S]/Year": cpp.expected_annual_impact,
                "VaR 95%": cpp.var_95,
                "CVaR 95%": cpp.cvar_95,
            })
        except Exception:
            continue
    
    return pd.DataFrame(results)


def simulate_compound_poisson_paths(
    cpp_result: CompoundPoissonResult,
    n_paths: int = 1000,
    horizon_days: int = 252,
    dt: float = 1.0,
) -> np.ndarray:
    """Simulate sample paths of the cumulative shock process.
    
    Parameters
    ----------
    cpp_result : CompoundPoissonResult
        Fitted CPP parameters
    n_paths : int
        Number of paths to simulate
    horizon_days : int
        Simulation horizon in days
    dt : float
        Time step (default 1 day)
        
    Returns
    -------
    Array of shape (n_paths, horizon_days) with cumulative impact paths
    """
    np.random.seed(42)
    n_steps = int(horizon_days / dt)
    paths = np.zeros((n_paths, n_steps))
    
    for i in range(n_paths):
        cumulative = 0
        for t in range(n_steps):
            # Probability of shock in this time step
            if np.random.random() < cpp_result.arrival_rate * dt:
                # A shock occurs - sample jump size
                jump = _sample_from_distribution(
                    cpp_result.jump_distribution,
                    cpp_result.jump_params,
                    1
                )[0]
                cumulative += jump
            paths[i, t] = cumulative
    
    return paths


def compute_cpp_percentiles(
    paths: np.ndarray,
    percentiles: List[float] = None,
) -> pd.DataFrame:
    """Compute percentile bands from simulated CPP paths.
    
    Parameters
    ----------
    paths : np.ndarray
        Simulated paths of shape (n_paths, n_steps)
    percentiles : List[float]
        Percentiles to compute (default: 5, 25, 50, 75, 95)
        
    Returns
    -------
    DataFrame with percentile values at each time step
    """
    if percentiles is None:
        percentiles = [5, 25, 50, 75, 95]
    
    n_steps = paths.shape[1]
    result = {"time": np.arange(n_steps)}
    
    for p in percentiles:
        result[f"p{p}"] = np.percentile(paths, p, axis=0)
    
    return pd.DataFrame(result)


# ---------------------------------------------------------------------------
# CPP Train-Test Split and Out-of-Sample Evaluation
# ---------------------------------------------------------------------------


@dataclass
class CPPForecastResult:
    """Results from CPP out-of-sample forecast evaluation."""
    train_end_date: pd.Timestamp
    test_start_date: pd.Timestamp
    train_cpp: CompoundPoissonResult
    test_n_shocks_actual: int
    test_n_shocks_predicted: float
    test_total_impact_actual: float
    test_total_impact_predicted: float
    shock_count_error: float  # (predicted - actual) / actual
    impact_error: float       # (predicted - actual) / actual
    var_95_exceeded: bool     # Did actual exceed VaR?
    cvar_coverage: float      # Fraction of test period where actual < CVaR


def cpp_train_test_split(
    returns: pd.Series,
    shock_indicator: pd.Series,
    train_fraction: float = 0.75,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Split returns and shock indicator into train and test sets.
    
    Parameters
    ----------
    returns : pd.Series
        Full sample returns
    shock_indicator : pd.Series
        Full sample shock indicator
    train_fraction : float
        Fraction of data to use for training (default 0.75)
        
    Returns
    -------
    Tuple of (train_returns, test_returns, train_shocks, test_shocks)
    """
    n = len(returns)
    split_idx = int(n * train_fraction)
    
    train_returns = returns.iloc[:split_idx]
    test_returns = returns.iloc[split_idx:]
    
    # Align shock indicator with returns
    train_shocks = shock_indicator.loc[train_returns.index]
    test_shocks = shock_indicator.loc[test_returns.index]
    
    return train_returns, test_returns, train_shocks, test_shocks


def evaluate_cpp_forecast(
    returns: pd.Series,
    shock_indicator: pd.Series,
    train_fraction: float = 0.75,
    n_simulations: int = 10000,
) -> CPPForecastResult:
    """Evaluate CPP model out-of-sample.
    
    Trains CPP on training set, then evaluates forecast accuracy on test set.
    
    Parameters
    ----------
    returns : pd.Series
        Full sample returns
    shock_indicator : pd.Series
        Full sample shock indicator
    train_fraction : float
        Fraction of data to use for training
    n_simulations : int
        Number of Monte Carlo simulations for VaR estimation
        
    Returns
    -------
    CPPForecastResult with forecast evaluation metrics
    """
    # Split data
    train_ret, test_ret, train_shocks, test_shocks = cpp_train_test_split(
        returns, shock_indicator, train_fraction
    )
    
    train_end_date = train_ret.index[-1]
    test_start_date = test_ret.index[0]
    
    # Fit CPP on training data
    train_cpp = fit_compound_poisson(train_ret, train_shocks, n_simulations=n_simulations)
    
    # Compute test period metrics
    test_n_days = len(test_shocks)
    test_years = test_n_days / 252.0
    
    # Actual test period outcomes
    test_shock_dates = test_shocks[test_shocks == 1].index
    test_n_shocks_actual = int(test_shocks.sum())
    test_shock_magnitudes = np.abs(test_ret.loc[test_shock_dates].values)
    test_total_impact_actual = float(np.sum(test_shock_magnitudes))
    
    # Predicted outcomes using trained CPP
    test_n_shocks_predicted = train_cpp.arrival_rate * test_n_days
    test_total_impact_predicted = train_cpp.arrival_rate * train_cpp.mean_jump * test_n_days
    
    # Compute errors
    shock_count_error = (test_n_shocks_predicted - test_n_shocks_actual) / max(test_n_shocks_actual, 1)
    impact_error = (test_total_impact_predicted - test_total_impact_actual) / max(test_total_impact_actual, 0.01)
    
    # Check if VaR was exceeded (scale VaR to test period length)
    # VaR is annual, so scale by test_years
    var_threshold = train_cpp.var_95 * test_years
    var_95_exceeded = test_total_impact_actual > var_threshold
    
    # Simulate to get CVaR coverage (fraction of simulated paths that exceed actual)
    np.random.seed(42)
    simulated_impacts = []
    for _ in range(n_simulations):
        n_shocks_sim = np.random.poisson(train_cpp.arrival_rate * test_n_days)
        if n_shocks_sim > 0:
            jumps = _sample_from_distribution(
                train_cpp.jump_distribution,
                train_cpp.jump_params,
                n_shocks_sim
            )
            simulated_impacts.append(np.sum(jumps))
        else:
            simulated_impacts.append(0)
    simulated_impacts = np.array(simulated_impacts)
    
    # CVaR coverage: fraction of simulations where simulated impact >= actual impact
    cvar_coverage = float(np.mean(simulated_impacts >= test_total_impact_actual))
    
    return CPPForecastResult(
        train_end_date=train_end_date,
        test_start_date=test_start_date,
        train_cpp=train_cpp,
        test_n_shocks_actual=test_n_shocks_actual,
        test_n_shocks_predicted=test_n_shocks_predicted,
        test_total_impact_actual=test_total_impact_actual,
        test_total_impact_predicted=test_total_impact_predicted,
        shock_count_error=shock_count_error,
        impact_error=impact_error,
        var_95_exceeded=var_95_exceeded,
        cvar_coverage=cvar_coverage,
    )


def cpp_rolling_forecast(
    returns: pd.Series,
    shock_indicator: pd.Series,
    train_fraction: float = 0.75,
    window_months: int = 3,
    n_simulations: int = 5000,
) -> pd.DataFrame:
    """Rolling window CPP forecast evaluation.
    
    Fits CPP on expanding window, forecasts next window_months, and evaluates.
    
    Parameters
    ----------
    returns : pd.Series
        Full sample returns
    shock_indicator : pd.Series
        Full sample shock indicator
    train_fraction : float
        Initial training fraction
    window_months : int
        Forecast horizon in months
    n_simulations : int
        Monte Carlo simulations per forecast
        
    Returns
    -------
    DataFrame with rolling forecast results
    """
    train_ret, test_ret, train_shocks, test_shocks = cpp_train_test_split(
        returns, shock_indicator, train_fraction
    )
    
    results = []
    
    # Get unique months in test period
    test_months = test_ret.resample("ME").size().index
    
    for i, forecast_end in enumerate(test_months[window_months-1:]):
        # Determine forecast window
        forecast_start_idx = i
        forecast_start = test_months[forecast_start_idx]
        
        # Get forecast window data
        window_mask = (test_ret.index >= forecast_start) & (test_ret.index <= forecast_end)
        window_returns = test_ret[window_mask]
        window_shocks = test_shocks[window_mask]
        
        if len(window_shocks) < 10:
            continue
        
        # Training data: everything before forecast window
        train_end = forecast_start - pd.Timedelta(days=1)
        combined_train_ret = returns[returns.index <= train_end]
        combined_train_shocks = shock_indicator.loc[combined_train_ret.index]
        
        if combined_train_shocks.sum() < 20:
            continue
        
        try:
            # Fit CPP on training data
            cpp_fit = fit_compound_poisson(
                combined_train_ret, 
                combined_train_shocks,
                n_simulations=n_simulations
            )
            
            # Compute actual outcomes in forecast window
            n_days = len(window_shocks)
            actual_shocks = int(window_shocks.sum())
            shock_dates = window_shocks[window_shocks == 1].index
            actual_impact = float(np.abs(window_returns.loc[shock_dates]).sum()) if len(shock_dates) > 0 else 0
            
            # Predicted outcomes
            pred_shocks = cpp_fit.arrival_rate * n_days
            pred_impact = cpp_fit.arrival_rate * cpp_fit.mean_jump * n_days
            
            results.append({
                "forecast_start": forecast_start,
                "forecast_end": forecast_end,
                "n_days": n_days,
                "actual_shocks": actual_shocks,
                "pred_shocks": pred_shocks,
                "shock_error": (pred_shocks - actual_shocks) / max(actual_shocks, 1),
                "actual_impact": actual_impact,
                "pred_impact": pred_impact,
                "impact_error": (pred_impact - actual_impact) / max(actual_impact, 0.01),
                "arrival_rate": cpp_fit.arrival_rate,
                "mean_jump": cpp_fit.mean_jump,
                "jump_dist": cpp_fit.jump_distribution,
            })
        except Exception:
            continue
    
    return pd.DataFrame(results)


def cpp_forecast_summary(forecast_result: CPPForecastResult) -> pd.DataFrame:
    """Create summary table from CPP forecast results.
    
    Parameters
    ----------
    forecast_result : CPPForecastResult
        Output from evaluate_cpp_forecast
        
    Returns
    -------
    DataFrame with summary statistics
    """
    cpp = forecast_result.train_cpp
    
    data = {
        "Metric": [
            "Training Period End",
            "Test Period Start",
            "Arrival Rate (λ/day)",
            "Annual Arrival Rate",
            "Jump Distribution",
            "Mean Jump E[J]",
            "Std Jump",
            "Test Shocks (Actual)",
            "Test Shocks (Predicted)",
            "Shock Count Error",
            "Test Impact (Actual)",
            "Test Impact (Predicted)",
            "Impact Error",
            "VaR 95% Exceeded",
            "Tail Coverage (P(sim ≥ actual))",
        ],
        "Value": [
            forecast_result.train_end_date.strftime("%Y-%m-%d"),
            forecast_result.test_start_date.strftime("%Y-%m-%d"),
            f"{cpp.arrival_rate:.4f}",
            f"{cpp.arrival_rate_annual:.2f}",
            cpp.jump_distribution.title(),
            f"{cpp.mean_jump:.4f}",
            f"{cpp.std_jump:.4f}",
            f"{forecast_result.test_n_shocks_actual}",
            f"{forecast_result.test_n_shocks_predicted:.1f}",
            f"{forecast_result.shock_count_error*100:.1f}%",
            f"{forecast_result.test_total_impact_actual:.3f}",
            f"{forecast_result.test_total_impact_predicted:.3f}",
            f"{forecast_result.impact_error*100:.1f}%",
            "Yes" if forecast_result.var_95_exceeded else "No",
            f"{forecast_result.cvar_coverage*100:.1f}%",
        ],
    }
    
    return pd.DataFrame(data)
