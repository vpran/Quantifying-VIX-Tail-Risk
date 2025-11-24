"""Run the full VIX shock workflow sequentially."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import (
    data_pipeline,
    volatility_models,
    shock_modeling,
    forecast_evaluation,
    visualization,
)

PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    print("Preparing VIX series...")
    vix_data = data_pipeline.prepare_series()
    df = data_pipeline.engineer_features(vix_data.frame)
    returns = df["dlog_vix"].dropna()
    print(
        f"Loaded {len(df):,} rows from {df.index.min().date()}"
        f" to {df.index.max().date()}"
    )

    print("\nFitting GARCH/EGARCH models...")
    garch_t = volatility_models.fit_garch(returns, distribution="t")
    egarch_t = volatility_models.fit_egarch(returns, distribution="t")
    summary = volatility_models.summarize_fits(garch_t, egarch_t)
    print(summary)

    print("\nShock identification at 95th percentile...")
    shock_def = shock_modeling.define_shocks(returns, quantile=0.95)
    print(
        f"Threshold={shock_def.threshold:.4f}, shocks={int(shock_def.indicator.sum())}"
    )

    interarrival = shock_modeling.interarrival_series(shock_def.indicator)
    hpp = shock_modeling.fit_hpp(interarrival)
    print(
        f"HPP rate/year={hpp.rate_per_year:.2f},"
        f" 95% CI={hpp.ci_95}"
    )

    monthly = shock_modeling.monthly_counts(shock_def.indicator)
    nhpp = shock_modeling.fit_nhpp(monthly)
    print("\nNHPP summary:")
    print(nhpp.result.summary().tables[0])

    print("\nForecast evaluation snapshot...")
    cond_vol = garch_t["result"].conditional_volatility / 100.0
    phi0 = garch_t["result"].params.get("Const", 0.0) / 100.0
    phi1 = garch_t["result"].params.get("ar[1]", 0.0)
    mean_hat = phi0 + phi1 * returns.shift(1)

    garch_df = pd.DataFrame(
        {"mean": mean_hat, "variance": cond_vol**2}, index=returns.index
    ).dropna()

    ewma_var = forecast_evaluation.ewma_variance(returns)
    roll_var = forecast_evaluation.rolling_variance(returns)

    combined = pd.concat(
        [garch_df[["mean", "variance"]], ewma_var, roll_var], axis=1
    ).dropna()
    combined.columns = ["mean", "garch_var", "ewma_var", "roll_var"]

    actual = returns.loc[combined.index]
    zero_mean = pd.Series(0.0, index=actual.index)

    scores = {
        "garch_log": forecast_evaluation.log_score(
            actual, combined["mean"], combined["garch_var"]
        ),
        "ewma_log": forecast_evaluation.log_score(
            actual, zero_mean, combined["ewma_var"]
        ),
        "roll_log": forecast_evaluation.log_score(
            actual, zero_mean, combined["roll_var"]
        ),
    }
    print("Log-score comparison:", scores)

    coverage = forecast_evaluation.coverage_rate(
        actual, combined["mean"], combined["garch_var"]
    )
    print(f"95% coverage (GARCH): {coverage:.3f}")

    pit = forecast_evaluation.pit_values(
        actual, combined["mean"], combined["garch_var"]
    )
    print("PIT summary:")
    print(pd.Series(pit).describe())

    visualization.plot_vix_series(df, save_as="run_vix_series.png")
    visualization.plot_shock_arrivals(monthly, save_as="run_shock_counts.png")
    visualization.plot_pit(pd.Series(pit), save_as="run_pit.png")

    pit_series = pd.Series(pit, index=actual.index)
    loss_garch = -np.log(pit_series.clip(lower=1e-9, upper=1 - 1e-9))
    pit_ewma = forecast_evaluation.pit_values(
        actual, zero_mean, combined["ewma_var"]
    )
    loss_ewma = -np.log(pd.Series(pit_ewma, index=actual.index).clip(lower=1e-9, upper=1 - 1e-9))
    dm_p = forecast_evaluation.diebold_mariano(
        loss_garch,
        loss_ewma,
    )
    print(f"Diebold-Mariano p-value (log PIT loss vs EWMA): {dm_p}")


if __name__ == "__main__":
    main()
