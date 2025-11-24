# The Dynamics of Market Fear: Shock Persistence and Shock Frequency in VIX

This repository operationalizes the research agenda described in the project proposal. It offers reproducible data ingestion, volatility modeling, shock-arrival analysis, and forecast calibration workflows built in Python.

## Project Structure

```
Shock-Persistence-and-Shock-Frequency-in-VIX/
├── data/                      # Cached downloads and intermediate data
│   └── raw/
├── notebooks/                 # Stepwise analysis notebooks
│   ├── 01_data_and_volatility.ipynb
│   ├── 02_shock_arrivals.ipynb
│   └── 03_forecast_evaluation.ipynb
├── src/                       # Reusable project modules
│   ├── config.py              # shared paths/tunings
│   ├── data_pipeline.py       # download & cleaning
│   ├── features.py            # helper transforms
│   ├── volatility_models.py   # AR/(E)GARCH fitting
│   ├── shock_modeling.py      # shock definitions + Poisson
│   ├── forecast_evaluation.py # forecasting + scoring
│   └── visualization.py       # plotting utilities
├── requirements.txt
└── README.md
```

## Quick Start

1. **Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run the notebooks** in order:
   - `notebooks/01_data_and_volatility.ipynb`: data ingestion, AR-GARCH/EGARCH estimation, half-life diagnostics.
   - `notebooks/02_shock_arrivals.ipynb`: shock labeling, inter-arrival analysis, Poisson GLM (NHPP) fits.
   - `notebooks/03_forecast_evaluation.ipynb`: forecast generation, calibration checks, Diebold–Mariano tests.

   Behind the scenes, notebooks rely on the `src/` modules in the following conceptual order:
   1. `config` → shared constants/paths.
   2. `data_pipeline` → download, cleaning, log transforms.
   3. `features` → helper columns (abs returns, calendar fields).
   4. `volatility_models` → AR-GARCH/EGARCH estimation.
   5. `shock_modeling` → quantile thresholds, Poisson modeling.
   6. `forecast_evaluation` → scoring, PIT, Diebold–Mariano.
   7. `visualization` → charts used throughout the notebooks.

   Each notebook caches Yahoo Finance pulls (`data/raw/vix_history.parquet`) for reproducibility. Set `force_download=True` in `data_pipeline.prepare_series` to refresh.

## Methodological Highlights

- **Volatility Dynamics**: AR(1)-mean with both GARCH(1,1) and EGARCH(1,1) variance structures under Normal vs Student-t errors. Persistence (`α+β` or `β`) and half-life diagnostics per proposal.
- **Shock Detection**: Quantile-based thresholds (configurable via `config.SHOCK_QUANTILES`) with optional EVT refinement through Peaks-Over-Threshold and GPD fits.
- **Arrival Modeling**: Homogeneous Poisson (inter-arrival MLE & CI) vs non-homogeneous Poisson via Poisson GLM with monthly seasonality and level covariates.
- **Forecast Evaluation**: ARCH-based one-step forecasts vs EWMA/rolling baselines, predictive log scores, PIT histograms, coverage checks, and Diebold–Mariano comparisons.

## Reproducibility Tips

- Set environment variables for proxies or store Yahoo cookies if necessary; otherwise `yfinance` handles requests anonymously.
- All randomness uses the global `RANDOM_STATE` from `src/config.py` where stochastic components are needed.
- To integrate with CI, convert notebooks to scripts via `jupyter nbconvert --to script notebooks/*.ipynb`.

## Next Steps

- Extend to intraday or alternative implied-volatility indices.
- Add bootstrap confidence intervals for half-life estimates.
- Integrate PIT and backtest plots into a lightweight report (e.g., Quarto or LaTeX).
