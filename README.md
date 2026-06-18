# Quantitative Portfolio Construction under Parameter Uncertainty

## Overview

Portfolio optimization is a fundamental problem in quantitative finance. Classical portfolio theory assumes that future expected returns and risks are known. In practice, however, expected returns must be estimated from historical data and are therefore subject to substantial forecasting uncertainty.

This project investigates whether advanced portfolio optimization techniques can outperform simple diversification when expected returns are estimated rather than known.

The framework combines:

* Financial data engineering
* Machine learning return forecasting
* Portfolio optimization under parameter uncertainty
* Walk-forward backtesting
* Performance evaluation

---

## Research Question

**Can advanced portfolio optimization methods outperform naive diversification when expected returns must be estimated from historical data and are therefore subject to forecasting uncertainty?**

---

## Asset Universe

The analysis is based on a diversified multi-asset ETF universe:

| Ticker | Asset Class             |
| ------ | ----------------------- |
| SPY    | US Equities             |
| EFA    | Developed Markets       |
| EEM    | Emerging Markets        |
| TLT    | Long-Term Treasuries    |
| IEF    | Intermediate Treasuries |
| GLD    | Gold                    |
| VNQ    | Real Estate             |
| LQD    | Investment Grade Bonds  |
| HYG    | High Yield Bonds        |
| XLF    | Financials              |
| XLV    | Healthcare              |
| XLE    | Energy                  |

---

## Feature Engineering

Monthly return forecasts are generated using:

### Momentum Features

* 1-Month Momentum
* 3-Month Momentum
* 6-Month Momentum
* 12-Month Momentum

### Risk Features

* 3-Month Volatility
* 6-Month Volatility
* 12-Month Volatility

### Drawdown Feature

* 12-Month Maximum Drawdown

### Target Variable

* Next-month ETF return

---

## Forecasting Models

Three forecasting approaches were compared:

* Ordinary Least Squares (OLS)
* Random Forest
* XGBoost

Forecast quality was evaluated using:

* RMSE
* MAE
* Information Coefficient (IC)
* Rank Information Coefficient (Rank IC)

### Validation Results (2019–2022)

Model selection was initially performed using a dedicated validation period.

| Rank | Model         | RMSE     | MAE      | Avg IC   | Avg Rank IC |
| ---- | ------------- | -------- | -------- | -------- | ----------- |
| 1    | XGBoost       | 0.056600 | 0.039687 | 0.076390 | 0.036398    |
| 2    | Random Forest | 0.057445 | 0.040091 | 0.033891 | -0.007626   |
| 3    | OLS           | 0.057557 | 0.040205 | 0.026253 | 0.025204    |

XGBoost achieved the strongest validation performance and was therefore selected as the leading candidate during the model selection stage.

### Independent Test Results (2023–2026)

The selected models were subsequently evaluated on an independent out-of-sample test period.

| Rank | Model         | RMSE     | MAE      | Avg IC    | Avg Rank IC |
| ---- | ------------- | -------- | -------- | --------- | ----------- |
| 1    | Random Forest | 0.041205 | 0.031276 | 0.088586  | 0.065042    |
| 2    | OLS           | 0.041402 | 0.031472 | 0.021135  | 0.000920    |
| 3    | XGBoost       | 0.041488 | 0.031511 | -0.025185 | -0.080060   |

Although XGBoost achieved the strongest validation performance, Random Forest demonstrated superior generalization on the independent test period. Random Forest achieved the lowest forecast error, the strongest Information Coefficient, and the strongest Rank Information Coefficient.

### Final Forecasting Model

The final forecasting model selected for portfolio construction was Random Forest.

Selected parameters:

* max_depth = 2
* min_samples_leaf = 10
* n_estimators = 200

### Key Finding

Monthly ETF returns proved difficult to predict using historical momentum, volatility, and drawdown information.

While XGBoost achieved the strongest validation performance, Random Forest delivered the strongest independent out-of-sample forecasting performance and was therefore selected for the portfolio construction stage.

---

## Portfolio Construction Models

The following portfolio allocation approaches were evaluated:

### Equal Weight (EQ)

Simple diversification across all assets.

### Mean-Variance Optimization (MV)

Optimizes the trade-off between forecasted expected return and portfolio variance.

### Ellipsoidal Robust Optimization (ELL)

Optimizes against the worst-case expected return realization within an ellipsoidal uncertainty set.

### Distributionally Robust Optimization (DRO)

Uses an additional robustness penalty to reduce portfolio sensitivity to uncertainty in asset returns.

### Stochastic Optimization (STOCH)

Optimizes portfolio weights by maximizing expected return across multiple return scenarios while controlling scenario volatility.

---

## Portfolio Constraints

The optimization framework incorporates realistic portfolio management constraints:

* Fully invested portfolio
* Long-only allocation
* Maximum position size: 30%
* Turnover constraint
* Transaction costs
* Diversification penalty

---

## Backtesting Framework

A walk-forward backtest was conducted:

1. Forecast expected returns using Random Forest
2. Estimate covariance matrices
3. Optimize portfolio weights
4. Apply turnover constraints and transaction costs
5. Observe realized next-month returns
6. Repeat monthly until the end of the sample

All reported results are strictly out-of-sample.

---

## Results

Forecast Model: Random Forest

Parameters:

* max_depth = 2
* min_samples_leaf = 10
* n_estimators = 200

| Rank | Model                                      | Annualized Return | Annualized Volatility | Sharpe Ratio |
| ---- | ------------------------------------------ | ----------------- | --------------------- | ------------ |
| 1    | Equal Weight (EQ)                          | 9.16%             | 10.41%                | 0.880        |
| 2    | Stochastic Optimization (STOCH)            | 6.03%             | 7.61%                 | 0.792        |
| 3    | Distributionally Robust Optimization (DRO) | 6.92%             | 9.85%                 | 0.703        |
| 4    | Mean-Variance Optimization (MV)            | 7.83%             | 11.41%                | 0.686        |
| 5    | Ellipsoidal Robust Optimization (ELL)      | 7.35%             | 10.83%                | 0.678        |

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* CVXPY
* SciPy
* Financial Time Series Analysis

---

## Conclusion

The results do not support the existence of a universally superior portfolio construction method under parameter uncertainty.

Expected returns were first estimated using a Random Forest forecasting model and subsequently incorporated into several advanced portfolio optimization frameworks. Despite the use of machine-learning-based forecasts and sophisticated optimization techniques, none of the optimized portfolios consistently outperformed the naive Equal Weight benchmark on a risk-adjusted basis.

Across all tested approaches, the Equal Weight portfolio achieved the highest out-of-sample Sharpe Ratio of 0.880 and generated the strongest risk-adjusted performance over the observation period. This finding highlights the remarkable robustness of simple diversification when expected returns must be estimated rather than assumed to be known.

Among the optimization-based approaches, Stochastic Optimization delivered the strongest performance, achieving a Sharpe Ratio of 0.792 while simultaneously generating the lowest portfolio volatility. Distributionally Robust Optimization, Mean-Variance Optimization, and Ellipsoidal Robust Optimization produced weaker risk-adjusted results despite explicitly accounting for estimation uncertainty.

The findings suggest that parameter uncertainty remains a major challenge in practical portfolio construction. Even when expected returns are generated using machine-learning forecasts, advanced optimization techniques remain highly sensitive to estimation error.

Consequently, the choice of portfolio construction method should be aligned with the investor's objectives and risk preferences rather than with the expectation that increasingly sophisticated optimization techniques will automatically generate superior investment outcomes.

Overall, the study provides empirical evidence that simple diversification remains a highly competitive benchmark, even when advanced forecasting models and portfolio optimization frameworks are employed.
