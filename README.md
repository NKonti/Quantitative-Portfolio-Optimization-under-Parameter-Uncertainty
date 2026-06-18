# Quantitative Portfolio Construction under Parameter Uncertainty

## Overview

Portfolio optimization is a fundamental problem in quantitative finance. Classical portfolio theory assumes that future expected returns and risks are known. In practice, however, expected returns must be estimated from historical data and are therefore subject to substantial forecasting uncertainty.

This project investigates whether advanced portfolio optimization techniques can outperform simple diversification when expected returns are estimated using machine learning models rather than assumed to be known.

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

XGBoost achieved the strongest validation performance and was therefore selected as the leading candidate model during the model selection stage.

### Independent Test Results (2023–2026)

The selected models were subsequently evaluated on an independent out-of-sample test period.

| Rank | Model         | RMSE     | MAE      | Avg IC    | Avg Rank IC |
| ---- | ------------- | -------- | -------- | --------- | ----------- |
| 1    | Random Forest | 0.041205 | 0.031276 | 0.088586  | 0.065042    |
| 2    | OLS           | 0.041402 | 0.031472 | 0.021135  | 0.000920    |
| 3    | XGBoost       | 0.041488 | 0.031511 | -0.025185 | -0.080060   |

Although XGBoost achieved the strongest validation performance, Random Forest demonstrated superior generalization on the independent test period. Random Forest achieved the lowest forecast error, the strongest Information Coefficient, and the strongest Rank Information Coefficient.

### Final Forecasting Model

The primary portfolio optimization analysis uses Random Forest forecasts because Random Forest demonstrated the strongest independent out-of-sample forecasting performance.

Selected model:

* Random Forest
* max_depth = 2
* min_samples_leaf = 10
* n_estimators = 200

### Key Finding

Monthly ETF returns proved difficult to predict using historical momentum, volatility, and drawdown information.

While XGBoost achieved the strongest validation performance, Random Forest delivered the strongest independent out-of-sample forecasting performance and was therefore selected as the primary forecasting model.

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

## Results (Random Forest Forecasts)

Forecast Model: Random Forest

Parameters:

* max_depth = 2
* min_samples_leaf = 10
* n_estimators = 200

| Rank | Model                           | Annualized Return | Annualized Volatility | Sharpe Ratio |
| ---- | ------------------------------- | ----------------- | --------------------- | ------------ |
| 1    | Equal Weight (EQ)               | 9.16%             | 10.41%                | 0.880        |
| 2    | Stochastic Optimization (STOCH) | 6.03%             | 7.61%                 | 0.792        |
| 3    | Distributionally Robust (DRO)   | 6.92%             | 9.85%                 | 0.703        |
| 4    | Mean-Variance (MV)              | 7.83%             | 11.41%                | 0.686        |
| 5    | Ellipsoidal Robust (ELL)        | 7.35%             | 10.83%                | 0.678        |

---

## Forecast Model Sensitivity Analysis

As a robustness check, the portfolio optimization framework was also evaluated using XGBoost forecasts.

### Results (XGBoost Forecasts)

Parameters:

* learning_rate = 0.03
* max_depth = 1
* n_estimators = 500
* subsample = 0.8
* colsample_bytree = 0.8
* reg_alpha = 1.0
* reg_lambda = 1.0

| Rank | Model                           | Annualized Return | Annualized Volatility | Sharpe Ratio |
| ---- | ------------------------------- | ----------------- | --------------------- | ------------ |
| 1    | Equal Weight (EQ)               | 9.16%             | 10.41%                | 0.880        |
| 2    | Mean-Variance (MV)              | 8.76%             | 10.41%                | 0.841        |
| 3    | Ellipsoidal Robust (ELL)        | 8.24%             | 9.85%                 | 0.837        |
| 4    | Stochastic Optimization (STOCH) | 6.22%             | 7.46%                 | 0.834        |
| 5    | Distributionally Robust (DRO)   | 7.57%             | 9.11%                 | 0.830        |

Across both forecasting specifications, the Equal Weight benchmark remained the best-performing portfolio on a risk-adjusted basis. This suggests that the central findings are robust to changes in the forecasting model used to estimate expected returns.

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

Across all tested optimization frameworks, the Equal Weight portfolio achieved the highest out-of-sample Sharpe Ratio and consistently generated the strongest risk-adjusted performance. This result highlights the remarkable robustness of simple diversification when expected returns must be estimated rather than known.

While advanced optimization techniques incorporated forecasts, covariance estimates, transaction costs, turnover constraints, and robustness adjustments, none of the optimized portfolios were able to consistently outperform the naive Equal Weight benchmark on a risk-adjusted basis.

Among the optimization-based approaches, Stochastic Optimization achieved the strongest Sharpe Ratio and the lowest volatility under the Random Forest forecasting specification. Mean-Variance, Ellipsoidal Robust, and Distributionally Robust Optimization produced intermediate results, reflecting different trade-offs between expected return and risk control.

An additional finding of the study is that improvements in forecasting accuracy do not necessarily translate into superior portfolio performance. Although Random Forest achieved the strongest forecasting performance on the independent test set, the resulting optimized portfolios still failed to outperform the Equal Weight benchmark. This observation further illustrates the challenges associated with parameter uncertainty in practical portfolio optimization.

The robustness analysis using XGBoost forecasts led to the same overarching conclusion: Equal Weight remained the strongest portfolio on a risk-adjusted basis despite noticeable changes in the relative ranking of the optimization-based approaches.

Overall, the findings suggest that portfolio optimization remains highly sensitive to estimation error and forecasting uncertainty. Consequently, the choice of portfolio construction method should be aligned with the investor's objectives, risk tolerance, and confidence in return forecasts rather than with a single optimization framework.

The results provide empirical support for the view that simple diversification remains a highly competitive benchmark, even when advanced forecasting and optimization techniques are employed.
