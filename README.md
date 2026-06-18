# Quantitative Portfolio Construction under Parameter Uncertainty

## Overview

Portfolio optimization is a fundamental problem in quantitative finance. Classical portfolio theory assumes that future expected returns and risks are known. In practice, however, these parameters must be estimated from historical data and are therefore subject to considerable uncertainty.

This project investigates whether advanced portfolio optimization techniques can outperform simple diversification when expected returns are forecasted rather than known.

The framework combines:

* Financial data engineering
* Machine learning return forecasting
* Portfolio optimization
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

| Rank | Model         | RMSE     | MAE      | Avg IC   |
| ---- | ------------- | -------- | -------- | -------- |
| 1    | XGBoost       | 0.056600 | 0.039687 | 0.076390 |
| 2    | Random Forest | 0.057445 | 0.040091 | 0.033891 |
| 3    | OLS           | 0.057557 | 0.040205 | 0.026253 |

XGBoost achieved the strongest validation performance and was therefore selected as the leading candidate model.

### Independent Test Results (2023–2026)

The selected models were subsequently evaluated on an independent out-of-sample test period.

| Rank | Model         | RMSE     | MAE      | Avg IC    | Avg Rank IC |
| ---- | ------------- | -------- | -------- | --------- | ----------- |
| 1    | Random Forest | 0.041205 | 0.031276 | 0.088586  | 0.065042    |
| 2    | OLS           | 0.041402 | 0.031472 | 0.021135  | 0.000920    |
| 3    | XGBoost       | 0.041488 | 0.031511 | -0.025185 | -0.080060   |

Although XGBoost achieved the strongest validation performance, Random Forest demonstrated superior generalization on the independent test period. In particular, Random Forest achieved the lowest forecast error and substantially stronger Information Coefficients, indicating a more reliable ability to rank assets according to future returns.

### Key Finding

Monthly ETF returns proved difficult to predict using historical momentum, volatility, and drawdown information.

While XGBoost achieved the best validation performance, Random Forest delivered the strongest independent out-of-sample results and was therefore selected as the final forecasting model for portfolio construction.

Final forecasting model:

* Random Forest
* max_depth = 2
* min_samples_leaf = 10
* n_estimators = 200

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

| Rank | Model                           | Annualized Return | Annualized Volatility | Sharpe Ratio |
| ---- | ------------------------------- | ----------------- | --------------------- | ------------ |
| 1    | Equal Weight (EQ)               | 9.16%             | 10.41%                | 0.880        |
| 2    | Stochastic Optimization (STOCH) | 6.03%             | 7.61%                 | 0.792        |
| 3    | Distributionally Robust (DRO)   | 6.92%             | 9.85%                 | 0.703        |
| 4    | Mean-Variance (MV)              | 7.83%             | 11.41%                | 0.686        |
| 5    | Ellipsoidal Robust (ELL)        | 7.35%             | 10.83%                | 0.678        |

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

The results do not support the existence of a universally superior portfolio construction method.

The Equal Weight portfolio achieved the highest out-of-sample Sharpe Ratio and therefore generated the strongest risk-adjusted performance over the observation period. This finding highlights the remarkable robustness of simple diversification in the presence of forecasting uncertainty.

At the same time, the stochastic optimization framework produced the lowest portfolio volatility and the most stable risk profile among all optimization-based approaches. Although this came at the expense of lower annualized returns, the resulting reduction in portfolio risk may be attractive for more risk-averse investors.

The remaining optimization frameworks occupied an intermediate position, providing different trade-offs between return maximization and risk control.

Consequently, the preferred portfolio construction method depends on the investor's objective function rather than on a universally optimal solution.

Investors seeking the highest risk-adjusted return would prefer the Equal Weight portfolio. Investors prioritizing stability and volatility reduction may prefer the Stochastic Optimization approach. Investors seeking more conservative portfolio allocations may favor Robust or Distributionally Robust Optimization approaches.

Overall, the results suggest that the choice of portfolio construction method should be aligned with the investor's risk preferences, investment horizon, and tolerance for estimation uncertainty rather than with a single performance metric.
