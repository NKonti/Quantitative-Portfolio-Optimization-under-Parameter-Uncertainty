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

### Interpretation of Optimization Results

#### Equal Weight Portfolio (EQ)

The Equal Weight portfolio serves as the naïve diversification benchmark and allocates capital uniformly across all available assets:

```text
w_i = 1 / n
```

Unlike the optimization-based approaches, Equal Weight does not rely on expected return forecasts, covariance estimates or uncertainty sets. As a result, it is largely immune to estimation errors in both expected returns and risk parameters.

The strong performance of the Equal Weight portfolio highlights an important insight from the portfolio optimization literature: when expected returns are difficult to estimate accurately, avoiding forecast-driven allocations can be advantageous. By distributing capital evenly across all assets, the portfolio benefits from broad diversification while avoiding the concentration risk that often arises when optimization models overreact to noisy forecasts.

The results suggest that, within the context of this study, the robustness gained from avoiding parameter estimation outweighed the potential benefits of more sophisticated optimization techniques. Consequently, Equal Weight achieved the highest out-of-sample Sharpe Ratio and delivered the strongest overall risk-adjusted performance.

---

#### Mean-Variance Optimization (MV)

Mean-Variance Optimization allocates capital according to the forecasted risk-return trade-off and is therefore highly sensitive to estimation errors. Even small deviations in expected return estimates may lead to materially different portfolio weights and can deteriorate out-of-sample performance.

The model assumes that forecasted expected returns contain sufficiently accurate information to justify overweighting and underweighting individual assets. In practice, however, expected returns are notoriously difficult to estimate and forecasting errors can dominate the optimization process. As a result, the theoretically optimal allocation may become suboptimal once implemented on unseen future data.

---

#### Ellipsoidal Robust Optimization (ELL)

Ellipsoidal Robust Optimization explicitly acknowledges parameter uncertainty by assuming that expected returns lie within a predefined uncertainty set. Rather than optimizing for a single estimated return vector, the model seeks a portfolio that performs reasonably well across a range of plausible return realizations.

As a consequence, the optimizer adopts a more conservative allocation and penalizes portfolios that are highly dependent on specific return forecasts. While this improves robustness against estimation errors, it may also reduce exposure to genuinely attractive investment opportunities, resulting in lower realized returns.

The empirical results indicate that the reduction in forecast sensitivity was not sufficient to fully compensate for the loss of return potential associated with the additional robustness penalty.

---

#### Distributionally Robust Optimization (DRO)

Distributionally Robust Optimization extends the concept of robustness beyond the expected return vector and explicitly accounts for uncertainty in the entire return distribution.

Rather than protecting only against errors in expected returns, the model seeks allocations that remain resilient under a broad range of possible probability distributions. Consequently, the resulting portfolios tend to be even more conservative and place a stronger emphasis on stability and protection against adverse scenarios.

While this approach reduces exposure to model misspecification and distributional uncertainty, it also limits the optimizer's ability to exploit forecasted return opportunities. In the present study, the additional robustness reduced portfolio flexibility and ultimately resulted in lower risk-adjusted performance than the Equal Weight benchmark.

---

#### Stochastic Optimization (STOCH)

Stochastic Optimization differs fundamentally from the robust optimization frameworks. Instead of optimizing against a worst-case realization of uncertain parameters, the model constructs a large set of plausible future scenarios and seeks a portfolio that performs well on average across those scenarios.

The optimization therefore balances expected return and risk across many potential future market environments rather than focusing on a single forecast or a predefined uncertainty set. This generally leads to more diversified and stable allocations, as the portfolio is designed to remain effective under a variety of market conditions.

The results show that Stochastic Optimization generated the lowest annualized portfolio volatility among all tested approaches. Although this came at the cost of lower annualized returns and a slightly lower Sharpe Ratio, the substantial reduction in portfolio risk demonstrates the effectiveness of the scenario-based framework.

From an economic perspective, Stochastic Optimization may therefore represent the most attractive solution for highly risk-averse investors, institutional investors, pension funds or insurance companies whose primary objective is long-term stability rather than maximum risk-adjusted return.

---

### Overall Interpretation

The empirical results suggest that no portfolio construction method was universally superior. The Equal Weight portfolio benefited from its complete independence from parameter estimation and achieved the highest out-of-sample Sharpe Ratio. In contrast, the optimization-based approaches attempted to exploit forecasted return information and explicitly account for uncertainty, but remained dependent on the quality of the underlying return forecasts.


