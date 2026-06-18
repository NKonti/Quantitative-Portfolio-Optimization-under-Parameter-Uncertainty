# Quantitative-Portfolio-Optimization-under-Parameter-Uncertainty
Portfolio optimization is one of the fundamental problems in quantitative finance. Classical portfolio theory assumes that future expected returns and risks are known. In reality, however, both quantities must be estimated from historical data and are therefore subject to substantial estimation error.

This project investigates how different portfolio optimization frameworks perform when expected returns are not known but must first be forecasted using machine learning techniques.

The analysis combines return forecasting, portfolio optimization and walk-forward backtesting in a fully reproducible quantitative research framework.

Research Question

Theoretical portfolio optimization models are typically developed under the assumption that expected returns are known:

μ

In practice, portfolio managers only observe estimates:

μ̂

where:

μ = true expected return vector
μ̂ = estimated expected return vector

The central question of this project is:

Can advanced portfolio optimization methods outperform simple diversification when expected returns must be estimated from historical data and are therefore subject to estimation uncertainty?

Motivation

Mean-Variance Optimization is theoretically optimal if expected returns and risks are known.

In practice:

μ̂ = μ + εμ

and

Σ̂ = Σ + εΣ

where:

μ = true expected return vector
μ̂ = estimated expected return vector
εμ = return estimation error
Σ = true covariance matrix
Σ̂ = estimated covariance matrix
εΣ = covariance estimation error

Small forecasting errors can lead to materially different portfolio allocations.

To address this issue, robust optimization methods have been developed to explicitly account for parameter uncertainty.

Project Structure
Market Data
    ↓
Feature Engineering
    ↓
Return Forecasting
    ↓
Portfolio Optimization
    ↓
Walk-Forward Backtest
    ↓
Performance Evaluation

Repository structure:

01_data_pipeline.py

02_forecasting_model_selection.py

03_portfolio_optimization_backtest.py

README.md

docs/
    Quant_Portfolio_Optimization_Report.pdf
Stage 1 – Data Engineering
Asset Universe
Ticker	Asset Class
SPY	US Equities
EFA	Developed Markets
EEM	Emerging Markets
TLT	Long-Term Treasuries
IEF	Intermediate Treasuries
GLD	Gold
VNQ	Real Estate
LQD	Investment Grade Bonds
HYG	High Yield Bonds
XLF	Financials
XLV	Healthcare
XLE	Energy
Return Calculation
r(t) = P(t) / P(t−1) − 1

where:

P(t) = current price
P(t−1) = previous month's price
r(t) = monthly return
Feature Construction
Momentum Features
mom_1
mom_3
mom_6
mom_12
Volatility Features
vol_3
vol_6
vol_12
Drawdown Feature
drawdown_12
Prediction Target
y(t) = r(t+1)

The objective is to forecast next-month returns using only information available at time t.

Stage 2 – Expected Return Forecasting

Expected returns are estimated as:

μ̂ = f(X)

where:

X = feature vector
μ̂ = forecasted expected return
Forecasting Models
Ordinary Least Squares (OLS)
r̂ = β₀ + β₁x₁ + ... + βₚxₚ

where:

β = regression coefficients
x = feature values
Random Forest
r̂ = fRF(X)
XGBoost
r̂ = fXGB(X)
Forecast Evaluation
Root Mean Squared Error
RMSE = √( average( (r − r̂)² ) )
Mean Absolute Error
MAE = average( |r − r̂| )
Information Coefficient
IC = corr(r̂ , r)
Rank Information Coefficient
RankIC = corr(rank(r̂), rank(r))
Forecasting Findings

Three forecasting models were compared:

OLS
Random Forest
XGBoost

The forecasting results revealed that monthly ETF returns are difficult to predict using momentum, volatility and drawdown features.

Predictive power remained weak across all evaluated models.

Among the tested approaches, XGBoost achieved the lowest out-of-sample forecast error and was therefore selected as the expected-return estimator used in the portfolio optimization stage.

Stage 3 – Portfolio Optimization

The forecast vector

μ̂ = (μ̂₁, ..., μ̂ₙ)

serves as input for all portfolio construction models.

where:

n = number of assets
μ̂ᵢ = expected return estimate of asset i

The covariance matrix is denoted by:

Σ

where:

Σᵢⱼ = covariance between assets i and j
Portfolio Constraints
Fully Invested Portfolio
Σ wi = 1
Long-Only Constraint
wi ≥ 0
Maximum Position Size
wi ≤ 0.30
Turnover Constraint
||wt − wt−1||₁ ≤ 0.50
Transaction Costs
0.002 × turnover
Diversification Penalty
0.10 × Σ wi²
Portfolio Construction Models
1. Equal Weight Portfolio
wi = 1 / n

where:

n = number of assets

No forecasting or optimization is applied.

2. Mean-Variance Optimization

Objective:

max μ̂ᵀw − λwᵀΣw

where:

w = portfolio weight vector
μ̂ = expected return vector
Σ = covariance matrix
λ = risk-aversion parameter
3. Ellipsoidal Robust Optimization

Theoretical formulation:

maxw minμ∈U ( μᵀw − λwᵀΣw )

Uncertainty set:

U = { μ : (μ−μ̂)ᵀΩ⁻¹(μ−μ̂) ≤ ρ² }

where:

Ω = uncertainty covariance matrix
ρ = uncertainty radius

This is a genuine max-min optimization problem.

The implementation solves the mathematically equivalent robust counterpart:

max μ̂ᵀw − ρ||Ω½w||₂ − λwᵀΣw

Interpretation:

The optimizer chooses portfolio weights that perform best under the worst admissible return realization inside the uncertainty set.

4. Distributionally Robust Optimization (DRO)

Objective:

max μ̂ᵀw − ρ||Σ½w||₂ − λwᵀΣw

where:

ρ = robustness parameter
Σ½ = covariance matrix square root

Unlike ellipsoidal robustness, DRO protects against uncertainty in the entire return distribution.

5. Stochastic Optimization

Scenario returns:

Rs = μ̂ + (rs − r̄)

where:

Rs = scenario return vector
rs = historical scenario return
r̄ = historical average return

Objective:

max E(Rp) − λσ(Rp)

where:

Rp = portfolio return
E(Rp) = expected portfolio return across scenarios
σ(Rp) = portfolio volatility across scenarios

Unlike robust optimization, stochastic optimization does not optimize against a worst-case outcome. Instead, it seeks attractive performance across many plausible future states.

Backtesting Framework

For every month:

Forecast expected returns using XGBoost.
Estimate covariance matrices.
Optimize portfolio weights.
Apply turnover constraints.
Apply transaction costs.
Observe realized next-month returns.
Repeat until the end of the sample.

All reported performance statistics are strictly out-of-sample.

Results

The following portfolio construction methods are compared:

Equal Weight
Mean-Variance Optimization
Ellipsoidal Robust Optimization
Distributionally Robust Optimization
Stochastic Optimization

Performance is evaluated using:

Annualized Return
Annualized Volatility
Sharpe Ratio

============================================================
FINAL REALISTIC MODEL COMPARISON
============================================================
      model  annualized_return  annualized_volatility  sharpe_ratio
rank                                                               
1        eq           0.091577               0.104076      0.879900
2        mv           0.087573               0.104097      0.841266
3       ell           0.082396               0.098495      0.836547
4     stoch           0.062241               0.074616      0.834155
5       dro           0.075691               0.091148      0.830426


Final Conclusion

The results do not support the existence of a universally superior portfolio construction method.

The Equal Weight portfolio achieved the highest out-of-sample Sharpe Ratio and therefore generated the strongest risk-adjusted performance over the observation period. This finding highlights the remarkable robustness of simple diversification in the presence of forecasting uncertainty.

At the same time, the stochastic optimization framework produced the lowest portfolio volatility and the most stable risk profile among all optimization-based approaches. Although this came at the expense of lower annualized returns, the resulting reduction in portfolio risk may be attractive for more risk-averse investors.

The remaining optimization frameworks occupied an intermediate position, providing different trade-offs between return maximization and risk control.

Consequently, the preferred portfolio construction method depends on the investor's objective function rather than on a universally optimal solution.

Investors seeking the highest risk-adjusted return would prefer the Equal Weight portfolio.
Investors prioritizing stability and volatility reduction may prefer the Stochastic Optimization approach.
Investors concerned about parameter uncertainty may favor Robust or Distributionally Robust Optimization frameworks.

Overall, the results suggest that the choice of portfolio construction method should be aligned with the investor's risk preferences, investment horizon, and tolerance for estimation uncertainty rather than with a single performance metric.
