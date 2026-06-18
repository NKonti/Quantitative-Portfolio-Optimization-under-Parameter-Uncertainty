# ============================================================
# FINAL BACKTEST
#
# Benchmark:
#   - Equal Weight
#
# Optimization Models:
#   - Mean-Variance Optimization
#   - Ellipsoidal Robust Optimization
#   - Distributionally Robust Optimization (DRO)
#   - Stochastic Optimization
#
# Realistic Features:
#   - Budget Allocation
#   - Transaction Costs
#   - Turnover Constraint
#   - Diversification Penalty
#   - Maximum Weight Constraint
#   - Solver Tolerance Correction
# ============================================================

import warnings
warnings.filterwarnings("ignore")



import numpy as np
import pandas as pd
import cvxpy as cp
from xgboost import XGBRegressor

# ============================================================
# SETTINGS
# ============================================================

budget = 10000

window = 36
start_idx = 60

risk_aversion = 3.0
transaction_cost = 0.002
turnover_limit = 0.50
div_penalty = 0.10
max_weight = 0.30

# ============================================================
# DATA
# ============================================================

dataset = pd.read_csv("forecast_dataset_monthly.csv", parse_dates=["date"])
dataset = dataset.sort_values(["date", "asset"])

returns_monthly = pd.read_csv(
    "returns_monthly.csv",
    index_col=0,
    parse_dates=True
)

assets = sorted(dataset["asset"].unique())
dates = sorted(dataset["date"].unique())

features = [
    "mom_1", "mom_3", "mom_6", "mom_12",
    "vol_3", "vol_6", "vol_12", "drawdown_12"
]

target = "target_return_next_month"

# ============================================================
# KPI FUNCTIONS
# ============================================================

def ann_ret(r):
    return (np.prod(1 + r) ** (12 / len(r))) - 1

def ann_vol(r):
    return np.std(r) * np.sqrt(12)

def sharpe(r):
    vol = ann_vol(r)
    return ann_ret(r) / vol if vol > 0 else np.nan

# ============================================================
# STORAGE
# ============================================================

results = []

prev_weights = {
    "mv": None,
    "ell": None,
    "dro": None,
    "stoch": None
}

# ============================================================
# BACKTEST LOOP
# ============================================================

for i in range(start_idx, len(dates) - 1):

    date = dates[i]
    next_date = dates[i + 1]

    train = dataset[dataset["date"] < date]

    current = (
        dataset[dataset["date"] == date]
        .set_index("asset")
        .reindex(assets)
        .reset_index()
    )

    if current[features].isna().any().any():
        continue

    hist = (
        returns_monthly
        .loc[returns_monthly.index < date, assets]
        .dropna()
        .tail(window)
    )

    if len(hist) < window:
        continue

    Sigma = hist.cov().values
    Sigma = 0.5 * (Sigma + Sigma.T) + np.eye(len(assets)) * 1e-10

    hist_mean = hist.mean().values

    # ========================================================
    # FORECAST MODEL
    # ========================================================

    model = XGBRegressor(
        n_estimators=300,
        max_depth=1,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )

    model.fit(train[features], train[target])

    mu = model.predict(current[features])

    n = len(assets)

    # ========================================================
    # SOLVER FUNCTION
    # ========================================================

    def solve_model(objective, prev_w):

        w = cp.Variable(n)

        turnover = 0 if prev_w is None else cp.norm(w - prev_w, 1)

        obj = cp.Maximize(
            objective(w)
            - transaction_cost * turnover
            - div_penalty * cp.sum_squares(w)
        )

        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            w <= max_weight
        ]

        if prev_w is not None:
            constraints.append(turnover <= turnover_limit)

        problem = cp.Problem(obj, constraints)
        problem.solve(solver=cp.CLARABEL, verbose=False)

        if w.value is None:
            raise RuntimeError(f"Optimization failed at date {date}")

        weights = np.asarray(w.value).flatten()

        weights = np.maximum(weights, 0)
        weights = weights / weights.sum()

        return weights

    # ========================================================
    # MODEL 1: MEAN-VARIANCE OPTIMIZATION
    # ========================================================

    mv = solve_model(
        lambda w:
            mu @ w
            - risk_aversion * cp.quad_form(w, Sigma),
        prev_weights["mv"]
    )

    # ========================================================
    # MODEL 2: ELLIPSOIDAL ROBUST OPTIMIZATION
    # ========================================================

    Omega = Sigma / len(hist)
    Omega = 0.5 * (Omega + Omega.T) + np.eye(len(assets)) * 1e-10

    eigvals, eigvecs = np.linalg.eigh(Omega)

    Omega_sqrt = (
        eigvecs
        @ np.diag(np.sqrt(np.clip(eigvals, 1e-12, None)))
        @ eigvecs.T
    )

    ell = solve_model(
        lambda w:
            mu @ w
            - 1.0 * cp.norm(Omega_sqrt @ w, 2)
            - risk_aversion * cp.quad_form(w, Sigma),
        prev_weights["ell"]
    )

    # ========================================================
    # MODEL 3: DISTRIBUTIONALLY ROBUST OPTIMIZATION
    # ========================================================

    eigvals, eigvecs = np.linalg.eigh(Sigma)

    Sigma_sqrt = (
        eigvecs
        @ np.diag(np.sqrt(np.clip(eigvals, 1e-12, None)))
        @ eigvecs.T
    )

    dro = solve_model(
        lambda w:
            mu @ w
            - 0.5 * cp.norm(Sigma_sqrt @ w, 2)
            - risk_aversion * cp.quad_form(w, Sigma),
        prev_weights["dro"]
    )

    # ========================================================
    # MODEL 4: STOCHASTIC OPTIMIZATION
    # ========================================================

    deviations = hist.values - hist_mean
    scenarios = mu + deviations

    probs = np.repeat(1 / len(hist), len(hist))

    def stoch_obj(w):

        scen_returns = scenarios @ w
        mean_return = probs @ scen_returns
        centered = scen_returns - mean_return

        std_return = cp.norm(
            cp.multiply(np.sqrt(probs), centered),
            2
        )

        return mean_return - risk_aversion * std_return

    stoch = solve_model(
        stoch_obj,
        prev_weights["stoch"]
    )

    # ========================================================
    # BENCHMARK: EQUAL WEIGHT
    # ========================================================

    eq = np.ones(n) / n

    # ========================================================
    # REALIZED NEXT-MONTH RETURNS
    # ========================================================

    r_next = returns_monthly.loc[next_date, assets].values

    results.append({
        "date": next_date,
        "mv": np.dot(mv, r_next),
        "ell": np.dot(ell, r_next),
        "dro": np.dot(dro, r_next),
        "stoch": np.dot(stoch, r_next),
        "eq": np.dot(eq, r_next)
    })

    prev_weights["mv"] = mv
    prev_weights["ell"] = ell
    prev_weights["dro"] = dro
    prev_weights["stoch"] = stoch

# ============================================================
# PERFORMANCE SUMMARY
# ============================================================

df = pd.DataFrame(results)
df = df.set_index("date")

summary = []

for col in df.columns:

    r = df[col].values

    summary.append({
        "model": col,
        "annualized_return": ann_ret(r),
        "annualized_volatility": ann_vol(r),
        "sharpe_ratio": sharpe(r)
    })

summary_df = (
    pd.DataFrame(summary)
    .sort_values("sharpe_ratio", ascending=False)
    .reset_index(drop=True)
)

summary_df.index = summary_df.index + 1
summary_df.index.name = "rank"

summary_df = summary_df[
    [
        "model",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio"
    ]
]

print("\n============================================================")
print("FINAL REALISTIC MODEL COMPARISON")
print("============================================================")
print(summary_df)
print("============================================================")

# ============================================================
# FINAL TARGET ALLOCATION BASED ON LAST OPTIMAL WEIGHTS
# ============================================================

allocation_rows = []

for model_name, weights in prev_weights.items():

    allocation = pd.DataFrame({
        "model": model_name,
        "asset": assets,
        "weight": weights,
        "target_investment": weights * budget
    })

    allocation_rows.append(allocation)

eq_weights = np.ones(len(assets)) / len(assets)

eq_allocation = pd.DataFrame({
    "model": "eq",
    "asset": assets,
    "weight": eq_weights,
    "target_investment": eq_weights * budget
})

allocation_rows.append(eq_allocation)

allocation_df = (
    pd.concat(allocation_rows, ignore_index=True)
    .sort_values(["model", "weight"], ascending=[True, False])
    .reset_index(drop=True)
)

print("\n============================================================")
print(f"FINAL TARGET ALLOCATION BASED ON BUDGET = {budget:,.2f}")
print("============================================================")
print(allocation_df)
print("============================================================")