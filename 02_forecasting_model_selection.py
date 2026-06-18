import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# ============================================================
# 1) DATEN LADEN
# ============================================================

dataset = pd.read_csv("forecast_dataset_monthly.csv", parse_dates=["date"])
dataset = dataset.sort_values(["date", "asset"]).reset_index(drop=True)

returns_monthly = pd.read_csv(
    "returns_monthly.csv",
    index_col=0,
    parse_dates=True
).sort_index()

# ============================================================
# 2) FEATURES / PARAMETER
# ============================================================

features = [
    "mom_1",
    "mom_3",
    "mom_6",
    "mom_12",
    "vol_3",
    "vol_6",
    "vol_12",
    "drawdown_12"
]

target_col = "target_return_next_month"

window = 48
min_train_obs = 36

dates = sorted(dataset["date"].unique())
assets = sorted(dataset["asset"].unique())

# ============================================================
# 3) MODELLE DEFINIEREN
# ============================================================

models = {
    "OLS": LinearRegression(),

    "RandomForest": RandomForestRegressor(
        n_estimators=300,
        max_depth=3,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBRegressor(
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
}

# ============================================================
# 4) ROLLING OUT-OF-SAMPLE PREDICTIONS
# ============================================================

all_predictions = []
feature_importance_rows = []

for i in range(window, len(dates)):

    current_date = dates[i]

    train_df = dataset[dataset["date"] < current_date].copy()
    current_df = dataset[dataset["date"] == current_date].copy()

    train_df = train_df.dropna(subset=features + [target_col]).copy()
    current_df = current_df.dropna(subset=features + [target_col]).copy()

    if len(train_df) < min_train_obs:
        continue

    if train_df.empty or current_df.empty:
        continue

    X_train = train_df[features].values
    y_train = train_df[target_col].values
    X_current = current_df[features].values

    for estimator_name, model in models.items():

        model.fit(X_train, y_train)
        y_pred = model.predict(X_current)

        for j, (_, row) in enumerate(current_df.iterrows()):

            all_predictions.append({
                "date": current_date,
                "asset": row["asset"],
                "estimator": estimator_name,
                "predicted_return": y_pred[j],
                "realized_return": row[target_col]
            })

        if hasattr(model, "feature_importances_"):
            for feat, importance in zip(features, model.feature_importances_):
                feature_importance_rows.append({
                    "date": current_date,
                    "estimator": estimator_name,
                    "feature": feat,
                    "importance": importance
                })

pred_df = (
    pd.DataFrame(all_predictions)
    .sort_values(["estimator", "date", "asset"])
    .reset_index(drop=True)
)

print("\n============================================================")
print("FORECAST OUTPUT PREVIEW")
print("============================================================")
print(pred_df.head(20).to_string(index=False))

# ============================================================
# 5) FEHLERMETRIKEN
# ============================================================

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean(np.abs(y_true - y_pred))

global_results = []

for estimator_name, group in pred_df.groupby("estimator"):

    global_results.append({
        "estimator": estimator_name,
        "rmse": rmse(group["realized_return"], group["predicted_return"]),
        "mae": mae(group["realized_return"], group["predicted_return"])
    })

global_results_df = pd.DataFrame(global_results)

# ============================================================
# 6) MONATLICHER IC / RANK IC
# ============================================================

monthly_ic_rows = []

for (estimator_name, dt), group in pred_df.groupby(["estimator", "date"]):

    group = group.dropna(subset=["predicted_return", "realized_return"]).copy()

    if len(group) < 3:
        continue

    try:
        ic = pearsonr(
            group["predicted_return"],
            group["realized_return"]
        )[0]
    except Exception:
        ic = np.nan

    try:
        rank_ic = spearmanr(
            group["predicted_return"],
            group["realized_return"]
        )[0]
    except Exception:
        rank_ic = np.nan

    monthly_ic_rows.append({
        "date": dt,
        "estimator": estimator_name,
        "ic": ic,
        "rank_ic": rank_ic
    })

monthly_ic_df = (
    pd.DataFrame(monthly_ic_rows)
    .sort_values(["estimator", "date"])
    .reset_index(drop=True)
)

# ============================================================
# 7) IC SUMMARY
# ============================================================

ic_summary_rows = []

for estimator_name, group in monthly_ic_df.groupby("estimator"):

    ic_summary_rows.append({
        "estimator": estimator_name,
        "avg_ic": group["ic"].mean(),
        "median_ic": group["ic"].median(),
        "avg_rank_ic": group["rank_ic"].mean(),
        "median_rank_ic": group["rank_ic"].median()
    })

ic_summary_df = pd.DataFrame(ic_summary_rows)

# ============================================================
# 8) FINAL MODEL COMPARISON
# ============================================================
# Für absolute Renditeprognosen ist RMSE/MAE zentral,
# weil die Optimierungsmodelle die Höhe von mu_hat verwenden.
# IC / Rank IC werden zusätzlich berichtet.
# ============================================================

model_comparison_df = (
    global_results_df
    .merge(ic_summary_df, on="estimator", how="left")
)

model_comparison_df = (
    model_comparison_df
    .sort_values(
        ["rmse", "mae", "avg_ic"],
        ascending=[True, True, False]
    )
    .reset_index(drop=True)
)

model_comparison_df.index = model_comparison_df.index + 1
model_comparison_df.index.name = "rank"

print("\n============================================================")
print("FINAL FORECAST MODEL COMPARISON")
print("============================================================")
print(model_comparison_df.to_string())
print("============================================================")

best_model = model_comparison_df.iloc[0]["estimator"]

print("\n============================================================")
print("SELECTED FORECAST MODEL")
print("============================================================")
print(f"Best model based on absolute return prediction error: {best_model}")
print("Selection criterion: lowest RMSE, then lowest MAE, then highest Avg IC")
print("============================================================")

# ============================================================
# 9) FEATURE IMPORTANCE SUMMARY
# ============================================================

if feature_importance_rows:

    feature_importance_df = pd.DataFrame(feature_importance_rows)

    feature_importance_summary = (
        feature_importance_df
        .groupby(["estimator", "feature"])["importance"]
        .mean()
        .reset_index()
        .sort_values(["estimator", "importance"], ascending=[True, False])
    )

    print("\n============================================================")
    print("FEATURE IMPORTANCE SUMMARY")
    print("============================================================")
    print(feature_importance_summary.to_string(index=False))
    print("============================================================")

else:
    feature_importance_df = pd.DataFrame()
    feature_importance_summary = pd.DataFrame()

# ============================================================
# 10) DATEIEN SPEICHERN
# ============================================================

pred_df.to_csv("forecast_model_predictions.csv", index=False)
global_results_df.to_csv("forecast_global_error_metrics.csv", index=False)
monthly_ic_df.to_csv("forecast_monthly_ic.csv", index=False)
ic_summary_df.to_csv("forecast_ic_summary.csv", index=False)
model_comparison_df.reset_index().to_csv("forecast_model_comparison.csv", index=False)

best_model_df = pd.DataFrame({
    "selected_model": [best_model],
    "selection_criterion": ["lowest RMSE, then lowest MAE, then highest Avg IC"],
    "target_used": ["absolute next-month return"]
})

best_model_df.to_csv("selected_forecast_model.csv", index=False)

if not feature_importance_df.empty:
    feature_importance_df.to_csv("forecast_feature_importance_by_date.csv", index=False)
    feature_importance_summary.to_csv("forecast_feature_importance_summary.csv", index=False)

print("\nDateien gespeichert:")
print("- forecast_model_predictions.csv")
print("- forecast_global_error_metrics.csv")
print("- forecast_monthly_ic.csv")
print("- forecast_ic_summary.csv")
print("- forecast_model_comparison.csv")
print("- selected_forecast_model.csv")

if not feature_importance_df.empty:
    print("- forecast_feature_importance_by_date.csv")
    print("- forecast_feature_importance_summary.csv")
