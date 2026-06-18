# ============================================================
# QUANT PROJECT - FORECASTING MODEL SELECTION
# TRAIN / VALIDATION / TEST + SMALL GRID SEARCH
# ============================================================
# Ziel:
# 1) Absolute Return Forecasting
# 2) OLS, Random Forest und XGBoost vergleichen
# 3) Hyperparameter nur auf Validation auswählen
# 4) Test Set unangetastet lassen
# 5) Bestes Modell als Expected-Return-Schätzer auswählen
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# ============================================================
# 1) DATA
# ============================================================

dataset = pd.read_csv("forecast_dataset_monthly.csv", parse_dates=["date"])
dataset = dataset.sort_values(["date", "asset"]).reset_index(drop=True)

# ============================================================
# 2) FEATURES / TARGET
# ============================================================

features = [
    "mom_1", "mom_3", "mom_6", "mom_12",
    "vol_3", "vol_6", "vol_12", "drawdown_12"
]

target_col = "target_return_next_month"

dates = sorted(dataset["date"].unique())
assets = sorted(dataset["asset"].unique())

min_train_obs = 36

# ============================================================
# 3) TIME SPLIT
# ============================================================
# Train:      until 2018-12-31
# Validation: 2019-01-01 to 2022-12-31
# Test:       from 2023-01-01 onward
# ============================================================

train_end = pd.Timestamp("2018-12-31")
val_start = pd.Timestamp("2019-01-01")
val_end = pd.Timestamp("2022-12-31")
test_start = pd.Timestamp("2023-01-01")

validation_dates = [
    d for d in dates
    if val_start <= pd.Timestamp(d) <= val_end
]

test_dates = [
    d for d in dates
    if pd.Timestamp(d) >= test_start
]

print("\n============================================================")
print("TIME SPLIT")
print("============================================================")
print(f"Train period      : < {val_start.date()}")
print(f"Validation period : {val_start.date()} to {val_end.date()}")
print(f"Test period       : >= {test_start.date()}")
print(f"Validation months : {len(validation_dates)}")
print(f"Test months       : {len(test_dates)}")
print("============================================================")

# ============================================================
# 4) SMALL GRID
# ============================================================

model_specs = []

# ------------------------------------------------------------
# OLS Benchmark
# ------------------------------------------------------------

model_specs.append({
    "estimator_family": "OLS",
    "model_id": "OLS_base",
    "model": LinearRegression(),
    "params": {}
})

# ------------------------------------------------------------
# Random Forest Grid
# ------------------------------------------------------------

rf_param_grid = {
    "n_estimators": [200, 500],
    "max_depth": [2, 4],
    "min_samples_leaf": [3, 10]
}

for idx, params in enumerate(ParameterGrid(rf_param_grid), start=1):
    model_specs.append({
        "estimator_family": "RandomForest",
        "model_id": f"RandomForest_grid_{idx}",
        "model": RandomForestRegressor(
            **params,
            random_state=42,
            n_jobs=-1
        ),
        "params": params
    })

# ------------------------------------------------------------
# XGBoost Grid
# ------------------------------------------------------------

xgb_param_grid = {
    "n_estimators": [200, 500],
    "max_depth": [1, 2],
    "learning_rate": [0.03, 0.08]
}

for idx, params in enumerate(ParameterGrid(xgb_param_grid), start=1):
    model_specs.append({
        "estimator_family": "XGBoost",
        "model_id": f"XGBoost_grid_{idx}",
        "model": XGBRegressor(
            **params,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            verbosity=0
        ),
        "params": params
    })

print("\n============================================================")
print("GRID SEARCH SETUP")
print("============================================================")
print(f"Number of model configurations: {len(model_specs)}")
print("1 OLS + 8 RandomForest + 8 XGBoost = 17 configurations")
print("============================================================")

# ============================================================
# 5) METRIC FUNCTIONS
# ============================================================

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean(np.abs(y_true - y_pred))

# ============================================================
# 6) ROLLING PREDICTION FUNCTION
# ============================================================

def rolling_predictions(eval_dates, specs, phase_name):

    all_predictions = []
    feature_importance_rows = []

    for current_date in eval_dates:

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

        for spec in specs:

            estimator_family = spec["estimator_family"]
            model_id = spec["model_id"]
            model_params = spec["params"]

            model = clone(spec["model"])

            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_current)

            except Exception as e:
                print(f"Model failed: {model_id} at {current_date}. Error: {e}")
                continue

            for j, (_, row) in enumerate(current_df.iterrows()):

                all_predictions.append({
                    "phase": phase_name,
                    "date": current_date,
                    "asset": row["asset"],
                    "estimator_family": estimator_family,
                    "model_id": model_id,
                    "params": str(model_params),
                    "predicted_return": y_pred[j],
                    "realized_return": row[target_col]
                })

            if hasattr(model, "feature_importances_"):
                for feat, importance in zip(features, model.feature_importances_):
                    feature_importance_rows.append({
                        "phase": phase_name,
                        "date": current_date,
                        "estimator_family": estimator_family,
                        "model_id": model_id,
                        "feature": feat,
                        "importance": importance
                    })

    pred_df = (
        pd.DataFrame(all_predictions)
        .sort_values(["model_id", "date", "asset"])
        .reset_index(drop=True)
    )

    fi_df = pd.DataFrame(feature_importance_rows)

    return pred_df, fi_df

# ============================================================
# 7) EVALUATION FUNCTION
# ============================================================

def evaluate_predictions(pred_df):

    global_results = []

    for model_id, group in pred_df.groupby("model_id"):

        estimator_family = group["estimator_family"].iloc[0]
        params = group["params"].iloc[0]

        global_results.append({
            "estimator_family": estimator_family,
            "model_id": model_id,
            "params": params,
            "rmse": rmse(group["realized_return"], group["predicted_return"]),
            "mae": mae(group["realized_return"], group["predicted_return"])
        })

    global_results_df = pd.DataFrame(global_results)

    monthly_ic_rows = []

    for (model_id, dt), group in pred_df.groupby(["model_id", "date"]):

        group = group.dropna(subset=["predicted_return", "realized_return"]).copy()

        if len(group) < 3:
            continue

        estimator_family = group["estimator_family"].iloc[0]

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
            "estimator_family": estimator_family,
            "model_id": model_id,
            "ic": ic,
            "rank_ic": rank_ic
        })

    monthly_ic_df = (
        pd.DataFrame(monthly_ic_rows)
        .sort_values(["model_id", "date"])
        .reset_index(drop=True)
    )

    ic_summary_rows = []

    for model_id, group in monthly_ic_df.groupby("model_id"):

        estimator_family = group["estimator_family"].iloc[0]

        ic_summary_rows.append({
            "estimator_family": estimator_family,
            "model_id": model_id,
            "avg_ic": group["ic"].mean(),
            "median_ic": group["ic"].median(),
            "avg_rank_ic": group["rank_ic"].mean(),
            "median_rank_ic": group["rank_ic"].median()
        })

    ic_summary_df = pd.DataFrame(ic_summary_rows)

    comparison_df = (
        global_results_df
        .merge(
            ic_summary_df[
                [
                    "model_id",
                    "avg_ic",
                    "median_ic",
                    "avg_rank_ic",
                    "median_rank_ic"
                ]
            ],
            on="model_id",
            how="left"
        )
    )

    comparison_df = (
        comparison_df
        .sort_values(
            ["rmse", "mae", "avg_ic"],
            ascending=[True, True, False]
        )
        .reset_index(drop=True)
    )

    comparison_df.index = comparison_df.index + 1
    comparison_df.index.name = "rank"

    return comparison_df, monthly_ic_df

# ============================================================
# 8) VALIDATION GRID SEARCH
# ============================================================

print("\n============================================================")
print("RUNNING VALIDATION GRID SEARCH")
print("============================================================")

validation_pred_df, validation_fi_df = rolling_predictions(
    validation_dates,
    model_specs,
    "validation"
)

validation_comparison_df, validation_monthly_ic_df = evaluate_predictions(
    validation_pred_df
)

print("\n============================================================")
print("VALIDATION MODEL COMPARISON")
print("============================================================")
print(validation_comparison_df.to_string())
print("============================================================")

# ============================================================
# 9) SELECT BEST MODELS ON VALIDATION ONLY
# ============================================================

best_overall_row = validation_comparison_df.iloc[0]

best_overall_family = best_overall_row["estimator_family"]
best_overall_model_id = best_overall_row["model_id"]
best_overall_params = best_overall_row["params"]

best_by_family_df = (
    validation_comparison_df
    .reset_index()
    .sort_values(
        ["estimator_family", "rmse", "mae", "avg_ic"],
        ascending=[True, True, True, False]
    )
    .groupby("estimator_family")
    .head(1)
    .sort_values("rmse", ascending=True)
    .reset_index(drop=True)
)

best_by_family_df.index = best_by_family_df.index + 1
best_by_family_df.index.name = "rank"

print("\n============================================================")
print("BEST CONFIGURATION PER FAMILY - SELECTED ON VALIDATION")
print("============================================================")
print(best_by_family_df.to_string())
print("============================================================")

print("\n============================================================")
print("SELECTED FORECAST MODEL BASED ON VALIDATION")
print("============================================================")
print(f"Best estimator family : {best_overall_family}")
print(f"Best model id         : {best_overall_model_id}")
print(f"Best parameters       : {best_overall_params}")
print("Selection criterion   : lowest validation RMSE, then MAE, then Avg IC")
print("============================================================")

# ============================================================
# 10) BUILD FINAL TEST MODEL SPECS
# ============================================================

best_model_ids = best_by_family_df["model_id"].tolist()

test_model_specs = [
    spec for spec in model_specs
    if spec["model_id"] in best_model_ids
]

# ============================================================
# 11) TEST EVALUATION WITH SELECTED CONFIGURATIONS ONLY
# ============================================================

print("\n============================================================")
print("RUNNING TEST EVALUATION")
print("============================================================")

test_pred_df, test_fi_df = rolling_predictions(
    test_dates,
    test_model_specs,
    "test"
)

test_comparison_df, test_monthly_ic_df = evaluate_predictions(
    test_pred_df
)

print("\n============================================================")
print("TEST MODEL COMPARISON")
print("============================================================")
print(test_comparison_df.to_string())
print("============================================================")

# ============================================================
# 12) FEATURE IMPORTANCE SUMMARY
# ============================================================

feature_importance_df = pd.concat(
    [validation_fi_df, test_fi_df],
    ignore_index=True
)

if not feature_importance_df.empty:

    selected_tree_ids = [
        model_id for model_id in best_model_ids
        if "RandomForest" in model_id or "XGBoost" in model_id
    ]

    feature_importance_summary = (
        feature_importance_df[
            feature_importance_df["model_id"].isin(selected_tree_ids)
        ]
        .groupby(["phase", "estimator_family", "model_id", "feature"])["importance"]
        .mean()
        .reset_index()
        .sort_values(
            ["phase", "estimator_family", "importance"],
            ascending=[True, True, False]
        )
    )

    print("\n============================================================")
    print("FEATURE IMPORTANCE SUMMARY - SELECTED TREE MODELS")
    print("============================================================")
    print(feature_importance_summary.to_string(index=False))
    print("============================================================")

else:
    feature_importance_summary = pd.DataFrame()

# ============================================================
# 13) SAVE OUTPUTS
# ============================================================

validation_pred_df.to_csv("forecast_validation_predictions.csv", index=False)
validation_comparison_df.reset_index().to_csv("forecast_validation_model_comparison.csv", index=False)
validation_monthly_ic_df.to_csv("forecast_validation_monthly_ic.csv", index=False)

best_by_family_df.reset_index().to_csv("forecast_best_by_family_validation.csv", index=False)

test_pred_df.to_csv("forecast_test_predictions.csv", index=False)
test_comparison_df.reset_index().to_csv("forecast_test_model_comparison.csv", index=False)
test_monthly_ic_df.to_csv("forecast_test_monthly_ic.csv", index=False)

selected_model_df = pd.DataFrame({
    "selected_estimator_family": [best_overall_family],
    "selected_model_id": [best_overall_model_id],
    "selected_params": [best_overall_params],
    "selection_criterion": ["lowest validation RMSE, then validation MAE, then validation Avg IC"],
    "target_used": ["absolute next-month return"]
})

selected_model_df.to_csv("selected_forecast_model_validation_based.csv", index=False)

if not feature_importance_df.empty:
    feature_importance_df.to_csv("forecast_feature_importance_by_date.csv", index=False)
    feature_importance_summary.to_csv("forecast_feature_importance_summary.csv", index=False)

print("\nDateien gespeichert:")
print("- forecast_validation_predictions.csv")
print("- forecast_validation_model_comparison.csv")
print("- forecast_validation_monthly_ic.csv")
print("- forecast_best_by_family_validation.csv")
print("- forecast_test_predictions.csv")
print("- forecast_test_model_comparison.csv")
print("- forecast_test_monthly_ic.csv")
print("- selected_forecast_model_validation_based.csv")

if not feature_importance_df.empty:
    print("- forecast_feature_importance_by_date.csv")
    print("- forecast_feature_importance_summary.csv")
