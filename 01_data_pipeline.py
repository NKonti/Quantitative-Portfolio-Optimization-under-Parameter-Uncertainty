# ============================================================
# QUANT PROJECT - DATA PIPELINE + DATA LOADER
# ============================================================
# Ziel:
# 1) Falls der Forecast-Datensatz schon existiert -> direkt laden
# 2) Falls nicht -> Rohdaten laden, Features bauen, CSV speichern
# 3) Danach Exploratory Data Analysis / Korrelationen ausgeben
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")   # robustes Backend, speichert Plots sauber

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 0) PLOT SETTINGS
# ============================================================
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["axes.grid"] = True

# ============================================================
# 1) DATEIPFADE
# ============================================================
DATASET_FILE = "forecast_dataset_monthly.csv"
PRICES_FILE = "prices_monthly.csv"
RETURNS_FILE = "returns_monthly.csv"

# ============================================================
# 2) ETF-UNIVERSUM
# ============================================================
tickers = [
    "SPY",  # US equities
    "EFA",  # Developed markets ex US
    "EEM",  # Emerging markets
    "TLT",  # Long-term Treasuries
    "IEF",  # Intermediate Treasuries
    "GLD",  # Gold
    "VNQ",  # Real Estate
    "LQD",  # Investment grade corporate bonds
    "HYG",  # High yield bonds
    "XLF",  # Financials
    "XLV",  # Healthcare
    "XLE"   # Energy
]

start_date = "2010-01-01"
end_date = None  # None = bis heute

# ============================================================
# 3) HILFSFUNKTIONEN
# ============================================================
def momentum(prices: pd.DataFrame, months: int) -> pd.DataFrame:
    """
    Momentum über 'months':
    P_t / P_{t-months} - 1
    """
    return prices / prices.shift(months) - 1

def rolling_vol(returns: pd.DataFrame, months: int) -> pd.DataFrame:
    """
    Rolling Standardabweichung der Monatsrenditen
    """
    return returns.rolling(months).std()

def drawdown(prices: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    """
    Rollierender Drawdown:
    current price / rolling max - 1
    """
    rolling_max = prices.rolling(window).max()
    return prices / rolling_max - 1

def save_plot(filename: str) -> None:
    """
    Plot sauber speichern und schließen
    """
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

# ============================================================
# 4) DATENSATZ ERZEUGEN ODER LADEN
# ============================================================
if os.path.exists(DATASET_FILE) and os.path.exists(PRICES_FILE) and os.path.exists(RETURNS_FILE):
    print("CSV-Dateien gefunden -> Datensatz wird direkt geladen.\n")

    dataset = pd.read_csv(DATASET_FILE, parse_dates=["date"])
    prices_monthly = pd.read_csv(PRICES_FILE, index_col=0, parse_dates=True)
    returns_monthly = pd.read_csv(RETURNS_FILE, index_col=0, parse_dates=True)

else:
    print("Keine vorhandenen CSV-Dateien gefunden -> Datensatz wird neu erzeugt.\n")

    # ========================================================
    # 4A) TÄGLICHE PREISE LADEN
    # ========================================================
    prices_daily = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )["Close"]

    prices_daily = prices_daily.dropna(how="all")

    print("Daily price data shape:", prices_daily.shape)
    print(prices_daily.tail())

    # ========================================================
    # 4B) MONATSULTIMO PREISE
    # ========================================================
    prices_monthly = prices_daily.resample("M").last()

    print("\nMonthly price data shape:", prices_monthly.shape)
    print(prices_monthly.tail())

    # ========================================================
    # 4C) MONATSRENDITEN
    # ========================================================
    returns_monthly = prices_monthly.pct_change()

    print("\nMonthly returns shape:", returns_monthly.shape)
    print(returns_monthly.tail())

    # ========================================================
    # 4D) FEATURES
    # ========================================================
    mom_1 = momentum(prices_monthly, 1)
    mom_3 = momentum(prices_monthly, 3)
    mom_6 = momentum(prices_monthly, 6)
    mom_12 = momentum(prices_monthly, 12)

    vol_3 = rolling_vol(returns_monthly, 3)
    vol_6 = rolling_vol(returns_monthly, 6)
    vol_12 = rolling_vol(returns_monthly, 12)

    dd_12 = drawdown(prices_monthly, 12)

    # ========================================================
    # 4E) ZIELVARIABLE
    #     nächste Monatsrendite
    # ========================================================
    target = returns_monthly.shift(-1)

    # ========================================================
    # 4F) LONG-FORMAT DATENSATZ BAUEN
    # ========================================================
    dataset_list = []

    for ticker in tickers:
        df = pd.DataFrame({
            "date": prices_monthly.index,
            "asset": ticker,
            "price": prices_monthly[ticker],
            "return_t": returns_monthly[ticker],
            "mom_1": mom_1[ticker],
            "mom_3": mom_3[ticker],
            "mom_6": mom_6[ticker],
            "mom_12": mom_12[ticker],
            "vol_3": vol_3[ticker],
            "vol_6": vol_6[ticker],
            "vol_12": vol_12[ticker],
            "drawdown_12": dd_12[ticker],
            "target_return_next_month": target[ticker]
        })
        dataset_list.append(df)

    dataset = pd.concat(dataset_list, ignore_index=True)

    # Fehlende Werte entfernen
    dataset = dataset.dropna().reset_index(drop=True)

    print("\nFinal dataset shape:", dataset.shape)
    print(dataset.head())

    # ========================================================
    # 4G) CSV SPEICHERN
    # ========================================================
    dataset.to_csv(DATASET_FILE, index=False)
    prices_monthly.to_csv(PRICES_FILE)
    returns_monthly.to_csv(RETURNS_FILE)

    print("\nCSV-Dateien gespeichert:")
    print(f"- {DATASET_FILE}")
    print(f"- {PRICES_FILE}")
    print(f"- {RETURNS_FILE}")

# ============================================================
# 5) BASISPRÜFUNG DES GELADENEN/ERZEUGTEN DATENSATZES
# ============================================================
print("\n================ DATA CHECK ================\n")
print("Final dataset shape:", dataset.shape)
print(dataset.head())

print("\nDate range in final dataset:")
print(dataset["date"].min(), "to", dataset["date"].max())

print("\nNumber of observations per asset:")
print(dataset["asset"].value_counts())

print("\nSummary statistics:")
print(dataset.describe(include="all").T)

# ============================================================
# 6) FEATURE-LISTE DEFINIEREN
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

# ============================================================
# 7) FEATURE-FEATURE KORRELATIONEN
# ============================================================
corr_features_pearson = dataset[features].corr(method="pearson")
corr_features_spearman = dataset[features].corr(method="spearman")

print("\nFeature-Feature Correlation Matrix (Pearson):")
print(corr_features_pearson)

print("\nFeature-Feature Correlation Matrix (Spearman):")
print(corr_features_spearman)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_features_pearson, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Feature-Feature Correlation Matrix (Pearson)")
save_plot("feature_feature_corr_pearson.png")

plt.figure(figsize=(10, 8))
sns.heatmap(corr_features_spearman, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Feature-Feature Correlation Matrix (Spearman)")
save_plot("feature_feature_corr_spearman.png")

# ============================================================
# 8) FEATURE-TARGET KORRELATIONEN
# ============================================================
corr_target_pearson = dataset[features + [target_col]].corr(method="pearson")[target_col].drop(target_col)
corr_target_pearson = corr_target_pearson.sort_values(ascending=False)

corr_target_spearman = dataset[features + [target_col]].corr(method="spearman")[target_col].drop(target_col)
corr_target_spearman = corr_target_spearman.sort_values(ascending=False)

print("\nFeature-Target Correlation (Pearson):")
print(corr_target_pearson)

print("\nFeature-Target Correlation (Spearman):")
print(corr_target_spearman)

plt.figure(figsize=(10, 5))
corr_target_pearson.plot(kind="bar")
plt.title("Feature vs Target Correlation (Pearson)")
plt.ylabel("Correlation")
save_plot("feature_target_corr_pearson.png")

plt.figure(figsize=(10, 5))
corr_target_spearman.plot(kind="bar")
plt.title("Feature vs Target Correlation (Spearman)")
plt.ylabel("Correlation")
save_plot("feature_target_corr_spearman.png")

# ============================================================
# 9) ZIELVARIABLE VISUALISIEREN
# ============================================================
plt.figure(figsize=(10, 5))
sns.histplot(dataset[target_col], bins=50, kde=True)
plt.title("Distribution of Target Variable: Next Month Return")
plt.xlabel("Next Month Return")
save_plot("target_distribution.png")

# ============================================================
# 10) FEATURE VS TARGET SCATTERPLOTS
# ============================================================
plot_sample = dataset.sample(min(2000, len(dataset)), random_state=42)

for feat in features:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=plot_sample, x=feat, y=target_col, alpha=0.5)
    plt.title(f"{feat} vs {target_col}")
    save_plot(f"scatter_{feat}_vs_target.png")

# ============================================================
# 11) FEATURE-TARGET KORRELATION PRO ASSET
# ============================================================
asset_corrs = []

for asset in tickers:
    asset_df = dataset[dataset["asset"] == asset]
    corr_series = asset_df[features + [target_col]].corr(method="pearson")[target_col].drop(target_col)
    corr_series.name = asset
    asset_corrs.append(corr_series)

asset_corr_df = pd.DataFrame(asset_corrs)

print("\nFeature-Target Correlation by Asset (Pearson):")
print(asset_corr_df)

plt.figure(figsize=(12, 6))
sns.heatmap(asset_corr_df, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Feature-Target Correlation by Asset (Pearson)")
save_plot("feature_target_corr_by_asset.png")

# ============================================================
# 12) OPTIONAL: DATENSATZ-VORSCHAU EXPORTIEREN
# ============================================================
dataset.head(20).to_csv("dataset_preview.csv", index=False)

print("\nPlots gespeichert:")
#print("- feature_feature_corr_pearson.png")
#print("- feature_feature_corr_spearman.png")
#print("- feature_target_corr_pearson.png")
#print("- feature_target_corr_spearman.png")
#print("- target_distribution.png")
#print("- scatter_[feature]_vs_target.png")
#print("- feature_target_corr_by_asset.png")

print("\nZusätzliche Datei gespeichert:")
print("- dataset_preview.csv")

print("\nFertig. Der Datensatz wird künftig direkt aus CSV geladen, solange du nichts an Features, Zeitraum oder ETFs änderst.")