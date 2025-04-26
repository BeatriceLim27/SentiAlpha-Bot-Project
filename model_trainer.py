import pandas as pd
import numpy as np
import xgboost as xgb
import os
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import classification_report
from features import build_features

# === Load Funding Rate and Price Data ===
fr = pd.read_csv(os.path.join("timeseries", "binance_btcusdt_fr.csv"), parse_dates=["timestamp"])
px = pd.read_csv(os.path.join("timeseries", "binance_btcusdt_price.csv"), parse_dates=["timestamp"])

# Ensure proper datetime conversion and set index
fr["timestamp"] = pd.to_datetime(fr["timestamp"], utc=True, errors="coerce")
px["timestamp"] = pd.to_datetime(px["timestamp"], utc=True, errors="coerce")
fr.dropna(subset=["timestamp"], inplace=True)
px.dropna(subset=["timestamp"], inplace=True)
fr.set_index("timestamp", inplace=True)
px.set_index("timestamp", inplace=True)

# === Merge using merge_asof (tolerance 30 minutes) ===
df = pd.merge_asof(
    fr.sort_index(),
    px.rename(columns={"close": "price"}).sort_index(),
    left_index=True,
    right_index=True,
    direction="nearest",
    tolerance=pd.Timedelta("30min")
).dropna()

print("[DEBUG] Raw merged shape:", df.shape)

# === Feature Engineering ===
df = build_features(df)
print("[DEBUG] Feature output shape before dropna:", df.shape)
df = df.dropna()  # Drop rows with NaNs introduced by rolling windows, shifts, etc.
print("[DEBUG] Feature output shape after dropna:", df.shape)
os.makedirs("timeseries", exist_ok=True)
df.to_csv(os.path.join("timeseries", "feature_output.csv"))
print("[DEBUG] Feature output saved. Shape:", df.shape)

# === Define Features and Target ===
features = [
    "fundingRate", "fr_delta", "fr_rolling_mean_3h",
    "price_return_1h", "price_volatility_3h",
    "fr_zscore", "fr_momentum", "price_bandwidth",
    "avg_sentiment", "is_bullish_sentiment",
    "price_deviation", "price_ma_3h"
]
target = "fr_sign_flip_next_3h"

X = df[features]
y = df[target].loc[X.index]

print("[DEBUG] X shape:", X.shape, "y shape:", y.shape)

# === Handle Class Imbalance ===
pos = y.sum()
neg = len(y) - pos
scale_pos_weight = neg / pos if pos > 0 else 1
print("[DEBUG] scale_pos_weight:", scale_pos_weight)

if len(X) == 0:
    raise ValueError("No data left after feature engineering. Consider extending the date range or adjusting rolling windows.")

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# === Hyperparameter Tuning with TimeSeriesSplit ===
tscv = TimeSeriesSplit(n_splits=5)
param_grid = {
    "n_estimators": [100, 150],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.05, 0.1],
    "scale_pos_weight": [scale_pos_weight]
}
grid = GridSearchCV(
    estimator=xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
    param_grid=param_grid,
    cv=tscv,
    scoring="f1_macro",
    n_jobs=-1
)
grid.fit(X_train, y_train)
print("Best parameters found:", grid.best_params_)

model = grid.best_estimator_

# === Evaluate Model ===
y_pred = model.predict(X_test)
print("\n[Classification Report]")
print(classification_report(y_test, y_pred))

# === Save the Trained Model ===
os.makedirs("models", exist_ok=True)
model.save_model(os.path.join("models", "fr_flip_xgb.json"))
print("[✅] Model saved to models/fr_flip_xgb.json")
