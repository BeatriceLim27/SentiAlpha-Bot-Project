import pandas as pd
import numpy as np
import os

def build_features(df):
    df = df.copy()
    
    # Ensure the index is datetime and timezone-aware (UTC)
    df.index = pd.to_datetime(df.index, errors="coerce", utc=True)
    df = df.sort_index()
    
    # --- FUNDING RATE FEATURES ---
    df["fr_delta"] = df["fundingRate"].diff()
    df["fr_rolling_mean_3h"] = df["fundingRate"].rolling(window=3, min_periods=1).mean()
    # 12-hour rolling mean for divergence calculation
    df["fr_rolling_mean_12h"] = df["fundingRate"].rolling(window=12, min_periods=1).mean()
    df["fr_divergence"] = df["fundingRate"] - df["fr_rolling_mean_12h"]
    df["fr_zscore"] = (df["fundingRate"] - df["fundingRate"].rolling(window=24, min_periods=1).mean()) / df["fundingRate"].rolling(window=24, min_periods=1).std()
    df["fr_momentum"] = df["fundingRate"] - df["fundingRate"].shift(3)
    
    # --- PRICE FEATURES ---
    df["price_return_1h"] = df["price"].pct_change()
    df["price_volatility_3h"] = df["price"].rolling(window=3, min_periods=1).std()
    df["price_ma_3h"] = df["price"].rolling(window=3, min_periods=1).mean()
    df["price_deviation"] = df["price"] - df["price_ma_3h"]
    df["price_bandwidth"] = (df["price"].rolling(window=3, min_periods=1).max() - df["price"].rolling(window=3, min_periods=1).min()) / df["price"]
    df["price_momentum"] = df["price"] - df["price"].shift(3)
    # Price range as a proxy for ATR-like volatility
    df["price_range"] = df["price"].rolling(window=3, min_periods=1).max() - df["price"].rolling(window=3, min_periods=1).min()
    
    # --- TARGET LABEL: Funding Rate Flip in Next 3 Intervals ---
    df["fr_sign"] = np.sign(df["fundingRate"])
    df["fr_sign_shifted"] = df["fr_sign"].shift(-3)
    df["fr_sign_flip_next_3h"] = (df["fr_sign"] != df["fr_sign_shifted"]).astype(int)
    
    # --- SENTIMENT FEATURES ---
    # Use a relative path for sentiment CSV files
    base_path = os.path.join(os.getcwd(), "timeseries")
    for source in ["reddit", "twitter"]:
        path = os.path.join(base_path, f"{source}_sentiment_hourly.csv")
        if os.path.exists(path):
            sent_df = pd.read_csv(path, parse_dates=["timestamp"])
            # Ensure sentiment timestamps are in UTC
            sent_df["timestamp"] = pd.to_datetime(sent_df["timestamp"], errors="coerce", utc=True)
            sent_df = sent_df.dropna(subset=["timestamp"]).sort_values("timestamp")
            # Merge sentiment with tolerance of 1 hour
            df = pd.merge_asof(
                df.sort_index(),
                sent_df.set_index("timestamp").sort_index()[["compound"]],
                left_index=True,
                right_index=True,
                direction="nearest",
                tolerance=pd.Timedelta("1h")
            )
            df.rename(columns={"compound": f"compound_{source}"}, inplace=True)
    
    # Ensure missing sentiment columns are filled with 0
    for col in ["compound_reddit", "compound_twitter"]:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = df[col].fillna(0)
    
    df["avg_sentiment"] = df[["compound_reddit", "compound_twitter"]].mean(axis=1)
    # Sentiment delta: change in average sentiment over 3 intervals
    df["sentiment_delta"] = df["avg_sentiment"] - df["avg_sentiment"].shift(3)
    df["is_bullish_sentiment"] = (df["avg_sentiment"] > 0.1).astype(int)
    
    # Optionally, fill any remaining NaNs with forward fill to preserve as many rows as possible
    df = df.fillna(method="ffill")
    
    return df
