import requests
import pandas as pd
import os
import time

# === Binance Funding Rates ===
def fetch_binance_funding(symbol="BTCUSDT", start_date="2024-03-30", end_date="2025-03-30"):
    print("[INFO] Fetching Binance funding rates...")
    start_time = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)
    end_time = int(pd.Timestamp(end_date, tz="UTC").timestamp() * 1000)
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    all_data = []
    while start_time < end_time:
        params = {
            "symbol": symbol,
            "startTime": start_time,
            "endTime": end_time,
            "limit": 1000
        }
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print("[ERROR] Binance funding fetch failed.")
            break
        data = response.json()
        if not data:
            break
        all_data.extend(data)
        last_time = data[-1]["fundingTime"]
        start_time = last_time + 1
        time.sleep(0.2)  # Rate limit handling
    df = pd.DataFrame(all_data)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
        df["fundingRate"] = df["fundingRate"].astype(float)
        df = df[["timestamp", "fundingRate"]].set_index("timestamp")
    return df

# === Binance Price Data (1m candles) ===
def fetch_binance_price(symbol="BTCUSDT", interval="1m", start_date="2024-03-30", end_date="2025-03-30"):
    print("[INFO] Fetching Binance 1m price data...")
    start_time = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)
    end_time = int(pd.Timestamp(end_date, tz="UTC").timestamp() * 1000)
    url = "https://fapi.binance.com/fapi/v1/klines"
    all_data = []
    limit = 1500
    while start_time < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time,
            "limit": limit
        }
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print("[ERROR] Binance price fetch failed.")
            break
        data = response.json()
        if not data:
            break
        all_data.extend(data)
        last_time = data[-1][0]
        start_time = last_time + 1
        time.sleep(0.2)
    df = pd.DataFrame(all_data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close"] = df["close"].astype(float)
        df = df[["timestamp", "close"]].set_index("timestamp")
    return df

# === Save All Data ===
def save_all_data():
    os.makedirs("timeseries", exist_ok=True)

    binance_fr = fetch_binance_funding()
    binance_fr.to_csv("timeseries/binance_btcusdt_fr.csv")
    print(f"[✅] Saved Binance funding rates: {binance_fr.shape[0]} rows")

    binance_px = fetch_binance_price()
    binance_px.to_csv("timeseries/binance_btcusdt_price.csv")
    print(f"[✅] Saved Binance price data: {binance_px.shape[0]} rows")

    return binance_fr, binance_px

if __name__ == "__main__":
    save_all_data()
