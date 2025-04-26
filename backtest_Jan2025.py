import pandas as pd
import numpy as np
import xgboost as xgb
import os
import matplotlib.pyplot as plt

class StrategyRunner:
    def __init__(self, model_path, feature_path, confidence_threshold=0.7, respect_sentiment=True, vol_exit_multiplier=1.2):
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        self.confidence_threshold = confidence_threshold
        self.respect_sentiment = respect_sentiment
        self.vol_exit_multiplier = vol_exit_multiplier

        # Load feature data
        self.df = pd.read_csv(feature_path, parse_dates=["timestamp"])
        self.df.set_index("timestamp", inplace=True)
        self.df = self.df[(self.df.index >= "2025-01-01") & (self.df.index < "2025-02-01")].copy()
        print(f"[DEBUG] Filtered row count (Jan 2025): {len(self.df)}")

        self.df[["position", "entry_price", "pnl"]] = None
        self.df["trade_type"] = None  # LONG or SHORT

    def run(self):
        position = None
        entry_price = None
        entry_index = None
        position_type = None
        trade_log = []
        prev_price = None
        short_count = 0
        long_count = 0

        for i in range(len(self.df)):
            row = self.df.iloc[i]
            timestamp = self.df.index[i]
            current_price = row["price"]
            avg_sentiment = row.get("avg_sentiment", 0)
            bullish_sentiment = row.get("is_bullish_sentiment", 0)
            current_vol = row.get("price_volatility_3h", 0)

            if prev_price is not None:
                price_momentum = current_price - prev_price
            else:
                price_momentum = 0
            prev_price = current_price

            features = row[[
                "fundingRate", "fr_delta", "fr_rolling_mean_3h",
                "price_return_1h", "price_volatility_3h",
                "fr_zscore", "fr_momentum", "price_bandwidth",
                "avg_sentiment", "is_bullish_sentiment",
                "price_deviation", "price_ma_3h"
            ]].values.reshape(1, -1)

            try:
                proba = self.model.predict_proba(features)[0]
                predicted_label = self.model.predict(features)[0]
            except Exception as e:
                print(f"[{timestamp}] ⚠️ Prediction error: {e}")
                continue

            # === Entry Logic ===
            if position is None and proba[predicted_label] >= self.confidence_threshold:
                if predicted_label == 1:  # SHORT
                    if self.respect_sentiment and avg_sentiment > 0.4:
                        trade_log.append(f"[{timestamp}] Skipped SHORT due to strong bullish sentiment (avg_sent={avg_sentiment:.2f})")
                    elif price_momentum < 0:
                        position = -1
                        position_type = "SHORT"
                        short_count += 1
                    elif proba[1] >= 0.8:
                        position = -1
                        position_type = "SHORT"
                        short_count += 1
                        trade_log.append(f"[{timestamp}] SHORT forced by high model confidence (proba={proba[1]:.2f})")
                    else:
                        trade_log.append(f"[{timestamp}] Skipped SHORT due to weak momentum (mom={price_momentum:.2f})")

                elif predicted_label == 0:  # LONG
                    if self.respect_sentiment and avg_sentiment < -0.2:
                        trade_log.append(f"[{timestamp}] Skipped LONG due to strong bearish sentiment (avg_sent={avg_sentiment:.2f})")
                    elif price_momentum > 0:
                        position = 1
                        position_type = "LONG"
                        long_count += 1
                    else:
                        trade_log.append(f"[{timestamp}] Skipped LONG due to weak momentum (mom={price_momentum:.2f})")

                if position is not None:
                    entry_price = current_price
                    entry_index = i
                    self.df.at[timestamp, "position"] = position
                    self.df.at[timestamp, "entry_price"] = entry_price
                    self.df.at[timestamp, "trade_type"] = position_type
                    trade_log.append(f"[{timestamp}] {position_type} entered @ {entry_price:.2f}")

            # === Exit Logic ===
            elif position is not None:
                elapsed = i - entry_index
                price_move = abs(current_price - entry_price)
                if elapsed >= 3 or (current_vol > 0 and price_move >= self.vol_exit_multiplier * current_vol):
                    exit_price = current_price
                    pnl = (entry_price - exit_price) if position == -1 else (exit_price - entry_price)
                    self.df.at[timestamp, "pnl"] = pnl
                    trade_log.append(f"[{timestamp}] {position_type} closed @ {exit_price:.2f} | PnL: {pnl:.2f}")
                    position = None
                    entry_price = None
                    entry_index = None
                    position_type = None

        self.df["cumulative_pnl"] = self.df["pnl"].fillna(0).cumsum()
        os.makedirs("backtests", exist_ok=True)
        out_file = os.path.join("backtests", "backtest_results_Jan2025.csv")
        self.df.to_csv(out_file)

        print("\n--- Trade Log (Jan 2025) ---")
        for log in trade_log:
            print(log)
        total_trades = self.df["pnl"].notna().sum()
        final_pnl = self.df["cumulative_pnl"].iloc[-1] if not self.df["cumulative_pnl"].empty else 0
        print("\n--- Summary ---")
        print(f"Total Trades: {total_trades}")
        print(f"LONG Trades: {long_count}")
        print(f"SHORT Trades: {short_count}")
        print(f"[✅] Strategy PnL (Jan 2025): {final_pnl:.2f} USDT")

        plt.figure(figsize=(12, 6))
        plt.plot(self.df["cumulative_pnl"], linewidth=2)
        plt.title("Cumulative PnL (Jan 2025)")
        plt.xlabel("Time")
        plt.ylabel("Cumulative PnL")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    runner = StrategyRunner(
        model_path=os.path.join("models", "fr_flip_xgb.json"),
        feature_path=os.path.join("timeseries", "feature_output.csv"),
        confidence_threshold=0.35,
        respect_sentiment=True,
        vol_exit_multiplier=1.2
    )
    runner.run()

