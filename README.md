# 🚀 SentiAlpha-Bot Project
- SentiAlpha-Bot is an AI-driven trading bot designed to predict cryptocurrency funding rate flips (short/long) by merging market data with real-time sentiment analysis from Twitter and Reddit.  
- The bot leverages advanced machine learning (XGBoost) and dynamic feature engineering to simulate trading strategies with high precision and profitability.  
- Built with Python, Binance API and NLP sentiment models (VADER), it achieves consistent monthly PnL and high trade win rates across multiple months of backtesting.

---

### 📊 Key Performance Metrics
- XGBoost classifier achieved 83% precision on non-flip periods.
- Simulated trading strategy generated consistent positive monthly PnL (Jan - Mar 2025).
- Achieved 65 - 75% trade win rates across volatile and stable market conditions.

---

### 📂 What's Inside
- `data_fetcher.py`: Fetch funding rates and prices data from Binance
- `twitter_sentiment_json.py`, `reddit_sentiment_json.py`: Scrape and score market sentiment
- `features.py`: Feature engineering
- `model_trainer.py`: Train and tune XGBoost model
- `strategy_runner.py`: Trade simulation and backtest trading strategy on Mar 2025 BTCUSDT data
- `backtest_Jan2025.py`, `backtest_Feb2025.py`: Month-specific backtesting PnL evaluations
