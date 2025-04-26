# reddit_sentiment_json.py

import requests
import pandas as pd
from datetime import datetime, timezone
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import os

# Download VADER lexicon for sentiment scoring
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

def fetch_reddit_sentiment(subreddit="CryptoCurrency", limit=100):
    headers = {"User-Agent": "Mozilla/5.0 (funding-rate-bot)"}
    url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={limit}"

    # Fetch top posts
    res = requests.get(url, headers=headers)
    if res.status_code != 200:
        print("[❌] Reddit JSON error:", res.text)
        return

    posts = res.json().get("data", {}).get("children", [])
    if not posts:
        print("[⚠️] No posts returned from Reddit JSON.")
        return

    data = []
    for post in posts:
        title = post.get("data", {}).get("title", "")
        if not title:
            continue
        print("Post:", title)  # optional: view scraped titles
        score = sia.polarity_scores(title)
        data.append({
            "title": title,
            "compound": score["compound"],
            "pos": score["pos"],
            "neu": score["neu"],
            "neg": score["neg"]
        })

    df = pd.DataFrame(data)
    df["timestamp"] = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    # Only average the numeric sentiment scores
    numeric_cols = ["compound", "pos", "neu", "neg"]
    sentiment_summary = df.groupby("timestamp")[numeric_cols].mean().reset_index()

    # Save to CSV
    os.makedirs("timeseries", exist_ok=True)
    sentiment_summary.to_csv("timeseries/reddit_sentiment_hourly.csv", index=False)
    print("[✅] Reddit JSON sentiment saved:", sentiment_summary)

if __name__ == "__main__":
    fetch_reddit_sentiment()
