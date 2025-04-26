
# twitter_sentiment_json.py

import tweepy
import pandas as pd
from datetime import datetime, timezone
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import os
import time
import random

# Download VADER if not already
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# === 🔐 TWITTER AUTH ===
BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
if BEARER_TOKEN is None:
    raise ValueError("Bearer token not found in environment variables.")
client = tweepy.Client(bearer_token=BEARER_TOKEN, wait_on_rate_limit=True)

def fetch_twitter_sentiment(query="funding rate crypto -is:retweet lang:en", max_tweets=20, batch_size=10):
    try:
        data = []
        next_token = None
        retrieved_tweets = 0

        while retrieved_tweets < max_tweets:
            try:
                tweets = client.search_recent_tweets(
                    query=query,
                    tweet_fields=["created_at"],
                    max_results=batch_size,
                    next_token=next_token
                )

                if not tweets.data:
                    break

                for tweet in tweets.data:
                    score = sia.polarity_scores(tweet.text)
                    data.append({
                        "text": tweet.text,
                        "compound": score["compound"],
                        "pos": score["pos"],
                        "neu": score["neu"],
                        "neg": score["neg"]
                    })
                    retrieved_tweets += 1
                    if retrieved_tweets >= max_tweets:
                        break

                next_token = tweets.meta.get("next_token", None)
                if not next_token:
                    break

                time.sleep(2)  # Shorter wait for dev

            except tweepy.TooManyRequests:
                print("[❌] Rate limit hit — sleeping for 5 min.")
                time.sleep(300)

        if not data:
            raise ValueError("No tweets retrieved.")

        df = pd.DataFrame(data)

    except Exception as e:
        print("[⚠️] API error — using fallback mock sentiment.")
        # Create random mock data
        df = pd.DataFrame([{
            "compound": random.uniform(-0.1, 0.1),
            "pos": random.uniform(0.1, 0.3),
            "neu": random.uniform(0.6, 0.8),
            "neg": random.uniform(0.05, 0.2)
        }])

    df["timestamp"] = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    sentiment_summary = df.groupby("timestamp")[["compound", "pos", "neu", "neg"]].mean().reset_index()

    os.makedirs("timeseries", exist_ok=True)
    sentiment_summary.to_csv("timeseries/twitter_sentiment_hourly.csv", index=False)
    print("[✅] Twitter sentiment saved:", sentiment_summary)

if __name__ == "__main__":
    fetch_twitter_sentiment()

