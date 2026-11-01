import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sentiment.news_fetcher import NewsFetcher
from sentiment.sentiment_analyzer import SentimentAnalyzer

async def main():
    print("Testing NewsFetcher + SentimentAnalyzer Pipeline...")
    
    fetcher = NewsFetcher()
    fetcher.connect()
    
    analyzer = SentimentAnalyzer()
    loaded = analyzer.load_model()
    if not loaded:
        print("FinBERT model failed to load. Do you have torch/transformers installed?")
        return
        
    print("\nFetching live news...")
    news_items = await fetcher.fetch_forex_news()
    print(f"Fetched {len(news_items)} news articles.")
    
    if not news_items:
        return
        
    headlines = [n["headline"] for n in news_items[:10]] # Analyze top 10
    
    print("\nAnalyzing sentiment (this may take a moment)...")
    results = await analyzer.analyze_batch(headlines)
    
    for i, res in enumerate(results):
        print(f"[{res.label.name}] ({res.confidence:.2f}) {res.text}")
        
    agg_score = analyzer.aggregate_score(results)
    
    print(f"\n--- OVERALL MARKET SENTIMENT SCORE ---")
    print(f"Score: {agg_score:+.2f} (-1.0 to +1.0)")
    
    if agg_score > 0.3:
        print("Conclusion: BULLISH")
    elif agg_score < -0.3:
        print("Conclusion: BEARISH")
    else:
        print("Conclusion: NEUTRAL")

if __name__ == "__main__":
    asyncio.run(main())
