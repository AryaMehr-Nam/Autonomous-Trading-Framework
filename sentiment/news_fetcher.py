"""
TradeAgent News Fetcher.

Ingests headlines from free RSS feeds using feedparser.
"""

import asyncio
from datetime import datetime
from typing import List, Dict

try:
    import feedparser
    HAS_FEEDPARSER = True
except ImportError:
    HAS_FEEDPARSER = False

from core.logger import get_logger

logger = get_logger("news_fetcher")


class NewsFetcher:
    def __init__(self):
        # High quality free RSS feeds for Forex & Gold
        self.rss_feeds = {
            "forexlive": "https://www.forexlive.com/feed/news",
            "yahoo_gold": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=GC=F",
            "yahoo_forex": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=EURUSD=X"
        }

    def connect(self) -> bool:
        if not HAS_FEEDPARSER:
            logger.error("feedparser is not installed. Run: pip install feedparser")
            return False
        logger.info("NewsFetcher configured with RSS feeds.")
        return True

    async def fetch_forex_news(self) -> List[Dict]:
        """Fetch general forex and gold news from RSS feeds."""
        if not HAS_FEEDPARSER:
            return []
            
        news = []
        
        for source, url in self.rss_feeds.items():
            try:
                def _fetch():
                    return feedparser.parse(url)
                
                feed = await asyncio.to_thread(_fetch)
                
                for item in feed.entries[:10]:  # Top 10 from each feed
                    news.append({
                        "id": item.get("link", ""),
                        "headline": item.get("title", ""),
                        "summary": item.get("summary", ""),
                        "source": source,
                        "timestamp": datetime.utcnow() # Approximation for headline
                    })
            except Exception as e:
                logger.error(f"RSS fetch error for {source}: {e}")
                
        # Deduplicate
        seen_headlines = set()
        dedup = []
        for n in news:
            if n["headline"] and n["headline"] not in seen_headlines:
                seen_headlines.add(n["headline"])
                dedup.append(n)
                
        # Sort newest first (using our utcnow approximation, so order is rough)
        return sorted(dedup, key=lambda x: x["timestamp"], reverse=True)
