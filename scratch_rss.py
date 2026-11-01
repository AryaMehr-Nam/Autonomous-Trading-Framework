import feedparser
import json

def fetch_rss(url):
    feed = feedparser.parse(url)
    print(f"Feed: {url}")
    print(f"Entries found: {len(feed.entries)}")
    for i, entry in enumerate(feed.entries[:3]):
        print(f"{i+1}. {entry.title}")
        print(f"   {entry.published}")

fetch_rss("https://feeds.finance.yahoo.com/rss/2.0/headline?s=GC=F")
fetch_rss("https://www.forexlive.com/feed/news")
