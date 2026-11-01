"""
TradeAgent Economic Calendar.

Fetches upcoming economic events from Finnhub API.
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Optional

import json
import urllib.request
from dateutil import parser as date_parser

from config.settings import settings, trading_params
from core.logger import get_logger
from core.models import EconomicEvent, EventImpact

logger = get_logger("economic_calendar")


class EconomicCalendar:
    """Manages upcoming macroeconomic events."""
    
    def __init__(self):
        self.config = trading_params.fundamentals
        self.api_key = settings.finnhub_api_key
        self.client = None
        self.events_cache: List[EconomicEvent] = []
        self.last_fetch: Optional[datetime] = None

    def connect(self) -> bool:
        # We use the free ForexFactory JSON API now, no API key needed.
        self.api_url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        return True

    async def fetch_events(self, from_date: datetime, to_date: datetime) -> List[EconomicEvent]:
        """Fetch calendar events from ForexFactory JSON feed."""
        try:
            def _fetch():
                req = urllib.request.Request(self.api_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response:
                    return json.loads(response.read().decode())
                
            data = await asyncio.to_thread(_fetch)
            
            events = []
            for item in data:
                impact_str = item.get("impact", "").lower()
                if "high" in impact_str: impact = EventImpact.HIGH
                elif "medium" in impact_str: impact = EventImpact.MEDIUM
                else: impact = EventImpact.LOW
                
                try:
                    # e.g. "2026-07-20T08:30:00-04:00"
                    dt = date_parser.parse(item.get("date", "")).astimezone(tz=None)
                    # Convert to UTC naive datetime for consistent comparison with datetime.utcnow()
                    from datetime import timezone
                    dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
                except Exception as e:
                    dt = datetime.utcnow()
                    
                events.append(EconomicEvent(
                    timestamp=dt,
                    country=item.get("country", ""),
                    event_name=item.get("title", ""),
                    impact=impact,
                    actual=None,
                    forecast=None,
                    previous=None
                ))
                
            self.events_cache = events
            self.last_fetch = datetime.utcnow()
            return events
            
        except Exception as e:
            logger.error(f"Failed to fetch ForexFactory calendar: {e}")
            return []

    def get_upcoming_high_impact(self, hours_ahead: int = 24) -> List[EconomicEvent]:
        """Get high impact events happening soon."""
        now = datetime.utcnow()
        horizon = now + timedelta(hours=hours_ahead)
        
        upcoming = [
            e for e in self.events_cache
            if e.impact == EventImpact.HIGH and now <= e.timestamp <= horizon
        ]
        return sorted(upcoming, key=lambda x: x.timestamp)

    def is_safe_to_trade(self, symbol: str) -> bool:
        """Check if we are safely outside the blackout window for high impact events."""
        if not self.config.get("enabled", True):
            return True
            
        now = datetime.utcnow()
        pause_before = timedelta(minutes=self.config.get("event_pause_minutes_before", 30))
        pause_after = timedelta(minutes=self.config.get("event_pause_minutes_after", 15))
        
        for event in self.events_cache:
            if event.impact != EventImpact.HIGH:
                continue
                
            # Basic currency matching (e.g. US event affects USD pairs)
            # Finnhub uses 2-letter country codes (US, EU, GB, JP, AU, CA, CH, NZ)
            country_currency_map = {
                "US": "USD", "EU": "EUR", "GB": "GBP", "JP": "JPY", 
                "AU": "AUD", "CA": "CAD", "CH": "CHF", "NZ": "NZD"
            }
            
            event_ccy = country_currency_map.get(event.country, "")
            if event_ccy and event_ccy not in symbol:
                continue # Event doesn't affect this pair
                
            window_start = event.timestamp - pause_before
            window_end = event.timestamp + pause_after
            
            if window_start <= now <= window_end:
                logger.info(f"Trading paused for {symbol} due to {event.event_name} at {event.timestamp}")
                return False
                
        return True
