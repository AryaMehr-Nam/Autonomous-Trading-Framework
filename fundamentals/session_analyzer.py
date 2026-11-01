"""
TradeAgent Session Analyzer.

Determines the current Forex trading session (Sydney, Tokyo, London, NY)
and filters signals based on session volume/volatility expectations.
"""

from datetime import datetime, time
import pytz

from core.models import ForexSession


class SessionAnalyzer:
    """Tracks global Forex market sessions."""
    
    def __init__(self):
        self.utc = pytz.UTC
        
        # Session UTC hours (Approximate, not accounting for DST shifts dynamically here for brevity)
        self.sessions = {
            ForexSession.SYDNEY: (time(22, 0), time(7, 0)),
            ForexSession.TOKYO: (time(0, 0), time(9, 0)),
            ForexSession.LONDON: (time(8, 0), time(17, 0)),
            ForexSession.NEW_YORK: (time(13, 0), time(22, 0))
        }

    def get_current_session(self, dt: datetime = None) -> ForexSession:
        if dt is None:
            dt = datetime.utcnow().replace(tzinfo=self.utc)
        elif dt.tzinfo is None:
            dt = dt.replace(tzinfo=self.utc)
            
        # Weekend check (Friday 22:00 to Sunday 22:00 UTC)
        if dt.weekday() == 5 or (dt.weekday() == 4 and dt.hour >= 22) or (dt.weekday() == 6 and dt.hour < 22):
            return ForexSession.CLOSED
            
        current_time = dt.time()
        
        is_london = self._is_in_range(current_time, *self.sessions[ForexSession.LONDON])
        is_ny = self._is_in_range(current_time, *self.sessions[ForexSession.NEW_YORK])
        
        if is_london and is_ny:
            return ForexSession.OVERLAP_LONDON_NY
        elif is_ny:
            return ForexSession.NEW_YORK
        elif is_london:
            return ForexSession.LONDON
        elif self._is_in_range(current_time, *self.sessions[ForexSession.TOKYO]):
            return ForexSession.TOKYO
        else:
            return ForexSession.SYDNEY

    def _is_in_range(self, t: time, start: time, end: time) -> bool:
        if start <= end:
            return start <= t <= end
        else:
            # Crosses midnight
            return start <= t or t <= end

    def is_optimal_for_pair(self, symbol: str, session: ForexSession) -> bool:
        """
        Determines if current session has high liquidity/volatility for a specific pair.
        """
        if session == ForexSession.CLOSED:
            return False
            
        # Usually overlap is best for everything
        if session == ForexSession.OVERLAP_LONDON_NY:
            return True
            
        # Pair specific logic
        is_eur_gbp_chf = any(c in symbol for c in ["EUR", "GBP", "CHF"])
        is_usd_cad = any(c in symbol for c in ["USD", "CAD"])
        is_jpy_aud_nzd = any(c in symbol for c in ["JPY", "AUD", "NZD"])
        
        if session == ForexSession.LONDON and is_eur_gbp_chf: return True
        if session == ForexSession.NEW_YORK and is_usd_cad: return True
        if session in [ForexSession.TOKYO, ForexSession.SYDNEY] and is_jpy_aud_nzd: return True
        
        # If no specific match, it's sub-optimal but not restricted
        # We return True but strategies might want to filter this
        return True
