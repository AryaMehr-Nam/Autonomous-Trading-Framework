"""
TradeAgent Async Event Bus.

A lightweight publish/subscribe event system for decoupling components.
Components publish events (e.g., NewCandle, NewSignal) and other components
subscribe to react to those events asynchronously.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, Callable, Coroutine, Dict, List, Optional

from core.logger import get_logger

logger = get_logger("event_bus")


# ═══════════════════════════════════════════════════════════════
# Event Types
# ═══════════════════════════════════════════════════════════════

class EventType:
    """Constants for all event types in the system."""
    # Data events
    NEW_TICK = "new_tick"
    NEW_CANDLE = "new_candle"
    DATA_ERROR = "data_error"

    # Analysis events
    INDICATORS_UPDATED = "indicators_updated"
    SIGNAL_GENERATED = "signal_generated"
    REGIME_CHANGED = "regime_changed"

    # Fundamental events
    ECONOMIC_EVENT = "economic_event"
    HIGH_IMPACT_EVENT = "high_impact_event"
    MACRO_UPDATE = "macro_update"

    # Sentiment events
    SENTIMENT_UPDATE = "sentiment_update"
    NEWS_ALERT = "news_alert"

    # ML events
    ML_PREDICTION = "ml_prediction"
    MODEL_RETRAINED = "model_retrained"

    # Trading events
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    POSITION_MODIFIED = "position_modified"

    # Risk events
    RISK_ALERT = "risk_alert"
    DRAWDOWN_WARNING = "drawdown_warning"
    TRADING_PAUSED = "trading_paused"
    TRADING_RESUMED = "trading_resumed"
    KILL_SWITCH_ACTIVATED = "kill_switch_activated"

    # System events
    AGENT_STARTED = "agent_started"
    AGENT_STOPPED = "agent_stopped"
    HEARTBEAT = "heartbeat"
    ERROR = "error"


# Type alias for event handler callbacks
EventHandler = Callable[..., Coroutine[Any, Any, None]]


class Event:
    """An event that can be published through the event bus."""

    def __init__(self, event_type: str, data: Any = None, source: str = ""):
        self.event_type = event_type
        self.data = data
        self.source = source
        self.timestamp = asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else 0

    def __repr__(self) -> str:
        return f"Event(type={self.event_type}, source={self.source})"


class EventBus:
    """
    Async publish/subscribe event bus.

    Components subscribe to specific event types and get notified
    when those events are published.

    Usage:
        bus = EventBus()

        async def on_new_candle(event: Event):
            candle = event.data
            print(f"New candle: {candle.symbol}")

        bus.subscribe(EventType.NEW_CANDLE, on_new_candle)
        await bus.publish(Event(EventType.NEW_CANDLE, data=candle))
    """

    def __init__(self):
        self._subscribers: Dict[str, List[EventHandler]] = defaultdict(list)
        self._event_history: List[Event] = []
        self._max_history: int = 1000
        self._paused: bool = False

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Subscribe a handler to an event type."""
        self._subscribers[event_type].append(handler)
        logger.debug(f"Subscribed {handler.__name__} to {event_type}")

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """Unsubscribe a handler from an event type."""
        if event_type in self._subscribers:
            self._subscribers[event_type] = [
                h for h in self._subscribers[event_type] if h != handler
            ]
            logger.debug(f"Unsubscribed {handler.__name__} from {event_type}")

    async def publish(self, event: Event) -> None:
        """
        Publish an event to all subscribers.
        Handlers are called concurrently using asyncio.gather.
        Errors in individual handlers are caught and logged.
        """
        if self._paused and event.event_type not in (
            EventType.KILL_SWITCH_ACTIVATED,
            EventType.AGENT_STOPPED,
            EventType.ERROR,
        ):
            return

        # Store in history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history:]

        handlers = self._subscribers.get(event.event_type, [])
        if not handlers:
            return

        logger.debug(f"Publishing {event.event_type} to {len(handlers)} handler(s)")

        tasks = []
        for handler in handlers:
            tasks.append(self._safe_call(handler, event))

        await asyncio.gather(*tasks)

    async def _safe_call(self, handler: EventHandler, event: Event) -> None:
        """Call a handler with error catching."""
        try:
            await handler(event)
        except Exception as e:
            logger.error(
                f"Error in event handler {handler.__name__} "
                f"for {event.event_type}: {e}"
            )

    def pause(self) -> None:
        """Pause event delivery (except critical events)."""
        self._paused = True
        logger.warning("Event bus PAUSED")

    def resume(self) -> None:
        """Resume event delivery."""
        self._paused = False
        logger.info("Event bus RESUMED")

    def get_history(
        self,
        event_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[Event]:
        """Get recent event history, optionally filtered by type."""
        events = self._event_history
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events[-limit:]

    @property
    def subscriber_count(self) -> Dict[str, int]:
        """Get count of subscribers per event type."""
        return {k: len(v) for k, v in self._subscribers.items() if v}

    def clear(self) -> None:
        """Clear all subscribers and history."""
        self._subscribers.clear()
        self._event_history.clear()
        logger.info("Event bus cleared")


# ── Global Event Bus Instance ────────────────────────────────
event_bus = EventBus()
