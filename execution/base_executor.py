"""
TradeAgent Abstract Execution Engine.

Defines the interface for broker execution adapters.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from core.models import AccountInfo, Order, Position, Tick


class BaseExecutor(ABC):
    """Abstract base class for all execution engines."""
    
    def __init__(self, name: str):
        self.name = name
        self.is_connected = False

    @abstractmethod
    async def connect(self) -> bool:
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        pass

    @abstractmethod
    async def get_account_info(self) -> AccountInfo:
        pass

    @abstractmethod
    async def open_order(self, order: Order) -> Order:
        """Submit a new order to the broker."""
        pass

    @abstractmethod
    async def close_position(self, position: Position, current_price: float) -> bool:
        """Close an existing position."""
        pass

    @abstractmethod
    async def modify_position(self, position: Position, new_sl: float, new_tp: float) -> bool:
        """Modify SL/TP of an existing position."""
        pass

    @abstractmethod
    async def get_open_positions(self) -> List[Position]:
        """Fetch current open positions from the broker."""
        pass

    # Optional hook for simulated executors
    async def update_simulated_price(self, tick: Tick) -> None:
        pass
