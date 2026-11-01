"""
TradeAgent Paper Executor.

Simulates order execution and position management locally for risk-free testing.
Matches against live ticks to simulate slippage and fills.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional

from config.settings import trading_params
from core.logger import get_logger
from core.models import (AccountInfo, Order, OrderSide, OrderStatus, Position,
                         PositionStatus, Tick)
from execution.base_executor import BaseExecutor

logger = get_logger("paper_executor")


class PaperExecutor(BaseExecutor):
    """Simulates broker execution locally."""
    
    def __init__(self):
        super().__init__(name="paper")
        # In-memory storage
        self.account = AccountInfo(balance=200.0, equity=200.0, leverage=100.0)
        self.open_orders: Dict[str, Order] = {}
        self.open_positions: Dict[str, Position] = {}
        self.current_prices: Dict[str, Tick] = {}

    async def connect(self) -> bool:
        self.is_connected = True
        logger.info("Paper Executor started.")
        return True

    async def disconnect(self) -> None:
        self.is_connected = False

    async def get_account_info(self) -> AccountInfo:
        # Recalculate equity
        unrealized = 0.0
        margin = 0.0
        
        for pos in self.open_positions.values():
            unrealized += pos.unrealized_pnl
            # Simplified margin calculation (Value / Leverage)
            margin += (pos.entry_price * pos.quantity) / self.account.leverage
            
        self.account.unrealized_pnl = unrealized
        self.account.equity = self.account.balance + unrealized
        self.account.margin_used = margin
        self.account.margin_available = self.account.equity - margin
        self.account.open_positions = len(self.open_positions)
        
        return self.account

    async def update_simulated_price(self, tick: Tick) -> None:
        """Called externally whenever a new tick arrives."""
        self.current_prices[tick.symbol] = tick
        
        # Update PnL for open positions
        for pos in self.open_positions.values():
            if pos.symbol == tick.symbol:
                if pos.side == OrderSide.BUY:
                    pos.current_price = tick.bid
                else:
                    pos.current_price = tick.ask
                    
                # Check Stop Loss / Take Profit
                await self._check_sl_tp(pos, tick)

    async def _check_sl_tp(self, pos: Position, tick: Tick):
        """Simulate SL/TP hits."""
        if pos.side == OrderSide.BUY:
            if pos.stop_loss > 0 and tick.bid <= pos.stop_loss:
                logger.info(f"Stop Loss hit for {pos.symbol} (Buy)")
                await self.close_position(pos, tick.bid)
            elif pos.take_profit > 0 and tick.bid >= pos.take_profit:
                logger.info(f"Take Profit hit for {pos.symbol} (Buy)")
                await self.close_position(pos, tick.bid)
        else: # SELL
            if pos.stop_loss > 0 and tick.ask >= pos.stop_loss:
                logger.info(f"Stop Loss hit for {pos.symbol} (Sell)")
                await self.close_position(pos, tick.ask)
            elif pos.take_profit > 0 and tick.ask <= pos.take_profit:
                logger.info(f"Take Profit hit for {pos.symbol} (Sell)")
                await self.close_position(pos, tick.ask)

    async def open_order(self, order: Order) -> Order:
        """Simulate order execution."""
        tick = self.current_prices.get(order.symbol)
        
        if not tick:
            logger.warning(f"Cannot execute order: No price data for {order.symbol}")
            order.status = OrderStatus.REJECTED
            return order

        # Simulate spread and slippage
        slippage_pct = trading_params.backtesting_config.get("slippage_pips", 1) * 0.0001
        
        if order.side == OrderSide.BUY:
            fill_price = tick.ask * (1 + slippage_pct)
        else:
            fill_price = tick.bid * (1 - slippage_pct)
            
        order.filled_price = fill_price
        order.filled_time = datetime.utcnow()
        order.status = OrderStatus.FILLED
        order.commission = (order.quantity * fill_price) * trading_params.backtesting_config.get("commission_pct", 0.001)
        
        # Create Position
        pos = Position(
            order_id=order.id,
            signal_id=order.signal_id,
            symbol=order.symbol,
            asset_class=order.asset_class,
            side=order.side,
            quantity=order.quantity,
            entry_price=fill_price,
            current_price=fill_price,
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
            commission=order.commission
        )
        
        self.open_positions[pos.id] = pos
        self.account.balance -= order.commission
        
        logger.info(f"Paper filled: {order.side.value} {order.quantity:.4f} {order.symbol} @ {fill_price:.5f}")
        return order

    async def close_position(self, position: Position, current_price: float) -> bool:
        """Simulate closing a position."""
        if position.id not in self.open_positions:
            return False
            
        slippage_pct = trading_params.backtesting_config.get("slippage_pips", 1) * 0.0001
        
        if position.side == OrderSide.BUY:
            exit_price = current_price * (1 - slippage_pct)
        else:
            exit_price = current_price * (1 + slippage_pct)
            
        close_comm = (position.quantity * exit_price) * trading_params.backtesting_config.get("commission_pct", 0.001)
        
        position.close_price = exit_price
        position.close_time = datetime.utcnow()
        position.status = PositionStatus.CLOSED
        
        # Realize PnL
        pnl = position.realized_pnl
        self.account.balance += (pnl - close_comm)
        
        del self.open_positions[position.id]
        logger.info(f"Paper closed: {position.side.value} {position.symbol} @ {exit_price:.5f} | PnL: {pnl:.2f}")
        
        return True

    async def modify_position(self, position: Position, new_sl: float, new_tp: float) -> bool:
        if position.id in self.open_positions:
            pos = self.open_positions[position.id]
            pos.stop_loss = new_sl
            pos.take_profit = new_tp
            return True
        return False

    async def get_open_positions(self) -> List[Position]:
        return list(self.open_positions.values())
