"""
TradeAgent OANDA Executor.

Executes trades on OANDA v20 REST API.
"""

import asyncio
from datetime import datetime
from typing import List, Optional

try:
    from oandapyV20 import API
    import oandapyV20.endpoints.accounts as accounts
    import oandapyV20.endpoints.orders as orders
    import oandapyV20.endpoints.positions as positions
    import oandapyV20.endpoints.trades as trades
    HAS_OANDA = True
except ImportError:
    HAS_OANDA = False

from config.settings import settings
from core.logger import get_logger
from core.models import (AccountInfo, Order, OrderSide, OrderStatus, Position,
                         PositionStatus)
from execution.base_executor import BaseExecutor

logger = get_logger("oanda_executor")


class OandaExecutor(BaseExecutor):
    def __init__(self):
        super().__init__(name="oanda_exec")
        self.account_id = settings.oanda_account_id
        self.api = None

    async def connect(self) -> bool:
        if not HAS_OANDA or not settings.has_oanda:
            return False
        try:
            self.api = API(
                access_token=settings.oanda_api_key,
                environment=settings.oanda_environment
            )
            self.is_connected = True
            return True
        except Exception as e:
            logger.error(f"OANDA connect error: {e}")
            return False

    async def disconnect(self) -> None:
        self.is_connected = False

    async def get_account_info(self) -> AccountInfo:
        if not self.is_connected: return AccountInfo()
        try:
            req = accounts.AccountSummary(self.account_id)
            res = await asyncio.to_thread(self.api.request, req)
            acc = res.get("account", {})
            return AccountInfo(
                balance=float(acc.get("balance", 0)),
                equity=float(acc.get("NAV", 0)),
                margin_used=float(acc.get("marginUsed", 0)),
                margin_available=float(acc.get("marginAvailable", 0)),
                unrealized_pnl=float(acc.get("unrealizedPL", 0)),
                currency=acc.get("currency", "USD"),
                open_positions=int(acc.get("openPositionCount", 0))
            )
        except Exception as e:
            logger.error(f"OANDA Account fetch error: {e}")
            return AccountInfo()

    def _normalize_symbol(self, symbol: str) -> str:
        return symbol.replace("/", "_")

    def _standardize_symbol(self, symbol: str) -> str:
        return symbol.replace("_", "/")

    async def open_order(self, order: Order) -> Order:
        if not self.is_connected:
            order.status = OrderStatus.REJECTED
            return order

        oanda_sym = self._normalize_symbol(order.symbol)
        
        # OANDA units: positive for BUY, negative for SELL
        units = int(order.quantity) if order.side == OrderSide.BUY else -int(order.quantity)
        
        data = {
            "order": {
                "units": str(units),
                "instrument": oanda_sym,
                "timeInForce": "FOK", # Fill or Kill
                "type": "MARKET",
                "positionFill": "DEFAULT"
            }
        }
        
        if order.stop_loss > 0:
            data["order"]["stopLossOnFill"] = {"price": f"{order.stop_loss:.5f}"}
        if order.take_profit > 0:
            data["order"]["takeProfitOnFill"] = {"price": f"{order.take_profit:.5f}"}

        try:
            req = orders.OrderCreate(self.account_id, data=data)
            res = await asyncio.to_thread(self.api.request, req)
            
            fill = res.get("orderFillTransaction", {})
            if fill:
                order.filled_price = float(fill.get("price", 0))
                order.broker_order_id = fill.get("tradeOpened", {}).get("tradeID")
                order.status = OrderStatus.FILLED
                order.filled_time = datetime.utcnow()
            else:
                cancel = res.get("orderCancelTransaction")
                if cancel:
                    order.status = OrderStatus.REJECTED
                    logger.warning(f"OANDA Order rejected: {cancel.get('reason')}")
                    
            return order
        except Exception as e:
            logger.error(f"OANDA Order open error: {e}")
            order.status = OrderStatus.REJECTED
            return order

    async def close_position(self, position: Position, current_price: float) -> bool:
        if not self.is_connected or not position.metadata.get("broker_trade_id"):
            return False
            
        trade_id = position.metadata.get("broker_trade_id")
        
        try:
            req = trades.TradeClose(self.account_id, tradeID=trade_id)
            res = await asyncio.to_thread(self.api.request, req)
            return True
        except Exception as e:
            logger.error(f"OANDA close position error: {e}")
            return False

    async def modify_position(self, position: Position, new_sl: float, new_tp: float) -> bool:
        if not self.is_connected or not position.metadata.get("broker_trade_id"):
            return False
            
        trade_id = position.metadata.get("broker_trade_id")
        data = {}
        if new_sl > 0: data["stopLoss"] = {"price": f"{new_sl:.5f}"}
        if new_tp > 0: data["takeProfit"] = {"price": f"{new_tp:.5f}"}
        
        try:
            req = trades.TradeCRCDO(self.account_id, tradeID=trade_id, data=data)
            await asyncio.to_thread(self.api.request, req)
            return True
        except Exception as e:
            logger.error(f"OANDA modify position error: {e}")
            return False

    async def get_open_positions(self) -> List[Position]:
        if not self.is_connected: return []
        try:
            req = trades.OpenTrades(self.account_id)
            res = await asyncio.to_thread(self.api.request, req)
            
            open_pos = []
            for t in res.get("trades", []):
                units = float(t.get("currentUnits", 0))
                side = OrderSide.BUY if units > 0 else OrderSide.SELL
                
                p = Position(
                    id=t["id"],
                    symbol=self._standardize_symbol(t["instrument"]),
                    side=side,
                    quantity=abs(units),
                    entry_price=float(t.get("price", 0)),
                    current_price=float(t.get("price", 0)), # Update required externally
                    stop_loss=float(t.get("stopLossOrder", {}).get("price", 0)),
                    take_profit=float(t.get("takeProfitOrder", {}).get("price", 0)),
                    metadata={"broker_trade_id": t["id"]}
                )
                open_pos.append(p)
            return open_pos
        except Exception as e:
            logger.error(f"OANDA get positions error: {e}")
            return []
