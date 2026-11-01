"""
TradeAgent CCXT Executor.

Executes trades on Crypto exchanges via CCXT.
"""

import asyncio
from datetime import datetime
from typing import List, Optional

try:
    import ccxt.async_support as ccxt
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False

from config.settings import settings
from core.logger import get_logger
from core.models import (AccountInfo, Order, OrderSide, OrderStatus, Position,
                         PositionStatus)
from execution.base_executor import BaseExecutor

logger = get_logger("ccxt_executor")


class CCXTExecutor(BaseExecutor):
    def __init__(self, exchange_id: str = "binance"):
        super().__init__(name=f"ccxt_{exchange_id}_exec")
        self.exchange_id = exchange_id
        self.exchange = None

    async def connect(self) -> bool:
        if not HAS_CCXT:
            return False
            
        try:
            exchange_class = getattr(ccxt, self.exchange_id)
            config = {"enableRateLimit": True}
            if self.exchange_id == "binance" and settings.has_binance:
                config["apiKey"] = settings.binance_api_key
                config["secret"] = settings.binance_secret
                
            self.exchange = exchange_class(config)
            await self.exchange.load_markets()
            self.is_connected = True
            return True
        except Exception as e:
            logger.error(f"CCXT connect error: {e}")
            return False

    async def disconnect(self) -> None:
        if self.exchange:
            await self.exchange.close()
        self.is_connected = False

    async def get_account_info(self) -> AccountInfo:
        if not self.is_connected or not self.exchange:
            return AccountInfo()
        try:
            bal = await self.exchange.fetch_balance()
            total_usd = 0.0
            free_usd = 0.0
            
            # Simple approximation assuming USDT base or retrieving total info
            # Real implementation needs to fetch ticker prices to convert all assets to USD
            if "USDT" in bal:
                total_usd = float(bal["USDT"]["total"])
                free_usd = float(bal["USDT"]["free"])
                
            return AccountInfo(
                balance=total_usd,
                equity=total_usd, # Simplification
                margin_used=total_usd - free_usd,
                margin_available=free_usd,
                currency="USDT"
            )
        except Exception as e:
            logger.error(f"CCXT Account fetch error: {e}")
            return AccountInfo()

    async def open_order(self, order: Order) -> Order:
        if not self.is_connected or not self.exchange:
            order.status = OrderStatus.REJECTED
            return order

        side = "buy" if order.side == OrderSide.BUY else "sell"
        order_type = "market"
        
        params = {}
        if order.stop_loss > 0:
            # Note: Stop loss format varies heavily by crypto exchange
            params["stopLossPrice"] = order.stop_loss
        if order.take_profit > 0:
            params["takeProfitPrice"] = order.take_profit
            
        try:
            res = await self.exchange.create_order(
                order.symbol, order_type, side, float(order.quantity), None, params
            )
            
            order.filled_price = float(res.get("price", res.get("average", 0)))
            order.broker_order_id = res.get("id")
            order.status = OrderStatus.FILLED
            order.filled_time = datetime.utcnow()
                    
            return order
        except Exception as e:
            logger.error(f"CCXT Order open error: {e}")
            order.status = OrderStatus.REJECTED
            return order

    async def close_position(self, position: Position, current_price: float) -> bool:
        if not self.is_connected or not self.exchange:
            return False
            
        side = "sell" if position.side == OrderSide.BUY else "buy"
        
        try:
            # Simple market close. For futures, might need reduceOnly=True
            params = {"reduceOnly": True} if self.exchange.has.get("createMarketOrder", False) else {}
            await self.exchange.create_order(
                position.symbol, "market", side, float(position.quantity), None, params
            )
            return True
        except Exception as e:
            logger.error(f"CCXT close position error: {e}")
            return False

    async def modify_position(self, position: Position, new_sl: float, new_tp: float) -> bool:
        # Crypto exchanges usually require cancelling and replacing SL/TP orders
        logger.warning("CCXT modify position is exchange-specific and requires complex logic.")
        return False

    async def get_open_positions(self) -> List[Position]:
        if not self.is_connected or not self.exchange: return []
        try:
            # Only works for futures/margin accounts typically
            if self.exchange.has.get("fetchPositions"):
                positions_data = await self.exchange.fetch_positions()
                open_pos = []
                for p in positions_data:
                    if float(p.get("contracts", 0)) > 0:
                        side = OrderSide.BUY if p["side"] == "long" else OrderSide.SELL
                        pos = Position(
                            id=p["id"] or p["symbol"],
                            symbol=p["symbol"],
                            side=side,
                            quantity=float(p["contracts"]),
                            entry_price=float(p["entryPrice"]),
                            current_price=float(p.get("markPrice", p["entryPrice"])),
                            metadata={"raw": p}
                        )
                        open_pos.append(pos)
                return open_pos
            return []
        except Exception as e:
            logger.error(f"CCXT get positions error: {e}")
            return []
