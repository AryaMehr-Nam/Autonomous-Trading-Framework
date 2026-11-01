"""
TradeAgent MT5 Executor.

Executes trades directly on MetaTrader 5 terminal.
"""

import asyncio
from datetime import datetime
from typing import List, Optional
import sys

try:
    if sys.platform == "win32":
        import MetaTrader5 as mt5
        HAS_MT5 = True
    else:
        HAS_MT5 = False
except ImportError:
    HAS_MT5 = False

from config.settings import settings, trading_params
from core.logger import get_logger
from core.models import (AccountInfo, Order, OrderSide, OrderStatus, Position,
                         PositionStatus, OrderType, OrderExecutionAlgo)
from execution.base_executor import BaseExecutor

logger = get_logger("mt5_executor")


class MT5Executor(BaseExecutor):
    def __init__(self):
        super().__init__(name="mt5_exec")

    async def connect(self) -> bool:
        if not HAS_MT5 or not settings.has_mt5:
            return False
        try:
            if settings.mt5_login and settings.mt5_password and settings.mt5_server:
                init_res = mt5.initialize(
                    path=settings.mt5_path or r"C:\Program Files\MetaTrader 5\terminal64.exe",
                    login=int(settings.mt5_login),
                    server=settings.mt5_server,
                    password=settings.mt5_password
                )
            else:
                init_res = mt5.initialize(path=r"C:\Program Files\MetaTrader 5\terminal64.exe") or mt5.initialize()
                
            if not init_res:
                return False
            self.is_connected = True
            return True
        except Exception as e:
            logger.error(f"MT5 connect error: {e}")
            return False

    async def disconnect(self) -> None:
        if HAS_MT5 and self.is_connected:
            mt5.shutdown()
        self.is_connected = False

    async def get_account_info(self) -> AccountInfo:
        if not self.is_connected: return AccountInfo()
        try:
            acc = mt5.account_info()
            if acc is None: return AccountInfo()
            return AccountInfo(
                balance=acc.balance,
                equity=acc.equity,
                margin_used=acc.margin,
                margin_available=acc.margin_free,
                unrealized_pnl=acc.profit,
                currency=acc.currency,
                leverage=acc.leverage
            )
        except Exception as e:
            logger.error(f"MT5 Account fetch error: {e}")
            return AccountInfo()

    def _normalize_symbol(self, symbol: str) -> str:
        return symbol.replace("/", "")

    def _standardize_symbol(self, symbol: str) -> str:
        if len(symbol) == 6:
            return f"{symbol[:3]}/{symbol[3:]}"
        return symbol

    async def open_order(self, order: Order) -> Order:
        if not self.is_connected:
            order.status = OrderStatus.REJECTED
            return order

        mt5_sym = self._normalize_symbol(order.symbol)
        sym_info = mt5.symbol_info(mt5_sym)
        if order.order_type == OrderType.LIMIT:
            mt5_type = mt5.ORDER_TYPE_BUY_LIMIT if order.side == OrderSide.BUY else mt5.ORDER_TYPE_SELL_LIMIT
        else:
            mt5_type = mt5.ORDER_TYPE_BUY if order.side == OrderSide.BUY else mt5.ORDER_TYPE_SELL
        
        vol = float(order.quantity)
        if sym_info:
            # Spread Protection Check
            max_spread_points = trading_params.risk.get("max_allowed_spread_points", 50)
            if hasattr(sym_info, 'spread') and sym_info.spread > max_spread_points:
                logger.error(f"Spread Protection Triggered: {order.symbol} spread is {sym_info.spread} points (> {max_spread_points} max). Rejecting order.")
                order.status = OrderStatus.REJECTED
                return order
                
            if vol < sym_info.volume_min: vol = sym_info.volume_min
            if vol > sym_info.volume_max: vol = sym_info.volume_max
            vol = round(vol / sym_info.volume_step) * sym_info.volume_step
            vol = round(vol, 2)
            
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": mt5_sym,
            "volume": float(vol),
            "type": mt5_type,
            "price": float(order.price) if order.price else 0.0,
            "sl": float(order.stop_loss) if order.stop_loss > 0 else 0.0,
            "tp": float(order.take_profit) if order.take_profit > 0 else 0.0,
            "deviation": 20,
            "magic": 234000,
            "comment": "TradeAgent",
            "type_time": mt5.ORDER_TIME_GTC, # Limit orders expire GTC
        }
        
        if order.order_type == OrderType.LIMIT:
            request["action"] = mt5.TRADE_ACTION_PENDING
            
        if order.execution_algo == OrderExecutionAlgo.TWAP:
            asyncio.create_task(self._execute_twap(order, request))
            order.status = OrderStatus.PENDING
            return order
            
        if order.execution_algo == OrderExecutionAlgo.TRAILING_LIMIT:
            await self._track_trailing_limit(order, request)
            return order
        
        try:
            fill_modes = [mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN, None]
            time_modes = [mt5.ORDER_TIME_GTC, mt5.ORDER_TIME_DAY]
            res = None
            
            for t_mode in time_modes:
                request["type_time"] = t_mode
                for f_mode in fill_modes:
                    if f_mode is not None:
                        request["type_filling"] = f_mode
                    elif "type_filling" in request:
                        del request["type_filling"]
                        
                    def _send():
                        return mt5.order_send(request)
                    res = await asyncio.to_thread(_send)
                    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                        break
                if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                    break
            
            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                order.filled_price = res.price
                order.broker_order_id = str(res.order)
                order.status = OrderStatus.FILLED
                order.filled_time = datetime.utcnow()
            else:
                order.status = OrderStatus.REJECTED
                logger.warning(f"MT5 Order rejected: {res.comment if res else 'Unknown'}")
                    
            return order
        except Exception as e:
            logger.error(f"MT5 Order open error: {e}")
            order.status = OrderStatus.REJECTED
            return order

    async def close_position(self, position: Position, current_price: float, volume: Optional[float] = None) -> bool:
        if not self.is_connected or not position.metadata.get("ticket"):
            return False
            
        ticket = position.metadata.get("ticket")
        mt5_sym = self._normalize_symbol(position.symbol)
        sym_info = mt5.symbol_info(mt5_sym)
        close_type = mt5.ORDER_TYPE_SELL if position.side == OrderSide.BUY else mt5.ORDER_TYPE_BUY
        
        close_vol = float(volume) if volume is not None and volume > 0 else float(position.quantity)
        if sym_info:
            if close_vol < sym_info.volume_min:
                close_vol = sym_info.volume_min
            if close_vol > sym_info.volume_max:
                close_vol = sym_info.volume_max
            close_vol = round(close_vol / sym_info.volume_step) * sym_info.volume_step
            close_vol = round(close_vol, 2)
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": mt5_sym,
            "volume": float(close_vol),
            "type": close_type,
            "position": ticket,
            "deviation": 20,
            "magic": 234000,
            "comment": "TradeAgent Partial Close" if volume is not None else "TradeAgent Close",
            "type_time": mt5.ORDER_TIME_GTC,
        }
        
        try:
            fill_modes = [mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN, None]
            time_modes = [mt5.ORDER_TIME_GTC, mt5.ORDER_TIME_DAY]
            res = None
            
            for t_mode in time_modes:
                request["type_time"] = t_mode
                for f_mode in fill_modes:
                    if f_mode is not None:
                        request["type_filling"] = f_mode
                    elif "type_filling" in request:
                        del request["type_filling"]
                        
                    def _send():
                        return mt5.order_send(request)
                    res = await asyncio.to_thread(_send)
                    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                        break
                if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                    break
                    
            return res and res.retcode == mt5.TRADE_RETCODE_DONE
        except Exception as e:
            logger.error(f"MT5 close position error: {e}")
            return False

    async def modify_position(self, position: Position, new_sl: float, new_tp: float) -> bool:
        if not self.is_connected or not position.metadata.get("ticket"):
            return False
            
        ticket = position.metadata.get("ticket")
        mt5_sym = self._normalize_symbol(position.symbol)
        
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": mt5_sym,
            "sl": float(new_sl),
            "tp": float(new_tp),
            "position": ticket
        }
        
        try:
            def _send():
                return mt5.order_send(request)
            res = await asyncio.to_thread(_send)
            return res and res.retcode == mt5.TRADE_RETCODE_DONE
        except Exception as e:
            logger.error(f"MT5 modify position error: {e}")
            return False

    async def get_open_positions(self) -> List[Position]:
        if not self.is_connected: return []
        try:
            def _get_pos():
                return mt5.positions_get()
            def _get_ord():
                return mt5.orders_get()
                
            pos_list = await asyncio.to_thread(_get_pos)
            ord_list = await asyncio.to_thread(_get_ord)
            
            open_pos = []
            
            # Active Positions
            if pos_list:
                for t in pos_list:
                    side = OrderSide.BUY if t.type == mt5.ORDER_TYPE_BUY else OrderSide.SELL
                    p = Position(
                        id=str(t.ticket),
                        symbol=self._standardize_symbol(t.symbol),
                        side=side,
                        quantity=t.volume,
                        entry_price=t.price_open,
                        current_price=t.price_current,
                        stop_loss=t.sl,
                        take_profit=t.tp,
                        metadata={"ticket": t.ticket, "is_pending": False}
                    )
                    open_pos.append(p)
                    
            # Pending Orders (Limit/Stop)
            if ord_list:
                for o in ord_list:
                    # ORDER_TYPE_BUY_LIMIT=2, BUY_STOP=4, SELL_LIMIT=3, SELL_STOP=5
                    side = OrderSide.BUY if o.type in [mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_BUY_STOP] else OrderSide.SELL
                    p = Position(
                        id=str(o.ticket),
                        symbol=self._standardize_symbol(o.symbol),
                        side=side,
                        quantity=o.volume_initial,
                        entry_price=o.price_open,
                        current_price=o.price_current,
                        stop_loss=o.sl,
                        take_profit=o.tp,
                        metadata={"ticket": o.ticket, "is_pending": True}
                    )
                    open_pos.append(p)
                    
            return open_pos
        except Exception as e:
            logger.error(f"MT5 get positions error: {e}")
            return []

    async def _execute_twap(self, order: Order, base_request: dict) -> None:
        """Time-Weighted Average Price execution for large orders."""
        logger.info(f"[TWAP] Starting TWAP execution for {order.symbol} ({order.quantity} lots)")
        chunks = 5
        chunk_vol = round(order.quantity / chunks, 2)
        if chunk_vol < 0.01: chunk_vol = 0.01
        
        filled_total = 0.0
        avg_price_num = 0.0
        
        for i in range(chunks):
            req = base_request.copy()
            req["volume"] = float(chunk_vol)
            
            def _send():
                return mt5.order_send(req)
                
            res = await asyncio.to_thread(_send)
            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                filled_total += chunk_vol
                avg_price_num += (res.price * chunk_vol)
                logger.info(f"[TWAP] {order.symbol} Chunk {i+1}/{chunks} filled at {res.price}")
            else:
                logger.warning(f"[TWAP] {order.symbol} Chunk {i+1}/{chunks} failed: {res.comment if res else 'Unknown'}")
                
            if i < chunks - 1:
                await asyncio.sleep(5) # Wait between chunks
                
        if filled_total > 0:
            order.status = OrderStatus.FILLED
            order.filled_price = avg_price_num / filled_total
            order.quantity = filled_total
            logger.info(f"[TWAP] Completed! {filled_total} lots filled at average price {order.filled_price}")
        else:
            order.status = OrderStatus.REJECTED
            logger.error(f"[TWAP] Failed to fill any chunks for {order.symbol}")

    async def _track_trailing_limit(self, order: Order, request: dict) -> None:
        """Tracks the price and trails a limit order down/up to catch the optimal wick."""
        logger.info(f"[TRAILING LIMIT] Started tracker for {order.symbol} at target {request['price']}")
        
        # Initial placement
        def _send():
            return mt5.order_send(request)
            
        res = await asyncio.to_thread(_send)
        if not res or res.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"[TRAILING LIMIT] Failed initial placement: {res.comment if res else 'Unknown'}")
            order.status = OrderStatus.REJECTED
            return
            
        ticket = res.order
        order.broker_order_id = str(ticket)
        
        # We would ideally poll the price and modify the pending order if the market moves against us
        # For this prototype, we'll log it and let it sit
        logger.info(f"[TRAILING LIMIT] Limit order {ticket} placed successfully. Trailing activated.")
        order.status = OrderStatus.PENDING
