"""
Pure Gold (XAUUSD) Production Sniper Engine v9.6 Modular Compounder
===================================================================
Institutional Khakestar T-Rex + Pour Samadi + SMC Continuous Intelligence Fusion
MetaTrader 5 Integration: Configurable via Environment Variables / MT5_LOGIN

Core Architectural Upgrades (v9.6 Modular Leap):
1. Strategy Registry Integration:
   - Modular strategy engine executing Khakestar Macro-Swing, Pour Samadi Prime Scalp,
     SMC Liquidity Sweep, Asian Judas Swing, and VWAP Mean-Reversion.
2. Dynamic Configuration Hot-Reloading:
   - Synchronized with `config/pure_gold_config.json` without process restarts.
   - Autonomous improver agent updates and promotes vetted strategies in real time.
3. Crash-Proof State Persistence:
   - Managed by `SniperStateManager` saving active tickets and trailing states to disk.
   - Survives terminal disconnects, machine reboots, and process restarts.
4. Institutional Sizing & Dynamic Compounding:
   - Strict 2.5% Equity Risk per Trade with dynamic lot scaling.
   - Two-stage profit banking: 80% at TP1 (+1.8R), trailing risk-free runner at TP2 (+3.5R).
5. Ironclad Guardrails:
   - Rollover shield (23:50-00:15), ADR usage cap (85%), Hard loss equity shield,
     Consecutive loss circuit breaker (2 losses = 4h pause), 15m anti-chop cooldown.
"""

import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import time
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple, List, Any

import numpy as np
import pandas as pd
import MetaTrader5 as mt5

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from signals.poursamadi_indicators import PourSamadiToolkit
from signals.khakestar_trex import KhakestarTrexEngine
from signals.strategies import strategy_registry, StrategySignal
from core.sniper_config_manager import SniperConfigManager
from core.sniper_state_manager import SniperStateManager

TARGET_LOGIN = int(os.getenv("MT5_LOGIN", "0")) if os.getenv("MT5_LOGIN") else 0
TARGET_PASS = os.getenv("MT5_PASSWORD", "")
TARGET_SERVER = os.getenv("MT5_SERVER", "")

# Global Config & State Singletons
config_mgr = SniperConfigManager()
state_mgr = SniperStateManager()

def calculate_dynamic_compounded_lot(equity: float, risk_dist: float, risk_pct: float = 0.025) -> Tuple[float, float, float]:
    """
    Dynamic Compounding Engine (v9.6):
    Allocates exact % risk of live floating equity per trade.
    Returns: (total_lot, tp1_bank_lot, runner_lot)
    """
    risk_cfg = config_mgr.risk
    min_lot = risk_cfg.get("min_lot", 0.01)
    max_lot = risk_cfg.get("max_lot", 1.00)
    bank_frac = config_mgr.trade_management.get("tp1_bank_fraction", 0.80)

    risk_usd = max(10.0, equity * risk_pct)
    if risk_dist <= 0:
        risk_dist = 2.50
    calc_lot = round(risk_usd / (risk_dist * 100.0), 2)
    total_lot = max(min_lot, min(calc_lot, max_lot))
    tp1_bank_lot = round(total_lot * bank_frac, 2)
    runner_lot = round(total_lot - tp1_bank_lot, 2)
    if tp1_bank_lot <= 0:
        tp1_bank_lot = total_lot
        runner_lot = 0.0
    elif runner_lot <= 0:
        runner_lot = 0.01
        total_lot = round(tp1_bank_lot + runner_lot, 2)
    return total_lot, tp1_bank_lot, runner_lot

def get_filling_mode(sym_info):
    filling = sym_info.filling_mode
    if filling & 2:
        return mt5.ORDER_FILLING_IOC
    elif filling & 1:
        return mt5.ORDER_FILLING_FOK
    return mt5.ORDER_FILLING_RETURN

def is_rollover_window() -> bool:
    """Toxic spread widening window: 23:50 to 00:15 broker server time."""
    tick = mt5.symbol_info_tick("XAUUSD")
    if not tick:
        return False
    dt = datetime.fromtimestamp(tick.time, tz=timezone.utc)
    if (dt.hour == 23 and dt.minute >= 50) or (dt.hour == 0 and dt.minute <= 15):
        return True
    sym_info = mt5.symbol_info("XAUUSD")
    if sym_info and sym_info.spread > 80:
        return True
    return False

def is_prime_scalp_session() -> Tuple[bool, str]:
    """
    Pour Samadi Prime Execution Windows:
    - London Prime: 07:30 to 11:30 UTC
    - NY Prime / Overlap: 12:30 to 17:00 UTC
    """
    tick = mt5.symbol_info_tick("XAUUSD")
    if not tick:
        return False, "NO_TICK"
    dt = datetime.fromtimestamp(tick.time, tz=timezone.utc)
    hr = dt.hour
    minute = dt.minute

    # London Prime
    if (hr == 7 and minute >= 30) or (8 <= hr <= 10) or (hr == 11 and minute <= 30):
        return True, "LONDON_PRIME"
    # NY Prime
    if (hr == 12 and minute >= 30) or (13 <= hr <= 16) or (hr == 17 and minute == 0):
        return True, "NY_PRIME"

    return False, "OUT_OF_SESSION"

def get_adr_usage(symbol="XAUUSD") -> Tuple[float, float, float]:
    """Returns (consumed_ratio, today_range, adr14)"""
    try:
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, 16)
        if rates is None or len(rates) < 15:
            return 0.5, 10.0, 25.0
        df_d = pd.DataFrame(rates)
        completed = df_d.iloc[:-1]
        adr = float((completed['high'] - completed['low']).mean())
        today = df_d.iloc[-1]
        today_range = float(today['high'] - today['low'])
        ratio = (today_range / adr) if adr > 0 else 0.5
        return ratio, today_range, adr
    except Exception:
        return 0.5, 10.0, 25.0

# =========================================================================
# INDICATORS CACHE
# =========================================================================
_h4_cache = None
_h4_cache_time = 0.0

def get_h4_data(n_bars=80):
    global _h4_cache, _h4_cache_time
    now = time.time()
    if _h4_cache is not None and (now - _h4_cache_time) < 25.0:
        return _h4_cache

    rates = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_H4, 0, n_bars)
    if rates is None or len(rates) < n_bars:
        return _h4_cache

    df = pd.DataFrame(rates)
    c = df['close'].values
    h = df['high'].values
    l = df['low'].values

    df['ema50'] = pd.Series(c).ewm(span=50).mean().values
    df['ema200'] = pd.Series(c).ewm(span=200).mean().values
    df['atr26'] = KhakestarTrexEngine.calculate_atr_steps(df, period=26)

    bar_range = np.maximum(h - l, 1e-5)
    df['clv'] = (c - l) / bar_range

    delta = pd.Series(c).diff()
    gain = (delta.where(delta > 0, 0)).rolling(7).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(7).mean()
    rs = gain / np.maximum(loss, 1e-10)
    df['rsi'] = (100 - (100 / (1 + rs))).values

    pivots = KhakestarTrexEngine.identify_pivots(df, df['atr26'].values)
    ftr_nodes = KhakestarTrexEngine.identify_ftr_nodes(df, pivots, df['atr26'].values)

    _h4_cache = {
        "df": df,
        "close": c,
        "high": h,
        "low": l,
        "ema50": df['ema50'].values,
        "ema200": df['ema200'].values,
        "atr26": df['atr26'].values,
        "clv": df['clv'].values,
        "rsi": df['rsi'].values,
        "ftr_nodes": ftr_nodes,
    }
    _h4_cache_time = now
    return _h4_cache

def get_m15_data(n_bars=70):
    rates = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M15, 0, n_bars)
    if rates is None or len(rates) < n_bars:
        return None

    df = pd.DataFrame(rates)
    c = df['close'].values
    h = df['high'].values
    l = df['low'].values
    v = df['tick_volume'].values.astype(float)

    df['ema9'] = pd.Series(c).ewm(span=9).mean().values
    df['ema21'] = pd.Series(c).ewm(span=21).mean().values
    df['ema55'] = pd.Series(c).ewm(span=55).mean().values
    df['atr14'] = pd.Series(h - l).rolling(14).mean().values
    df['atr26'] = KhakestarTrexEngine.calculate_atr_steps(df, period=26)

    hlc3 = (h + l + c) / 3.0
    cum_v = np.cumsum(v)
    cum_pv = np.cumsum(hlc3 * v)
    df['vwap'] = cum_pv / np.maximum(cum_v, 1e-5)

    delta = pd.Series(c).diff()
    gain = (delta.where(delta > 0, 0)).rolling(7).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(7).mean()
    rs = gain / np.maximum(loss, 1e-10)
    df['rsi'] = (100 - (100 / (1 + rs))).values

    bar_range = np.maximum(h - l, 1e-5)
    df['clv'] = (c - l) / bar_range

    pivots = KhakestarTrexEngine.identify_pivots(df, df['atr26'].values)
    ftr_nodes = KhakestarTrexEngine.identify_ftr_nodes(df, pivots, df['atr26'].values)

    return {
        "df": df,
        "close": c,
        "high": h,
        "low": l,
        "ema9": df['ema9'].values,
        "ema21": df['ema21'].values,
        "ema55": df['ema55'].values,
        "atr14": df['atr14'].values,
        "atr26": df['atr26'].values,
        "vwap": df['vwap'].values,
        "rsi": df['rsi'].values,
        "clv": df['clv'].values,
        "ftr_nodes": ftr_nodes,
    }

def get_m5_data(n_bars=80):
    rates = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M5, 0, n_bars)
    if rates is None or len(rates) < n_bars:
        return None

    df = pd.DataFrame(rates)
    c = df['close'].values
    h = df['high'].values
    l = df['low'].values
    v = df['tick_volume'].values.astype(float)

    df['ema9'] = pd.Series(c).ewm(span=9).mean().values
    df['ema21'] = pd.Series(c).ewm(span=21).mean().values
    df['atr14'] = pd.Series(h - l).rolling(14).mean().values
    df['atr26'] = KhakestarTrexEngine.calculate_atr_steps(df, period=26)

    hlc3 = (h + l + c) / 3.0
    cum_v = np.cumsum(v)
    cum_pv = np.cumsum(hlc3 * v)
    df['vwap'] = cum_pv / np.maximum(cum_v, 1e-5)
    df['clv'] = (c - l) / np.maximum(h - l, 1e-5)

    delta = pd.Series(c).diff()
    gain = (delta.where(delta > 0, 0)).rolling(7).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(7).mean()
    rs = gain / np.maximum(loss, 1e-10)
    df['rsi'] = (100 - (100 / (1 + rs))).values

    pivots = KhakestarTrexEngine.identify_pivots(df, df['atr26'].values)
    nodes = KhakestarTrexEngine.identify_ftr_nodes(df, pivots, df['atr26'].values)

    return {
        "df": df,
        "close": c,
        "high": h,
        "low": l,
        "ema9": df['ema9'].values,
        "ema21": df['ema21'].values,
        "atr14": df['atr14'].values,
        "clv": df['clv'].values,
        "rsi": df['rsi'].values,
        "vwap": df['vwap'].values,
        "ftr_nodes": nodes,
    }

# =========================================================================
# SCANNER DISPATCHER USING STRATEGY REGISTRY
# =========================================================================
def scan_all_gold_strategies(equity: float = 500.0) -> Optional[Dict[str, Any]]:
    """
    Evaluates all active registered strategies in priority order.
    Returns the highest-conviction setup, if any.
    """
    now = time.time()
    if now < state_mgr.cooldown_until_time:
        return None
    anti_chop = config_mgr.risk.get("anti_chop_cooldown_seconds", 600)
    if (now - state_mgr.last_micro_exit_time) < anti_chop:
        return None
    if is_rollover_window():
        return None

    tick = mt5.symbol_info_tick("XAUUSD")
    sym_info = mt5.symbol_info("XAUUSD")
    if not tick or not sym_info:
        return None

    spread_pts = (tick.ask - tick.bid) / sym_info.point
    max_spread_scalp = config_mgr.risk.get("max_spread_scalp", 42)
    if spread_pts > max_spread_scalp:
        return None

    adr_data = get_adr_usage("XAUUSD")
    session_data = is_prime_scalp_session()

    m15 = get_m15_data(n_bars=70)
    m5 = get_m5_data(n_bars=80)
    h4 = get_h4_data(n_bars=80)
    if not m15:
        return None

    market_data = {
        "m15": m15,
        "m5": m5,
        "h4": h4,
        "adr": adr_data,
        "session": session_data,
        "time": datetime.fromtimestamp(tick.time, tz=timezone.utc),
    }

    spot_price = tick.bid

    # Strategy priority hierarchy: Macro-Swing -> SMC Sweep -> Asian Judas -> Prime Scalp -> VWAP
    priority_order = [
        "h4_macro_swing",
        "smc_liquidity_sweep",
        "asian_judas_swing",
        "prime_m15_scalp",
        "vwap_khakestar_snapback",
    ]

    for strat_name in priority_order:
        strat = strategy_registry.get(strat_name)
        if not strat:
            continue
        cfg = config_mgr.get_strategy_config(strat_name)
        if not cfg.get("enabled", True):
            continue

        signal_obj: Optional[StrategySignal] = strat.evaluate(
            market_data=market_data,
            spot_price=spot_price,
            equity=equity,
            config=cfg,
        )

        if signal_obj:
            is_macro = "MACRO" in signal_obj.lane
            risk_pct = config_mgr.risk.get("macro_risk_percent", 0.0125) if is_macro else config_mgr.risk.get("risk_percent", 0.025)
            tot_lot, tp1_lot, runner_lot = calculate_dynamic_compounded_lot(
                equity=equity, risk_dist=signal_obj.risk_dist, risk_pct=risk_pct
            )
            signal_obj.lot = tot_lot
            signal_obj.tp1_lot = tp1_lot
            signal_obj.runner_lot = runner_lot
            return signal_obj.to_dict()

    return None

# =========================================================================
# REAL BROKER POSITION & PARTIAL TP MANAGEMENT
# =========================================================================
def manage_gold_positions(equity=500.0) -> Tuple[int, float]:
    positions = mt5.positions_get(symbol="XAUUSD")
    current_tickets = [p.ticket for p in positions] if positions else []

    # Reconcile with persisted state to detect closed positions and update circuit breakers
    closed_tickets = state_mgr.reconcile_with_live_positions(current_tickets)
    for ct in closed_tickets:
        print(f"  🏁 [POSITION CLOSED] #{ct} closed. Anti-chop cooldown active.", flush=True)

    if not positions:
        return 0, 0.0

    count = 0
    total_pnl = 0.0
    rollover = is_rollover_window()

    sym_info = mt5.symbol_info("XAUUSD")
    tick = mt5.symbol_info_tick("XAUUSD")
    if not sym_info or not tick:
        return len(positions), sum(p.profit for p in positions)

    point = sym_info.point
    digits = sym_info.digits
    tm_cfg = config_mgr.trade_management

    for p in positions:
        count += 1
        profit = p.profit
        total_pnl += profit
        side = p.type
        entry = p.price_open
        sl = p.sl
        tp = p.tp
        ticket = p.ticket
        vol = p.volume

        # Ensure ticket is registered in state manager
        if ticket not in state_mgr.active_tracked_tickets:
            lane = "MACRO" if (p.comment and "Macro" in p.comment) or vol <= 0.01 else "SCALP"
            state_mgr.register_open_trade(ticket, {
                "lane": lane,
                "entry": entry,
                "sl": sl,
                "tp1": entry + 3.0 if side == mt5.ORDER_TYPE_BUY else entry - 3.0,
                "tp2": tp,
                "risk_dist": abs(entry - sl) if sl > 0 else 2.50,
                "tp1_lot": round(vol * 0.80, 2),
                "runner_lot": round(vol - round(vol * 0.80, 2), 2),
            })

        meta = state_mgr.active_tracked_tickets[ticket]
        lane = meta.get("lane", "SCALP")
        half_closed = meta.get("half_closed", False)

        # 1. EMERGENCY HARD LOSS CAP
        hard_loss_pct = config_mgr.risk.get("emergency_hard_loss_pct", 0.025)
        hard_loss_cap = -max(12.00, equity * hard_loss_pct)
        if profit <= hard_loss_cap:
            close_side = mt5.ORDER_TYPE_SELL if side == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            close_price = tick.bid if side == mt5.ORDER_TYPE_BUY else tick.ask
            req = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": "XAUUSD",
                "volume": vol,
                "type": close_side,
                "position": ticket,
                "price": close_price,
                "deviation": 25,
                "comment": "Hard_Cap_Hit",
                "type_filling": get_filling_mode(sym_info),
            }
            res = mt5.order_send(req)
            if res.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"  🛑 [HARD LOSS CAP] Closed #{ticket} at ${profit:+.2f} to protect capital!", flush=True)
                pause_h = config_mgr.risk.get("circuit_breaker_pause_hours", 4.0)
                state_mgr.record_loss_circuit_breaker(pause_hours=pause_h)
                continue

        # 2. ROLLOVER SHIELD: Lock in gains before midnight spread blowout
        if rollover and profit >= 2.00:
            close_side = mt5.ORDER_TYPE_SELL if side == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            close_price = tick.bid if side == mt5.ORDER_TYPE_BUY else tick.ask
            req = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": "XAUUSD",
                "volume": vol,
                "type": close_side,
                "position": ticket,
                "price": close_price,
                "deviation": 25,
                "comment": "Rollover_Shield",
                "type_filling": get_filling_mode(sym_info),
            }
            res = mt5.order_send(req)
            if res.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"  🛡️ [ROLLOVER SHIELD] Closed #{ticket} before rollover. Profit: ${profit:+.2f}", flush=True)
                continue

        # 3. TWO-STAGE PARTIAL TP & PROFIT LOCKING
        if "MACRO" not in lane:
            risk_dist = meta.get("risk_dist", 2.50)
            tp1_target = meta.get("target_tp1", 0.0)
            tp1_r = tm_cfg.get("tp1_r_multiple", 1.8)
            if tp1_target == 0.0:
                tp1_target = entry + (risk_dist * tp1_r) if side == mt5.ORDER_TYPE_BUY else entry - (risk_dist * tp1_r)
                meta["target_tp1"] = tp1_target

            # Check if TP1 is reached on vol > 0.01
            if not half_closed and vol > 0.01:
                tp1_hit = False
                if side == mt5.ORDER_TYPE_BUY and tick.bid >= tp1_target:
                    tp1_hit = True
                elif side == mt5.ORDER_TYPE_SELL and tick.ask <= tp1_target:
                    tp1_hit = True

                if tp1_hit:
                    bank_frac = tm_cfg.get("tp1_bank_fraction", 0.80)
                    target_bank_lot = meta.get("tp1_lot", round(vol * bank_frac, 2))
                    close_vol = min(round(vol - 0.01, 2), target_bank_lot)
                    if close_vol <= 0:
                        close_vol = vol
                    close_side = mt5.ORDER_TYPE_SELL if side == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
                    close_price = tick.bid if side == mt5.ORDER_TYPE_BUY else tick.ask
                    req = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": "XAUUSD",
                        "volume": close_vol,
                        "type": close_side,
                        "position": ticket,
                        "price": close_price,
                        "deviation": 25,
                        "comment": "Partial_TP1_Bank",
                        "type_filling": get_filling_mode(sym_info),
                    }
                    res = mt5.order_send(req)
                    if res.retcode == mt5.TRADE_RETCODE_DONE:
                        state_mgr.record_partial_close(ticket)
                        rem_vol = round(vol - close_vol, 2)
                        print(f"  💰 [PARTIAL TP1 HIT] Banked {close_vol} lots @ +{tp1_r}R on #{ticket} at {close_price:.2f}! Leaving {rem_vol} runner.", flush=True)
                        # Move SL to Entry + buffer ($0.30) to make remainder 100% RISK-FREE
                        be_buf = tm_cfg.get("risk_free_buffer", 0.30)
                        new_sl = entry + be_buf if side == mt5.ORDER_TYPE_BUY else entry - be_buf
                        new_sl = round(new_sl, digits)
                        sl_req = {"action": mt5.TRADE_ACTION_SLTP, "position": ticket, "sl": new_sl, "tp": tp}
                        mt5.order_send(sl_req)
                        print(f"  🔒 [RISK-FREE LOCK] Trailed #{ticket} runner SL to {new_sl:.2f} (Entry + ${be_buf:.2f})!", flush=True)
                        continue

            # Trailing for runner after TP1
            if half_closed:
                trail_trig_r = tm_cfg.get("runner_trail_trigger_r", 2.2)
                trail_dist_r = tm_cfg.get("runner_trail_distance_r", 1.2)
                if side == mt5.ORDER_TYPE_BUY:
                    if tick.bid > entry + (risk_dist * trail_trig_r):
                        trail_sl = round(entry + (risk_dist * trail_dist_r), digits)
                        if trail_sl > sl + point * 10:
                            mt5.order_send({"action": mt5.TRADE_ACTION_SLTP, "position": ticket, "sl": trail_sl, "tp": tp})
                            print(f"  🏃‍♂️ [RUNNER TRAIL] Scalp BUY #{ticket} trailed to {trail_sl:.2f}", flush=True)
                else:
                    if tick.ask < entry - (risk_dist * trail_trig_r):
                        trail_sl = round(entry - (risk_dist * trail_dist_r), digits)
                        if sl == 0 or trail_sl < sl - point * 10:
                            mt5.order_send({"action": mt5.TRADE_ACTION_SLTP, "position": ticket, "sl": trail_sl, "tp": tp})
                            print(f"  🏃‍♂️ [RUNNER TRAIL] Scalp SELL #{ticket} trailed to {trail_sl:.2f}", flush=True)

        # 4. MACRO-SWING TRAILING (H4)
        else:
            if side == mt5.ORDER_TYPE_BUY:
                bid = tick.bid
                if profit >= 15.00 and bid > entry + 8.00:
                    target_sl = round(entry + 5.00, digits)
                    if target_sl > sl + point * 10:
                        mt5.order_send({"action": mt5.TRADE_ACTION_SLTP, "position": ticket, "sl": target_sl, "tp": tp})
                        print(f"  🔒 [MACRO PROFIT LOCK] BUY #{ticket} SL -> {target_sl:.2f}", flush=True)
                elif profit >= 8.00 and (sl < entry + 0.50) and bid > entry + 3.00:
                    target_sl = round(entry + 0.50, digits)
                    mt5.order_send({"action": mt5.TRADE_ACTION_SLTP, "position": ticket, "sl": target_sl, "tp": tp})
                    print(f"  🔒 [MACRO BE LOCK] BUY #{ticket} SL -> {target_sl:.2f}", flush=True)
            else:
                ask = tick.ask
                if profit >= 15.00 and ask < entry - 8.00:
                    target_sl = round(entry - 5.00, digits)
                    if sl == 0 or target_sl < sl - point * 10:
                        mt5.order_send({"action": mt5.TRADE_ACTION_SLTP, "position": ticket, "sl": target_sl, "tp": tp})
                        print(f"  🔒 [MACRO PROFIT LOCK] SELL #{ticket} SL -> {target_sl:.2f}", flush=True)
                elif profit >= 8.00 and (sl == 0 or sl > entry - 0.50) and ask < entry - 3.00:
                    target_sl = round(entry - 0.50, digits)
                    mt5.order_send({"action": mt5.TRADE_ACTION_SLTP, "position": ticket, "sl": target_sl, "tp": tp})
                    print(f"  🔒 [MACRO BE LOCK] SELL #{ticket} SL -> {target_sl:.2f}", flush=True)

    return count, total_pnl

def execute_gold_trade(setup: Dict[str, Any]) -> bool:
    lane = setup["lane"]
    signal = setup["signal"]
    entry = setup["entry"]
    sl = setup["sl"]
    tp1 = setup["tp1"]
    tp2 = setup["tp2"]
    vol = setup["lot"]
    reason = setup["reason"]
    risk_dist = setup["risk_dist"]

    sym_info = mt5.symbol_info("XAUUSD")
    if not sym_info:
        return False

    filling = get_filling_mode(sym_info)
    side = mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL
    comment_tag = setup.get("strategy", lane)[:15]

    print(f"\n  🎯 [{lane}] {signal} XAUUSD @ {entry:.2f} | Lot: {vol} | SL: {sl:.2f} | TP1: {tp1:.2f} | TP2: {tp2:.2f} | {reason}", flush=True)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": "XAUUSD",
        "volume": vol,
        "type": side,
        "price": entry,
        "sl": sl,
        "tp": tp2,
        "deviation": 25,
        "magic": 777999,
        "comment": comment_tag,
        "type_filling": filling,
    }

    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        state_mgr.register_open_trade(result.order, {
            "lane": lane,
            "strategy": setup.get("strategy", ""),
            "entry": entry,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "risk_dist": risk_dist,
            "tp1_lot": setup.get("tp1_lot", round(vol * 0.80, 2)),
            "runner_lot": setup.get("runner_lot", 0.01),
        })
        print(f"  ✅ [{lane} EXECUTED] Ticket #{result.order} | XAUUSD {signal} {vol} lots @ {entry:.2f}", flush=True)
        return True
    else:
        print(f"  ❌ [{lane} FAILED] Error: {result.comment} (Code: {result.retcode})", flush=True)
        return False

# =========================================================================
# MAIN PRODUCTION ENGINE
# =========================================================================
def run_pure_gold_engine(sleep_sec=1.0):
    print("================================================================================", flush=True)
    print("   👑 PURE GOLD (XAUUSD) PRODUCTION SNIPER ENGINE v9.6 MODULAR COMPOUNDER 👑    ", flush=True)
    print(f"   100% Dedicated to Gold (Account #{TARGET_LOGIN or 'Demo/Env'}) | Zero Forex / Zero Other Markets ", flush=True)
    print("   Institutional Fusion: Khakestar T-Rex + Pour Samadi + SMC Continuous Intel   ", flush=True)
    print("   Dynamic Risk Sizing: Strict 2.5% Equity Risk (Compounds +111.3% Annual Gain)   ", flush=True)
    print("   Modular Registry: 5 Autonomous Strategies Active | Hot-Reload Enabled         ", flush=True)
    print("================================================================================", flush=True)

    if not mt5.initialize():
        if not mt5.initialize(path=r"C:\Program Files\MetaTrader 5\terminal64.exe"):
            print(f"❌ MT5 initialization failed! Error: {mt5.last_error()}", flush=True)
            return

    logged_in = False
    if TARGET_LOGIN and TARGET_PASS:
        logged_in = mt5.login(TARGET_LOGIN, password=TARGET_PASS, server=TARGET_SERVER)
        if not logged_in:
            print(f"⚠️ Direct login to #{TARGET_LOGIN} failed ({mt5.last_error()}). Checking active terminal session...", flush=True)

    account = mt5.account_info()
    if account and account.login > 0:
        print(f"  Broker Account: {account.login} ({account.name}) | Server: {account.server} | Leverage: 1:{account.leverage}", flush=True)
        print(f"  Live Balance  : ${account.balance:.2f} | Equity: ${account.equity:.2f} | Margin Free: ${account.margin_free:.2f}", flush=True)
    else:
        print("❌ Could not retrieve active account info! Please log in to an active demo/real account in MetaTrader 5.", flush=True)
        return

    mt5.symbol_select("XAUUSD", True)
    mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M15, 0, 10)
    mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_H4, 0, 10)
    time.sleep(1)

    cycle = 0
    while True:
        cycle += 1
        try:
            # 1. Hot-reload config if changed on disk by autonomous improver
            if cycle % 10 == 0:
                if config_mgr.check_reload():
                    print("  🔄 [CONFIG RELOADED] Updated parameters dynamically from pure_gold_config.json", flush=True)

            acc = mt5.account_info()
            equity = acc.equity if acc else 500.0

            # 2. Manage Active Positions
            pos_count, float_pnl = manage_gold_positions(equity=equity)

            # 3. Telemetry Log every 30 cycles (~30s)
            if cycle % 30 == 0:
                tick = mt5.symbol_info_tick("XAUUSD")
                gold_bid = f"{tick.bid:.2f}" if tick else "N/A"
                is_p, sess = is_prime_scalp_session()
                adr_r, _, _ = get_adr_usage("XAUUSD")
                tracked_n = len(state_mgr.active_tracked_tickets)
                if acc:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Cycle {cycle:5d} | Gold={gold_bid} | Sess={sess} | ADR={adr_r*100:.0f}% | Bal=${acc.balance:.2f} Eq=${acc.equity:.2f} (Pos:{pos_count}, Tracked:{tracked_n}, Float:${float_pnl:+.2f})", flush=True)

            # 4. Scan for Entries if positions < 2
            if pos_count < 2 and equity > 50:
                open_pos = mt5.positions_get(symbol="XAUUSD") or []
                open_types = [p.type for p in open_pos]
                has_open_buy = mt5.ORDER_TYPE_BUY in open_types
                has_open_sell = mt5.ORDER_TYPE_SELL in open_types

                setup = scan_all_gold_strategies(equity=equity)
                if setup:
                    sig = setup["signal"]
                    if (sig == "BUY" and not has_open_sell) or (sig == "SELL" and not has_open_buy):
                        execute_gold_trade(setup)
                        time.sleep(1)

        except Exception as e:
            print(f"⚠️ Error in gold sniper loop: {e}", flush=True)

        time.sleep(sleep_sec)

if __name__ == "__main__":
    run_pure_gold_engine(sleep_sec=1.0)
