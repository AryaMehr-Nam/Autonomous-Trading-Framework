import os
import re
import sys
import json
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from datetime import datetime

# Optional: Try to import MT5, fallback if not available
try:
    if sys.platform == "win32":
        import MetaTrader5 as mt5
        HAS_MT5 = True
    else:
        HAS_MT5 = False
except ImportError:
    HAS_MT5 = False

app = FastAPI(title="TradeAgent Dashboard API")

# Mount the static directory to serve HTML/CSS/JS
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.on_event("startup")
async def startup_event():
    if HAS_MT5:
        if not mt5.initialize():
            print("Dashboard: MT5 initialization failed!")
        else:
            print("Dashboard: MT5 initialized successfully.")

@app.on_event("shutdown")
async def shutdown_event():
    if HAS_MT5:
        mt5.shutdown()

@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")

@app.get("/api/account")
async def get_account():
    if not HAS_MT5:
        return {"balance": 0, "equity": 0, "margin": 0, "margin_free": 0, "margin_level": 0, "currency": "USD", "leverage": 1}

    acc = mt5.account_info()
    if acc is None:
        return {"error": "Failed to get account info"}

    return {
        "balance": acc.balance,
        "equity": acc.equity,
        "margin": acc.margin,
        "margin_free": acc.margin_free,
        "margin_level": acc.margin_level,
        "currency": acc.currency,
        "leverage": acc.leverage
    }

@app.get("/api/positions")
async def get_positions():
    if not HAS_MT5:
        return {"positions": []}

    positions = mt5.positions_get()
    orders = mt5.orders_get()

    pos_list = []
    if positions:
        for p in positions:
            pos_list.append({
                "ticket": p.ticket,
                "symbol": p.symbol,
                "type": "BUY" if p.type == 0 else "SELL",
                "volume": p.volume,
                "open_price": p.price_open,
                "current_price": p.price_current,
                "profit": p.profit,
                "is_pending": False
            })

    if orders:
        for o in orders:
            side = "BUY LIMIT" if o.type in [2, 4] else "SELL LIMIT"
            pos_list.append({
                "ticket": o.ticket,
                "symbol": o.symbol,
                "type": side,
                "volume": o.volume_initial,
                "open_price": o.price_open,
                "current_price": o.price_current,
                "profit": 0.0,
                "is_pending": True
            })

    return {"positions": pos_list}

@app.get("/api/sentiment")
async def get_sentiment():
    """Reads the latest sentiment score directly from the central log file."""
    log_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "tradeagent.log")

    score = 0.0
    latest_update = "Awaiting Data"

    if os.path.exists(log_file):
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()[-500:]

            for line in reversed(lines):
                if "[SENTIMENT] Market Score Updated:" in line:
                    match = re.search(r'Updated:\s*([+-]?\d+\.\d+)', line)
                    if match:
                        score = float(match.group(1))
                        time_match = re.match(r'^(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})', line)
                        if time_match:
                            latest_update = time_match.group(1)
                        break
        except Exception as e:
            print(f"Error reading log: {e}")

    if score > 0.3: tone = "BULLISH"
    elif score < -0.3: tone = "BEARISH"
    else: tone = "NEUTRAL"

    return {
        "score": score,
        "tone": tone,
        "last_update": latest_update
    }

@app.get("/api/strategies")
async def get_strategies():
    """Reads the optimized strategy files and live bot log to display intelligence state."""
    strategies = []
    base_dir = os.path.dirname(os.path.dirname(__file__))
    strat_dir = os.path.join(base_dir, "data_store", "strategies")

    if os.path.exists(strat_dir):
        for file in sorted(os.listdir(strat_dir)):
            if file.startswith("optimized_") and file.endswith(".json") and file != "optimized_strategies.json":
                try:
                    with open(os.path.join(strat_dir, file), "r") as f:
                        data = json.load(f)
                        strategies.append(data)
                except Exception as e:
                    print(f"Failed to read {file}: {e}")

    # Read bot log for recent signal activity
    log_file = os.path.join(base_dir, "logs", "tradeagent.log")
    recent_signals = []
    if os.path.exists(log_file):
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()[-1000:]
            for line in reversed(lines):
                if "APPROVED by RiskManager" in line or "Rejecting signal" in line or "Starting analysis cycle" in line:
                    recent_signals.append(line.strip())
                if len(recent_signals) >= 10:
                    break
        except Exception:
            pass

    return {
        "engine": "Optuna Walk-Forward AI",
        "status": "Active",
        "strategies": strategies,
        "recent_activity": recent_signals
    }

@app.get("/api/bot_status")
async def get_bot_status():
    """Read the bot's recent log to show last N lines of activity."""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    log_file = os.path.join(base_dir, "logs", "tradeagent.log")

    lines_out = []
    if os.path.exists(log_file):
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                raw = f.readlines()[-50:]
            for l in raw:
                lines_out.append(l.strip())
        except Exception as e:
            lines_out = [f"Error reading log: {e}"]
    else:
        lines_out = ["Log file not found — bot may not be running."]

    return {"log_lines": lines_out}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
