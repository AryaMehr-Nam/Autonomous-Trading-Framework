import asyncio
import os
from dotenv import load_dotenv

# Load .env file before anything else
load_dotenv(dotenv_path="config/.env")

from execution.brokers.mt5_executor import MT5Executor

async def show_results():
    exec = MT5Executor()
    connected = await exec.connect()
    if not connected:
        print("Failed to connect to MT5.")
        return
        
    acc = await exec.get_account_info()
    print(f"\n--- Account Info ---")
    print(f"Balance: {acc.balance} {acc.currency}")
    print(f"Equity: {acc.equity} {acc.currency}")
    print(f"Unrealized PnL: {acc.unrealized_pnl} {acc.currency}")
    
    positions = await exec.get_open_positions()
    print(f"\n--- Open Positions ({len(positions)}) ---")
    for p in positions:
        print(f"[{p.symbol}] {p.side.name} | Qty: {p.quantity} | Entry: {p.entry_price} | Current: {p.current_price} | PnL: {(p.current_price - p.entry_price) * p.quantity if p.side.name == 'BUY' else (p.entry_price - p.current_price) * p.quantity} (raw estimation)")

    await exec.disconnect()

if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(show_results())
